import base64
import io
import json
import numpy as np
import qiskit
import requests
import secrets
import time
import uuid
import webbrowser

from IPython.display import display, HTML
from keycloak import KeycloakOpenID
from urllib.parse import urlencode

from qiskit import qpy
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Pauli, PauliList, SparsePauliOp
from qiskit.quantum_info.operators.linear_op import LinearOp
from qiskit_nature.second_q.hamiltonians.lattices import (
    KagomeLattice,
    Lattice,
    LineLattice,
    HexagonalLattice,
    HyperCubicLattice,
    SquareLattice,
    TriangularLattice,
)
from qiskit_nature.second_q.hamiltonians.lattices.boundary_condition import (
    BoundaryCondition,
)
from qiskit_optimization import QuadraticProgram


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, BoundaryCondition):
            # Convert enum to string
            return str(obj)
        return super().default(obj)


def all_numbers(lst):
    return all(isinstance(x, (int, float, complex)) for x in lst)


def serialize_circuit(circuit):
    buffer = io.BytesIO()
    qpy.dump(circuit, buffer)
    qpy_binary_data = buffer.getvalue()
    base64_encoded_circuit = base64.b64encode(qpy_binary_data).decode("utf-8")
    return base64_encoded_circuit


class AuthenticationFailure(Exception):
    def __init__(self, message):
        self.message = message


class AuthorizationFailure(Exception):
    def __init__(self, message):
        self.message = message


class Job:
    def __init__(self, job_id):
        self._job_id = job_id

    def id(self):
        return self._job_id


class WorkflowJob:
    def __init__(self, job_id):
        self._job_id = job_id

    def id(self):
        return self._job_id


class InputData:
    def __init__(self, label=None, content=None):
        self.data = {}
        if label:
            self.add_data(label, content)

    def __str__(self):
        return json.dumps(self.data, indent=2, cls=CustomJSONEncoder)

    def add_data(self, label, content):
        self.check_label(label, self.data)
        try:
            if label == "operator":
                operator = content
                self.validate_operator(operator)
                coeffs = None
                if type(content) == tuple:
                    operator, coeffs = content
                sparse_pauli_operator = self.to_sparse_pauli_operator(
                    operator, coeffs=coeffs
                )
                pauli_terms, coefficients = self.serialize_sparse_pauli_operator(
                    sparse_pauli_operator
                )
                self.data["operator"] = {
                    "pauli-terms": pauli_terms,
                    "coefficients": coefficients,
                    "operator-string-representation": str(operator),
                }
            elif label == "pub":
                content = self.validate_and_serialize_pub(content)
                if not "pubs" in self.data.keys():
                    self.data["pubs"] = []
                self.data["pubs"].append(content)
            elif label == "molecule-info":
                self.validate_molecule_info(content)
                self.data[label] = content
            elif label == "lattice":
                self.data[label] = self.lattice_to_dict(content)
            elif label == "ising-model":
                self.validate_ising_model(content)
                self.data[label] = content
            elif label == "training-data":
                self.validate_training_data(content)
                self.data[label] = content
            elif label == "inference-data":
                self.validate_inference_data(content)
                self.data[label] = content
            elif label == "quadratic-program":
                self.validate_quadratic_program(content)
                lp_string = content.export_as_lp_string()
                self.data[label] = lp_string
            else:
                self.data[label] = content
        except (OverflowError, TypeError, ValueError):
            raise Exception("Input data content must be JSON serializable.")

    def check_label(self, label, data):
        if type(label) != str:
            raise Exception("Input data label must be string.")
        if label not in [
            "ansatz-parameters",
            "inference-data",
            "ising-model",
            "lattice",
            "lp-model",
            "max-iterations",
            "molecule-info",
            "operator",
            "pub",
            "quadratic-program",
            "training-data",
        ]:
            raise Exception(
                f"Input data of type {label} is not supported. Please choose one of the following options: 'ansatz-parameters', 'inference-data' 'ising-model', 'lattice', 'lp-model', 'molecule-info', 'operator', 'pub', 'training-data'."
            )
        if label != "pub" and label in data.keys():
            raise Exception(
                f"An input data item of type '{label}' has already been added to the job input data. Multiple data items of same category are allowed only for PUBs."
            )

    def lattice_to_dict(self, lattice):
        if isinstance(lattice, LineLattice):
            lattice_data = {
                "type": "LineLattice",
                "num_nodes": lattice.num_nodes,
                "boundary_condition": lattice.boundary_condition[0].name,
                "edge_parameter": lattice.edge_parameter,
                "onsite_parameter": lattice.onsite_parameter,
            }
            return lattice_data
        elif isinstance(lattice, TriangularLattice):
            lattice_data = {
                "type": "TriangularLattice",
                "rows": lattice.rows,
                "cols": lattice.cols,
                "boundary_condition": lattice.boundary_condition.name,
                "edge_parameter": lattice.edge_parameter,
                "onsite_parameter": lattice.onsite_parameter,
            }
            return lattice_data
        elif isinstance(lattice, (SquareLattice, KagomeLattice, HyperCubicLattice)):
            lattice_data = {
                "type": type(lattice).__name__,
                "rows": lattice.rows,
                "cols": lattice.cols,
                "boundary_condition": [
                    bc.name
                    for bc in (
                        lattice.boundary_condition
                        if isinstance(lattice.boundary_condition, tuple)
                        else (lattice.boundary_condition,)
                    )
                ],
                "edge_parameter": lattice.edge_parameter,
                "onsite_parameter": lattice.onsite_parameter,
            }
            return lattice_data
        elif isinstance(lattice, HexagonalLattice):
            lattice_data = {
                "type": "HexagonalLattice",
                "rows": lattice._rows,
                "cols": lattice._cols,
                "edge_parameter": lattice.edge_parameter,
                "onsite_parameter": lattice.onsite_parameter,
            }
            return lattice_data
        elif isinstance(lattice, Lattice):
            graph = lattice.graph
            nodes = list(graph.node_indexes())
            edges = [
                {"source": edge[0], "target": edge[1], "weight": edge[2]}
                for edge in graph.weighted_edge_list()
            ]
            lattice_data = {
                "type": "Lattice",
                "nodes": nodes,
                "edges": edges,
                "num_nodes": lattice.num_nodes,
            }
            return lattice_data
        else:
            raise Exception(
                "This input lattice object is not supported. Please use an object of the following types: Lattice, LineLattice, TriangularLattice, SquareLattice, KagomeLattice, HyperCubicLattice or HexagonalLattice. All of them are available in the qiskit_nature library."
            )

    def validate_molecule_info(self, molecule_info):
        if not isinstance(molecule_info["symbols"], list):
            raise Exception("The 'symbols' must be a list of nuclei.")
        if not isinstance(molecule_info["coords"], list):
            raise Exception(
                "The 'coords' must be a list of tuples representing the x, y, z position of each nuclei."
            )
        if "mutiplicity" in molecule_info and not isinstance(
            molecule_info["mutiplicity"], int
        ):
            raise Exception("The 'multiplicity' must be an integer.")
        if "charge" in molecule_info and not isinstance(molecule_info["charge"], int):
            raise Exception("The 'charge' must be an integer.")
        if (
            "units" in molecule_info
            and molecule_info["units"].lower() != "angstrom"
            and molecule_info["units"].lower() != "bohr"
        ):
            raise Exception("The 'units' must be either 'Angstrom' or 'Bohr'.")
        if "masses" in molecule_info and not all(
            isinstance(m, (int, float)) for m in molecule_info["masses"]
        ):
            raise Exception(
                "The 'masses' must be a list of floats, one for each nucleus in the molecule."
            )

    def validate_ising_model(self, ising_model):
        if not isinstance(ising_model, dict):
            raise Exception("The 'ising_model' must be a dictionary.")

        for key in ising_model.keys():
            if key not in ["h", "J"]:
                raise Exception(
                    "The 'ising_model' dictionary can only contain the keys: 'h' and 'J'."
                )

        if "h" in ising_model:
            if not isinstance(ising_model["h"], list):
                raise Exception("The 'h' field must be a list of numeric values.")
            if not all(isinstance(h, (int, float)) for h in ising_model["h"]):
                raise Exception("Each element in 'h' must be an int or float.")

        if "J" in ising_model:
            if not isinstance(ising_model["J"], list):
                raise Exception("The 'J' field must be a list of dictionaries.")
            for interaction in ising_model["J"]:
                if not isinstance(interaction, dict):
                    raise Exception(
                        "Each item in 'J' must be a dictionary with 'pair' and 'value' keys."
                    )
                if "pair" not in interaction or "value" not in interaction:
                    raise Exception(
                        "Each item in 'J' must contain 'pair' and 'value' keys."
                    )
                if (
                    not isinstance(interaction["pair"], list)
                    or len(interaction["pair"]) != 2
                    or not all(isinstance(i, int) for i in interaction["pair"])
                ):
                    raise Exception("'pair' must be a list of two integers.")
                if not isinstance(interaction["value"], (int, float)):
                    raise Exception("'value' must be a numeric type (int or float).")

    def validate_training_data(self, training_data):
        vector_size = None
        if not isinstance(training_data, list):
            raise Exception("The 'training_data' must be a list of dictionaries.")
        for data in training_data:
            if not isinstance(data, dict):
                raise Exception("The 'training_data' must be a list of dictionaries.")
            if not "data-point" in data:
                raise Exception(
                    "Each dictionary in the list 'training_data' must contain a 'data-point' key."
                )
            vector = data["data-point"]
            data_tags = data["data-tags"] if "data-tags" in data else None
            if not isinstance(vector, list):
                raise Exception(
                    "The 'data-point' value must be a list of numeric values."
                )
            if data_tags is not None and not isinstance(data_tags, list):
                raise Exception(
                    "The optional 'data-tags' value must be a list of strings."
                )
            if not all(isinstance(item, (int, float)) for item in vector):
                raise Exception(
                    "The 'data-point' value must be a list of numeric values (int or float)."
                )
            if vector_size is None:
                vector_size = len(vector)
            if len(vector) != vector_size:
                raise Exception(
                    "All 'data-point' vectors in training data entries must have the same length."
                )
            if data_tags is not None and len(data_tags) != vector_size:
                raise Exception(
                    "If provided, the 'data-tags' list must have the same length as the 'data-point' vector."
                )
            if not "label" in data:
                raise Exception(
                    "Each dictionary in the list of training data points must contain a 'label' key."
                )
            label = data["label"]
            if not isinstance(label, (int, float)):
                raise Exception(
                    "The 'label' value must be a numeric type (int or float)."
                )

    def validate_inference_data(self, inference_data):
        vector_size = None
        if not isinstance(inference_data, list):
            raise Exception("The 'inference_data' must be a list of dictionaries.")
        for data in inference_data:
            if not isinstance(data, dict):
                raise Exception("The 'inference_data' must be a list of dictionaries.")
            if not "data-point" in data:
                raise Exception(
                    "Each dictionary in the list of inference data points must contain a 'data-point' key."
                )
            vector = data["data-point"]
            data_tags = data["data-tags"] if "data-tags" in data else None
            if not isinstance(vector, list):
                raise Exception(
                    "The 'data-point' value must be a list of numeric values."
                )
            if data_tags is not None and not isinstance(data_tags, list):
                raise Exception(
                    "The optional 'data-tags' value must be a list of strings."
                )
            if not all(isinstance(item, (int, float)) for item in vector):
                raise Exception(
                    "The 'data-point' value must be a list of numeric values (int or float)."
                )
            if vector_size is None:
                vector_size = len(vector)
            if len(vector) != vector_size:
                raise Exception(
                    "All 'data-point' vectors in inference data entries must have the same length."
                )
            if data_tags is not None and len(data_tags) != vector_size:
                raise Exception(
                    "If provided, the 'data-tags' list must have the same length as the 'data-point' vector."
                )

    def validate_quadratic_program(self, qp):
        if not isinstance(qp, QuadraticProgram):
            raise Exception(
                "The input object must be an instance of QuadraticProgram class from Qiskit Optimization module."
            )

    def validate_and_serialize_pub(self, pub):
        shots = None
        paramaters = None
        if type(pub) == QuantumCircuit:
            quantum_circuit = pub
        elif type(pub) != tuple:
            raise Exception(
                "A pub can be either a quantum circuit or a tuple containing a quantum circuit, optionally second a list of circuit parameters and optionally third a number of shots."
            )
        elif len(pub) == 3:
            quantum_circuit, paramaters, shots = pub
        elif len(pub) == 2:
            quantum_circuit, paramaters = pub
        elif len(pub) == 1:
            quantum_circuit = pub[0]
        else:
            raise Exception(
                "A pub can be a tuple with at most 3 elements: a quantum circuit, a list of circuit paramaters and a number of shots."
            )
        if shots is not None and type(shots) != int:
            raise Exception(
                "The 'shots' setting in a PUB must be an integer and be positioned as the third element of a tuple specifying a PUB."
            )
        if paramaters is not None and type(paramaters) != list:
            raise Exception(
                "The 'paramaters' in a PUB must be a list of numbers and be positioned as the second element of a tuple specifying a PUB."
            )
        if quantum_circuit.num_parameters == 0 and (
            paramaters is not None and len(paramaters) != 0
        ):
            raise Exception(
                "A circuit with zero parameters must have 'paramaters' argument 'None' or an empty list."
            )
        elif paramaters is not None and quantum_circuit.num_parameters != len(
            paramaters
        ):
            raise Exception(
                f"The number of paramaters for a quantum circuit {quantum_circuit.num_parameters} is different from the length {len(paramaters)} of the list of aruguments."
            )
        if paramaters is not None and not all(
            isinstance(item, (int, float)) for item in paramaters
        ):
            raise Exception(
                "The 'paramaters' setting in a PUB must be a list of numbers."
            )

        return (serialize_circuit(quantum_circuit), paramaters, shots)

    def validate_operator(self, operator):
        if (
            not isinstance(operator, Operator)
            and not isinstance(operator, Pauli)
            and not isinstance(operator, SparsePauliOp)
            and not (
                isinstance(operator, tuple)
                and isinstance(operator[0], PauliList)
                and isinstance(operator[1], list)
                and (operator[1] and not all_numbers(operator[1]))
            )
        ):
            raise Exception(
                "The operator must be an instance of the Operator, Pauli, SparsePauliOp class or a tuple containing a PauliList and a possible empty list of numeric coefficents."
            )

        if isinstance(operator, Operator):
            matrix = operator.data
            if not np.allclose(matrix, matrix.conj().T):
                print("WARNING: The operator you supplied is not Hermitian!")

        if (
            isinstance(operator, tuple)
            and isinstance(operator[0], PauliList)
            and isinstance(operator[1], list)
        ):
            pauli_list = operator[0]
            coefficients = operator[1]
            if (
                coefficients is not None
                and len(coefficients) > 0
                and len(pauli_list) != len(coefficients)
            ):
                raise Exception(
                    "The number of Pauli terms in the Pauli list must match the number of coefficients or list of coefficients must be empty."
                )

    def to_sparse_pauli_operator(self, operator, coeffs=None):
        if isinstance(operator, SparsePauliOp):
            return operator

        elif isinstance(operator, Pauli):
            return SparsePauliOp(operator)

        elif isinstance(operator, PauliList):
            if coeffs is not None and len(coeffs) > 0 and len(coeffs) != len(operator):
                raise ValueError(
                    "Number of coefficients must match number of Pauli operators in PauliList"
                )

            coefficients = (
                coeffs
                if (coeffs is not None and len(coeffs) > 0)
                else [1.0] * len(operator)
            )
            pauli_strings = [str(pauli) for pauli in operator]
            return SparsePauliOp(pauli_strings, coeffs=coefficients)

        elif isinstance(operator, Operator):
            return SparsePauliOp.from_operator(operator)

    def serialize_sparse_pauli_operator(self, sparse_op):
        if not isinstance(sparse_op, SparsePauliOp):
            raise ValueError("Input must be a SparsePauliOp")

        pauli_data = sparse_op.to_list()
        pauli_terms = [term[0] for term in pauli_data]
        coefficients = [term[1] for term in pauli_data]
        return pauli_terms, coefficients


class QuantumFlowsProvider:

    _asp_net_port_dev = "5001"
    _keycloak_port = "8080"
    _client_id = "straful-client"
    _realm_name = "straful-realm"
    _provider_url_dev = "https://localhost"
    _provider_url_prod = "https://quantum-flows.transilvania-quantum.com"
    _keycloak_url_dev = "http://localhost"
    _keycloak_url_prod = "https://keycloak.transilvania-quantum.com"

    def __init__(self, use_https=True, debug=False):
        self._use_https = use_https
        self._debug = debug
        self._state = None
        self._access_token = None
        self._refresh_token = None
        self._token_expiration_time = None
        self._refresh_token_expiration_time = None
        self._asp_net_url = (
            f"{self._provider_url_dev}:{self._asp_net_port_dev}"
            if self._debug
            else f"{self._provider_url_prod}"
        )
        self._auth_call_back_url = f"{self._asp_net_url}/auth/callback"
        self._show_code_callback_url = f"{self._asp_net_url}/auth/showcode"
        self._keycloak_server_url = (
            f"{self._keycloak_url_dev}:{self._keycloak_port}"
            if self._debug
            else f"{self._keycloak_url_prod}"
        )

        if not self._is_server_online(self._keycloak_server_url):
            raise SystemExit(
                f"The service you are trying to access at: {self._asp_net_url}, is not responding. \
In case the service has been recently started please wait 5 minutes for it to become fully functional."
            )

        self._keycloak_openid = KeycloakOpenID(
            server_url=self._keycloak_server_url,
            client_id=self._client_id,
            realm_name=self._realm_name,
            verify=self._use_https,
        )

    def authenticate(self):
        try:
            self._access_token = None
            self._refresh_token = None
            self._token_expiration_time = None
            self._refresh_token_expiration_time = None
            self._store_state()
            auth_url = self._get_authentication_url()
            opened = webbrowser.open(auth_url)
            if not opened:
                display(
                    HTML(
                        f"<p>Please click to authenticate: "
                        f'<a href="{auth_url}" target="_blank">{auth_url}</a></p>'
                    )
                )
            auth_code = self._get_authentication_code()
            token_response = self._keycloak_openid.token(
                grant_type="authorization_code",
                code=auth_code,
                redirect_uri=self._auth_call_back_url,
            )
            self._access_token = token_response["access_token"]
            self._refresh_token = token_response["refresh_token"]
            self._token_expiration_time = (
                time.time() + token_response["expires_in"] - 5
            )  # seconds
            self._refresh_token_expiration_time = (
                time.time() + token_response["refresh_expires_in"] - 5
            )  # seconds
            print("Authentication successful.")
        except AuthenticationFailure as ex:
            print(ex.message)
        except AuthorizationFailure as ex:
            print(
                "Failed to authenticate with the quantum provider. Make sure you are using the correct Gmail account."
            )
            if self._debug:
                print("More details: ", ex.message)
        except Exception as ex:
            print("Failed to authenticate with the quantum provider.")
            if "Connection refused" in str(ex):
                print("The remote service does not respond. Please try again later.")
            if self._debug:
                print("Unexpected exception: ", ex)

    def submit_job(
        self, *, backend=None, circuit=None, circuits=None, shots=None, comments=""
    ):
        if not self._verify_user_is_authenticated():
            return
        if not backend:
            print("Please specify the backend name.")
            return
        if circuit is None and circuits is None:
            print(
                "An quantum circuit to be executed or a list of quantum circuits to be executed must be specified."
            )
            return
        if circuit is not None and circuits is not None:
            print(
                "You can use either 'circuit' or 'circuits' as input arguments but not both at the same time."
            )
            return
        if circuit is not None and not isinstance(circuit, QuantumCircuit):
            print(
                "The 'circuit' argument must be an instance of QuantumCircuit or deriving from it."
            )
            return
        if circuits is not None and (
            not isinstance(circuits, list)
            or not all(isinstance(circ, QuantumCircuit) for circ in circuits)
        ):
            print(
                "The 'circuits' argument must be a list of QuantumCircuit instances or objects deriving from QuantumCircuit."
            )
            return
        if shots is None:
            print("Please specify the number of shots.")
            return
        if not isinstance(shots, int):
            print("The number of shots must be specified as an integer number.")
            return
        try:
            if circuit is not None:
                job_data = {
                    "BackendName": backend,
                    "Circuit": serialize_circuit(circuit),
                    "Circuits": [],
                    "Shots": shots,
                    "Comments": comments,
                    "QiskitVersion": qiskit.__version__,
                }
            elif circuits is not None:
                job_data = {
                    "BackendName": backend,
                    "Circuit": None,
                    "Circuits": [serialize_circuit(circuit) for circuit in circuits],
                    "Shots": shots,
                    "Comments": comments,
                    "QiskitVersion": qiskit.__version__,
                }
            (status_code, result) = self._make_post_request(
                f"{self._asp_net_url}/api/job", job_data
            )
            if status_code == 201:
                return Job(result["id"])
            elif status_code == 401:
                print(
                    "You are not authorized to access this service. Please try to authenticate first and make sure you have signed on on our web-site with a Google email account."
                )
            elif "Under Maintenance" in result:
                print(
                    "The remote service is currently under maintenance. Please try again later."
                )
            else:
                print(
                    f"Job submission has failed with http status code: {status_code}. \nRemote server response: '{result}'"
                )
                return Job(None)
        except Exception as ex:
            print(str(ex))

    def submit_workflow_job(
        self,
        *,
        backend=None,
        shots=None,
        workflow_id=None,
        comments="",
        max_iterations=None,
        input_data=InputData(),
    ):
        if not self._verify_user_is_authenticated():
            return
        if not backend:
            print("Please specify a backend name.")
            return
        if not workflow_id:
            print("Please specify a workflow Id.")
            return
        if shots is not None and not isinstance(shots, int):
            print("The optional number of shots input argument must be an integer.")
            return
        if not self.is_valid_uuid(workflow_id):
            print("The specified workflow Id is not a valid GUID.")
            return
        if max_iterations is not None:
            if not isinstance(max_iterations, int) or max_iterations <= 0:
                print(
                    "The optional 'max-iterations' input argument must be a positive integer."
                )
                return
        try:
            input_data_labels = []
            input_data_items = []
            input_data_labels.append("backend")
            input_data_items.append(backend)
            input_data_labels.append("shots")
            input_data_items.append(str(shots))
            input_data_labels.append("max-iterations")
            input_data_items.append(str(max_iterations))
            for input_data_label in input_data.data.keys():
                input_data_labels.append(input_data_label)
                content = input_data.data[input_data_label]
                input_data_items.append(
                    json.dumps(content, indent=2, cls=CustomJSONEncoder)
                )
            job_data = {
                "BackendName": backend,
                "WorkflowId": workflow_id,
                "Shots": shots,
                "Comments": comments,
                "MaxIterations": max_iterations,
                "InputDataLabels": input_data_labels,
                "InputDataItems": input_data_items,
                "QiskitVersion": qiskit.__version__,
            }
            (status_code, result) = self._make_post_request(
                f"{self._asp_net_url}/api/workflow-job", job_data
            )
            if status_code == 201:
                return WorkflowJob(result["id"])
            elif status_code == 401:
                print(
                    "You are not authorized to access this service. Please try to authenticate first and make sure you have signed on on our web-site with a Google email account."
                )
            else:
                print(
                    f"Workflow job submission has failed with http status code: {status_code}. \nRemote server response: '{result}'"
                )
                return WorkflowJob(None)
        except Exception as ex:
            print(str(ex))

    def get_backends(self):
        if not self._verify_user_is_authenticated():
            return
        try:
            response = self._make_get_request(f"{self._asp_net_url}/api/backends")
            status_code = response.status_code
            if status_code == 200:
                backends = response.json()
                for backend in backends["$values"]:
                    print(
                        backend["name"],
                        "-",
                        f"no qubits: {backend['noQubits']}",
                        "-",
                        "Online" if backend["online"] else "Offline",
                    )
            else:
                print(f"Request has failed with http status code: {status_code}.")
        except Exception as ex:
            print(str(ex))

    def get_job_status(self, job):
        if not self._verify_user_is_authenticated():
            return
        if job is None or job.id is None:
            print("This job is not valid.")
            return
        if type(job) == Job:
            try:
                response = self._make_get_request(
                    f"{self._asp_net_url}/api/job/status/{job.id()}"
                )
                status_code = response.status_code
                if status_code == 200:
                    print("Job status: ", response.text)
                else:
                    print(f"Request has failed with http status code: {status_code}.")
            except Exception as ex:
                print(str(ex))
        elif type(job) == WorkflowJob:
            try:
                response = self._make_get_request(
                    f"{self._asp_net_url}/api/workflow-job/status/{job.id()}"
                )
                status_code = response.status_code
                if status_code == 200:
                    print("Job status: ", response.text)
                else:
                    print(f"Request has failed with http status code: {status_code}.")
            except Exception as ex:
                print(str(ex))

    def get_job_result(self, job):
        if not self._verify_user_is_authenticated():
            return
        if job is None or job.id is None:
            print("This job is not valid.")
            return
        if type(job) == Job:
            try:
                response = self._make_get_request(
                    f"{self._asp_net_url}/api/job/result/{job.id()}"
                )
                status_code = response.status_code
                if status_code == 200:
                    print(response.text)
                else:
                    print(f"Request has failed with http status code: {status_code}.")
            except Exception as ex:
                print(str(ex))
        elif type(job) == WorkflowJob:
            print("Operation not supported for workflow jobs.")

    def _verify_user_is_authenticated(self):
        if (
            self._access_token is None
            or self._refresh_token is None
            or self._refresh_token_expiration_time is None
        ):
            print(
                "You are not authorized to access this service. Please try to authenticate first and make sure you have signed on on our web-site with a Google email account."
            )
            return False
        if self.is_refresh_token_expired():
            print("You session timed out, you need to re-authenticate!")
            return False
        return True

    def _make_get_request(self, api_url):
        if self.is_token_expired():
            self._try_refresh_tokens()
        return requests.get(
            api_url,
            headers={"Authorization": f"Bearer {self._access_token}"},
            verify=self._use_https,
        )

    def _make_post_request(self, api_url, data):
        if self.is_token_expired():
            self._try_refresh_tokens()
        response = requests.post(
            api_url,
            json=data,
            headers={"Authorization": f"Bearer {self._access_token}"},
            verify=self._use_https,
        )
        try:
            json = response.json()
            return (response.status_code, json)
        except:
            return (response.status_code, response.text)

    def is_token_expired(self):
        if self._token_expiration_time is None:
            return True
        return time.time() > self._token_expiration_time

    def is_refresh_token_expired(self):
        if self._refresh_token_expiration_time is None:
            return True
        return time.time() > self._refresh_token_expiration_time

    def _try_refresh_tokens(self):
        try:
            token_response = self._keycloak_openid.token(
                grant_type="refresh_token", refresh_token=self._refresh_token
            )
            self._access_token = token_response["access_token"]
            self._refresh_token = token_response["refresh_token"]
            self._token_expiration_time = (
                time.time() + token_response["expires_in"] - 5
            )  # seconds
            self._refresh_token_expiration_time = (
                time.time() + token_response["refresh_expires_in"] - 5
            )  # seconds
        except:
            pass

    def _is_server_online(self, url):
        try:
            response = requests.get(url, verify=self._use_https)
            if response.status_code == 200:
                return True
            return False
        except requests.exceptions.RequestException as e:
            return False

    def _is_server_under_maintenance(self, url):
        try:
            response = requests.get(url, verify=self._use_https)
            if "Under Maintenance" in response.text:
                return True
            return False
        except requests.exceptions.RequestException as e:
            return False

    def _store_state(self):
        state = secrets.token_urlsafe(64)
        response = requests.post(
            f"{self._asp_net_url}/auth/storestate",
            json={"state": state},
            verify=self._use_https,
        )
        if response.status_code != 200:
            if not self._is_server_online(self._asp_net_url):
                raise AuthenticationFailure(
                    f"The service you are trying to access at: {self._asp_net_url} is not online."
                )
            elif self._is_server_under_maintenance(self._asp_net_url):
                raise AuthenticationFailure(
                    f"The service you are trying to access at: {self._asp_net_url} is under maintenance."
                )
            else:
                raise AuthenticationFailure(
                    "Cannot initiate authentication, the authentication provider does not respond."
                )
        self._state = state

    def _get_authentication_url(self):
        auth_url_params = {
            "client_id": self._client_id,
            "redirect_uri": self._auth_call_back_url,
            "response_type": "code",
            "scope": "openid profile email",
            "kc_idp_hint": "google",
            "state": self._state,
        }
        return f"{self._keycloak_server_url}/realms/{self._realm_name}/protocol/openid-connect/auth?{urlencode(auth_url_params)}"

    def _get_authentication_code(self):

        timeout_seconds = 16
        start_time = time.time()

        try:
            while (time.time() - start_time) < timeout_seconds:
                response = requests.get(
                    self._show_code_callback_url,
                    params={"state": self._state},
                    verify=self._use_https,
                )
                # TODO: what if I use a wrong email account
                if response.status_code == 400:
                    if response.text == "Authorization state is missing.":
                        raise Exception()
                    time.sleep(1)
                    continue
                data = response.text
                auth_code = data.split(": ")[1]
                return auth_code
        except Exception as e:
            if self._debug:
                print(
                    f"Failed to retrieve the authorization code from the authentication provider: {e}"
                )
            pass

        raise AuthorizationFailure(
            "Authorization code was not received. Please make sure you are using a Google account which you have signed-on our web-site. If our website is not online please try again later."
        )

    def is_valid_uuid(self, value: str) -> bool:
        try:
            uuid_obj = uuid.UUID(value)
            return str(uuid_obj) == value.lower()
        except (ValueError, TypeError):
            return False
