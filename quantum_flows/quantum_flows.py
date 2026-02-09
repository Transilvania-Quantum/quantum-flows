import base64
import io
import json
import numpy as np
import qiskit
import requests
import secrets
import time
import traceback
import uuid
import webbrowser

from collections.abc import Sequence
from IPython.display import display, HTML
from keycloak import KeycloakOpenID
from urllib.parse import urlencode

from qiskit import qpy
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Pauli, PauliList, SparsePauliOp
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


DEFAULT_TIMEOUT = (3.05, 10)  # (connect timeout, read timeout)


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.generic):
            if np.iscomplexobj(obj):
                c = complex(obj)
                return {"real": c.real, "imag": c.imag}
            return obj.item()
        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, BoundaryCondition):
            return obj.name
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        if isinstance(obj, tuple):
            return list(obj)

        return super().default(obj)


def _is_number(x) -> bool:
    return isinstance(x, (int, float, complex, np.generic))


def _to_complex_dict(c):
    if isinstance(c, np.generic):
        c = c.item()
    if isinstance(c, complex):
        return {"real": float(c.real), "imag": float(c.imag)}
    return {"real": float(c), "imag": 0.0}


def serialize_circuit(circuit):
    buffer = io.BytesIO()
    qpy.dump(circuit, buffer)
    qpy_binary_data = buffer.getvalue()
    base64_encoded_circuit = base64.b64encode(qpy_binary_data).decode("utf-8")
    return base64_encoded_circuit


class AuthenticationFailure(Exception):
    pass


class AuthorizationFailure(Exception):
    pass


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

    def _assert_jsonable(self, value, label: str):
        try:
            json.dumps(value, cls=CustomJSONEncoder)
        except (TypeError, OverflowError, ValueError) as e:
            raise Exception(
                f"Input data '{label}' is not JSON-serializable: {e}"
            ) from e

    def __str__(self):
        try:
            return json.dumps(self.data, indent=2, cls=CustomJSONEncoder)
        except ValueError as e:
            raise Exception(f"Invalid input data format: {e}")
        except (OverflowError, TypeError) as e:
            raise Exception(f"Input data content must be JSON serializable: {e}.")

    def _normalize_operator_input(self, content):
        if isinstance(content, tuple):
            if len(content) != 2:
                raise Exception(
                    "Operator tuple must be exactly (PauliList, coeffs_or_none)."
                )
            op, coeffs = content
            if not isinstance(op, PauliList):
                raise Exception(
                    "Operator tuple is only supported for (PauliList, coeffs_or_none)."
                )
            return op, coeffs
        return content, None

    def add_data(self, label, content):
        self.check_label(label, self.data)
        try:
            if label == "operator":
                operator, coeffs = self._normalize_operator_input(content)
                self.validate_operator(
                    (operator, coeffs) if coeffs is not None else operator
                )
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
            elif label == "pubs":
                content = self.validate_and_serialize_pub(content)
                if "pubs" not in self.data:
                    self.data["pubs"] = []
                self._assert_jsonable(content, label)
                self.data["pubs"].append(content)
            elif label == "molecule-info":
                self.validate_molecule_info(content)
                self._assert_jsonable(content, label)
                self.data[label] = content
            elif label == "lattice":
                lattice_dict = self.lattice_to_dict(content)
                self._assert_jsonable(lattice_dict, label)
                self.data[label] = lattice_dict
            elif label == "ising-model":
                self.validate_ising_model(content)
                self._assert_jsonable(content, label)
                self.data[label] = content
            elif label == "training-data":
                self.validate_training_data(content)
                self._assert_jsonable(content, label)
                self.data[label] = content
            elif label == "inference-data":
                self.validate_inference_data(content)
                self._assert_jsonable(content, label)
                self.data[label] = content
            elif label == "quadratic-program":
                self.validate_quadratic_program(content)
                lp_string = content.export_as_lp_string()
                self._assert_jsonable(lp_string, label)
                self.data[label] = lp_string
            else:
                self._assert_jsonable(content, label)
                self.data[label] = content

        except ValueError as e:
            raise Exception(f"Invalid input data format: {e}")
        except (OverflowError, TypeError) as e:
            raise Exception(f"Input data content must be JSON serializable: {e}")

    def check_label(self, label, data):
        if not isinstance(label, str):
            raise Exception("Input data label must be string.")
        if label not in [
            "ansatz-parameters",
            "inference-data",
            "ising-model",
            "lattice",
            "lp-model",
            "max-fun-evaluations",
            "molecule-info",
            "operator",
            "pubs",
            "quadratic-program",
            "training-data",
        ]:
            raise Exception(
                f"Input data of type {label} is not supported. Please choose one of the following options: 'ansatz-parameters', 'inference-data' 'ising-model', 'lattice', 'lp-model', 'molecule-info', 'operator', 'pub', 'pubs', 'training-data'."
            )
        if label != "pubs" and label in data.keys():
            raise Exception(
                f"An input data item of type '{label}' has already been added to the job input data. Multiple data items of same category are allowed only for PUBs."
            )

    def lattice_to_dict(self, lattice):
        if isinstance(lattice, LineLattice):
            lattice_data = {
                "type": "LineLattice",
                "num_nodes": lattice.num_nodes,
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
        elif isinstance(lattice, TriangularLattice):
            lattice_data = {
                "type": "TriangularLattice",
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
        if not isinstance(molecule_info, dict):
            raise Exception("The 'molecule_info' must be a dictionary.")
        if (
            "symbols" not in molecule_info
            or not isinstance(molecule_info["symbols"], list)
            or not all(isinstance(x, str) for x in molecule_info["symbols"])
        ):
            raise Exception("Add 'symbols' as a list of nuclei name strings.")
        if (
            "coords" not in molecule_info
            or not isinstance(molecule_info["coords"], list)
            or len(molecule_info["coords"]) != len(molecule_info["symbols"])
            or not all(
                isinstance(c, tuple) and len(c) == 3 for c in molecule_info["coords"]
            )
            or not all(
                all(isinstance(i, (int, float)) for i in x)
                for x in molecule_info["coords"]
            )
        ):
            raise Exception(
                "The 'coords' must be a list of tuples with numbers representing the x, y, z position of each nuclei."
            )
        if "multiplicity" in molecule_info and not isinstance(
            molecule_info["multiplicity"], int
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
        if "masses" in molecule_info:
            if not isinstance(molecule_info["masses"], list):
                raise Exception("The 'masses' must be a list of numbers.")
            if not all(isinstance(m, (int, float)) for m in molecule_info["masses"]):
                raise Exception(
                    "The 'masses' must be a list of numbers, one for each nucleus in the molecule."
                )
            if len(molecule_info["masses"]) != len(molecule_info["symbols"]):
                raise Exception(
                    "The 'masses' list must have the same length as the 'symbols' list."
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
        is_classification = False
        is_regression = False
        line = 0
        output_length = None
        if not isinstance(training_data, list):
            raise Exception("The 'training_data' must be a list of dictionaries.")
        for data in training_data:
            line += 1
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
                    f"The optional 'data-tags' value must be a list of strings (check line {line})."
                )
            if not all(isinstance(item, (int, float)) for item in vector):
                raise Exception(
                    f"The 'data-point' value must be a list of numeric values (int or float) (check line {line})."
                )
            if vector_size is None:
                vector_size = len(vector)
            if len(vector) != vector_size:
                raise Exception(
                    "All 'data-point' vectors in training data entries must have the same length."
                )
            if data_tags is not None and len(data_tags) != vector_size:
                raise Exception(
                    f"If provided, the 'data-tags' list must have the same length as the 'data-point' vector (check line {line})."
                )
            if data_tags is not None and not all(
                isinstance(tag, str) for tag in data_tags
            ):
                raise Exception(
                    f"If provided, the 'data-tags' list must contain only strings (check line {line})."
                )
            if "label" in data:
                is_classification = True
                label = data["label"]
                if not isinstance(label, int):
                    raise Exception(
                        f"The 'label' value must be an integer (check line {line})."
                    )
            if "output" in data:
                is_regression = True
                output = data["output"]
                if not isinstance(output, (int, float, list)) or (
                    isinstance(output, list)
                    and not all(isinstance(value, (int, float)) for value in output)
                ):
                    raise Exception(
                        f"The 'output' value must be a numeric type (int, float) or a list of numeric values (check line {line})."
                    )
                values = output if isinstance(output, list) else [output]
                for value in values:
                    if value < -1 or value > 1:
                        raise Exception(
                            f"The 'output' numeric values must be in [-1, 1] range (check line {line})."
                        )
                if output_length is None:
                    output_length = len(values)
                elif output_length != len(values):
                    raise Exception(
                        f"All 'output' lists in training data entries must have the same length. Choose either a consistent sized list or a numeric value (check line {line})."
                    )
            if is_classification and is_regression:
                raise Exception(
                    "The training data cannot contain both 'label' and 'output' keys. Please choose either classification or regression data template."
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
            if data_tags is not None:
                if not isinstance(data_tags, list):
                    raise Exception(
                        "The optional 'data-tags' value must be a list of strings."
                    )
                if not all(isinstance(tag, str) for tag in data_tags):
                    raise Exception(
                        "If provided, 'data-tags' must contain only strings."
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

    def _normalize_numeric_sequence(self, name: str, value):
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            seq = value.ravel().tolist()
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            seq = list(value)
        else:
            raise Exception(
                f"'{name}' must be a numeric sequence (list/tuple/np.ndarray) or None."
            )
        if not all(_is_number(x) for x in seq):
            raise Exception(f"'{name}' must contain only numbers.")
        return seq

    def validate_and_serialize_pub(self, pub):
        shots = None
        parameters = None

        if isinstance(pub, QuantumCircuit):
            quantum_circuit = pub
        elif isinstance(pub, tuple):
            if len(pub) == 3:
                quantum_circuit, parameters, shots = pub
            elif len(pub) == 2:
                quantum_circuit, parameters = pub
            elif len(pub) == 1:
                (quantum_circuit,) = pub
            else:
                raise Exception("A pub tuple must have 1..3 elements.")
            if not isinstance(quantum_circuit, QuantumCircuit):
                raise Exception("First element must be a QuantumCircuit.")
        else:
            raise Exception(
                "A pub must be a QuantumCircuit or a tuple (circuit[, params[, shots]])."
            )

        if shots is not None:
            if not isinstance(shots, int) or shots <= 0:
                raise Exception("'shots' must be a positive integer.")

        if not isinstance(quantum_circuit, QuantumCircuit):
            raise Exception("First element must be a QuantumCircuit.")

        parameters = self._normalize_numeric_sequence("parameters", parameters)

        if quantum_circuit.num_parameters == 0 and parameters not in (None, []):
            raise Exception(
                "Circuit has zero parameters; parameters must be None or []."
            )

        if parameters is not None and quantum_circuit.num_parameters != len(parameters):
            raise Exception(
                f"Parameter count mismatch: circuit expects {quantum_circuit.num_parameters}, got {len(parameters)}."
            )

        return (serialize_circuit(quantum_circuit), parameters, shots)

    def validate_operator(self, operator):
        if isinstance(operator, (Operator, Pauli, SparsePauliOp, PauliList)):
            if isinstance(operator, Operator):
                matrix = operator.data
                if not np.allclose(matrix, matrix.conj().T):
                    print("WARNING: The operator you supplied is not Hermitian!")
            return

        if isinstance(operator, tuple):
            if len(operator) != 2 or not isinstance(operator[0], PauliList):
                raise Exception(
                    "Operator tuple is only supported for (PauliList, coeffs_or_none)."
                )

            pauli_list, coeffs = operator
            if coeffs is None:
                return

            if isinstance(coeffs, np.ndarray):
                coeffs_seq = coeffs.ravel().tolist()
            elif isinstance(coeffs, Sequence) and not isinstance(coeffs, (str, bytes)):
                coeffs_seq = list(coeffs)
            else:
                raise Exception(
                    "Coefficients must be a numeric sequence (list/tuple/np.ndarray) or None."
                )

            if len(coeffs_seq) == 0:
                return
            if not all(_is_number(c) for c in coeffs_seq):
                raise Exception("Operator coefficients must be numeric.")
            if len(coeffs_seq) != len(pauli_list):
                raise Exception(
                    "Number of coefficients must match number of Pauli terms (or empty/None for all-ones)."
                )
            return

        raise Exception(
            "Operator must be one of: Operator, Pauli, SparsePauliOp, PauliList, "
            "or the tuple form (PauliList, coeffs_or_none)."
        )

    def to_sparse_pauli_operator(self, operator, coeffs=None):
        if isinstance(operator, SparsePauliOp):
            return operator

        elif isinstance(operator, Pauli):
            return SparsePauliOp(operator)

        elif isinstance(operator, PauliList):

            if coeffs is not None:
                if not hasattr(coeffs, "__len__"):
                    raise ValueError("Coefficients must be a sequence (or None).")
                if len(coeffs) > 0 and len(coeffs) != len(operator):
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
        raise ValueError(
            "Unsupported operator type. Supported types are: Operator, Pauli, SparsePauliOp or PauliList and a possible empty list of numeric coefficents."
        )

    def serialize_sparse_pauli_operator(self, sparse_op):
        if not isinstance(sparse_op, SparsePauliOp):
            raise ValueError("Input must be a SparsePauliOp")

        pauli_data = sparse_op.to_list()
        pauli_terms = [term for (term, _) in pauli_data]
        coefficients = [_to_complex_dict(coeff) for (_, coeff) in pauli_data]
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

    def __init__(self, verify_tls=True, debug=False):
        self._verify_tls = verify_tls
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

        if not self._is_server_reachable(self._keycloak_server_url):
            raise SystemExit(
                f"The service you are trying to access at: {self._asp_net_url}, is not responding. \
In case the service has been recently started please wait 5 minutes for it to become fully functional."
            )

        self._keycloak_openid = KeycloakOpenID(
            server_url=self._keycloak_server_url,
            client_id=self._client_id,
            realm_name=self._realm_name,
            verify=self._verify_tls,
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
        except AuthorizationFailure as ex:
            print(
                "Failed to authenticate with the quantum provider. Make sure you are using the correct Gmail account."
            )
            if self._debug:
                print("More details: ", str(ex))
                traceback.print_exc()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as ex:
            print("Failed to authenticate with the quantum provider.")
            if "Connection refused" in str(ex):
                print("The remote service does not respond. Please try again later.")
            if self._debug:
                print("Unexpected exception: ", str(ex))

    def _ensure_access_token(self) -> None:
        # refresh token must exist and be valid
        if self._refresh_token is None or self._refresh_token_expiration_time is None:
            self._clear_tokens(clear_refresh=True)
            raise AuthorizationFailure("Not authenticated; please authenticate.")

        if self.is_refresh_token_expired():
            self._clear_tokens(clear_refresh=True)
            raise AuthorizationFailure("Session timed out; please re-authenticate.")

        # access token must exist and be fresh; otherwise refresh
        if self._access_token is None or self.is_token_expired():
            ok = self._try_refresh_tokens()
            if not ok or self._access_token is None:
                # if refresh failed, _try_refresh_tokens already cleared appropriately
                raise AuthorizationFailure("Session expired; please re-authenticate.")

    def submit_job(
        self, *, backend=None, circuit=None, circuits=None, shots=None, comments=""
    ):
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
                    "Circuits": [serialize_circuit(circ) for circ in circuits],
                    "Shots": shots,
                    "Comments": comments,
                    "QiskitVersion": qiskit.__version__,
                }
            (status_code, result) = self._make_post_request(
                f"{self._asp_net_url}/api/job", job_data
            )
            if status_code == 201:
                if not isinstance(result, dict):
                    raise Exception(
                        f"Expected JSON response for job creation, instead got:\n{str(result)}."
                    )
                return Job(result["id"])
            elif isinstance(result, str) and "Under Maintenance" in result:
                print(
                    "The remote service is currently under maintenance. Please try again later."
                )
            else:
                print(
                    f"Job submission has failed with http status code: {status_code}. \nRemote server response: '{result}'"
                )
                return Job(None)
        except (KeyboardInterrupt, SystemExit):
            raise
        except AuthorizationFailure as ex:
            print(str(ex))
        except requests.exceptions.RequestException as ex:
            print("Network error talking to provider:", ex)
        except Exception as ex:
            print(str(ex))

    def submit_workflow_job(
        self,
        *,
        backend=None,
        shots=None,
        workflow_id=None,
        tag="",
        comments="",
        max_fun_evaluations=None,
        input_data=None,
    ):
        if input_data is None:
            input_data = InputData()
        if shots is not None:
            if not isinstance(shots, int) or shots <= 0:
                print("The 'shots' input argument must be a positive integer.")
                return
        if not backend:
            print("Please specify a backend name.")
            return
        if not isinstance(backend, str):
            print("The 'backend' input argument must be a string.")
            return
        if not workflow_id:
            print("Please specify a workflow Id.")
            return
        if not self.is_valid_uuid(workflow_id):
            print("The specified workflow Id is not a valid GUID.")
            return
        if max_fun_evaluations is not None:
            if not isinstance(max_fun_evaluations, int) or max_fun_evaluations <= 0:
                print(
                    "The optional 'max-fun-evaluations' input argument must be a positive integer."
                )
                return
        if not isinstance(tag, str):
            print("The optional 'tag' input argument must be a string.")
            return
        if len(tag) > 100:
            print("The optional 'tag' input argument must be at most 100 characters.")
            return
        if not isinstance(comments, str):
            print("The optional 'comments' input argument must be a string.")
            return
        if len(comments) > 1000:
            print(
                "The optional 'comments' input argument must be at most 1000 characters."
            )
            return

        try:
            input_data_labels = []
            input_data_items = []
            input_data_labels.append("backend")
            input_data_items.append(backend)
            input_data_labels.append("shots")
            input_data_items.append(str(shots))
            input_data_labels.append("max-fun-evaluations")
            input_data_items.append(str(max_fun_evaluations))
            for input_data_label in input_data.data.keys():
                input_data_labels.append(input_data_label)
                content = input_data.data[input_data_label]
                try:
                    dumped = json.dumps(content, indent=2, cls=CustomJSONEncoder)
                except ValueError as e:
                    raise Exception(
                        f"Input data '{input_data_label}' has invalid format: {e}"
                    )
                except (OverflowError, TypeError) as e:
                    raise Exception(
                        f"Input data '{input_data_label}' content must be JSON serializable: {e}"
                    )
                input_data_items.append(dumped)
            job_data = {
                "BackendName": backend,
                "WorkflowId": workflow_id,
                "Shots": shots,
                "Tag": tag,
                "Comments": comments,
                "MaxFunEvaluations": max_fun_evaluations,
                "InputDataLabels": input_data_labels,
                "InputDataItems": input_data_items,
                "QiskitVersion": qiskit.__version__,
            }
            (status_code, result) = self._make_post_request(
                f"{self._asp_net_url}/api/workflow-job", job_data
            )
            if status_code == 201:
                if not isinstance(result, dict):
                    raise Exception(
                        f"Expected JSON response for job creation, instead got:\n{str(result)}."
                    )
                return WorkflowJob(result["id"])
            else:
                print(
                    f"Workflow job submission has failed with http status code: {status_code}. \nRemote server response: '{result}'"
                )
                return WorkflowJob(None)
        except (KeyboardInterrupt, SystemExit):
            raise
        except AuthorizationFailure as ex:
            print(str(ex))
        except requests.exceptions.RequestException as ex:
            print("Network error talking to provider:", ex)
        except Exception as ex:
            print(str(ex))

    def get_backends(self):
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
        except (KeyboardInterrupt, SystemExit):
            raise
        except AuthorizationFailure as ex:
            print(str(ex))
        except Exception as ex:
            print(str(ex))

    def get_job_status(self, job):
        if job is None or job.id() is None:
            print("This job is not valid.")
            return
        if isinstance(job, Job):
            try:
                response = self._make_get_request(
                    f"{self._asp_net_url}/api/job/status/{job.id()}"
                )
                status_code = response.status_code
                if status_code == 200:
                    print("Job status: ", response.text)
                else:
                    print(f"Request has failed with http status code: {status_code}.")
            except (KeyboardInterrupt, SystemExit):
                raise
            except AuthorizationFailure as ex:
                print(str(ex))
                return
            except Exception as ex:
                print(str(ex))
        elif isinstance(job, WorkflowJob):
            try:
                response = self._make_get_request(
                    f"{self._asp_net_url}/api/workflow-job/status/{job.id()}"
                )
                status_code = response.status_code
                if status_code == 200:
                    print("Job status: ", response.text)
                else:
                    print(f"Request has failed with http status code: {status_code}.")
            except (KeyboardInterrupt, SystemExit):
                raise
            except AuthorizationFailure as ex:
                print(str(ex))
            except Exception as ex:
                print(str(ex))

    def get_job_result(self, job):
        if job is None or job.id() is None:
            print("This job is not valid.")
            return
        if isinstance(job, Job):
            try:
                response = self._make_get_request(
                    f"{self._asp_net_url}/api/job/result/{job.id()}"
                )
                status_code = response.status_code
                if status_code == 200:
                    print(response.text)
                else:
                    print(f"Request has failed with http status code: {status_code}.")
            except (KeyboardInterrupt, SystemExit):
                raise
            except AuthorizationFailure as ex:
                print(str(ex))
            except Exception as ex:
                print(str(ex))
        elif isinstance(job, WorkflowJob):
            print("Operation not supported for workflow jobs.")

    def _make_get_request(self, api_url):
        self._ensure_access_token()
        response = requests.get(
            api_url,
            headers={"Authorization": f"Bearer {self._access_token}"},
            verify=self._verify_tls,
            timeout=DEFAULT_TIMEOUT,
        )
        if response.status_code == 401:
            # access token rejected; try refresh once
            self._access_token = None
            self._token_expiration_time = None
            self._ensure_access_token()
            response = requests.get(
                api_url,
                headers={"Authorization": f"Bearer {self._access_token}"},
                verify=self._verify_tls,
                timeout=DEFAULT_TIMEOUT,
            )
        # second 401 => hard fail
        if response.status_code == 401:
            self._clear_tokens(clear_refresh=True)
            raise AuthorizationFailure(
                "You are not authorized to access this service. Please try to authenticate first and make sure you have signed on on our web-site with a Google email account."
            )
        return response

    def _make_post_request(self, api_url, data):
        self._ensure_access_token()
        response = requests.post(
            api_url,
            json=data,
            headers={"Authorization": f"Bearer {self._access_token}"},
            verify=self._verify_tls,
            timeout=DEFAULT_TIMEOUT,
        )
        if response.status_code == 401:
            # access token rejected; try refresh once
            self._access_token = None
            self._token_expiration_time = None
            self._ensure_access_token()
            response = requests.post(
                api_url,
                json=data,
                headers={"Authorization": f"Bearer {self._access_token}"},
                verify=self._verify_tls,
                timeout=DEFAULT_TIMEOUT,
            )
        # second 401 => hard fail
        if response.status_code == 401:
            self._clear_tokens(clear_refresh=True)
            raise AuthorizationFailure(
                "You are not authorized to access this service. Please try to authenticate first and make sure you have signed on on our web-site with a Google email account."
            )
        try:
            payload = response.json()
            return (response.status_code, payload)
        except ValueError:
            return (response.status_code, response.text)

    def is_token_expired(self):
        if self._token_expiration_time is None:
            return True
        return time.time() > self._token_expiration_time

    def is_refresh_token_expired(self):
        if self._refresh_token_expiration_time is None:
            return True
        return time.time() > self._refresh_token_expiration_time

    def _clear_tokens(self, *, clear_refresh: bool = True) -> None:
        self._access_token = None
        self._token_expiration_time = None
        if clear_refresh:
            self._refresh_token = None
            self._refresh_token_expiration_time = None

    def _try_refresh_tokens(self) -> bool:
        try:
            token_response = self._keycloak_openid.token(
                grant_type="refresh_token",
                refresh_token=self._refresh_token,
            )
            self._access_token = token_response["access_token"]
            self._refresh_token = token_response["refresh_token"]
            self._token_expiration_time = time.time() + token_response["expires_in"] - 5
            self._refresh_token_expiration_time = (
                time.time() + token_response["refresh_expires_in"] - 5
            )
            return True
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as ex:
            clear_refresh = False
            is_transient = isinstance(ex, requests.exceptions.RequestException)
            msg = str(ex).lower()
            transient_markers = [
                "timeout",
                "timed out",
                "connection aborted",
                "connection refused",
                "connection reset",
                "temporary failure",
                "temporarily unavailable",
                "service unavailable",
                "bad gateway",
                "gateway timeout",
                "name or service not known",
                "dns",
            ]
            if any(m in msg for m in transient_markers):
                is_transient = True
            invalid_markers = [
                "invalid_grant",
                "refresh token is invalid",
                "refresh_token is invalid",
                "token is not active",
                "session not active",
            ]
            if (not is_transient) and any(m in msg for m in invalid_markers):
                clear_refresh = True
            self._clear_tokens(clear_refresh=clear_refresh)
            if self._debug:
                print("Failed to refresh authentication tokens:", ex)
                print(
                    "Classified as transient:",
                    is_transient,
                    "clear_refresh:",
                    clear_refresh,
                )
            return False

    def _is_server_reachable(self, url):
        try:
            requests.get(url, verify=self._verify_tls, timeout=DEFAULT_TIMEOUT)
            return True
        except requests.exceptions.RequestException as e:
            return False

    def _is_server_under_maintenance(self, url):
        try:
            response = requests.get(
                url, verify=self._verify_tls, timeout=DEFAULT_TIMEOUT
            )
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
            verify=self._verify_tls,
            timeout=DEFAULT_TIMEOUT,
        )
        if response.status_code != 200:
            if not self._is_server_reachable(self._asp_net_url):
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
        accept_missing_code_once = True

        try:
            while (time.time() - start_time) < timeout_seconds:
                response = requests.get(
                    self._show_code_callback_url,
                    params={"state": self._state},
                    verify=self._verify_tls,
                    timeout=DEFAULT_TIMEOUT,
                )
                if response.status_code != 200:
                    if response.text == "Authorization state is missing.":
                        raise Exception(
                            "The authentication process was not initiated properly. Please try to authenticate again."
                        )
                    time.sleep(1)
                    continue
                parts = response.text.split("Your authorization code is: ", 1)
                if len(parts) != 2 or not parts[1].strip():
                    if accept_missing_code_once:
                        accept_missing_code_once = False
                        time.sleep(3)
                        continue
                    else:
                        raise Exception(
                            "The server failed to provide a valid authorization code. Please try to authenticate again, if it keeps failing send a bug report using the feedback form."
                        )
                auth_code = parts[1].strip()
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

    def is_valid_uuid(self, value) -> bool:
        if not isinstance(value, str):
            return False
        try:
            uuid.UUID(value)  # parses many valid forms
            return True
        except (ValueError, TypeError, AttributeError):
            return False
