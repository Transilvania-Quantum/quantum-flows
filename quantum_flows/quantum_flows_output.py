class QuantumFlowsOutput:
    def __init__(self):
        self.result = {}
        self.ansatz = {}
        self.inferred_data = {}
        self.graphs = []
        self.job_metadata = None

    def add_workflow_title(self, title):
        if len(title) > 200:
            raise ValueError("Workflow title cannot exceed 200 characters.")
        self.result["result-title-string"] = title

    def add_workflow_result(self, result):
        self.result["result-text-string"] = result

    def add_inferred_training_data(self, inferred_data):
        self.inferred_data["inferred-training-data"] = inferred_data

    def add_inferred_inference_data(self, inferred_data):
        self.inferred_data["inferred-inference-data"] = inferred_data

    def add_graph(
        self,
        graph_title=None,
        graph_data_json=None,
        graph_layout_json="{}",
    ):
        if not graph_title:
            raise ValueError("Graph title is required.")
        if any(graph.get("title") == graph_title for graph in self.graphs):
            raise ValueError(
                f"A graph with title '{graph_title}' was already added once in this workflow."
            )
        if graph_data_json is None:
            raise ValueError("Graph data is required to add a graph.")
        self.graphs.append(
            {
                "title": graph_title,
                "graph-data-json": graph_data_json,
                "graph-layout-json": graph_layout_json,
            }
        )

    def add_ansatz(
        self,
        ansatz_name=None,
        printed_circuit=None,
        open_qasm_circuit=None,
        parameters_array=None,
    ):
        if printed_circuit is None:
            raise ValueError(
                "A string containing the ansatz printed circuit must be added to the ansatz."
            )
        if open_qasm_circuit is None:
            raise ValueError(
                "A string containing the Open QASM ansatz code is required to add an ansatz."
            )
        if parameters_array is None:
            raise ValueError(
                "An array of ansatz parameters is required to be added to an ansatz."
            )

        self.ansatz = {
            "ansatz-name": ansatz_name,
            "printed-circuit-string": printed_circuit,
            "open-qasm-string": open_qasm_circuit,
            "ansatz-parameters": parameters_array,
        }

    def add_job_metadata(self, job_metadata):
        if not isinstance(job_metadata, dict):
            raise ValueError("Job metadata should be a dictionary.")
        self.job_metadata = job_metadata
