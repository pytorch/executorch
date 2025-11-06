# Copyright 2025 Arm Limited and/or its affiliates.
# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import subprocess
import time
from typing import Any, Callable, Type

from executorch.exir import EdgeProgramManager, ExecutorchProgramManager  # type: ignore
from executorch.exir.program._program import (  # type: ignore
    _update_exported_program_graph_module,
)

from torch._export.verifier import Verifier
from torch.export.exported_program import ExportedProgram  # type: ignore
from torch.fx import GraphModule, Node  # type: ignore

try:
    from model_explorer import config, consts, visualize_from_config  # type: ignore
    from model_explorer.config import ModelExplorerConfig  # type: ignore
    from model_explorer.pytorch_exported_program_adater_impl import (  # type: ignore
        PytorchExportedProgramAdapterImpl,
    )
except ImportError:
    print(
        "Error: 'model_explorer' is not installed. Install using devtools/install_requirements.sh"
    )
    raise


class SingletonModelExplorerServer:
    """Singleton context manager for starting a model-explorer server.
    If multiple ModelExplorerServer contexts are nested, a single
    server is still used.
    """

    server: None | subprocess.Popen = None
    num_open: int = 0
    wait_after_start = 3.0

    def __init__(self, open_in_browser: bool = True, port: int | None = None):
        if SingletonModelExplorerServer.server is None:
            command = ["model-explorer"]
            if not open_in_browser:
                command.append("--no_open_in_browser")
            if port is not None:
                command.append("--port")
                command.append(str(port))
            SingletonModelExplorerServer.server = subprocess.Popen(command)

    def __enter__(self):
        SingletonModelExplorerServer.num_open = (
            SingletonModelExplorerServer.num_open + 1
        )
        time.sleep(SingletonModelExplorerServer.wait_after_start)
        return self

    def __exit__(self, type, value, traceback):
        SingletonModelExplorerServer.num_open = (
            SingletonModelExplorerServer.num_open - 1
        )
        if SingletonModelExplorerServer.num_open == 0:
            if SingletonModelExplorerServer.server is not None:
                SingletonModelExplorerServer.server.kill()
                try:
                    SingletonModelExplorerServer.server.wait(
                        SingletonModelExplorerServer.wait_after_start
                    )
                except subprocess.TimeoutExpired:
                    SingletonModelExplorerServer.server.terminate()
                SingletonModelExplorerServer.server = None


class ModelExplorerServer:
    """Context manager for starting a model-explorer server."""

    wait_after_start = 2.0

    def __init__(self, open_in_browser: bool = True, port: int | None = None):
        command = ["model-explorer"]
        if not open_in_browser:
            command.append("--no_open_in_browser")
        if port is not None:
            command.append("--port")
            command.append(str(port))
        self.server = subprocess.Popen(command)

    def __enter__(self):
        time.sleep(self.wait_after_start)

    def __exit__(self, type, value, traceback):
        self.server.kill()
        try:
            self.server.wait(self.wait_after_start)
        except subprocess.TimeoutExpired:
            self.server.terminate()


def _get_exported_program(
    visualizable: ExportedProgram | EdgeProgramManager | ExecutorchProgramManager,
) -> ExportedProgram:
    if isinstance(visualizable, ExportedProgram):
        return visualizable
    if isinstance(visualizable, (EdgeProgramManager, ExecutorchProgramManager)):
        return visualizable.exported_program()
    raise RuntimeError(f"Cannot get ExportedProgram from {visualizable}")


def visualize(
    visualizable: ExportedProgram | EdgeProgramManager | ExecutorchProgramManager,
    reuse_server: bool = True,
    no_open_in_browser: bool = False,
    **kwargs,
):
    """Wraps the visualize_from_config call from model_explorer.
    For convenience, figures out how to find the exported_program
    from EdgeProgramManager and ExecutorchProgramManager for you.

    See https://github.com/google-ai-edge/model-explorer/wiki/4.-API-Guide#visualize-pytorch-models
    for full documentation.
    """
    cur_config = config()
    settings = consts.DEFAULT_SETTINGS
    cur_config.add_model_from_pytorch(
        "Executorch",
        exported_program=_get_exported_program(visualizable),
        settings=settings,
    )
    if reuse_server:
        cur_config.set_reuse_server()
    visualize_model_explorer(
        config=kwargs.pop("config", cur_config),
        no_open_in_browser=no_open_in_browser,
        **kwargs,
    )


def visualize_model_explorer(
    **kwargs,
):
    """Wraps the visualize_from_config call from model_explorer."""
    visualize_from_config(
        **kwargs,
    )


def _save_model_as_json(cur_config: ModelExplorerConfig, file_name: str):
    """Save the graphs stored in the `cur_config` in JSON format, which can be loaded by the Model Explorer GUI.

    :param cur_config: ModelExplorerConfig containing the graph for visualization.
    :param file_name: Name of the JSON file for storage.
    """
    # Extract the graphs from the config file.
    graphs_list = json.loads(cur_config.get_transferrable_data()["graphs_list"])
    graphs = json.loads(graphs_list[0])["graphs"]

    # The returned dictionary is missing the `collectionLabel` entry. Add it manually.
    for graph in graphs:
        graph["collectionLabel"] = "Executorch"

    # Create the JSON according to the structure required by the Model Explorer GUI.
    json_data = {
        "label": "Executorch",
        "graphs": graphs,
        "graphsWithLevel": [
            {"graph": graph, "level": level} for level, graph in enumerate(graphs)
        ],
    }

    # Store the JSON.
    with open(file_name, "w") as f:
        json.dump(json_data, f)


def visualize_with_clusters(
    exported_program: ExportedProgram,
    json_file_name: str | None = None,
    reuse_server: bool = False,
    get_node_partition_name: Callable[[Node], str | None] = lambda node: node.meta.get(
        "delegation_tag", None
    ),
    get_node_qdq_cluster_name: Callable[
        [Node], str | None
    ] = lambda node: node.meta.get("cluster", None),
    **kwargs,
):
    """Visualize exported programs using the Model Explorer. The QDQ clusters and individual partitions are highlighted.

        To install the Model Explorer, run `devtools/install_requirements.sh`.
        To display a stored json file, first launch the Model Explorer server by running `model-explorer`, and then
         use the GUI to open the json.

        NOTE: FireFox seems to have issues rendering the graphs. Other browsers seem to work well.

    :param exported_program: Program to visualize.
    :param json_file_name: If not None, a JSON of the visualization will be stored in the provided file. The JSON can
                            then be opened in the Model Explorer GUI later.
                           If None, a Model Explorer instance will be launched with the model visualization.
    :param reuse_server: If True, an existing instance of the Model Explorer server will be used (if one exists).
                          Otherwise, a new instance at a separate port will start.
    :param get_node_partition_name: Function which takes a `Node` and returns a string with the name of the partition
                                     the `Node` belongs to, or `None` if it has no partition.
    :param get_node_qdq_cluster_name: Function which takes a `Node` and returns a string with the name of the QDQ
                                       cluster the `Node` belongs to, or `None` if it has no cluster.
    :param kwargs: Additional kwargs for the `visualize_from_config()` function.
    """

    cur_config = config()

    # Create a Model Explorer graph from the `exported_program`.
    adapter = PytorchExportedProgramAdapterImpl(
        exported_program, consts.DEFAULT_SETTINGS
    )
    graphs = adapter.convert()

    nodes = list(exported_program.graph.nodes)
    explorer_nodes = graphs["graphs"][0].nodes

    # Highlight QDQ clusters and individual partitions.
    known_partition_names = []
    for explorer_node, node in zip(explorer_nodes, nodes, strict=True):
        # Generate the `namespace` for the node, which will determine node grouping in the visualizer.
        # The character "/" is used as a divider when a node has multiple namespaces.
        namespace = ""

        if (partition_name := get_node_partition_name(node)) is not None:
            # If the nodes are tagged by the partitioner, highlight the tagged groups.

            # Create a custom naming for the partitions ("partition <i>" where `i` = 0, 1, 2, ...).
            if partition_name not in known_partition_names:
                known_partition_names.append(partition_name)
            partition_id = known_partition_names.index(partition_name)

            safe_partition_name = partition_name.replace(
                "/", ":"
            )  # Avoid using unwanted "/".
            namespace += f"partition {partition_id} ({safe_partition_name})"

        if (cluster_name := get_node_qdq_cluster_name(node)) is not None:
            # Highlight the QDQ cluster.

            # Add a separator, in case the namespace already contains the `partition`.
            if len(namespace) != 0:
                namespace += "/"

            # Create a custom naming for the clusters ("cluster (<old_cluster_name>)").
            safe_cluster_name = cluster_name.replace(
                "/", ":"
            )  # Avoid using unwanted "/".
            namespace += f"cluster ({safe_cluster_name})"

        explorer_node.namespace = namespace

    # Store the modified graph in the config.
    graphs_index = len(cur_config.graphs_list)
    cur_config.graphs_list.append(graphs)
    name = "Executorch"
    model_source: config.ModelSource = {"url": f"graphs://{name}/{graphs_index}"}
    cur_config.model_sources.append(model_source)

    if json_file_name is not None:
        # Just save the visualization.
        _save_model_as_json(cur_config, json_file_name)

    else:
        # Start the ModelExplorer server, and visualize the graph in the browser.
        if reuse_server:
            cur_config.set_reuse_server()
        visualize_from_config(
            cur_config,
            **kwargs,
        )


def visualize_graph(
    graph_module: GraphModule,
    exported_program: ExportedProgram | EdgeProgramManager | ExecutorchProgramManager,
    reuse_server: bool = True,
    no_open_in_browser: bool = False,
    **kwargs,
):
    """Overrides the graph_module of the supplied exported_program with 'graph_module' before visualizing.
    Also disables validating operators to allow visualizing graphs containing custom ops.

    A typical example is after running passes, which returns a graph_module rather than an ExportedProgram.
    """

    class _any_op(Verifier):
        dialect = "ANY_OP"

        def allowed_op_types(self) -> tuple[Type[Any], ...]:
            return (Callable,)  # type: ignore

    exported_program = _get_exported_program(exported_program)
    exported_program = _update_exported_program_graph_module(
        exported_program, graph_module, override_verifiers=[_any_op]
    )
    visualize(exported_program, reuse_server, no_open_in_browser, **kwargs)
