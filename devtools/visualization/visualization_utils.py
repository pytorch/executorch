# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import subprocess
import time
from typing import Any, Callable, Type

from executorch.exir import EdgeProgramManager, ExecutorchProgramManager
from executorch.exir.program._program import _update_exported_program_graph_module
from torch._export.verifier import Verifier
from torch.export.exported_program import ExportedProgram
from torch.fx import GraphModule

try:
    from model_explorer import config, consts, visualize_from_config  # type: ignore
except ImportError:
    print(
        "Error: 'model_explorer' is not installed. Install using devtools/install_requirement.sh"
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
