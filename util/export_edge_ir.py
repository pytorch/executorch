from typing import Tuple, Union

import executorch.exir as exir

import torch
from executorch.exir import EdgeProgramManager, ExecutorchProgramManager, to_edge
from executorch.exir.tracer import Value
from torch._export import capture_pre_autograd_graph
from torch.export import export, ExportedProgram


_EDGE_COMPILE_CONFIG = exir.EdgeCompileConfig(
    _check_ir_validity=False,
)


def _to_core_aten(
    model: Union[torch.fx.GraphModule, torch.nn.Module],
    example_inputs: Tuple[Value, ...],
) -> ExportedProgram:
    # post autograd export. eventually this will become .to_core_aten
    if not isinstance(model, torch.fx.GraphModule):
        raise ValueError(
            f"Expected passed in model to be an instance of fx.GraphModule, got {type(model)}"
        )
    core_aten_ep = export(model, example_inputs)
    print(f"Core ATen graph:\n{core_aten_ep.graph}")
    return core_aten_ep


def _core_aten_to_edge(
    core_aten_exir_ep: ExportedProgram,
    edge_compile_config=None,
) -> EdgeProgramManager:
    if not edge_compile_config:
        edge_compile_config = exir.EdgeCompileConfig(
            _check_ir_validity=False,  # quant ops currently break ir verification
        )
    edge_manager: EdgeProgramManager = to_edge(
        core_aten_exir_ep, compile_config=edge_compile_config
    )
    print(f"Exported graph:\n{edge_manager.exported_program().graph}")
    return edge_manager


def export_to_edge(
    model: Union[torch.fx.GraphModule, torch.nn.Module],
    example_inputs: Tuple[Value, ...],
    edge_compile_config=_EDGE_COMPILE_CONFIG,
) -> EdgeProgramManager:
    model = capture_pre_autograd_graph(model, example_inputs)
    core_aten_ep = _to_core_aten(model, example_inputs)
    return _core_aten_to_edge(core_aten_ep, edge_compile_config)


if __name__ == "__main__":
    class MyModule(torch.nn.Module):
        def forward(self, x):
            return torch.rand(1, 3, 224, 224) + x

    example_inputs = (torch.rand(1, 3, 224, 224),)
    m = MyModule().eval()
    export_to_edge(m, example_inputs)
