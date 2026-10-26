# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.arm._passes import InsertTableOpsPass, LowerInt16RsqrtPass
from executorch.backends.arm._passes.arm_pass_manager import ArmPassManager
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.exir import EdgeCompileConfig, ExportedProgram, to_edge
from torch.fx import GraphModule


class _SimpleIntModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + x


class _PipelineInspectionPassManager(ArmPassManager):
    def _transform(
        self, exported_program: ExportedProgram, graph_module: GraphModule
    ) -> GraphModule:
        return graph_module


def _backend_pass_types(tosa_spec: str) -> list[type]:
    exported_program = torch.export.export(
        _SimpleIntModule().eval(),
        (torch.randint(-8, 8, (4,), dtype=torch.int8),),
        strict=True,
    )
    edge_program = to_edge(exported_program, compile_config=EdgeCompileConfig())
    compile_spec = TosaCompileSpec(tosa_spec)
    pass_manager = _PipelineInspectionPassManager(compile_spec)
    pass_manager.transform_to_backend_pipeline(
        edge_program.exported_program(),
        edge_program.exported_program().graph_module,
    )
    return [type(pass_) for pass_ in pass_manager.passes]


def test_lower_int16_rsqrt_pass_only_added_for_int16_extension() -> None:
    no_int16_passes = _backend_pass_types("TOSA-1.0+INT")
    int16_passes = _backend_pass_types("TOSA-1.0+INT+int16")

    assert LowerInt16RsqrtPass not in no_int16_passes
    assert LowerInt16RsqrtPass in int16_passes
    assert InsertTableOpsPass in int16_passes
    assert int16_passes.index(LowerInt16RsqrtPass) < int16_passes.index(
        InsertTableOpsPass
    )
