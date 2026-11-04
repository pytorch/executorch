# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from executorch.backends.arm._passes import InsertTableOpsPass, LowerInt16RsqrtPass
from executorch.backends.arm._passes.arm_pass_manager import ArmPassManager
from executorch.backends.arm._passes.quant_args import QuantArgs
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir import EdgeCompileConfig, ExportedProgram, to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx import GraphModule


class _SimpleIntModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + x


class _RsqrtModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rsqrt(x)


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


@pytest.mark.parametrize("output_dtype", (torch.int16, torch.int32))
def test_lower_int16_rsqrt_mul_uses_explicit_zero_shift(output_dtype):
    exported_program = torch.export.export(
        _RsqrtModule(), (torch.ones(4),), strict=True
    )
    edge_program = to_edge(exported_program, compile_config=EdgeCompileConfig())
    program = edge_program.exported_program()
    rsqrt = program.graph_module.graph.find_nodes(
        op="call_function", target=exir_ops.edge.aten.rsqrt.default
    )[0]
    source = rsqrt.args[0]
    source.meta["val"] = source.meta["val"].to(dtype=torch.int16)
    rsqrt.meta["val"] = rsqrt.meta["val"].to(dtype=output_dtype)
    output_limit = torch.iinfo(output_dtype).max
    rsqrt.meta.update(
        input_qparams={0: QuantArgs(0.1, 0, -32767, 32767, torch.int16)},
        output_qparams={
            0: QuantArgs(0.1, 0, -output_limit, output_limit, output_dtype)
        },
    )

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.0+INT+int16")
    ):
        result = LowerInt16RsqrtPass(program)(program.graph_module)

    assert result is not None
    result.graph_module.graph.lint()
    multiplications = result.graph_module.graph.find_nodes(
        op="call_function", target=exir_ops.backend.tosa.MUL.default
    )
    assert len(multiplications) == 2
    shift = multiplications[0].args[2]
    assert all(node.args[2] is shift for node in multiplications)
    assert shift.target == exir_ops.backend.tosa.CONST.default
    assert shift.args == ([0], [1], torch.int8)
    assert shift.meta["val"].dtype == torch.int8
