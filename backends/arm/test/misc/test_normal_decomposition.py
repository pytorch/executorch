# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

import torch
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.tosa.partitioner import TOSAPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export


class Normal(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.normal(2.0, 3.0, size=x.shape)


def test_decomposed_normal_partitions_randn_to_portable() -> None:
    exported_program = export(Normal(), (torch.zeros(2, 3),))
    partitioner = TOSAPartitioner(TosaCompileSpec("TOSA-1.0+FP"))

    edge_manager = to_edge_transform_and_lower(
        exported_program,
        partitioner=[partitioner],
    )
    targets = {
        node.target
        for node in edge_manager.exported_program().graph.nodes
        if node.op == "call_function"
    }

    assert torch.ops.higher_order.executorch_call_delegate in targets
    assert operator.getitem in targets
    assert exir_ops.edge.aten.randn.default in targets
    assert not any("normal" in str(target) for target in targets)

    program = edge_manager.to_executorch().executorch_program
    operators = {(op.name, op.overload) for op in program.execution_plan[0].operators}
    assert ("aten::randn", "out") in operators
    assert not any("normal" in name for name, _ in operators)
