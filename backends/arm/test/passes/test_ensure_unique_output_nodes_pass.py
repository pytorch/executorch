# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm._passes import EnsureUniqueOutputNodesPass
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline
from executorch.backends.test.harness.stages import StageType
from executorch.exir.dialects._ops import ops as exir_ops


class DuplicateOutputModule(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        y = x + 1.0
        return y, y


class UniqueOutputModule(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        y = x + 1.0
        z = x + 2.0
        return y, z


def test_ensure_unique_output_nodes_no_target_inserts_identity_per_repeated_output() -> (
    None
):
    pipeline = PassPipeline[tuple[torch.Tensor]](
        DuplicateOutputModule(),
        (torch.rand(2, 2),),
        quantize=False,
        pass_list=[EnsureUniqueOutputNodesPass],
        ops_after_pass={
            "executorch_exir_dialects_backend__ops_tosa_IDENTITY_default": 2,
        },
    )
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()

    graph_module = (
        pipeline.tester.get_artifact(StageType.RUN_PASSES)
        .exported_program()
        .graph_module
    )
    output_node = graph_module.graph.output_node()
    outputs = list(output_node.args[0])

    assert outputs[0] is not outputs[1]
    assert outputs[0].target == exir_ops.backend.tosa.IDENTITY.default
    assert outputs[1].target == exir_ops.backend.tosa.IDENTITY.default
    assert outputs[0].args[0] is outputs[1].args[0]


def test_ensure_unique_output_nodes_no_target_keeps_unique_outputs_unchanged() -> None:
    pipeline = PassPipeline[tuple[torch.Tensor]](
        UniqueOutputModule(),
        (torch.rand(2, 2),),
        quantize=False,
        pass_list=[EnsureUniqueOutputNodesPass],
        ops_not_after_pass=[
            "executorch_exir_dialects_backend__ops_tosa_IDENTITY_default",
        ],
    )
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()
