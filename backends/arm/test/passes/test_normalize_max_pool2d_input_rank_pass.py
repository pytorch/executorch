# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm._passes import (
    NormalizeMaxPool2dInputRankPass,
    RemoveGetItemPass,
)
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline
from executorch.backends.test.harness.stages import StageType
from executorch.exir.dialects._ops import ops as exir_ops


input_t = Tuple[torch.Tensor]


class MaxPool2d(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.max_pool2d(
            x,
            kernel_size=(3, 2),
            stride=(2, 1),
            padding=(1, 0),
            dilation=(1, 1),
            ceil_mode=True,
        )


def test_normalize_rank3_max_pool2d_input() -> None:
    pipeline = PassPipeline[input_t](
        MaxPool2d(),
        (torch.rand(3, 9, 11),),
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_max_pool2d_with_indices_default": 1,
        },
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_unsqueeze_copy_default": 1,
            "executorch_exir_dialects_edge__ops_aten_max_pool2d_default": 1,
            "executorch_exir_dialects_edge__ops_aten_squeeze_copy_dims": 1,
        },
        pass_list=[RemoveGetItemPass, NormalizeMaxPool2dInputRankPass],
    )
    pipeline.run()

    exported_program = pipeline.tester.get_artifact(
        StageType.RUN_PASSES
    ).exported_program()
    pool_node = next(
        node
        for node in exported_program.graph.nodes
        if node.target == exir_ops.edge.aten.max_pool2d.default
    )
    unsqueeze_node = pool_node.args[0]
    assert isinstance(unsqueeze_node, torch.fx.Node)
    assert unsqueeze_node.target == exir_ops.edge.aten.unsqueeze_copy.default
    assert unsqueeze_node.args[1] == 0
    assert tuple(pool_node.args[1]) == (3, 2)
    assert tuple(pool_node.args[2]) == (2, 1)
    assert tuple(pool_node.args[3]) == (1, 0)
    assert tuple(pool_node.args[4]) == (1, 1)
    assert pool_node.args[5] is True

    squeeze_node = next(iter(pool_node.users))
    assert squeeze_node.target == exir_ops.edge.aten.squeeze_copy.dims
    assert squeeze_node.args == (pool_node, [0])


def test_normalize_rank4_max_pool2d_input_is_noop() -> None:
    PassPipeline[input_t](
        MaxPool2d(),
        (torch.rand(1, 3, 9, 11),),
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_max_pool2d_with_indices_default": 1,
        },
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_max_pool2d_default": 1,
        },
        ops_not_after_pass=[
            "executorch_exir_dialects_edge__ops_aten_unsqueeze_copy_default",
            "executorch_exir_dialects_edge__ops_aten_squeeze_copy_dims",
        ],
        pass_list=[RemoveGetItemPass, NormalizeMaxPool2dInputRankPass],
    ).run()
