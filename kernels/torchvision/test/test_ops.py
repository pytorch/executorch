# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass
from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
)
from torchvision.ops import nms


class WrapperModule(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class TestOps(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_nms(self) -> None:
        def apply_nms(boxes, scores, iou_threshold=0.5):
            keep = nms(boxes, scores, iou_threshold)
            torch._check(keep.shape[0] <= 200)
            return keep

        boxes = torch.tensor(
            [[10, 10, 50, 50], [20, 20, 60, 60], [40, 40, 70, 70]], dtype=torch.float32
        )
        scores = torch.tensor([0.9, 0.8, 0.7])
        inputs = (boxes, scores)
        ep = torch.export.export(WrapperModule(apply_nms), inputs)

        edge = to_edge(
            ep,
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
                _use_edge_ops=True,
            ),
        )
        executorch_program = edge.to_executorch(
            ExecutorchBackendConfig(
                sym_shape_eval_pass=ConstraintBasedSymShapeEvalPass()
            )
        )

        executorch_module = _load_for_executorch_from_buffer(executorch_program.buffer)
        et_output = executorch_module(boxes, scores)  # pyre-ignore
        eager_output = apply_nms(boxes, scores)
        torch.testing.assert_close(et_output, eager_output)
