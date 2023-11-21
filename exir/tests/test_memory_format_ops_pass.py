# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch import exir
from executorch.exir import CaptureConfig, EdgeCompileConfig
from executorch.exir.passes import MemoryFormatOpsPass
from torch.testing import FileCheck


class TestMemoryFormatOpsPass(unittest.TestCase):
    def test_op_to_copy_replacement(self) -> None:
        class F(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(
                self, x: torch.Tensor, mem_format: torch.memory_format
            ) -> torch.Tensor:
                return x.to(dtype=torch.double, memory_format=mem_format)

        module = F().eval()
        sample_inputs = [
            (torch.randn([2, 2], dtype=torch.float32), torch.contiguous_format),
            (torch.randn([2, 2, 2], dtype=torch.float32), torch.contiguous_format),
            (torch.randn([2, 2, 2, 2], dtype=torch.float32), torch.channels_last),
            (
                torch.rand_like(
                    torch.zeros([2, 2, 2, 2]),
                    dtype=torch.float32,
                    memory_format=torch.channels_last,
                ),
                torch.contiguous_format,
            ),
        ]

        aten_op_str = "torch.ops.aten._to_copy.default"
        edge_op_str = "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default"

        for sample_input in sample_inputs:
            before = exir.capture(
                module,
                sample_input,
                CaptureConfig(enable_dynamic_shape=True),
            )

            # check op strings before
            FileCheck().check_count(aten_op_str, 1, exactly=True).check_not(
                edge_op_str
            ).run(before.exported_program.graph_module.code)

            ep = before.to_edge(
                config=EdgeCompileConfig(_use_edge_ops=True)
            )  # Only replacing edge_ops

            # Run the pass - TODO move this in to_edge passes
            after = ep.transform([MemoryFormatOpsPass()], check_ir_validity=False)

            # check op strings
            FileCheck().check_not(aten_op_str).check_count(
                edge_op_str, 1, exactly=True
            ).run(after.exported_program.graph_module.code)

            # check EdgeOp and the new BackendOp should behave the same
            expected = before(*sample_input)
            actual = after(*sample_input)
            self.assertTrue(torch.allclose(actual, expected))

            # TODO - more
            after.to_executorch()
