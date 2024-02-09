# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import dataclass
from typing import Any, List, Tuple

import torch
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.passes import MemoryFormatOpsPass
from torch.export import export
from torch.testing import FileCheck


class TestMemoryFormatOpsPass(unittest.TestCase):
    def test_op_to_copy_replacement(self) -> None:
        class ContiguousModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.to(dtype=torch.double, memory_format=torch.contiguous_format)

        class ChannelsLastModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.to(dtype=torch.double, memory_format=torch.channels_last)

        @dataclass
        class TestSet:
            module: torch.nn.Module
            sample_input: Tuple[Any, ...]

        contiguous_module = ContiguousModule().eval()
        channels_last_module = ChannelsLastModule().eval()

        all_test_sets: List[TestSet] = [
            TestSet(
                module=contiguous_module,
                sample_input=(torch.randn([2, 2], dtype=torch.float32),),
            ),
            TestSet(
                module=contiguous_module,
                sample_input=(torch.randn([2, 2, 2], dtype=torch.float32),),
            ),
            TestSet(
                module=channels_last_module,
                sample_input=(torch.randn([2, 2, 2, 2], dtype=torch.float32),),
            ),
            TestSet(
                module=channels_last_module,
                sample_input=(
                    torch.rand_like(
                        torch.zeros([2, 2, 2, 2]),
                        dtype=torch.float32,
                        memory_format=torch.channels_last,
                    ),
                ),
            ),
        ]

        aten_op_str = "torch.ops.aten._to_copy.default"
        edge_op_str = "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default"

        for test_set in all_test_sets:
            before = export(test_set.module, test_set.sample_input)

            # check op strings before
            FileCheck().check_count(aten_op_str, 1, exactly=True).check_not(
                edge_op_str
            ).run(before.graph_module.code)

            # TODO(gasoonjia): make to_dim_copy pass verifier
            epm = to_edge(
                before,
                compile_config=EdgeCompileConfig(
                    _check_ir_validity=False, _skip_dim_order=False
                ),
            )

            # check op strings
            FileCheck().check_not(aten_op_str).check_count(
                edge_op_str, 1, exactly=True
            ).run(epm.exported_program().graph_module.code)

            # check EdgeOp and the new BackendOp should behave the same
            expected = before(*test_set.sample_input)
            actual = epm.exported_program()(*test_set.sample_input)
            self.assertTrue(torch.allclose(actual, expected))

            # TODO - more
            epm.to_executorch()
