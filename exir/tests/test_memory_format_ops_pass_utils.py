# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest
from dataclasses import dataclass
from typing import Any, Tuple

import torch

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge, to_edge_transform_and_lower
from executorch.exir.capture._config import EdgeCompileConfig

from executorch.exir.dim_order_utils import (
    is_channel_last_dim_order,
    is_contiguous_dim_order,
)

from torch.export import export
from torch.testing import FileCheck
from torch.utils._pytree import tree_flatten


@dataclass
class MemoryFormatTestSet:
    module: torch.nn.Module
    sample_input: Tuple[Any, ...]
    target_memory_format: torch.memory_format
    _load_for_executorch_from_buffer: Any
    op_level_check: bool = True
    use_xnnpack: bool = False
    rtol: float = 1e-05
    atol: float = 1e-08


class SimpleToCopyContiguousModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(dtype=torch.double, memory_format=torch.contiguous_format)


class SimpleToCopyChannelsLastModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(dtype=torch.double, memory_format=torch.channels_last)


class PropagateToCopyChannalsLastModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t1 = x.to(dtype=torch.double, memory_format=torch.channels_last)
        t2 = t1 + t1
        return t1 * t2


class MemoryFormatOpsPassTestUtils:
    @staticmethod
    def memory_format_test_runner(
        test_class: unittest.TestCase, test_set: MemoryFormatTestSet
    ):
        before = export(test_set.module, test_set.sample_input)

        if test_set.use_xnnpack:
            epm = to_edge_transform_and_lower(
                before,
                compile_config=EdgeCompileConfig(
                    _skip_dim_order=False, _check_ir_validity=False
                ),
                partitioner=[XnnpackPartitioner()],
            )
        else:
            epm = to_edge(
                before, compile_config=EdgeCompileConfig(_skip_dim_order=False)
            )

        # check memory format ops, if needed
        if test_set.op_level_check:
            aten_op_str = "torch.ops.aten._to_copy.default"
            edge_op_str = "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default"

            # check op strings before
            FileCheck().check_count(aten_op_str, 1, exactly=True).check_not(
                edge_op_str
            ).run(before.graph_module.code)

            # check op strings
            FileCheck().check_not(aten_op_str).check_count(
                edge_op_str, 1, exactly=True
            ).run(epm.exported_program().graph_module.code)

        # check EdgeOp and the new BackendOp should behave the same
        expected = before.module()(*test_set.sample_input)
        actual = epm.exported_program().module()(*test_set.sample_input)
        test_class.assertTrue(
            torch.allclose(actual, expected, atol=test_set.atol, rtol=test_set.rtol)
        )
        test_class.assertEqual(
            is_channel_last_dim_order(actual),
            is_channel_last_dim_order(expected),
        )
        if test_set.target_memory_format == torch.channels_last:
            test_class.assertTrue(is_channel_last_dim_order(actual))
        elif test_set.target_memory_format == torch.contiguous_format:
            test_class.assertTrue(is_contiguous_dim_order(actual))
        else:
            raise RuntimeError("Unknown memory format")

        # check EdgeOp and the new BackendOp should behave the same in the runtime
        executorch_prog = epm.to_executorch()

        executorch_module = test_set._load_for_executorch_from_buffer(
            executorch_prog.buffer
        )
        inputs_flattened = tree_flatten(test_set.sample_input)[0]
        runtime_output = executorch_module.run_method(
            "forward", tuple(inputs_flattened)
        )[0]
        test_class.assertTrue(
            torch.allclose(
                runtime_output, expected, atol=test_set.atol, rtol=test_set.rtol
            )
        )
        test_class.assertEqual(
            is_channel_last_dim_order(runtime_output),
            is_channel_last_dim_order(expected),
        )
