# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import dataclass
from typing import Any, Tuple

import torch
from executorch.exir import to_edge

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
    is_aten_mode: bool


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


class TestMemoryFormatOpsPass(unittest.TestCase):
    def memory_format_test_runner(self, test_set: MemoryFormatTestSet):
        aten_op_str = "torch.ops.aten._to_copy.default"
        edge_op_str = "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default"

        before = export(test_set.module, test_set.sample_input)

        # check op strings before
        FileCheck().check_count(aten_op_str, 1, exactly=True).check_not(
            edge_op_str
        ).run(before.graph_module.code)

        epm = to_edge(before)

        # check op strings
        FileCheck().check_not(aten_op_str).check_count(
            edge_op_str, 1, exactly=True
        ).run(epm.exported_program().graph_module.code)

        # check EdgeOp and the new BackendOp should behave the same
        expected = before.module()(*test_set.sample_input)
        actual = epm.exported_program().module()(*test_set.sample_input)
        self.assertTrue(torch.allclose(actual, expected))
        self.assertEqual(
            is_channel_last_dim_order(actual),
            is_channel_last_dim_order(expected),
        )
        if test_set.target_memory_format == torch.channels_last:
            self.assertTrue(is_channel_last_dim_order(actual))
        elif test_set.target_memory_format == torch.contiguous_format:
            self.assertTrue(is_contiguous_dim_order(actual))
        else:
            raise RuntimeError("Unknown memory format")

        # check EdgeOp and the new BackendOp should behave the same in the runtime
        executorch_prog = epm.to_executorch()

        if test_set.is_aten_mode:
            from executorch.extension.pybindings.aten_lib import (  # @manual
                _load_for_executorch_from_buffer,
            )
        else:
            from executorch.extension.pybindings.portable_lib import (  # @manual
                _load_for_executorch_from_buffer,
            )

        executorch_module = _load_for_executorch_from_buffer(executorch_prog.buffer)
        inputs_flattened = tree_flatten(test_set.sample_input)[0]
        runtime_output = executorch_module.run_method(
            "forward", tuple(inputs_flattened)
        )[0]
        self.assertTrue(torch.allclose(runtime_output, expected))
        self.assertEqual(
            is_channel_last_dim_order(runtime_output),
            is_channel_last_dim_order(expected),
        )

    def test_op_to_copy_replacement_2d(self) -> None:
        self.memory_format_test_runner(
            MemoryFormatTestSet(
                module=SimpleToCopyContiguousModule().eval(),
                sample_input=(torch.randn([3, 4, 5], dtype=torch.float32),),
                target_memory_format=torch.contiguous_format,
                is_aten_mode=False,
            )
        )

    def test_op_to_copy_replacement_2d_aten(self) -> None:
        self.memory_format_test_runner(
            MemoryFormatTestSet(
                module=SimpleToCopyContiguousModule().eval(),
                sample_input=(torch.randn([3, 4, 5], dtype=torch.float32),),
                target_memory_format=torch.contiguous_format,
                is_aten_mode=True,
            )
        )

    def test_op_to_copy_replacement_4d(self) -> None:
        self.memory_format_test_runner(
            MemoryFormatTestSet(
                module=SimpleToCopyContiguousModule().eval(),
                sample_input=(torch.randn([3, 4, 5, 6], dtype=torch.float32),),
                target_memory_format=torch.contiguous_format,
                is_aten_mode=False,
            )
        )

    def test_op_to_copy_replacement_4d_aten(self) -> None:
        self.memory_format_test_runner(
            MemoryFormatTestSet(
                module=SimpleToCopyContiguousModule().eval(),
                sample_input=(torch.randn([3, 4, 5, 6], dtype=torch.float32),),
                target_memory_format=torch.contiguous_format,
                is_aten_mode=True,
            )
        )

    def test_op_dim_order_update(self) -> None:
        self.memory_format_test_runner(
            MemoryFormatTestSet(
                module=SimpleToCopyChannelsLastModule().eval(),
                sample_input=(
                    torch.rand_like(
                        torch.zeros([2, 2, 2, 2]),
                        dtype=torch.float32,
                        memory_format=torch.contiguous_format,
                    ),
                ),
                target_memory_format=torch.channels_last,
                is_aten_mode=False,
            ),
        )

    def test_op_dim_order_update_aten(self) -> None:
        self.memory_format_test_runner(
            MemoryFormatTestSet(
                module=SimpleToCopyChannelsLastModule().eval(),
                sample_input=(
                    torch.rand_like(
                        torch.zeros([2, 2, 2, 2]),
                        dtype=torch.float32,
                        memory_format=torch.contiguous_format,
                    ),
                ),
                target_memory_format=torch.channels_last,
                is_aten_mode=True,
            ),
        )

    def test_op_dim_order_propagation(self) -> None:
        self.memory_format_test_runner(
            MemoryFormatTestSet(
                module=PropagateToCopyChannalsLastModule().eval(),
                sample_input=(
                    torch.rand_like(
                        torch.zeros([2, 2, 2, 2]),
                        dtype=torch.float32,
                        memory_format=torch.contiguous_format,
                    ),
                ),
                target_memory_format=torch.channels_last,
                is_aten_mode=False,
            )
        )

    def test_op_dim_order_propagation_aten(self) -> None:
        self.memory_format_test_runner(
            MemoryFormatTestSet(
                module=PropagateToCopyChannalsLastModule().eval(),
                sample_input=(
                    torch.rand_like(
                        torch.zeros([2, 2, 2, 2]),
                        dtype=torch.float32,
                        memory_format=torch.contiguous_format,
                    ),
                ),
                target_memory_format=torch.channels_last,
                is_aten_mode=True,
            )
        )
