# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torchvision

from executorch.exir.tests.test_memory_format_ops_pass_utils import (
    MemoryFormatOpsPassTestUtils,
    MemoryFormatTestSet,
    PropagateToCopyChannalsLastModule,
    SimpleToCopyChannelsLastModule,
    SimpleToCopyContiguousModule,
)

from executorch.extension.pybindings.aten_lib import (  # @manual
    _load_for_executorch_from_buffer,
)


class TestMemoryFormatOpsPass(unittest.TestCase):
    def test_op_to_copy_replacement_2d_aten(self) -> None:
        MemoryFormatOpsPassTestUtils.memory_format_test_runner(
            self,
            MemoryFormatTestSet(
                module=SimpleToCopyContiguousModule().eval(),
                sample_input=(torch.randn([3, 4, 5], dtype=torch.float32),),
                target_memory_format=torch.contiguous_format,
                _load_for_executorch_from_buffer=_load_for_executorch_from_buffer,
            ),
        )

    def test_op_to_copy_replacement_4d_aten(self) -> None:
        MemoryFormatOpsPassTestUtils.memory_format_test_runner(
            self,
            MemoryFormatTestSet(
                module=SimpleToCopyContiguousModule().eval(),
                sample_input=(torch.randn([3, 4, 5, 6], dtype=torch.float32),),
                target_memory_format=torch.contiguous_format,
                _load_for_executorch_from_buffer=_load_for_executorch_from_buffer,
            ),
        )

    def test_op_dim_order_update_aten(self) -> None:
        MemoryFormatOpsPassTestUtils.memory_format_test_runner(
            self,
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
                _load_for_executorch_from_buffer=_load_for_executorch_from_buffer,
            ),
        )

    def test_op_dim_order_propagation_aten(self) -> None:
        MemoryFormatOpsPassTestUtils.memory_format_test_runner(
            self,
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
                _load_for_executorch_from_buffer=_load_for_executorch_from_buffer,
            ),
        )

    def test_resnet18(self) -> None:
        model = torchvision.models.resnet18()
        MemoryFormatOpsPassTestUtils.memory_format_test_runner(
            self,
            MemoryFormatTestSet(
                module=model.eval(),
                sample_input=(torch.randn(1, 3, 224, 224),),
                target_memory_format=torch.contiguous_format,
                op_level_check=False,
                _load_for_executorch_from_buffer=_load_for_executorch_from_buffer,
                atol=1e-3,
                rtol=1e-3,
            ),
        )

    def test_mobilenet_v3(self) -> None:
        model = torchvision.models.mobilenetv3.mobilenet_v3_small(pretrained=True)
        MemoryFormatOpsPassTestUtils.memory_format_test_runner(
            self,
            MemoryFormatTestSet(
                module=model.eval(),
                sample_input=(torch.randn(1, 3, 224, 224),),
                target_memory_format=torch.contiguous_format,
                op_level_check=False,
                _load_for_executorch_from_buffer=_load_for_executorch_from_buffer,
                atol=1e-3,
                rtol=1e-3,
            ),
        )
