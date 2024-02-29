# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ctypes
import random
import unittest
from typing import List

import torch

from executorch.backends.vulkan.serialization.vulkan_graph_schema import VkGraph

from executorch.backends.vulkan.serialization.vulkan_graph_serialize import (
    serialize_vulkan_graph,
    VulkanDelegateHeader,
)


class TestSerialization(unittest.TestCase):
    def _generate_random_const_tensors(self, num_tensors: int) -> List[torch.Tensor]:
        """
        Helper function to generate `num_tensor` buffers of random sizes and random contents,
        we return a tuple of (list_of_buffers, list_of_mem_sizes),
        """
        tensors = []
        for _ in range(num_tensors):
            width = random.randint(4, 100)
            height = random.randint(4, 100)
            channels = random.randint(2, 8)

            tensor = torch.randn(channels, width, height)
            tensors.append(tensor)

        return tensors

    def test_serialize_vulkan_binary(self):
        vk_graph = VkGraph(
            version="0",
            chain=[],
            values=[],
            input_ids=[],
            output_ids=[],
            constants=[],
            shaders=[],
        )
        const_tensors = self._generate_random_const_tensors(5)

        serialized_binary = serialize_vulkan_graph(vk_graph, const_tensors, [])

        # Check header
        self.assertEqual(serialized_binary[0:4], b"\x00\x00\x00\x00")
        self.assertEqual(serialized_binary[VulkanDelegateHeader.MAGIC_IX], b"VH00")
        flatbuffer_offset = int.from_bytes(
            serialized_binary[VulkanDelegateHeader.FLATBUFFER_OFFSET_IX],
            byteorder="little",
        )
        constants_offset = int.from_bytes(
            serialized_binary[VulkanDelegateHeader.BYTES_OFFSET_IX],
            byteorder="little",
        )
        constants_size = int.from_bytes(
            serialized_binary[VulkanDelegateHeader.BYTES_SIZE_IX],
            byteorder="little",
        )

        # Flatbuffer magic should be in the same spot as the Header's magic
        self.assertEqual(
            serialized_binary[flatbuffer_offset:][VulkanDelegateHeader.MAGIC_IX],
            b"VK00",
        )

        constant_data_payload = serialized_binary[
            constants_offset : constants_offset + constants_size
        ]

        # We check that constant data indexes stored in the vk_graph correctly index
        # into the correct buffer in the constant data section
        self.assertEqual(len(vk_graph.constants), len(const_tensors))
        for bytes_range, tensor in zip(vk_graph.constants, const_tensors):
            offset = bytes_range.offset
            length = bytes_range.length

            constant_data_bytes = constant_data_payload[offset : offset + length]

            array_type = ctypes.c_char * tensor.untyped_storage().nbytes()
            array = ctypes.cast(
                tensor.untyped_storage().data_ptr(),
                ctypes.POINTER(array_type),
            ).contents

            tensor_bytes = bytes(array)
            self.assertEqual(constant_data_bytes, tensor_bytes)
