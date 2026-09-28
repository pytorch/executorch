# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import math

import operator

import os

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner

from executorch.backends.vulkan.serialization.vulkan_graph_schema import (
    VkDataType,
    VkStorageType,
    VkTensor,
)

from executorch.backends.vulkan.serialization.vulkan_graph_serialize import (
    extract_vk_flatbuffer,
    flatbuffer_to_vk_graph,
)

from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

from executorch.exir.lowered_backend_module import LoweredBackendModule

from torch.export import Dim, export

USING_SWIFTSHADER = os.environ.get("ETVK_USING_SWIFTSHADER") in ("1", "True")


def _vulkan_graphs(edge):
    return [
        flatbuffer_to_vk_graph(extract_vk_flatbuffer(module.processed_bytes))
        for module in edge.exported_program().graph_module.modules()
        if isinstance(module, LoweredBackendModule)
        and module.backend_id == "VulkanBackend"
    ]


class TestVulkanDynamic(unittest.TestCase):
    def _lower(
        self,
        model,
        inputs,
        dynamic_shapes=None,
        storage=VkStorageType.TEXTURE_3D,
        *,
        fully_delegated=True,
        downcast_64_bit=True,
    ):
        options = {
            "require_dynamic_shapes": True,
            "storage_type_override": storage,
            "downcast_64_bit": downcast_64_bit,
        }
        if storage == VkStorageType.BUFFER:
            options["texture_limits"] = (1, 1, 1)
        edge = to_edge_transform_and_lower(
            export(model.eval(), inputs, dynamic_shapes=dynamic_shapes, strict=False),
            partitioner=[VulkanPartitioner(options)],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        targets = [
            node.target
            for node in edge.exported_program().graph.nodes
            if node.op == "call_function" and node.target != operator.getitem
        ]
        if fully_delegated:
            self.assertEqual(targets, [torch.ops.higher_order.executorch_call_delegate])
        if storage == VkStorageType.BUFFER:
            for graph in _vulkan_graphs(edge):
                for value_id in graph.input_ids + graph.output_ids:
                    value = graph.values[value_id].value
                    if isinstance(value, VkTensor) and math.prod(value.dims) > 4:
                        self.assertEqual(value.storage_type, VkStorageType.BUFFER)
        return edge

    def _run(
        self,
        edge,
        model,
        inputs,
        *,
        atol=1e-5,
        rtol=1e-4,
        equal_nan=False,
        check_signed_zero=False,
    ):
        from executorch.extension.pybindings.portable_lib import (
            _load_for_executorch_from_buffer,
        )

        if USING_SWIFTSHADER and any(
            isinstance(value.value, VkTensor)
            and value.value.constant_id < 0
            and value.value.datatype == VkDataType.BOOL
            and value.value.storage_type == VkStorageType.BUFFER
            for graph in _vulkan_graphs(edge)
            for value in graph.values
        ):
            self.skipTest("SwiftShader does not support 8-bit storage buffers")

        program_buffer = edge.to_executorch().buffer
        module = _load_for_executorch_from_buffer(program_buffer)
        for sample in inputs:
            with self.subTest(shapes=[tuple(x.shape) for x in sample]):
                actual = module.run_method("forward", sample)
                expected = model(*sample)
                if isinstance(expected, torch.Tensor):
                    expected = (expected,)
                self.assertEqual(len(actual), len(expected))
                for output, reference in zip(actual, expected):
                    torch.testing.assert_close(
                        output, reference, atol=atol, rtol=rtol, equal_nan=equal_nan
                    )
                    if check_signed_zero:
                        zeros = reference == 0
                        self.assertTrue(
                            torch.equal(
                                torch.signbit(output[zeros]),
                                torch.signbit(reference[zeros]),
                            )
                        )

    def test_dynamic_gelu(self):
        for approximate in ("none", "tanh"):
            for storage in (VkStorageType.TEXTURE_3D, VkStorageType.BUFFER):
                for dtype in (torch.float32, torch.float16):
                    with self.subTest(
                        approximate=approximate, storage=storage, dtype=dtype
                    ):
                        model = torch.nn.GELU(approximate=approximate)
                        inputs = [
                            (torch.linspace(-6, 6, s, dtype=dtype).repeat(3, 1),)
                            for s in (257, 17, 511, 2, 257)
                        ]
                        edge = self._lower(
                            model, inputs[0], ({1: Dim("s", min=2, max=512)},), storage
                        )
                        tolerance = 5e-6 if dtype == torch.float32 else 1e-3
                        self._run(edge, model, inputs, atol=tolerance, rtol=tolerance)

    def test_gelu_with_singleton_dimensions(self):
        for approximate in ("none", "tanh"):
            for shape in ((6, 1, 3), (2, 1, 3, 5)):
                for storage in (VkStorageType.TEXTURE_3D, VkStorageType.BUFFER):
                    with self.subTest(
                        approximate=approximate, shape=shape, storage=storage
                    ):
                        model = torch.nn.GELU(approximate=approximate)
                        x = torch.linspace(-6, 6, math.prod(shape)).reshape(shape)
                        edge = self._lower(model, (x,), storage=storage)
                        self._run(edge, model, [(x,)], atol=5e-6, rtol=5e-6)

    def test_buffer_reduction_range(self):
        class Reduce(torch.nn.Module):
            def __init__(self, op):
                super().__init__()
                self.op = op

            def forward(self, x):
                return self.op(x, dim=-1, keepdim=True)

        for op in (torch.sum, torch.mean, torch.amax):
            with self.subTest(op=op):
                width, value = (20000, 4) if op == torch.sum else (8, 80000)
                x = torch.tensor([value, -value], dtype=torch.float32)[:, None].repeat(
                    1, width
                )
                model = Reduce(op)
                edge = self._lower(model, (x,), storage=VkStorageType.BUFFER)
                self._run(edge, model, [(x,)], atol=0, rtol=0)

    def test_reduction_special_values(self):
        class Reduce(torch.nn.Module):
            def __init__(self, op):
                super().__init__()
                self.op = op

            def forward(self, x):
                return self.op(x, dim=-1, keepdim=True)

        for dtype in (torch.float32, torch.float16):
            x = torch.tensor(
                [
                    [40000] * 9,
                    [-40000] * 9,
                    [1, torch.nan, 2, 3, torch.nan, 4, 5, 6, 7],
                ],
                dtype=dtype,
            )
            for op in (torch.sum, torch.mean, torch.amax, torch.amin):
                for storage in (VkStorageType.TEXTURE_3D, VkStorageType.BUFFER):
                    with self.subTest(dtype=dtype, op=op, storage=storage):
                        model = Reduce(op)
                        edge = self._lower(model, (x,), storage=storage)
                        self._run(edge, model, [(x,)], atol=0, rtol=0, equal_nan=True)

        x = torch.tensor([4, -4], dtype=torch.float16)[:, None].repeat(1, 70000)
        model = Reduce(torch.mean)
        edge = self._lower(model, (x,), storage=VkStorageType.BUFFER)
        self._run(edge, model, [(x,)], atol=0, rtol=0)

    def test_argreduce_first_nan(self):
        class Reduce(torch.nn.Module):
            def __init__(self, op):
                super().__init__()
                self.op = op

            def forward(self, x):
                return self.op(x, dim=-1, keepdim=True)

        for dtype in (torch.float32, torch.float16):
            x = torch.tensor(
                [
                    [1, torch.nan, 2, torch.nan, 3, 4, 5],
                    [1, 2, 3, 4, 5, torch.nan, torch.nan],
                ],
                dtype=dtype,
            )
            for op in (torch.argmax, torch.argmin):
                with self.subTest(dtype=dtype, op=op):
                    model = Reduce(op)
                    edge = self._lower(model, (x,), storage=VkStorageType.BUFFER)
                    self._run(edge, model, [(x,)], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
