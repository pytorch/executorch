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
import torch.nn.functional as F
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


class TransformerBlock(torch.nn.Module):
    def __init__(self, sdpa):
        super().__init__()
        self.sdpa = sdpa
        self.qkv = torch.nn.Linear(64, 192)
        self.ff = torch.nn.Linear(64, 64)

    def forward(self, x, lengths):
        b, s, _ = x.shape
        mask = (torch.arange(s)[None, :] < lengths[:, None])[:, None, None, :]
        q, k, v = self.qkv(x).view(b, s, 3, 2, 32).permute(2, 0, 3, 1, 4)
        if self.sdpa:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            bias = torch.where(mask, 0.0, -torch.inf)
            y = torch.softmax(q @ k.transpose(-1, -2) * 32**-0.5 + bias, -1) @ v
        return F.gelu(self.ff(y.transpose(1, 2).reshape(b, s, 64)))


class TestVulkanTransformer(unittest.TestCase):
    def _lower(
        self,
        model,
        inputs,
        dynamic_shapes=None,
        storage=VkStorageType.TEXTURE_3D,
        *,
        fully_delegated=True,
    ):
        options = {"require_dynamic_shapes": True, "storage_type_override": storage}
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

    def _run(self, edge, model, inputs, *, atol=1e-5, rtol=1e-4, equal_nan=False):
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

    def test_partition_transformer(self):
        for sdpa in (False, True):
            with self.subTest(sdpa=sdpa):
                self._lower(
                    TransformerBlock(sdpa),
                    (torch.randn(1, 16, 64), torch.tensor([16])),
                    ({1: Dim("s", min=2, max=1000)}, {}),
                )

    def test_dynamic_transformer(self):
        torch.manual_seed(0)
        for sdpa in (False, True):
            for storage in (VkStorageType.TEXTURE_3D, VkStorageType.BUFFER):
                with self.subTest(sdpa=sdpa, storage=storage):
                    model = TransformerBlock(sdpa).eval()
                    inputs = [
                        (torch.randn(1, s, 64), torch.tensor([length]))
                        for s, length in (
                            (16, 16),
                            (7, 4),
                            (31, 23),
                            (2, 1),
                            (16, 0 if sdpa else 5),
                        )
                    ]
                    edge = self._lower(
                        model, inputs[0], ({1: Dim("s", min=2, max=32)}, {}), storage
                    )
                    self._run(edge, model, inputs)

    def test_partition_any_unsupported_inputs(self):
        class AnyDim(torch.nn.Module):
            def forward(self, x):
                return torch.any(x, dim=0, keepdim=True)

        for x in (
            torch.tensor(True),
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.uint8),
            torch.tensor([0.0, 1.0]),
        ):
            with self.subTest(shape=x.shape, dtype=x.dtype):
                edge = to_edge_transform_and_lower(
                    export(AnyDim(), (x,)),
                    partitioner=[VulkanPartitioner({"require_dynamic_shapes": True})],
                )
                self.assertNotIn(
                    torch.ops.higher_order.executorch_call_delegate,
                    [node.target for node in edge.exported_program().graph.nodes],
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

    def test_scalar_tensor_values(self):
        class WhereScalars(torch.nn.Module):
            def __init__(self, positive, negative):
                super().__init__()
                self.positive = positive
                self.negative = negative

            def forward(self, x):
                return torch.where(x, self.positive, self.negative)

        inputs = [(torch.tensor([True, False, True, False]),)]
        for positive, negative in (
            (3, -7.0),
            (3.0, -7),
            (16777217, -7),
            (2**31 - 1, -(2**31)),
        ):
            with self.subTest(positive=positive, negative=negative):
                model = WhereScalars(positive, negative)
                edge = to_edge_transform_and_lower(
                    export(model, inputs[0]),
                    partitioner=[VulkanPartitioner()],
                )
                self.assertIn(
                    torch.ops.higher_order.executorch_call_delegate,
                    [node.target for node in edge.exported_program().graph.nodes],
                )
                self._run(edge, model, inputs, atol=0, rtol=0)

    def test_dynamic_expand(self):
        class Expand(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "offset", torch.arange(4, dtype=torch.float32)[None, :]
                )

            def forward(self, x):
                return x.expand(2, x.shape[1], 4) + self.offset.expand(2, 4)[:, None, :]

        for storage in (VkStorageType.TEXTURE_3D, VkStorageType.BUFFER):
            with self.subTest(storage=storage):
                model = Expand()
                inputs = [(torch.randn(1, s, 1),) for s in (16, 3, 31, 2, 16)]
                edge = self._lower(
                    model, inputs[0], ({1: Dim("s", min=2, max=32)},), storage
                )
                self._run(edge, model, inputs, atol=0, rtol=0)

    def test_dynamic_full(self):
        class Full(torch.nn.Module):
            def forward(self, x):
                return (
                    torch.full(x.shape, 2.5),
                    torch.zeros(x.shape),
                    torch.ones(x.shape),
                    torch.full_like(x, -1.5),
                    torch.zeros_like(x),
                    torch.ones_like(x),
                )

        for storage in (VkStorageType.TEXTURE_3D, VkStorageType.BUFFER):
            with self.subTest(storage=storage):
                model = Full()
                inputs = [(torch.randn(2, s, 3),) for s in (16, 3, 31, 2, 16)]
                edge = self._lower(
                    model, inputs[0], ({1: Dim("s", min=2, max=32)},), storage
                )
                self._run(edge, model, inputs, atol=0, rtol=0)

    def test_dynamic_any_dim(self):
        class AnyDim(torch.nn.Module):
            def __init__(self, dim, keepdim):
                super().__init__()
                self.dim = dim
                self.keepdim = keepdim

            def forward(self, x):
                return torch.any(x, dim=self.dim, keepdim=self.keepdim)

        for dim, keepdim in ((-1, True), (-1, False), (1, True)):
            for storage in (VkStorageType.TEXTURE_3D, VkStorageType.BUFFER):
                with self.subTest(dim=dim, keepdim=keepdim, storage=storage):
                    model = AnyDim(dim, keepdim)
                    inputs = []
                    for s in (16, 3, 31, 2, 16):
                        x = torch.zeros(2, s, 5, dtype=torch.bool)
                        if s != 3:
                            x[0, s // 2, 1] = True
                            x[1, -1, 3] = True
                            x[1, 0, 4] = True
                        inputs.append((x,))
                    seq = Dim("s", min=2, max=32)
                    unsupported = dim == 1 and storage == VkStorageType.BUFFER
                    edge = self._lower(
                        model,
                        inputs[0],
                        ({1: seq},),
                        storage,
                        fully_delegated=not unsupported,
                    )
                    if unsupported:
                        self.assertEqual(_vulkan_graphs(edge), [])
                    self._run(edge, model, inputs, atol=0, rtol=0)

    def test_dynamic_logical_not(self):
        class LogicalNot(torch.nn.Module):
            def forward(self, x):
                return torch.logical_not(x)

        model = LogicalNot()
        inputs = [
            ((torch.arange(3 * s).reshape(3, s) % 3 == 0),) for s in (7, 2, 15, 3, 7)
        ]
        for storage in (VkStorageType.TEXTURE_3D, VkStorageType.BUFFER):
            with self.subTest(storage=storage):
                edge = self._lower(
                    model, inputs[0], ({1: Dim("s", min=2, max=16)},), storage
                )
                self._run(edge, model, inputs, atol=0, rtol=0)

    def test_scalar_types_before_conv_and_view(self):
        class ScalarTypes(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 2, 3, padding=1)

            def forward(self, x):
                y = self.conv(torch.where(x > 0, 1.0, 0.5))
                return y.view(1, 2, x.shape[2], -1)

        torch.manual_seed(0)
        model = ScalarTypes().eval()
        inputs = [(torch.randn(1, 1, s, 5),) for s in (7, 2, 15, 7)]
        edge = self._lower(model, inputs[0], ({2: Dim("s", min=2, max=16)},))
        self._run(edge, model, inputs)

    def test_nan_scalars_fall_back(self):
        class NanScalar(torch.nn.Module):
            def __init__(self, kind):
                super().__init__()
                self.kind = kind

            def forward(self, x):
                if self.kind == "where":
                    return torch.where(x > 0, x, torch.nan)
                if self.kind == "masked_fill":
                    return x.masked_fill(x > 0, torch.nan)
                if self.kind == "full":
                    return torch.full_like(x, torch.nan)
                if self.kind == "scalar_tensor":
                    return torch.scalar_tensor(torch.nan)
                if self.kind == "pow":
                    return x**torch.nan
                return torch.ops.aten.mul.Scalar(x, torch.nan)

        inputs = [(torch.tensor([-1.0, 0.0, 1.0, 2.0]),)]
        for kind in ("where", "masked_fill", "full", "scalar_tensor", "pow", "mul"):
            with self.subTest(kind=kind):
                model = NanScalar(kind)
                edge = self._lower(model, inputs[0], fully_delegated=False)
                self._run(edge, model, inputs, atol=0, rtol=0, equal_nan=True)

    def test_integer_fill_values(self):
        class IntegerFill(torch.nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.dtype = dtype

            def forward(self, x):
                return (
                    torch.full(x.shape, 16777217, dtype=self.dtype),
                    torch.full_like(x, -(2**31), dtype=self.dtype),
                    torch.full(x.shape, 2**31 - 1, dtype=self.dtype),
                )

        inputs = [(torch.randn(2, s, 3),) for s in (7, 2, 15, 7)]
        for dtype in (torch.int32, torch.int64):
            for storage in (VkStorageType.TEXTURE_3D, VkStorageType.BUFFER):
                with self.subTest(dtype=dtype, storage=storage):
                    model = IntegerFill(dtype)
                    edge = self._lower(
                        model, inputs[0], ({1: Dim("s", min=2, max=16)},), storage
                    )
                    self._run(edge, model, inputs, atol=0, rtol=0)

    def test_large_integer_scalars_fall_back(self):
        class LargeScalar(torch.nn.Module):
            def __init__(self, kind, value):
                super().__init__()
                self.kind = kind
                self.value = value

            def forward(self, x):
                if self.kind == "scalar_tensor":
                    return torch.scalar_tensor(self.value, dtype=torch.int64)
                if self.kind == "full":
                    return torch.full(x.shape, self.value, dtype=torch.int64)
                return torch.full_like(x, self.value, dtype=torch.int64)

        inputs = [(torch.zeros(2, 3),)]
        for kind in ("scalar_tensor", "full", "full_like"):
            for value in (2**31, 2**40, 2**63 - 1, -(2**63)):
                with self.subTest(kind=kind, value=value):
                    model = LargeScalar(kind, value)
                    edge = self._lower(model, inputs[0], fully_delegated=False)
                    self.assertEqual(_vulkan_graphs(edge), [])
                    self._run(edge, model, inputs, atol=0, rtol=0)

    def test_bool_fill_values(self):
        class BoolFill(torch.nn.Module):
            def forward(self, x):
                return (
                    torch.full_like(x, 0.5, dtype=torch.bool),
                    torch.full_like(x, -1.5, dtype=torch.bool),
                    torch.full(x.shape, 0, dtype=torch.bool),
                )

        model = BoolFill()
        inputs = [(torch.zeros(3, 7),)]
        for storage in (VkStorageType.TEXTURE_3D, VkStorageType.BUFFER):
            with self.subTest(storage=storage):
                edge = self._lower(model, inputs[0], storage=storage)
                self._run(edge, model, inputs, atol=0, rtol=0)

    def test_dynamic_scalar_values_fall_back(self):
        class DynamicScalars(torch.nn.Module):
            def forward(self, x):
                n = x.shape[0]
                value = n * 2
                return (
                    x**value,
                    torch.ops.aten.mul.Scalar(x, value),
                    torch.full((n,), value),
                    torch.scalar_tensor(value, dtype=torch.int64),
                    torch.full((n,), 0.5),
                    F.gelu(x),
                )

        model = DynamicScalars()
        inputs = [(torch.linspace(0.1, 0.9, n),) for n in (4, 2, 7, 3, 4)]
        edge = self._lower(
            model, inputs[0], ({0: Dim("n", min=2, max=8)},), fully_delegated=False
        )
        self.assertTrue(_vulkan_graphs(edge))
        self._run(edge, model, inputs)

    def test_partition_4d_reductions(self):
        class Reduce(torch.nn.Module):
            def __init__(self, op, dim):
                super().__init__()
                self.op = op
                self.dim = dim

            def forward(self, x):
                return self.op(x, dim=self.dim, keepdim=True)

        for op in (torch.any, torch.sum, torch.mean, torch.amax):
            for batch, dim, supported in (
                (1, 0, False),
                (2, 0, False),
                (2, 1, False),
                (1, 1, True),
                (2, 2, True),
                (2, -1, True),
            ):
                with self.subTest(op=op, batch=batch, dim=dim):
                    x = torch.randn(batch, 3, 4, 5)
                    if op == torch.any:
                        x = x > 0
                    edge = self._lower(Reduce(op, dim), (x,), fully_delegated=supported)
                    if not supported:
                        self.assertEqual(_vulkan_graphs(edge), [])
                    edge.to_executorch()

    @unittest.skipUnless(USING_SWIFTSHADER, "requires a device without 8-bit buffers")
    def test_bool_buffers_fail_cleanly_without_8bit_storage(self):
        from executorch.extension.pybindings.portable_lib import (
            _load_for_executorch_from_buffer,
        )

        class LogicalNot(torch.nn.Module):
            def forward(self, x):
                return torch.logical_not(x)

        inputs = (torch.zeros(3, 7, dtype=torch.bool),)
        edge = self._lower(LogicalNot(), inputs, storage=VkStorageType.BUFFER)
        program_buffer = edge.to_executorch().buffer
        module = _load_for_executorch_from_buffer(program_buffer)
        with self.assertRaisesRegex(RuntimeError, r"0x:?10\b"):
            module.run_method("forward", inputs)


if __name__ == "__main__":
    unittest.main()
