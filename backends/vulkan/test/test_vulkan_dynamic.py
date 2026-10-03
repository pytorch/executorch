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

from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
    parse_compile_options,
    VulkanPartitioner,
)

from executorch.backends.vulkan.serialization.vulkan_graph_schema import (
    VkDataType,
    VkStorageType,
    VkTensor,
)

from executorch.backends.vulkan.serialization.vulkan_graph_serialize import (
    extract_vk_flatbuffer,
    flatbuffer_to_vk_graph,
)

from executorch.exir import EdgeCompileConfig, to_edge, to_edge_transform_and_lower

from executorch.exir.backend.backend_api import to_backend

from executorch.exir.dialects._ops import ops as exir_ops

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


class ConstantMask(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mask", torch.arange(21).reshape(3, 7) % 2 == 0)

    def forward(self, x):
        return torch.where(self.mask, x, -x)


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
                fully_delegated = isinstance(positive, float) or isinstance(
                    negative, float
                )
                edge = self._lower(model, inputs[0], fully_delegated=fully_delegated)
                if not fully_delegated:
                    self.assertEqual(
                        [
                            node.target
                            for node in edge.exported_program().graph.nodes
                            if node.op == "call_function"
                            and node.target != operator.getitem
                        ],
                        [
                            torch.ops.higher_order.executorch_call_delegate,
                            exir_ops.edge.aten.where.self,
                        ],
                    )
                self._run(edge, model, inputs, atol=0, rtol=0)

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

    def test_scalar_tensor_dtypes(self):
        class ScalarTensor(torch.nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.dtype = dtype

            def forward(self, x):
                return torch.scalar_tensor(2.5, dtype=self.dtype)

        inputs = [(torch.ones(1),)]
        for dtype in (torch.float16, torch.float32, torch.int32):
            with self.subTest(dtype=dtype):
                model = ScalarTensor(dtype)
                edge = self._lower(model, inputs[0])
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

    def test_any_without_keepdim_preserves_texture_delegates(self):
        class AnyDim(torch.nn.Module):
            def __init__(self, keepdim):
                super().__init__()
                self.keepdim = keepdim

            def forward(self, x):
                mask = x > 0
                if self.keepdim is None:
                    reduced = torch.any(mask, dim=-1)
                else:
                    reduced = torch.any(mask, dim=-1, keepdim=self.keepdim)
                return torch.logical_not(reduced)

        inputs = [
            ((torch.arange(2 * s * 5).reshape(2, s, 5) % 17).float() - 14,)
            for s in (16, 3, 31, 2, 16)
        ]
        for keepdim in (None, False):
            with self.subTest(keepdim=keepdim):
                model = AnyDim(keepdim)
                edge = self._lower(
                    model,
                    inputs[0],
                    ({1: Dim("s", min=2, max=32)},),
                    fully_delegated=False,
                )
                self.assertEqual(
                    [
                        node.target
                        for node in edge.exported_program().graph.nodes
                        if node.op == "call_function"
                        and node.target != operator.getitem
                    ],
                    [
                        torch.ops.higher_order.executorch_call_delegate,
                        exir_ops.edge.aten.any.dim,
                        torch.ops.higher_order.executorch_call_delegate,
                    ],
                )
                for graph in _vulkan_graphs(edge):
                    for value in graph.values:
                        tensor = value.value
                        if isinstance(tensor, VkTensor) and tensor.constant_id < 0:
                            self.assertEqual(
                                tensor.storage_type, VkStorageType.TEXTURE_3D
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
                    unsupported = not keepdim or (
                        dim == 1 and storage == VkStorageType.BUFFER
                    )
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

    def test_constant_bool_mask(self):
        model = ConstantMask()
        inputs = [(torch.linspace(-1, 1, 21).reshape(3, 7),)]
        for storage in (VkStorageType.TEXTURE_3D, VkStorageType.BUFFER):
            with self.subTest(storage=storage):
                edge = self._lower(model, inputs[0], storage=storage)
                self.assertTrue(
                    any(
                        isinstance(value.value, VkTensor)
                        and value.value.constant_id >= 0
                        and value.value.datatype == VkDataType.BOOL
                        for graph in _vulkan_graphs(edge)
                        for value in graph.values
                    )
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

    def test_signed_zero_scalars(self):
        class SignedZero(torch.nn.Module):
            def forward(self, x):
                return (
                    torch.ops.aten.mul.Scalar(x, -0.0),
                    torch.full_like(x, -0.0),
                    torch.scalar_tensor(-0.0, dtype=x.dtype),
                )

        model = SignedZero()
        for dtype in (torch.float32, torch.float16):
            for storage in (VkStorageType.TEXTURE_3D, VkStorageType.BUFFER):
                with self.subTest(dtype=dtype, storage=storage):
                    inputs = [(torch.arange(-7, 14, dtype=dtype).reshape(3, 7),)]
                    edge = self._lower(
                        model, inputs[0], storage=storage, fully_delegated=False
                    )
                    self.assertTrue(_vulkan_graphs(edge))
                    self.assertTrue(
                        all(
                            node.target
                            in (
                                operator.getitem,
                                torch.ops.higher_order.executorch_call_delegate,
                            )
                            for node in edge.exported_program().graph.nodes
                            if node.op == "call_function"
                        )
                    )
                    self._run(
                        edge, model, inputs, atol=0, rtol=0, check_signed_zero=True
                    )

    def test_fp16_chained_mul_scalar(self):
        class ChainedMul(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.mul.Scalar(
                    torch.ops.aten.mul.Scalar(x, 1.0006), 1000
                )

        model = ChainedMul()
        x = torch.ones(3, 7, dtype=torch.float16)
        for storage in (VkStorageType.TEXTURE_3D, VkStorageType.BUFFER):
            with self.subTest(storage=storage):
                edge = self._lower(model, (x,), storage=storage)
                operators = [
                    op.name for graph in _vulkan_graphs(edge) for op in graph.chain
                ]
                self.assertEqual(operators.count("aten.mul.Scalar"), 2)
                self._run(edge, model, [(x,)], atol=0, rtol=0)

    def test_integer_scalar_range_fallback(self):
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
            for value in (
                2**31 - 0.5,
                -(2**31) - 0.5,
                2**31,
                2**40,
                2**63 - 1,
                -(2**63),
            ):
                with self.subTest(kind=kind, value=value):
                    model = LargeScalar(kind, value)
                    edge = self._lower(model, inputs[0], fully_delegated=False)
                    self.assertEqual(_vulkan_graphs(edge), [])
                    self._run(edge, model, inputs, atol=0, rtol=0)

    def test_64_bit_arithmetic_without_downcasting(self):
        class Arithmetic(torch.nn.Module):
            def forward(self, x):
                return x + x, x.to(torch.float32) + 1

        for dtype in (torch.int64, torch.float64):
            with self.subTest(dtype=dtype):
                model = Arithmetic()
                inputs = [
                    (torch.arange(3 * s, dtype=dtype).reshape(3, s),) for s in (7, 2)
                ]
                edge = self._lower(
                    model,
                    inputs[0],
                    ({1: Dim("s", min=2, max=16)},),
                    fully_delegated=False,
                    downcast_64_bit=False,
                )
                graphs = _vulkan_graphs(edge)
                self.assertTrue(graphs)
                for graph in graphs:
                    for value in graph.values:
                        if isinstance(value.value, VkTensor):
                            self.assertNotIn(
                                value.value.datatype,
                                (VkDataType.INT64, VkDataType.FLOAT64),
                            )
                self._run(edge, model, inputs, atol=0, rtol=0)

    def test_integer_factories_without_downcasting(self):
        class IntegerFactories(torch.nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.dtype = dtype

            def forward(self, x):
                return (
                    torch.scalar_tensor(16777217, dtype=self.dtype),
                    torch.full(x.shape, 2**31 - 1, dtype=self.dtype),
                    torch.full_like(x, -(2**31), dtype=self.dtype),
                )

        inputs = [(torch.zeros(3, s),) for s in (7, 2, 15, 7)]
        for dtype in (torch.int32, torch.int64):
            for downcast in (True, False):
                with self.subTest(dtype=dtype, downcast=downcast):
                    model = IntegerFactories(dtype)
                    delegated = dtype == torch.int32 or downcast
                    edge = self._lower(
                        model,
                        inputs[0],
                        ({1: Dim("s", min=2, max=16)},),
                        fully_delegated=delegated,
                        downcast_64_bit=downcast,
                    )
                    self.assertEqual(bool(_vulkan_graphs(edge)), delegated)
                    self._run(edge, model, inputs, atol=0, rtol=0)

    def test_64_bit_inputs_without_downcasting(self):
        class Input64Bit(torch.nn.Module):
            def forward(self, x):
                return (
                    x + x,
                    torch.full_like(x, 3, dtype=torch.int32),
                    torch.ones(x.shape, dtype=torch.float32),
                )

        model = Input64Bit()
        for dtype in (torch.int64, torch.float64):
            with self.subTest(dtype=dtype):
                inputs = [
                    (torch.arange(3 * s, dtype=dtype).reshape(3, s),)
                    for s in (7, 2, 15, 7)
                ]
                edge = self._lower(
                    model,
                    inputs[0],
                    ({1: Dim("s", min=2, max=16)},),
                    fully_delegated=False,
                    downcast_64_bit=False,
                )
                graphs = _vulkan_graphs(edge)
                self.assertTrue(graphs)
                for graph in graphs:
                    for value in graph.values:
                        if isinstance(value.value, VkTensor):
                            self.assertNotIn(
                                value.value.datatype,
                                (VkDataType.INT64, VkDataType.FLOAT64),
                            )
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

    def test_64_bit_fusion_inputs_without_downcasting(self):
        class SelectScalar(torch.nn.Module):
            def __init__(self, narrow):
                super().__init__()
                self.narrow = narrow

            def forward(self, x, pos):
                value = pos[0].item()
                if self.narrow:
                    torch._check(value >= 0)
                    torch._check(value <= 6)
                    return x.narrow(1, value, 2) + 1
                return x * value

        inputs = [(torch.randn(2, 8), torch.tensor([pos])) for pos in (3, 5, 0)]
        for narrow in (False, True):
            for downcast in (False, True):
                with self.subTest(narrow=narrow, downcast=downcast):
                    model = SelectScalar(narrow)
                    edge = self._lower(
                        model,
                        inputs[0],
                        fully_delegated=False,
                        downcast_64_bit=downcast,
                    )
                    for graph in _vulkan_graphs(edge):
                        for value in graph.values:
                            if isinstance(value.value, VkTensor):
                                self.assertNotIn(
                                    value.value.datatype,
                                    (VkDataType.INT64, VkDataType.FLOAT64),
                                )
                    self._run(edge, model, inputs, atol=0, rtol=0)

    def test_quantized_embedding_without_downcasting(self):
        from torchao.quantization.granularity import PerGroup
        from torchao.quantization.quant_api import IntxWeightOnlyConfig, quantize_
        from torchao.utils import unwrap_tensor_subclass

        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Embedding(64, 128)).eval()
        quantize_(
            model,
            IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerGroup(32)),
            filter_fn=lambda module, fqn: isinstance(module, torch.nn.Embedding),
        )
        unwrap_tensor_subclass(model)
        inputs = [(torch.tensor(indices),) for indices in ([0, 5, 63, 7], [3, 3, 1, 0])]
        for downcast in (False, True):
            with self.subTest(downcast=downcast):
                if downcast and USING_SWIFTSHADER:
                    self.skipTest("Quantized embedding requires 8-bit storage buffers")
                edge = self._lower(
                    model,
                    inputs[0],
                    fully_delegated=downcast,
                    downcast_64_bit=downcast,
                )
                self.assertEqual(bool(_vulkan_graphs(edge)), downcast)
                if downcast:
                    self._run(edge, model, inputs)

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
                    torch.ops.aten.mul.Scalar(x, n * 0.5),
                    x + torch.full((n,), n * 0.5),
                    torch.full((n,), 0.5),
                    F.gelu(x),
                    torch.clamp(x, max=n * 0.5),
                    F.leaky_relu(x, negative_slope=n * 0.1),
                )

        model = DynamicScalars()
        inputs = [(torch.linspace(-0.9, 4.1, n),) for n in (4, 2, 7, 3, 4)]
        edge = self._lower(
            model, inputs[0], ({0: Dim("n", min=2, max=8)},), fully_delegated=False
        )
        self.assertTrue(_vulkan_graphs(edge))
        self._run(edge, model, inputs)

    def test_dynamic_compare_scalars_fall_back(self):
        class Compare(torch.nn.Module):
            def __init__(self, op):
                super().__init__()
                self.op = op

            def forward(self, x):
                return self.op(x, x.shape[1])

        inputs = [
            (torch.arange(2 * s, dtype=torch.float32).reshape(2, s),)
            for s in (16, 3, 31, 2, 16)
        ]
        for op in (torch.eq, torch.ne, torch.lt, torch.le, torch.gt, torch.ge):
            with self.subTest(op=op):
                model = Compare(op)
                edge = self._lower(
                    model,
                    inputs[0],
                    ({1: Dim("s", min=2, max=32)},),
                    fully_delegated=False,
                )
                self.assertEqual(_vulkan_graphs(edge), [])
                self._run(edge, model, inputs, atol=0, rtol=0)

    def test_compare_scalar_values_fall_back(self):
        class Compare(torch.nn.Module):
            def __init__(self, op, value):
                super().__init__()
                self.op = op
                self.value = value

            def forward(self, x):
                return self.op(x, self.value)

        for op in (torch.eq, torch.ne, torch.lt, torch.le, torch.gt, torch.ge):
            for x, value in (
                (torch.tensor([-1.0, 0.0, 1.0, 2.0]), torch.nan),
                (torch.tensor([-3, 0, 1, 7], dtype=torch.int32), 2**40),
                (torch.tensor([-(2**40), 0, 2**40, 2**40 + 1]), 2**40),
            ):
                with self.subTest(op=op, dtype=x.dtype, value=value):
                    model = Compare(op, value)
                    inputs = [(x,)]
                    edge = self._lower(model, inputs[0], fully_delegated=False)
                    self.assertEqual(_vulkan_graphs(edge), [])
                    self._run(edge, model, inputs, atol=0, rtol=0)

    def test_compare_scalar_values(self):
        class Compare(torch.nn.Module):
            def __init__(self, op, value):
                super().__init__()
                self.op = op
                self.value = value

            def forward(self, x):
                return self.op(x, self.value)

        for op in (torch.eq, torch.ne, torch.lt, torch.le, torch.gt, torch.ge):
            for dtype in (torch.int32, torch.float32):
                for value in (2.0, 2.5, -1.5):
                    with self.subTest(op=op, dtype=dtype, value=value):
                        x = torch.arange(-7, 14, dtype=dtype).reshape(3, 7)
                        model = Compare(op, value)
                        edge = self._lower(model, (x,))
                        self._run(edge, model, [(x,)], atol=0, rtol=0)

    def test_4d_reductions(self):
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
                    values = torch.arange(batch * 3 * 4 * 5).reshape(batch, 3, 4, 5)
                    x = (
                        values % 7 == 0
                        if op == torch.any
                        else -((values * 37 + 11) % values.numel() + 1).float() / 7
                    )
                    model = Reduce(op, dim)
                    edge = self._lower(model, (x,), fully_delegated=supported)
                    if not supported:
                        self.assertEqual(_vulkan_graphs(edge), [])
                    self._run(edge, model, [(x,)])

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

    def test_argreduce_dims(self):
        class Reduce(torch.nn.Module):
            def __init__(self, op, dim, keepdim):
                super().__init__()
                self.op = op
                self.dim = dim
                self.keepdim = keepdim

            def forward(self, x):
                return self.op(x, dim=self.dim, keepdim=self.keepdim)

        for op in (torch.argmax, torch.argmin):
            for keepdim in (True, False):
                for shape, dim, supported in (
                    ((1, 8), None, False),
                    ((3, 8), None, False),
                    ((3, 8), 0, False),
                    ((3, 8), -2, False),
                    ((8,), None, True),
                    ((3, 8), 1, True),
                    ((3, 8), -1, True),
                ):
                    with self.subTest(op=op, keepdim=keepdim, shape=shape, dim=dim):
                        x = ((torch.arange(math.prod(shape)) * 5 + 3) % 17).float()
                        x = x.reshape(shape)
                        model = Reduce(op, dim, keepdim)
                        edge = self._lower(model, (x,), fully_delegated=supported)
                        graphs = _vulkan_graphs(edge)
                        if supported:
                            (graph,) = graphs
                            for value_id in graph.input_ids + graph.output_ids:
                                self.assertEqual(
                                    graph.values[value_id].value.storage_type,
                                    VkStorageType.BUFFER,
                                )
                        else:
                            self.assertEqual(len(graphs), 0)
                        self._run(edge, model, [(x,)], atol=0, rtol=0)

    def test_unsupported_reduction_dims_fall_back(self):
        class Reduce(torch.nn.Module):
            def __init__(self, op, keepdim, dims):
                super().__init__()
                self.op = op
                self.keepdim = keepdim
                self.dims = dims

            def forward(self, x):
                return self.op(x, dim=self.dims, keepdim=self.keepdim)

        for op in (torch.sum, torch.mean, torch.amax, torch.amin):
            for keepdim in (False, True):
                for dims, shape in (
                    ([], (8,)),
                    ([], (2, 3, 5)),
                    (None, (8, 3)),
                    (None, (1, 8)),
                ):
                    if dims is None and op not in (torch.sum, torch.mean):
                        continue
                    with self.subTest(op=op, keepdim=keepdim, dims=dims, shape=shape):
                        x = torch.linspace(-4, 3, math.prod(shape)).reshape(shape)
                        model = Reduce(op, keepdim, dims)
                        edge = self._lower(model, (x,), fully_delegated=False)
                        self.assertEqual(_vulkan_graphs(edge), [])
                        self._run(edge, model, [(x,)])

    def test_power_special_values(self):
        class Power(torch.nn.Module):
            def __init__(self, exponent):
                super().__init__()
                self.exponent = exponent

            def forward(self, x):
                return torch.pow(x, self.exponent)

        for dtype in (torch.float32, torch.float16):
            x = torch.tensor(
                [-torch.inf, -10000, -4, -0.0, 0.0, 1, 10000, torch.inf, torch.nan],
                dtype=dtype,
            ).repeat(3, 1)
            for exponent in (
                -3,
                -0.5,
                0,
                0.5,
                2,
                2.0001,
                3,
                2049,
                torch.inf,
                -torch.inf,
            ):
                for storage in (VkStorageType.TEXTURE_3D, VkStorageType.BUFFER):
                    with self.subTest(dtype=dtype, exponent=exponent, storage=storage):
                        model = Power(exponent)
                        edge = self._lower(model, (x,), storage=storage)
                        self._run(
                            edge, model, [(x,)], equal_nan=True, check_signed_zero=True
                        )

    def test_fp16_scalar_rounding(self):
        class CreateTensor(torch.nn.Module):
            def __init__(self, value, scalar):
                super().__init__()
                self.value = value
                self.scalar = scalar

            def forward(self, x):
                if self.scalar:
                    return torch.scalar_tensor(self.value, dtype=x.dtype)
                return torch.full_like(x, self.value)

        x = torch.ones(3, 7, dtype=torch.float16)
        for value in (
            0.3,
            -1.00075,
            2**-24,
            -(2**-24),
            2**-25,
            3 * 2**-25,
            65519.0,
            65520.0,
            1e5,
            -1e5,
        ):
            for scalar in (False, True):
                if not scalar and abs(value) > 65504:
                    continue
                for storage in (VkStorageType.TEXTURE_3D, VkStorageType.BUFFER):
                    with self.subTest(value=value, scalar=scalar, storage=storage):
                        model = CreateTensor(value, scalar)
                        edge = self._lower(model, (x,), storage=storage)
                        self._run(edge, model, [(x,)], atol=0, rtol=0)

    def test_int32_buffer_reduction_shader_range(self):
        from executorch.extension.pybindings.portable_lib import (
            _load_for_executorch_from_buffer,
        )

        class Reduce(torch.nn.Module):
            def __init__(self, op):
                super().__init__()
                self.op = op

            def forward(self, x):
                return self.op(x, dim=-1, keepdim=True)

        x = torch.tensor(
            [[80000, 80001, 80002, 80003], [-80000, -80001, -80002, -80003]],
            dtype=torch.int32,
        )
        for op in (torch.amax, torch.amin):
            with self.subTest(op=op):
                model = Reduce(op)
                edge = to_edge(export(model, (x,)))
                # Integer reductions are excluded by the partitioner.
                lowered = to_backend(
                    "VulkanBackend",
                    edge.exported_program(),
                    parse_compile_options(
                        {
                            "storage_type_override": VkStorageType.BUFFER,
                            "texture_limits": (1, 1, 1),
                        }
                    ),
                )
                graph = flatbuffer_to_vk_graph(
                    extract_vk_flatbuffer(lowered.processed_bytes)
                )
                for value_id in graph.input_ids + graph.output_ids:
                    value = graph.values[value_id].value
                    self.assertEqual(value.datatype, VkDataType.INT32)
                    self.assertEqual(value.storage_type, VkStorageType.BUFFER)
                program_buffer = lowered.buffer()
                module = _load_for_executorch_from_buffer(program_buffer)
                (actual,) = module.run_method("forward", (x,))
                torch.testing.assert_close(actual, model(x), atol=0, rtol=0)

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

    def test_fp16_reduction_halfway_rounding(self):
        class Mean(torch.nn.Module):
            def forward(self, x):
                return torch.mean(x, dim=-1, keepdim=True)

        # Adjacent half values produce ties at even/odd mantissas, an exponent
        # carry, and the normal/subnormal boundary without rounding the inputs.
        x = torch.tensor(
            [
                [1, 1 + 2**-10],
                [1 + 2**-10, 1 + 2**-9],
                [2 - 2**-10, 2],
                [2**-14 - 2**-24, 2**-14],
                [0, 2**-24],
                [2**-24, 2**-23],
            ],
            dtype=torch.float16,
        )
        x = torch.cat((x, -x))
        model = Mean()
        edge = self._lower(model, (x,), storage=VkStorageType.TEXTURE_3D)
        (graph,) = _vulkan_graphs(edge)
        output = graph.values[graph.output_ids[0]].value
        self.assertEqual(output.datatype, VkDataType.FLOAT16)
        self.assertEqual(output.storage_type, VkStorageType.TEXTURE_3D)
        self._run(edge, model, [(x,)], atol=0, rtol=0, check_signed_zero=True)

    def test_fp16_reduction_overflow_rounding(self):
        class Sum(torch.nn.Module):
            def forward(self, x):
                return torch.sum(x, dim=-1, keepdim=True)

        x = torch.tensor([[65504, 15], [65504, 16], [65504, 17]], dtype=torch.float16)
        x = torch.cat((x, -x))
        model = Sum()
        edge = self._lower(model, (x,), storage=VkStorageType.TEXTURE_3D)
        (graph,) = _vulkan_graphs(edge)
        output = graph.values[graph.output_ids[0]].value
        self.assertEqual(output.datatype, VkDataType.FLOAT16)
        self.assertEqual(output.storage_type, VkStorageType.TEXTURE_3D)
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

    @unittest.skipUnless(USING_SWIFTSHADER, "requires a device without 8-bit buffers")
    def test_bool_buffers_fail_cleanly_without_8bit_storage(self):
        from executorch.extension.pybindings.portable_lib import (
            _load_for_executorch_from_buffer,
        )

        class LogicalNot(torch.nn.Module):
            def forward(self, x):
                return torch.logical_not(x)

        for model, inputs in (
            (LogicalNot(), (torch.zeros(3, 7, dtype=torch.bool),)),
            (ConstantMask(), (torch.zeros(3, 7),)),
        ):
            with self.subTest(model=type(model).__name__):
                edge = self._lower(model, inputs, storage=VkStorageType.BUFFER)
                program_buffer = edge.to_executorch().buffer
                module = _load_for_executorch_from_buffer(program_buffer)
                with self.assertRaisesRegex(RuntimeError, r"0x:?10\b"):
                    module.run_method("forward", inputs)


if __name__ == "__main__":
    unittest.main()
