# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
import unittest

import torch
import torch.nn.functional as F
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.backends.vulkan.serialization.vulkan_graph_schema import VkStorageType
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from torch.export import Dim, export


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
    def _lower(self, model, inputs, dynamic_shapes, storage=VkStorageType.TEXTURE_3D):
        edge = to_edge_transform_and_lower(
            export(model.eval(), inputs, dynamic_shapes=dynamic_shapes, strict=False),
            partitioner=[
                VulkanPartitioner(
                    {"require_dynamic_shapes": True, "storage_type_override": storage}
                )
            ],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
        targets = [
            node.target
            for node in edge.exported_program().graph.nodes
            if node.op == "call_function" and node.target != operator.getitem
        ]
        self.assertEqual(targets, [torch.ops.higher_order.executorch_call_delegate])
        return edge

    def _run(self, edge, model, inputs, *, atol=1e-5, rtol=1e-4):
        from executorch.extension.pybindings.portable_lib import (
            _load_for_executorch_from_buffer,
        )

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
                    torch.testing.assert_close(output, reference, atol=atol, rtol=rtol)

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
                        tolerance = 1e-6 if dtype == torch.float32 else 1e-3
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
        for positive, negative in ((3, -7.0), (3.0, -7), (16777217, -7)):
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
                        x = torch.zeros(2, s, s, dtype=torch.bool)
                        x[1, -1, -1] = True
                        inputs.append((x,))
                    seq = Dim("s", min=2, max=32)
                    edge = self._lower(model, inputs[0], ({1: seq, 2: seq},), storage)
                    self._run(edge, model, inputs, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
