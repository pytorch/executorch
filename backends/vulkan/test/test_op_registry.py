# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from unittest import TestCase

import torch
from executorch.backends.vulkan.op_registry import (
    get_op_features,
    is_custom_sdpa_node_supported,
    is_general_sdpa_node_supported,
    is_integer_remainder_scalar_node_supported,
    is_update_cache_with_indices_node_supported,
    vulkan_supported_ops,
)
from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
    VulkanSupportedOperators,
)
from executorch.backends.vulkan import utils
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops


class TestCustomSDPASupport(TestCase):
    def test_supported_causal_configuration(self) -> None:
        node = SimpleNamespace(
            args=(object(), object(), object(), 0, None, 0.0, True, None),
            kwargs={},
        )
        self.assertTrue(is_custom_sdpa_node_supported(node))

    def test_rejects_masked_non_causal_configuration(self) -> None:
        node = SimpleNamespace(
            args=(object(), object(), object(), 0, object()), kwargs={}
        )
        self.assertFalse(is_custom_sdpa_node_supported(node))


class TestUpdateCacheWithIndicesSupport(TestCase):
    @staticmethod
    def _tensor_node(shape, dtype=torch.float32):
        node = torch.fx.Graph().placeholder("tensor")
        node.meta["val"] = SimpleNamespace(shape=shape, dtype=dtype)
        return node

    def test_accepts_production_contract(self) -> None:
        node = SimpleNamespace(
            target=None,
            args=(
                self._tensor_node((1, 4, 8, 128)),
                self._tensor_node((1, 32, 8, 128)),
                0,
                self._tensor_node((1, 4), torch.int64),
            ),
        )
        self.assertTrue(is_update_cache_with_indices_node_supported(node))

    def test_rejects_batch_greater_than_one(self) -> None:
        node = SimpleNamespace(
            target=None,
            args=(
                self._tensor_node((2, 4, 8, 128)),
                self._tensor_node((2, 32, 8, 128)),
                0,
                self._tensor_node((2, 4), torch.int64),
            ),
        )
        self.assertFalse(is_update_cache_with_indices_node_supported(node))

    def test_accepts_auto_functionalized_v2_contract(self) -> None:
        value = self._tensor_node((1, 4, 8, 128))
        cache = self._tensor_node((1, 32, 8, 128))
        indices = self._tensor_node((1, 4), torch.int64)
        node = SimpleNamespace(
            target=torch.ops.higher_order.auto_functionalized_v2,
            args=(torch.ops.aten.add_.Tensor,),
            kwargs={
                "value": value,
                "indices": indices,
                "_cache_base_index": 0,
                "_all_bases": [cache],
            },
        )
        self.assertTrue(is_update_cache_with_indices_node_supported(node))

    def test_rejects_mismatched_dtype(self) -> None:
        node = SimpleNamespace(
            target=None,
            args=(
                self._tensor_node((1, 4, 8, 128)),
                self._tensor_node((1, 32, 8, 128), torch.float16),
                0,
                self._tensor_node((1, 4), torch.int64),
            ),
        )
        self.assertFalse(is_update_cache_with_indices_node_supported(node))

    def test_rejects_unsupported_shapes_and_index_dtype(self) -> None:
        cases = (
            (
                (1, 4, 8, 128),
                (1, 32, 8, 128),
                (1, 3),
                torch.int64,
            ),
            (
                (1, 4, 8, 128),
                (1, 32, 4, 128),
                (1, 4),
                torch.int64,
            ),
            (
                (1, 4, 8),
                (1, 32, 8, 128),
                (1, 4),
                torch.int64,
            ),
            (
                (1, 4, 8, 128),
                (1, 32, 8, 128),
                (1, 4),
                torch.int32,
            ),
        )
        for value_shape, cache_shape, indices_shape, indices_dtype in cases:
            with self.subTest(
                value_shape=value_shape,
                cache_shape=cache_shape,
                indices_shape=indices_shape,
                indices_dtype=indices_dtype,
            ):
                node = SimpleNamespace(
                    target=None,
                    args=(
                        self._tensor_node(value_shape),
                        self._tensor_node(cache_shape),
                        0,
                        self._tensor_node(indices_shape, indices_dtype),
                    ),
                )
                self.assertFalse(is_update_cache_with_indices_node_supported(node))


class TestGeneralSDPASupport(TestCase):
    @staticmethod
    def _tensor_node(shape, dtype=torch.float32):
        node = torch.fx.Graph().placeholder("tensor")
        node.meta["val"] = SimpleNamespace(shape=shape, dtype=dtype)
        return node

    def _node(
        self,
        q_shape=(1, 32, 4, 128),
        k_shape=(1, 8, 128, 128),
        v_shape=(1, 8, 128, 128),
        mask_shape=(4, 128),
        dtype=torch.float32,
        mask_dtype=None,
        scale=None,
    ):
        mask = None
        if mask_shape is not None:
            mask = self._tensor_node(mask_shape, mask_dtype or dtype)
        return SimpleNamespace(
            args=(
                self._tensor_node(q_shape, dtype),
                self._tensor_node(k_shape, dtype),
                self._tensor_node(v_shape, dtype),
                mask,
                scale,
            )
        )

    def test_accepts_equal_head_and_gqa_contracts(self) -> None:
        cases = (
            self._node(
                q_shape=(1, 32, 4, 64),
                k_shape=(1, 32, 128, 64),
                v_shape=(1, 32, 128, 64),
            ),
            self._node(),
            self._node(mask_shape=(1, 1, 4, 128), dtype=torch.float16),
            self._node(mask_shape=None, scale=0.125),
        )
        for node in cases:
            with self.subTest(args=node.args):
                self.assertTrue(is_general_sdpa_node_supported(node))

        shortened = self._node()
        shortened.args = shortened.args[:3]
        self.assertTrue(is_general_sdpa_node_supported(shortened))

    def test_rejects_unsupported_head_shape_mask_and_dtype(self) -> None:
        cases = (
            self._node(q_shape=(1, 30, 4, 128)),
            self._node(v_shape=(1, 4, 128, 128)),
            self._node(mask_shape=(2, 4, 128)),
            self._node(mask_shape=(4, 128), mask_dtype=torch.float16),
            self._node(scale=object()),
        )
        for node in cases:
            with self.subTest(args=node.args):
                self.assertFalse(is_general_sdpa_node_supported(node))


class TestIntegerRemainderScalarSupport(TestCase):
    def test_accepts_nonzero_int32_range_divisors(self) -> None:
        for divisor in (1, -1, 7, -7, 128, -(2**31), 2**31 - 1):
            with self.subTest(divisor=divisor):
                node = SimpleNamespace(args=(object(), divisor))
                self.assertTrue(is_integer_remainder_scalar_node_supported(node))

    def test_rejects_zero_noninteger_bool_and_out_of_range_divisors(self) -> None:
        for divisor in (0, 1.0, True, False, object(), -(2**31) - 1, 2**31):
            with self.subTest(divisor=divisor):
                node = SimpleNamespace(args=(object(), divisor))
                self.assertFalse(is_integer_remainder_scalar_node_supported(node))

        self.assertFalse(
            is_integer_remainder_scalar_node_supported(
                SimpleNamespace(args=(object(),))
            )
        )


class TestEmbeddingBufferLimit(TestCase):
    def test_partitioner_option_controls_embedding_weight_limit(self) -> None:
        class EmbeddingModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(5, 4))

            def forward(self, indices: torch.Tensor) -> torch.Tensor:
                return torch.nn.functional.embedding(indices, self.weight)

        edge = to_edge(
            torch.export.export(
                EmbeddingModule().eval(),
                (torch.tensor([[0, 1]], dtype=torch.int64),),
            )
        )
        embedding_node = next(
            node
            for node in edge.exported_program().graph.nodes
            if node.target == exir_ops.edge.aten.embedding.default
        )
        features = get_op_features(embedding_node.target)

        self.assertEqual(features.buffer_limit_args, (0,))
        self.assertFalse(
            VulkanSupportedOperators(
                utils.DEFAULT_TEXTURE_LIMITS, buffer_limit=19
            )._is_node_supported(embedding_node)
        )
        self.assertTrue(
            VulkanSupportedOperators(
                utils.DEFAULT_TEXTURE_LIMITS, buffer_limit=20
            )._is_node_supported(embedding_node)
        )


class TestQuantizedLinearFeatures(TestCase):
    def test_dynamic_8da4w_linear_supports_resize(self) -> None:
        features = get_op_features(exir_ops.edge.et_vk.linear_dq8ca_q4gsw.default)

        self.assertTrue(features.supports_resize)


class TestScalarTensorSupport(TestCase):
    def test_registers_edge_and_data_dependent_targets(self) -> None:
        self.assertIn(exir_ops.edge.aten.scalar_tensor.default, vulkan_supported_ops)
        self.assertIn(torch.ops.aten.scalar_tensor.default, vulkan_supported_ops)


class TestRegistryRuntimeParity(TestCase):
    def test_does_not_advertise_unimplemented_logical_not(self) -> None:
        self.assertNotIn(exir_ops.edge.aten.logical_not.default, vulkan_supported_ops)
