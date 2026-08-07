# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
import unittest
from unittest.mock import MagicMock

import torch

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
    CONTIGUOUS_BUFFER,
    FP_T,
    INT_T,
    NO_STORAGE,
    NONE_T,
    OpFeatures,
    VulkanPartitioner,
    VulkanSupportedOperators,
)
from torch._subclasses.fake_tensor import FakeTensorMode


class TestScopedOperatorFeatures(unittest.TestCase):
    def setUp(self) -> None:
        self.op = torch.ops.aten.exp2.default
        self.features = OpFeatures(
            inputs_storage=CONTIGUOUS_BUFFER,
            supports_resize=True,
        )

    def _node(self) -> torch.fx.Node:
        node = MagicMock(spec=torch.fx.Node)
        node.op = "call_function"
        node.target = self.op
        node.args = (MagicMock(),)
        with FakeTensorMode() as mode:
            node.meta = {"val": mode.from_tensor(torch.empty(1))}
        return node

    def _support(self, **kwargs) -> VulkanSupportedOperators:
        return VulkanSupportedOperators(
            (16384, 16384, 2048),
            1 << 30,
            **kwargs,
        )

    def test_public_feature_construction_surface(self) -> None:
        features = OpFeatures(
            inputs_dtypes=[FP_T, INT_T, NONE_T],
            outputs_dtypes=[FP_T, INT_T],
            inputs_storage=[
                CONTIGUOUS_BUFFER,
                CONTIGUOUS_BUFFER,
                NO_STORAGE,
            ],
            outputs_storage=[CONTIGUOUS_BUFFER, CONTIGUOUS_BUFFER],
            supports_resize=True,
        )
        self.assertTrue(features.supports_resize)

    def test_default_support_rejects_unregistered_operator(self) -> None:
        self.assertFalse(self._support()._is_node_supported(self._node()))

    def test_explicit_support_accepts_only_the_scoped_operator(self) -> None:
        support = self._support(extra_op_features={self.op: self.features})
        self.assertTrue(support._is_node_supported(self._node()))
        self.assertFalse(self._support()._is_node_supported(self._node()))

    def test_allowlist_and_blocklist_run_before_scoped_features(self) -> None:
        support = self._support(
            operator_blocklist={self.op},
            extra_op_features={self.op: self.features},
        )
        self.assertFalse(support._is_node_supported(self._node()))

    def test_partitioner_copies_caller_mapping(self) -> None:
        source = {self.op: self.features}
        partitioner = VulkanPartitioner(extra_op_features=source)
        source.clear()
        self.assertEqual(partitioner.extra_op_features, {self.op: self.features})
        self.assertFalse(self._support()._is_node_supported(self._node()))

    def test_partitioner_rejects_global_registry_overlap(self) -> None:
        with self.assertRaisesRegex(ValueError, "already registered"):
            VulkanPartitioner(
                extra_op_features={exir_ops.edge.aten.add.Tensor: self.features},
            )


class TestGuardOnlySymbolicNodes(unittest.TestCase):
    """`sym_min`/`sym_max` are non-tensor, so the guard-only check must run in
    `node_is_compatible` before the `is_tensor_node` dispatch; routing it through
    `op_node_is_compatible` made it unreachable."""

    def _support(self) -> VulkanSupportedOperators:
        return VulkanSupportedOperators((16384, 16384, 2048), 1 << 30)

    def _tensor_meta(self) -> dict:
        with FakeTensorMode() as mode:
            return {"val": mode.from_tensor(torch.empty(4))}

    def _sym_graph(self, sink: bool, tensor_user: bool):
        graph = torch.fx.Graph()
        a = graph.placeholder("a")
        b = graph.placeholder("b")
        sym = graph.call_function(torch.sym_max, (a, b))
        expr = graph.call_function(operator.add, (sym, 1))
        if sink:
            graph.call_function(torch.ops.aten._assert_scalar.default, (expr, "ok"))
        if tensor_user:
            live = graph.call_function(torch.ops.aten.full.default, ([sym], 1.0))
            live.meta.update(self._tensor_meta())
        return sym

    def test_guard_only_chain_reaching_a_sink_is_accepted(self) -> None:
        sym = self._sym_graph(sink=True, tensor_user=False)
        ok, reason = self._support().node_is_compatible(sym)
        self.assertTrue(ok, reason)
        self.assertIsInstance(reason, str)

    def test_live_tensor_consumer_is_rejected(self) -> None:
        sym = self._sym_graph(sink=True, tensor_user=True)
        ok, reason = self._support().node_is_compatible(sym)
        self.assertFalse(ok)
        self.assertIsInstance(reason, str)

    def test_missing_sink_is_rejected(self) -> None:
        sym = self._sym_graph(sink=False, tensor_user=False)
        ok, reason = self._support().node_is_compatible(sym)
        self.assertFalse(ok)

    def test_mixed_users_without_sink_are_rejected(self) -> None:
        sym = self._sym_graph(sink=False, tensor_user=True)
        ok, reason = self._support().node_is_compatible(sym)
        self.assertFalse(ok)

    def test_contract_is_always_a_bool_str_pair(self) -> None:
        for sink, tensor_user in ((True, False), (True, True), (False, False)):
            with self.subTest(sink=sink, tensor_user=tensor_user):
                result = self._support().node_is_compatible(
                    self._sym_graph(sink, tensor_user)
                )
                self.assertIsInstance(result, tuple)
                self.assertEqual(len(result), 2)
                self.assertIsInstance(result[0], bool)
                self.assertIsInstance(result[1], str)
