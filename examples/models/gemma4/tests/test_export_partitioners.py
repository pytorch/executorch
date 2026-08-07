# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from executorch.backends.vulkan.op_registry import vulkan_supported_ops
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.examples.models.gemma4.webgpu_partitioner import (
    _webgpu_allowlist,
    build_webgpu_partitioner,
)
from executorch.exir.dialects._ops import ops as exir_ops


class ExportPartitionersTest(unittest.TestCase):
    def test_plain_features_are_instance_scoped(self) -> None:
        registry_before = dict(vulkan_supported_ops)
        partitioner = build_webgpu_partitioner("8da4w+emb4")

        self.assertEqual(vulkan_supported_ops, registry_before)
        self.assertIn(
            exir_ops.edge.et_vk.apply_rotary_emb_hf_single.default,
            partitioner._inner.extra_op_features,
        )
        self.assertIn(
            exir_ops.edge.et_vk.gemma4_sdpa.default,
            partitioner._inner.extra_op_features,
        )
        # Assert against the GLOBAL registry, not a default partitioner's
        # instance map: that map is unconditionally empty, so the old form
        # passed even if the op were globally registered.
        self.assertNotIn(
            exir_ops.edge.et_vk.apply_rotary_emb_hf_single.default,
            vulkan_supported_ops,
        )
        self.assertNotIn(
            exir_ops.edge.et_vk.gemma4_sdpa.default,
            vulkan_supported_ops,
        )
        self.assertEqual(VulkanPartitioner().extra_op_features, {})

    def test_restricted_allowlist_includes_symbolic_select(self) -> None:
        allowlist = set(_webgpu_allowlist())
        self.assertIn(exir_ops.edge.et_vk.select_as_symint.default, allowlist)
        self.assertNotIn(exir_ops.edge.aten.mm.default, allowlist)
        self.assertNotIn(exir_ops.edge.aten.linear.default, allowlist)

    def test_emb8_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "emb8"):
            build_webgpu_partitioner("8da4w+emb8")
