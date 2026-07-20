# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Any, Dict

from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
    parse_compile_options,
)
from executorch.backends.vulkan.vulkan_preprocess import parse_compile_spec


class TestVulkanCompileOptions(unittest.TestCase):
    """Verify that compile options survive the partitioner -> backend round trip.

    The partitioner serializes the user-provided options into CompileSpecs
    (parse_compile_options) and the backend deserializes them at preprocess time
    (parse_compile_spec). Boolean options that are serialized but not handled on
    the deserialization side are silently dropped, which is a class of bug that
    previously hid the small_texture_limits desktop-compatibility option.
    """

    def _round_trip(self, options: Dict[str, Any]) -> Dict[str, Any]:
        return parse_compile_spec(parse_compile_options(options))

    def test_small_texture_limits_round_trips(self) -> None:
        round_tripped = self._round_trip({"small_texture_limits": True})
        self.assertTrue(round_tripped.get("small_texture_limits"))

    def test_skip_memory_planning_round_trips(self) -> None:
        round_tripped = self._round_trip({"skip_memory_planning": True})
        self.assertTrue(round_tripped.get("skip_memory_planning"))

    def test_force_fp16_round_trips(self) -> None:
        round_tripped = self._round_trip({"force_fp16": True})
        self.assertTrue(round_tripped.get("force_fp16"))

    def test_unset_options_are_absent(self) -> None:
        round_tripped = self._round_trip({})
        self.assertNotIn("small_texture_limits", round_tripped)
        self.assertNotIn("skip_memory_planning", round_tripped)


if __name__ == "__main__":
    unittest.main()
