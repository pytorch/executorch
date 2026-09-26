# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
    parse_compile_options,
)
from executorch.backends.vulkan.vulkan_preprocess import (
    parse_compile_spec,
    VulkanBackend,
)
from executorch.exir._serialize._named_data_store import NamedDataStore
from executorch.exir._serialize.data_serializer import DataEntry
from executorch.exir.backend.compile_spec_schema import CompileSpec


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

    def test_alias_buffer_mutations_round_trips(self) -> None:
        round_tripped = self._round_trip({"alias_buffer_mutations": True})
        self.assertTrue(round_tripped.get("alias_buffer_mutations"))

    def test_force_fp16_round_trips(self) -> None:
        round_tripped = self._round_trip({"force_fp16": True})
        self.assertTrue(round_tripped.get("force_fp16"))

    def test_external_constants_max_data_bytes_round_trips_uint64_bounds(
        self,
    ) -> None:
        for value in (1, (1 << 64) - 1):
            with self.subTest(value=value):
                self.assertEqual(
                    self._round_trip({"external_constants_max_data_bytes": value}).get(
                        "external_constants_max_data_bytes"
                    ),
                    value,
                )

    def test_external_constants_max_data_bytes_rejects_invalid_values(self) -> None:
        invalid_values: list[Any] = [True, 0, -1, 1 << 64, 1.5, "10"]
        for value in invalid_values:
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "positive uint64"
            ):
                parse_compile_options({"external_constants_max_data_bytes": value})

    def test_external_constants_max_data_bytes_rejects_invalid_encoding(
        self,
    ) -> None:
        for payload in (b"", b"\x01", b"\x01" * 7, b"\x01" * 9):
            with self.subTest(payload=payload), self.assertRaisesRegex(
                ValueError, "encoded as uint64"
            ):
                parse_compile_spec(
                    [CompileSpec("external_constants_max_data_bytes", payload)]
                )
        with self.assertRaisesRegex(ValueError, "positive uint64"):
            parse_compile_spec(
                [CompileSpec("external_constants_max_data_bytes", b"\x00" * 8)]
            )

    def _preprocess_named_data(self, options: Dict[str, Any]):
        store = NamedDataStore()
        graph_builder = MagicMock()
        graph_builder.named_data_store = store

        def build_graph():
            store.add_named_data("constant", b"constant", 16)
            return MagicMock()

        graph_builder.build_graph.side_effect = build_graph
        graph_builder.delegate_mapping_builder.get_delegate_mapping.return_value = {}
        program = MagicMock()

        with patch.object(
            store, "externalize_pte_data", wraps=store.externalize_pte_data
        ) as externalize_pte_data, patch(
            "executorch.backends.vulkan.vulkan_preprocess."
            "unsafe_remove_auto_functionalized_pass",
            side_effect=lambda value: value,
        ), patch(
            "executorch.backends.vulkan.vulkan_preprocess.apply_passes",
            side_effect=lambda value, _passes: value,
        ), patch(
            "executorch.backends.vulkan.vulkan_preprocess.VkGraphBuilder",
            return_value=graph_builder,
        ) as graph_builder_factory, patch(
            "executorch.backends.vulkan.vulkan_preprocess.serialize_vulkan_graph",
            return_value=b"vk_graph",
        ):
            result = VulkanBackend.preprocess(program, parse_compile_options(options))
        return result.data_store_output, externalize_pte_data, graph_builder_factory

    def test_external_constants_default_keeps_constants_inline(self) -> None:
        output, externalize_pte_data, _ = self._preprocess_named_data({})

        self.assertEqual(output.buffers, [b"constant"])
        self.assertEqual(output.pte_data, {"constant": DataEntry(0, 16, None)})
        self.assertEqual(output.external_data, {})
        externalize_pte_data.assert_not_called()

    def test_external_constants_option_externalizes_constants(self) -> None:
        output, externalize_pte_data, _ = self._preprocess_named_data(
            {"external_constants_max_data_bytes": 16}
        )

        self.assertEqual(output.buffers, [b"constant"])
        self.assertEqual(output.pte_data, {})
        self.assertEqual(len(output.external_data), 1)
        self.assertEqual(list(next(iter(output.external_data.values()))), ["constant"])
        externalize_pte_data.assert_called_once_with(16, "vulkan_constants")

    def test_alias_buffer_mutations_reaches_graph_builder(self) -> None:
        for options, expected in (
            ({}, False),
            ({"alias_buffer_mutations": True}, True),
        ):
            with self.subTest(options=options):
                _, _, graph_builder_factory = self._preprocess_named_data(options)
                self.assertIs(
                    graph_builder_factory.call_args.kwargs["alias_buffer_mutations"],
                    expected,
                )

    def test_unset_options_are_absent(self) -> None:
        round_tripped = self._round_trip({})
        self.assertNotIn("alias_buffer_mutations", round_tripped)
        self.assertNotIn("small_texture_limits", round_tripped)
        self.assertNotIn("skip_memory_planning", round_tripped)
        self.assertNotIn("external_constants_max_data_bytes", round_tripped)


if __name__ == "__main__":
    unittest.main()
