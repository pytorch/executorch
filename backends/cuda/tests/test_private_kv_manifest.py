# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import unittest

from executorch.backends.cuda.passes.lower_offgraph_kv import (
    offgraph_kv_constant_methods,
    OFFGRAPH_KV_SEQUENCE_INFO_ABI_METHOD,
    parse_offgraph_kv_manifest,
)


class PrivateKVManifestTest(unittest.TestCase):
    @staticmethod
    def manifest(**overrides):
        fields = {
            "version": 1,
            "layout": "batched-sequence",
            "maximum_capacity": 131072,
            "max_write": 4096,
            "max_seqs": 64,
            "max_segments": 128,
            "layers": [
                {"layer_id": 0, "policy": "flat"},
                {"layer_id": 1, "policy": "ring", "window": 512},
            ],
        }
        fields.update(overrides)
        return json.dumps(fields).encode()

    def test_absent_option_preserves_abi_one(self):
        parsed = parse_offgraph_kv_manifest(self.manifest())
        self.assertFalse(parsed["private_rows"])
        methods = offgraph_kv_constant_methods(self.manifest())
        self.assertEqual(methods[OFFGRAPH_KV_SEQUENCE_INFO_ABI_METHOD], 1)

    def test_existing_runtime_pitch_preserves_abi_two(self):
        methods = offgraph_kv_constant_methods(self.manifest(runtime_columns=True))
        self.assertEqual(methods[OFFGRAPH_KV_SEQUENCE_INFO_ABI_METHOD], 2)

    def test_private_rows_select_abi_three_without_runtime_pitch(self):
        value = self.manifest(private_rows=True)
        parsed = parse_offgraph_kv_manifest(value)
        self.assertTrue(parsed["private_rows"])
        self.assertFalse(parsed["runtime_columns"])
        methods = offgraph_kv_constant_methods(value)
        self.assertEqual(methods[OFFGRAPH_KV_SEQUENCE_INFO_ABI_METHOD], 3)
        self.assertEqual(methods["get_offgraph_kv_max_seqs"], 64)

    def test_rejects_mixed_addressing_abis(self):
        with self.assertRaisesRegex(ValueError, "without ABI 2"):
            parse_offgraph_kv_manifest(
                self.manifest(private_rows=True, runtime_columns=True)
            )

    def test_rejects_other_layouts(self):
        for layout in ("sequence", "cell"):
            with self.subTest(layout=layout), self.assertRaisesRegex(
                ValueError, "batched-sequence"
            ):
                parse_offgraph_kv_manifest(
                    self.manifest(private_rows=True, layout=layout)
                )

    def test_rejects_draft_attention(self):
        for overrides in ({"causal": False}, {"causal": False, "context_window": True}):
            with self.subTest(overrides=overrides), self.assertRaisesRegex(
                ValueError, "draft or non-causal"
            ):
                parse_offgraph_kv_manifest(
                    self.manifest(private_rows=True, **overrides)
                )

    def test_rejects_non_bool_option(self):
        for flag in (0, 1, "true", None):
            with self.subTest(flag=flag), self.assertRaisesRegex(
                ValueError, "private_rows must be a bool"
            ):
                parse_offgraph_kv_manifest(self.manifest(private_rows=flag))

    def test_rejects_inconsistent_segment_capacity(self):
        with self.assertRaisesRegex(ValueError, "max_segments must equal"):
            parse_offgraph_kv_manifest(
                self.manifest(private_rows=True, max_segments=64)
            )


if __name__ == "__main__":
    unittest.main()
