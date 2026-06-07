# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit + drift tests for the embedded-WGSL-header generator.

Loads the generator by file path (no package/namespace dependency).
"""

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

_GEN = Path(__file__).resolve().parents[1] / "scripts" / "gen_wgsl_headers.py"
_spec = importlib.util.spec_from_file_location("gen_wgsl_headers", _GEN)
g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g)


class WgslCodegenTest(unittest.TestCase):
    def test_symbol_base(self) -> None:
        self.assertEqual(g.symbol_base("binary_add"), "BinaryAdd")
        self.assertEqual(
            g.symbol_base("sdpa_compute_attn_weights"), "SdpaComputeAttnWeights"
        )
        self.assertEqual(g.symbol_base("update_cache"), "UpdateCache")
        self.assertEqual(g.symbol_base("rms_norm"), "RmsNorm")

    def test_parse_workgroup_literal(self) -> None:
        self.assertEqual(
            g.parse_workgroup_size("@compute @workgroup_size(64, 1, 1)\nfn main(){}"),
            (64, 1, 1),
        )

    def test_parse_workgroup_override_indirection(self) -> None:
        src = "override wg_size: u32 = 256;\n@compute @workgroup_size(wg_size)\nfn main(){}"
        self.assertEqual(g.parse_workgroup_size(src), (256, 1, 1))

    def test_parse_workgroup_suffix_typed_literal(self) -> None:
        self.assertEqual(
            g.parse_workgroup_size("@compute @workgroup_size(64u, 1, 1)\nfn main(){}"),
            (64, 1, 1),
        )

    def test_parse_workgroup_const_without_type_annotation(self) -> None:
        src = "const WG = 128u;\n@compute @workgroup_size(WG)\nfn main(){}"
        self.assertEqual(g.parse_workgroup_size(src), (128, 1, 1))

    def test_parse_workgroup_not_fooled_by_const(self) -> None:
        # rms_norm/softmax shape: a sibling `const WG_SIZE` beside a LITERAL size.
        src = (
            "const WG_SIZE: u32 = 64u;\n@compute @workgroup_size(64, 1, 1)\nfn main(){}"
        )
        self.assertEqual(g.parse_workgroup_size(src), (64, 1, 1))

    def test_render_header_shape(self) -> None:
        wgsl = "@compute @workgroup_size(64, 1, 1)\nfn main(){}\n"
        h = g.render_header(Path("runtime/ops/update_cache/update_cache.wgsl"), wgsl)
        self.assertIn("#pragma once", h)
        self.assertIn("#include <cstdint>", h)
        self.assertIn("namespace executorch::backends::webgpu {", h)
        self.assertIn("// @generated from update_cache.wgsl - DO NOT EDIT.", h)
        self.assertIn('inline constexpr const char* kUpdateCacheWGSL = R"(', h)
        self.assertIn("inline constexpr uint32_t kUpdateCacheWorkgroupSizeX = 64;", h)
        self.assertIn("inline constexpr uint32_t kUpdateCacheWorkgroupSizeY = 1;", h)
        self.assertIn("inline constexpr uint32_t kUpdateCacheWorkgroupSizeZ = 1;", h)
        self.assertNotIn("kUpdateCacheWorkgroupSize ", h)
        self.assertNotIn("Confidential", h)
        # the shader is embedded verbatim:
        body = h.split('R"(', 1)[1].split(')";', 1)[0]
        self.assertEqual(body, "\n" + wgsl)
        self.assertTrue(h.endswith("\n"))

    def test_render_header_embeds_sha256(self) -> None:
        wgsl = "@compute @workgroup_size(64, 1, 1)\nfn main(){}\n"
        h = g.render_header(Path("runtime/ops/update_cache/update_cache.wgsl"), wgsl)
        want = hashlib.sha256(wgsl.encode("utf-8")).hexdigest()
        self.assertIn(f"// wgsl-sha256: {want}", h)
        self.assertEqual(g.embedded_sha256(h), want)
        self.assertEqual(g.wgsl_sha256(wgsl), want)

    def test_embedded_sha256_missing_returns_empty(self) -> None:
        self.assertEqual(g.embedded_sha256("no sha line here\n"), "")

    def test_sha256_changes_with_shader(self) -> None:
        a = g.wgsl_sha256("@compute @workgroup_size(64, 1, 1)\nfn main(){}\n")
        b = g.wgsl_sha256("@compute @workgroup_size(256)\nfn main(){}\n")
        self.assertNotEqual(a, b)

    def test_committed_headers_match_generator(self) -> None:
        wgsls = g.discover()
        self.assertGreater(len(wgsls), 0, "no .wgsl shaders discovered")
        for wgsl in wgsls:
            want = g.render_header(wgsl, wgsl.read_text())
            got = wgsl.with_name(wgsl.stem + "_wgsl.h").read_text()
            self.assertEqual(
                got, want, f"{wgsl.stem}_wgsl.h stale; run scripts/gen_wgsl_headers.py"
            )

    def test_parse_workgroup_allows_space(self) -> None:
        # @workgroup_size (64) — the spec-legal spaced form must still parse.
        self.assertEqual(
            g.parse_workgroup_size("@compute @workgroup_size (64)\nfn main(){}"),
            (64, 1, 1),
        )

    def test_render_header_rejects_raw_string_terminator(self) -> None:
        # A shader body containing )" would close the R"( literal -> must reject.
        with self.assertRaises(ValueError):
            g.render_header(
                Path("bad.wgsl"), '@workgroup_size(64)\n// stray )" terminator\n'
            )

    def test_check_fails_on_stale_header(self) -> None:
        # --check must exit 1 when a committed header drifts (the build gate).
        with tempfile.TemporaryDirectory() as tmp:
            op_dir = Path(tmp) / "runtime/ops/foo"
            op_dir.mkdir(parents=True)
            (op_dir / "foo.wgsl").write_text(
                "@compute @workgroup_size(64)\nfn main() {}\n"
            )
            (op_dir / "foo_wgsl.h").write_text("// wgsl-sha256: " + "0" * 64 + "\n")
            orig = g.BACKEND_ROOT
            g.BACKEND_ROOT = Path(tmp)
            try:
                self.assertEqual(g.main(["--check"]), 1)
            finally:
                g.BACKEND_ROOT = orig

    def test_parse_workgroup_1d_defaults_yz(self) -> None:
        self.assertEqual(
            g.parse_workgroup_size("@compute @workgroup_size(64)\nfn main(){}"),
            (64, 1, 1),
        )

    def test_parse_workgroup_2d(self) -> None:
        self.assertEqual(
            g.parse_workgroup_size("@compute @workgroup_size(8, 4)\nfn main(){}"),
            (8, 4, 1),
        )

    def test_parse_workgroup_3d_full(self) -> None:
        self.assertEqual(
            g.parse_workgroup_size("@compute @workgroup_size(4, 4, 4)\nfn main(){}"),
            (4, 4, 4),
        )

    def test_parse_workgroup_override_in_y(self) -> None:
        src = "override wgy: u32 = 8;\n@compute @workgroup_size(16, wgy)\nfn main(){}"
        self.assertEqual(g.parse_workgroup_size(src), (16, 8, 1))

    def test_parse_workgroup_too_many_dims(self) -> None:
        with self.assertRaises(ValueError):
            g.parse_workgroup_size("@workgroup_size(1, 2, 3, 4)\nfn main(){}")

    def test_parse_workgroup_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            g.parse_workgroup_size("@compute @workgroup_size()\nfn main(){}")

    def test_parse_workgroup_suffix_typed_all_dims(self) -> None:
        self.assertEqual(
            g.parse_workgroup_size("@compute @workgroup_size(8u, 4u, 2u)\nfn main(){}"),
            (8, 4, 2),
        )

    def test_parse_workgroup_override_in_z(self) -> None:
        src = (
            "override wgz: u32 = 2;\n@compute @workgroup_size(8, 16, wgz)\nfn main(){}"
        )
        self.assertEqual(g.parse_workgroup_size(src), (8, 16, 2))

    def test_parse_workgroup_spaced_args(self) -> None:
        self.assertEqual(
            g.parse_workgroup_size("@compute @workgroup_size ( 8 , 4 )\nfn main(){}"),
            (8, 4, 1),
        )

    def test_render_header_3d_emits_xyz(self) -> None:
        wgsl = "@compute @workgroup_size(4, 8, 2)\nfn main(){}\n"
        h = g.render_header(Path("runtime/ops/foo/foo.wgsl"), wgsl)
        self.assertIn("inline constexpr uint32_t kFooWorkgroupSizeX = 4;", h)
        self.assertIn("inline constexpr uint32_t kFooWorkgroupSizeY = 8;", h)
        self.assertIn("inline constexpr uint32_t kFooWorkgroupSizeZ = 2;", h)


if __name__ == "__main__":
    unittest.main()
