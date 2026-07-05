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
import json
import re
import tempfile
import unittest
from pathlib import Path

_GEN = Path(__file__).resolve().parents[1] / "scripts" / "gen_wgsl_headers.py"
_spec = importlib.util.spec_from_file_location("gen_wgsl_headers", _GEN)
g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g)

# gen_wgsl_headers.py and backends/vulkan/runtime/gen_vulkan_spv.py share the
# same $-block transpiler helpers; the test below keeps them in sync with that
# source of truth. Resolve the path relative to the repo root (both
# backends/vulkan and backends/webgpu exist in pytorch/executorch) and compare
# the bodies as TEXT -- gen_vulkan_spv.py cannot be imported (it top-level
# `import yaml`s, absent on the codegen runtime).
_REPO_ROOT = g.BACKEND_ROOT.parents[1]
_VULKAN_SPV = _REPO_ROOT / "backends" / "vulkan" / "runtime" / "gen_vulkan_spv.py"
_SHARED_TRANSPILER_FNS = ("extract_leading_whitespace", "escape", "preprocess")


def _function_source(text: str, name: str) -> str:
    """Return a top-level function's source: its `def <name>` line through the
    last line before the next column-0 construct (def-line to next dedent).

    Two-phase so a multi-line signature -- whose closing `) -> str:` sits at
    column 0 -- is not mistaken for the next top-level construct.
    """
    lines = text.splitlines()
    start = next(
        (i for i, ln in enumerate(lines) if re.match(rf"^def {re.escape(name)}\b", ln)),
        None,
    )
    if start is None:
        raise AssertionError(f"def {name} not found")
    # Advance past the (possibly multi-line) signature to the line ending in ':'.
    sig = start
    while not lines[sig].rstrip().endswith(":"):
        sig += 1
    # The body ends at the next non-blank column-0 line.
    end = len(lines)
    for k in range(sig + 1, len(lines)):
        head = lines[k][:1]
        if head != "" and not head.isspace():
            end = k
            break
    return "\n".join(lines[start:end]).rstrip()


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
            # headers_for_shader handles both verbatim shaders and templates
            # (a template emits one header per expanded variant).
            for header, want in g.headers_for_shader(wgsl):
                got = header.read_text()
                self.assertEqual(
                    got, want, f"{header.name} stale; run scripts/gen_wgsl_headers.py"
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


class WgslTemplateEngineTest(unittest.TestCase):
    """Coverage for the $-block template engine + DTYPE/VEC variant matrix."""

    # --- transpiler helpers stay in sync with their source ---

    @unittest.skipUnless(
        _VULKAN_SPV.exists(), f"source of truth not present at {_VULKAN_SPV}"
    )
    def test_transpiler_helpers_stay_in_sync(self) -> None:
        # The shared $-block transpiler helpers must stay character-identical to
        # their source of truth so they cannot silently drift. Read both files as
        # TEXT (the source of truth cannot be imported -- it top-level
        # `import yaml`s).
        src_text = _VULKAN_SPV.read_text()
        gen_text = _GEN.read_text()
        for fn in _SHARED_TRANSPILER_FNS:
            self.assertEqual(
                _function_source(src_text, fn),
                _function_source(gen_text, fn),
                f"{fn} has drifted from its source of truth "
                f"({_VULKAN_SPV}) -- re-sync the shared transpiler helpers",
            )

    # --- preprocess -------------------------------------------------------

    def test_preprocess_if_else_selects_branch(self) -> None:
        tmpl = 'fn main() {\n  $if MODE == "a":\n    let x = 1;\n  $else:\n    let x = 2;\n}\n'
        self.assertEqual(
            g.preprocess(tmpl, {"MODE": "a"}), "fn main() {\n  let x = 1;\n}\n"
        )
        self.assertEqual(
            g.preprocess(tmpl, {"MODE": "b"}), "fn main() {\n  let x = 2;\n}\n"
        )

    def test_preprocess_inline_substitution_uses_helper(self) -> None:
        tmpl = "type: ${buffer_gvec_type(DTYPE, VEC)};\n"
        out = g.preprocess(tmpl, {**g.WGSL_HELPERS, "DTYPE": "float", "VEC": 4})
        self.assertEqual(out, "type: vec4<f32>;\n")

    def test_preprocess_guarded_body_indent_matches_control_column(self) -> None:
        # $if authored at column 2 with its body one 2-space level deeper -> the
        # guarded output line lands at column 2 (the control-line's column).
        tmpl = "fn main() {\n  $if VEC == 4:\n    let a = 1;\n  $else:\n    let b = 2;\n}\n"
        self.assertEqual(
            g.preprocess(tmpl, {"VEC": 4}), "fn main() {\n  let a = 1;\n}\n"
        )
        self.assertEqual(
            g.preprocess(tmpl, {"VEC": 1}), "fn main() {\n  let b = 2;\n}\n"
        )

    def test_preprocess_enable_f16_only_for_half(self) -> None:
        # DD-009: `enable f16;` is a literal line behind `$if DTYPE == "half":`,
        # NOT an inline ${} (which would print a stray blank line for float and
        # break byte-identity of the fp32 base).
        tmpl = '$if DTYPE == "half":\n  enable f16;\nfn main() {}\n'
        self.assertEqual(
            g.preprocess(tmpl, {"DTYPE": "half"}), "enable f16;\nfn main() {}\n"
        )
        self.assertEqual(g.preprocess(tmpl, {"DTYPE": "float"}), "fn main() {}\n")

    # --- generate_variant_combinations -----------------------------------

    def test_generate_variant_combinations_product(self) -> None:
        iterated = {
            "DTYPE": [{"VALUE": "float"}, {"VALUE": "half", "SUFFIX": "half"}],
            "VEC": [{"VALUE": 1, "SUFFIX": ""}, {"VALUE": 4, "SUFFIX": "vec4"}],
        }
        combos = g.generate_variant_combinations(iterated)
        self.assertEqual(len(combos), 4)
        flat = [tuple((s[0], s[1], s[2]) for s in combo) for combo in combos]
        self.assertIn((("DTYPE", "float", "float"), ("VEC", "", 1)), flat)
        self.assertIn((("DTYPE", "half", "half"), ("VEC", "vec4", 4)), flat)

    def test_generate_variant_combinations_suffix_empty_suppresses(self) -> None:
        combos = g.generate_variant_combinations({"VEC": [{"VALUE": 1, "SUFFIX": ""}]})
        self.assertEqual(combos, [(("VEC", "", 1),)])

    def test_generate_variant_combinations_suffix_defaults_to_value(self) -> None:
        # SUFFIX absent -> the suffix defaults to the VALUE (stringified in names).
        combos = g.generate_variant_combinations({"VEC": [{"VALUE": 4}]})
        self.assertEqual(len(combos), 1)
        ((name, suffix, value),) = combos[0]
        self.assertEqual(name, "VEC")
        self.assertEqual(value, 4)
        self.assertEqual(str(suffix), "4")

    def test_generate_variant_combinations_excludes_param(self) -> None:
        # A param already fixed by the variant is excluded from the forall product.
        combos = g.generate_variant_combinations(
            {"VEC": [{"VALUE": 1}, {"VALUE": 4}]}, {"VEC"}
        )
        self.assertEqual(combos, [()])

    # --- parse_template_spec ---------------------------------------------

    def _write_spec(self, tmp: str, name: str, spec_obj) -> Path:
        p = Path(tmp) / f"{name}.json"
        p.write_text(json.dumps(spec_obj))
        return p

    def test_parse_template_spec_minimal(self) -> None:
        spec_obj = {
            "op": {
                "parameter_names_with_default_values": {"DTYPE": "float", "VEC": 1},
                "generate_variant_forall": {
                    "VEC": [
                        {"VALUE": 1, "SUFFIX": ""},
                        {"VALUE": 4, "SUFFIX": "vec4"},
                    ]
                },
                "shader_variants": [{"NAME": "op"}],
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            parsed = g.parse_template_spec(self._write_spec(tmp, "op", spec_obj))
        self.assertEqual(list(parsed.keys()), ["op"])
        v1, v4 = parsed["op"]
        self.assertEqual((v1["NAME"], v1["VEC"], v1["DTYPE"]), ("op", 1, "float"))
        self.assertEqual(v1["VARIANT_NAME"], "op")
        self.assertEqual((v4["NAME"], v4["VEC"], v4["DTYPE"]), ("op_vec4", 4, "float"))
        self.assertEqual(v4["VARIANT_NAME"], "op")

    def test_parse_template_spec_default_suffix_str_value_in_name(self) -> None:
        # A forall value with no SUFFIX contributes str(VALUE) to the variant NAME.
        spec_obj = {
            "op": {
                "parameter_names_with_default_values": {"VEC": 1},
                "generate_variant_forall": {"VEC": [{"VALUE": 4}]},
                "shader_variants": [{"NAME": "op"}],
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            parsed = g.parse_template_spec(self._write_spec(tmp, "op", spec_obj))
        self.assertEqual(parsed["op"][0]["NAME"], "op_4")

    def test_parse_template_spec_duplicate_key_raises(self) -> None:
        # The dup-key object_pairs_hook rejects a repeated key anywhere in the spec.
        dup = '{"op": {"NAME": 1, "NAME": 2}}'
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "op.json"
            p.write_text(dup)
            with self.assertRaises(ValueError):
                g.parse_template_spec(p)

    def test_headers_for_shader_top_level_key_must_match_stem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            op_dir = Path(tmp) / "runtime/ops/op"
            op_dir.mkdir(parents=True)
            (op_dir / "op.wgsl").write_text("@workgroup_size(64)\nfn main(){}\n")
            # top-level key "WRONG" != stem "op" -> must raise.
            (op_dir / "op.json").write_text(
                '{"WRONG": {"parameter_names_with_default_values": {},'
                ' "shader_variants": [{"NAME": "op"}]}}'
            )
            with self.assertRaises(ValueError):
                list(g.headers_for_shader(op_dir / "op.wgsl"))

    def test_headers_for_shader_templating_without_sidecar_raises(self) -> None:
        # A $if/${ shader with no sibling .json spec is a hard error.
        with tempfile.TemporaryDirectory() as tmp:
            op_dir = Path(tmp) / "runtime/ops/op"
            op_dir.mkdir(parents=True)
            (op_dir / "op.wgsl").write_text(
                "$if VEC == 4:\n  x\n@workgroup_size(64)\nfn main(){}\n"
            )
            with self.assertRaises(ValueError):
                list(g.headers_for_shader(op_dir / "op.wgsl"))

    # --- WGSL type-helpers -----------------------------------------------

    def test_buffer_scalar_type(self) -> None:
        self.assertEqual(g.buffer_scalar_type("half"), "f16")
        self.assertEqual(g.buffer_scalar_type("float"), "f32")

    def test_buffer_gvec_type(self) -> None:
        self.assertEqual(g.buffer_gvec_type("float", 1), "f32")
        self.assertEqual(g.buffer_gvec_type("float", 4), "vec4<f32>")
        self.assertEqual(g.buffer_gvec_type("half", 4), "vec4<f16>")

    def test_accum_scalar_type(self) -> None:
        # The float family (incl. half) accumulates in f32.
        self.assertEqual(g.accum_scalar_type("float"), "f32")
        self.assertEqual(g.accum_scalar_type("half"), "f32")

    # --- byte-identity round-trip ----------------------------------------

    def test_rms_norm_template_roundtrip_byte_identical(self) -> None:
        # Expanding the committed rms_norm.wgsl template + embedding it must
        # reproduce the committed headers exactly (the dedup proof point).
        rms_dir = g.BACKEND_ROOT / "runtime/ops/rms_norm"
        template = (rms_dir / "rms_norm.wgsl").read_text()
        for name, vec, header_name in [
            ("rms_norm", 1, "rms_norm_wgsl.h"),
            ("rms_norm_vec4", 4, "rms_norm_vec4_wgsl.h"),
        ]:
            expanded = g.preprocess(
                template, {**g.WGSL_HELPERS, "DTYPE": "float", "VEC": vec}
            )
            want = g.render_header(name, expanded, "rms_norm")
            got = (rms_dir / header_name).read_text()
            self.assertEqual(
                got, want, f"{header_name} not reproduced from rms_norm.wgsl template"
            )

    def test_rms_norm_half_variant_is_type_correct(self) -> None:
        # A DTYPE=half expansion must emit compilable WGSL: `enable f16;`, an f32
        # accumulator, loads widened to f32 for the reduction, and the store
        # narrowed back to f16 -- f16 storage with f32 compute, no type mismatch.
        template = (g.BACKEND_ROOT / "runtime/ops/rms_norm/rms_norm.wgsl").read_text()
        cases = {
            1: ("array<f16>", "f32(v) * f32(v)", "= f16(f32(v) * rstd * f32(w));"),
            4: (
                "array<vec4<f16>>",
                "dot(vec4<f32>(v), vec4<f32>(v))",
                "= vec4<f16>(vec4<f32>(t_in[base4 + x4]) * rstd"
                " * vec4<f32>(t_weight[x4]));",
            ),
        }
        for vec, (buf, widened_accum, narrowed_store) in cases.items():
            out = g.preprocess(
                template, {**g.WGSL_HELPERS, "DTYPE": "half", "VEC": vec}
            )
            self.assertTrue(out.startswith("enable f16;\n"))
            self.assertIn(buf, out)
            self.assertIn("local_sq_sum: f32", out)  # f32 accumulator for both dtypes
            self.assertIn(widened_accum, out)
            self.assertIn(narrowed_store, out)


if __name__ == "__main__":
    unittest.main()
