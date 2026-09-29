# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit + drift tests for the embedded-WGSL-header generator.

Loads the generator by file path (no package/namespace dependency).
"""

import contextlib
import hashlib
import importlib.util
import io
import os
import re
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

_GEN = Path(__file__).resolve().parents[1] / "scripts" / "gen_wgsl_headers.py"
_spec = importlib.util.spec_from_file_location("gen_wgsl_headers", _GEN)
g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g)

# gen_wgsl_headers.py and backends/vulkan/runtime/gen_vulkan_spv.py share the
# same $-block transpiler helpers + the UniqueKeyLoader; the test below keeps
# them in sync with that source of truth. Resolve the path relative to the repo
# root (both backends/vulkan and backends/webgpu exist in pytorch/executorch)
# and compare the bodies as TEXT -- gen_vulkan_spv.py is the Vulkan backend's
# script and is not imported here.
_REPO_ROOT = g.BACKEND_ROOT.parents[1]
_VULKAN_SPV = _REPO_ROOT / "backends" / "vulkan" / "runtime" / "gen_vulkan_spv.py"
_SHARED_TRANSPILER_NAMES = (
    "extract_leading_whitespace",
    "escape",
    "preprocess",
    "UniqueKeyLoader",
)


def _function_source(text: str, name: str) -> str:
    """Return a top-level function/class's source: the `def`/`class <name>` line
    through the last line before the next column-0 construct (to next dedent).

    Two-phase so a multi-line signature -- whose closing `) -> str:` sits at
    column 0 -- is not mistaken for the next top-level construct.
    """
    lines = text.splitlines()
    start = next(
        (
            i
            for i, ln in enumerate(lines)
            if re.match(rf"^(?:def|class) {re.escape(name)}\b", ln)
        ),
        None,
    )
    if start is None:
        raise AssertionError(f"def/class {name} not found")
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
    def test_compare_word_count_does_not_overflow_u32(self) -> None:
        source = (g.BACKEND_ROOT / "runtime/ops/compare/compare.wgsl").read_text()
        expression = "(params.num_elements - 1u) / 4u + 1u"
        self.assertIn(expression, source)
        for num_elements in (1, 4, 5, (1 << 32) - 3, (1 << 32) - 2, (1 << 32) - 1):
            self.assertEqual(
                (num_elements - 1) // 4 + 1,
                (num_elements + 3) // 4,
            )

    def test_registry_entries_match_concrete_headers(self) -> None:
        entries = g.registry_entries()
        names = [entry.name for entry in entries]
        expected = sorted(
            header.name[: -len("_wgsl.h")]
            for wgsl in g.discover()
            for header, _ in g.headers_for_shader(wgsl)
        )
        self.assertEqual(names, expected)
        self.assertEqual(len(names), len(set(names)))

    def test_registry_render_is_deterministic(self) -> None:
        entries = g.registry_entries()
        self.assertEqual(
            g.render_registry(entries),
            g.render_registry(list(reversed(entries))),
        )

    def test_registry_rejects_duplicate_names(self) -> None:
        entry = g.registry_entries()[0]
        with self.assertRaisesRegex(ValueError, "duplicate shader registry name"):
            g.render_registry([entry, entry])

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

    def test_render_header_long_name_is_clang_format_stable(self) -> None:
        stem = "streaming_attention_qwen3_q32_k16_causal_bound"
        wgsl = "@compute @workgroup_size(32, 8, 1)\nfn main(){}\n"
        h = g.render_header(Path(f"runtime/ops/sdpa/{stem}.wgsl"), wgsl)

        self.assertIn(
            f"// @generated from {stem}.wgsl\n// DO NOT EDIT.",
            h,
        )
        self.assertIn(
            "inline constexpr uint32_t\n"
            "    kStreamingAttentionQwen3Q32K16CausalBoundWorkgroupSizeX = 32;",
            h,
        )
        self.assertEqual(g.embedded_sha256(h), g.wgsl_sha256(wgsl))

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

    def test_generated_output_manifest_digest(self) -> None:
        outputs = sorted(
            [
                *(g.BACKEND_ROOT / "runtime/ops").glob("**/*_wgsl.h"),
                g.registry_path(),
            ]
        )
        digest = hashlib.sha256()
        for output in outputs:
            digest.update(output.relative_to(g.BACKEND_ROOT).as_posix().encode())
            digest.update(b"\0")
            digest.update(output.read_bytes())
            digest.update(b"\0")
        self.assertEqual(len(outputs), 136)
        self.assertEqual(
            digest.hexdigest(),
            "0512f8d258952e446ffaedcb653b6a3a720eccf8a6b5327d95fd454a912214a3",
        )
        self.assertEqual(
            hashlib.sha256(g.registry_path().read_bytes()).hexdigest(),
            "28aaa7a8d3e916df43e407120e91d487d0d51cbc5ca93c56bd822d25d109890e",
        )

    def test_rope_hf_reconstructs_full_2d_grid_stride(self) -> None:
        shader = (
            g.BACKEND_ROOT / "runtime" / "ops" / "rope" / "rotary_embedding_hf.wgsl"
        ).read_text()
        self.assertIn("@builtin(num_workgroups) num_workgroups", shader)
        self.assertIn(
            "gid.x + gid.y * (num_workgroups.x * wg_size)",
            shader,
        )
        self.assertIn("let freqs_b_idx = freqs_a_idx + half_dim;", shader)
        self.assertIn("t_out[b_idx] = x_b * c_b + x_a * si_b;", shader)

        wg_size = 2
        workgroups_x = 2
        indices = [
            group_x * wg_size + lane + group_y * (workgroups_x * wg_size)
            for group_y in range(2)
            for group_x in range(workgroups_x)
            for lane in range(wg_size)
        ]
        self.assertEqual(indices, list(range(8)))

    def test_qwen3_runtime_eligibility_is_exact(self) -> None:
        sdpa = (g.BACKEND_ROOT / "runtime/ops/sdpa/Sdpa.cpp").read_text()
        self.assertIn("q/k/v/output must be fp32", sdpa)
        self.assertIn("cache dtype does not match the selected storage mode", sdpa)
        self.assertIn("scale == qwen3_expected_scale", sdpa)
        self.assertNotIn("std::fabs(scale - qwen3_expected_scale)", sdpa)

    def test_fp16_kv_graph_guards_transfer_and_topology(self) -> None:
        graph = (g.BACKEND_ROOT / "runtime/WebGPUGraph.cpp").read_text()
        self.assertIn("serialized cache tensor must be fp32", graph)
        self.assertIn("consumed through a ValueList", graph)
        self.assertIn("preserve it while changing storage", graph)

        copy_inputs = graph.index("void WebGPUGraph::copy_inputs")
        input_guard = graph.index(
            "fp16 device input requires an fp32 host tensor", copy_inputs
        )
        fast_path = graph.index("// Fast path", copy_inputs)
        self.assertLess(input_guard, fast_path)

        copy_outputs = graph.index("void WebGPUGraph::copy_outputs")
        output_guard = graph.index(
            "fp16 device output requires an fp32 host tensor", copy_outputs
        )
        map_request = graph.index("wgpuBufferMapAsync", copy_outputs)
        self.assertLess(output_guard, map_request)

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
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(g.main(["--check"]), 1)
                self.assertEqual(
                    output.getvalue().count("Stale embedded WGSL headers"), 1
                )
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


class WgslGenerationTransactionTest(unittest.TestCase):
    _VALID_SHADER = "@compute @workgroup_size(1)\nfn main() {}\n"

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "runtime/ops").mkdir(parents=True)
        self._original_root = g.BACKEND_ROOT
        g.BACKEND_ROOT = self.root

    def tearDown(self) -> None:
        g.BACKEND_ROOT = self._original_root
        self._tmp.cleanup()

    def _write_shader(
        self, directory: str, stem: str, text: str = _VALID_SHADER
    ) -> Path:
        op_dir = self.root / "runtime/ops" / directory
        op_dir.mkdir(parents=True, exist_ok=True)
        shader = op_dir / f"{stem}.wgsl"
        shader.write_text(text)
        return shader

    def _write_template(
        self, directory: str, stem: str, text: str, names: list[str]
    ) -> Path:
        shader = self._write_shader(directory, stem, text)
        spec = {
            stem: {
                "parameter_names_with_default_values": {},
                "shader_variants": [{"NAME": name} for name in names],
            }
        }
        shader.with_suffix(".yaml").write_text(yaml.safe_dump(spec))
        return shader

    def _snapshot(self):
        return {
            path.relative_to(self.root).as_posix(): (
                path.read_bytes(),
                stat.S_IMODE(path.stat().st_mode),
            )
            for path in sorted(self.root.rglob("*"))
            if path.is_file()
        }

    def _run(self, *args: str):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = g.main(list(args))
        return result, output.getvalue()

    def _assert_no_temps(self) -> None:
        self.assertEqual(list(self.root.rglob("*.tmp")), [])

    @staticmethod
    def _fail_nth(real_fn, n: int):
        calls = 0

        def wrapped(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == n:
                raise OSError(f"injected failure on call {n}")
            return real_fn(*args, **kwargs)

        return wrapped

    def test_late_malformed_shader_leaves_tree_unchanged(self) -> None:
        good = self._write_shader("a", "good")
        good.with_name("good_wgsl.h").write_text("stale\n")
        self._write_shader("z", "bad", "${MISSING\n")
        before = self._snapshot()

        result, _ = self._run()

        self.assertEqual(result, 1)
        self.assertEqual(self._snapshot(), before)

    def test_duplicate_registry_name_leaves_tree_unchanged(self) -> None:
        self._write_shader("a", "shared")
        self._write_shader("b", "shared")
        before = self._snapshot()

        result, _ = self._run()

        self.assertEqual(result, 1)
        self.assertEqual(self._snapshot(), before)

    def test_duplicate_registry_symbol_leaves_tree_unchanged(self) -> None:
        self._write_shader("a", "foo_bar")
        self._write_shader("b", "foo__bar")
        before = self._snapshot()

        result, _ = self._run()

        self.assertEqual(result, 1)
        self.assertEqual(self._snapshot(), before)

    def test_duplicate_output_path_leaves_tree_unchanged(self) -> None:
        self._write_template("op", "op", self._VALID_SHADER, ["duplicate", "duplicate"])
        before = self._snapshot()

        result, _ = self._run()

        self.assertEqual(result, 1)
        self.assertEqual(self._snapshot(), before)

    def test_second_stage_failure_leaves_tree_unchanged(self) -> None:
        self._write_shader("a", "first")
        self._write_shader("b", "second")
        before = self._snapshot()
        real_mkstemp = tempfile.mkstemp

        with mock.patch(
            "tempfile.mkstemp", side_effect=self._fail_nth(real_mkstemp, 2)
        ):
            result, _ = self._run()

        self.assertEqual(result, 1)
        self.assertEqual(self._snapshot(), before)
        self._assert_no_temps()

    def test_staging_interrupt_leaves_tree_unchanged(self) -> None:
        self._write_shader("a", "first")
        self._write_shader("b", "second")
        before = self._snapshot()
        real_chmod = Path.chmod

        def interrupt_second(path, mode, **kwargs):
            interrupt_second.calls += 1
            if interrupt_second.calls == 2:
                raise KeyboardInterrupt("injected staging interruption")
            return real_chmod(path, mode, **kwargs)

        interrupt_second.calls = 0
        with mock.patch.object(
            Path, "chmod", autospec=True, side_effect=interrupt_second
        ):
            with self.assertRaises(KeyboardInterrupt):
                self._run()

        self.assertEqual(self._snapshot(), before)
        self._assert_no_temps()

    def test_replace_failure_restores_existing_destination(self) -> None:
        self._write_shader("op", "op")
        registry = g.registry_path()
        registry.write_text("old registry\n")
        registry.chmod(0o600)
        before = self._snapshot()
        real_replace = os.replace

        with mock.patch("os.replace", side_effect=self._fail_nth(real_replace, 2)):
            result, _ = self._run()

        self.assertEqual(result, 1)
        self.assertEqual(self._snapshot(), before)
        self._assert_no_temps()

    def test_replace_failure_removes_new_destination(self) -> None:
        self._write_shader("op", "op")
        before = self._snapshot()
        real_replace = os.replace

        with mock.patch("os.replace", side_effect=self._fail_nth(real_replace, 2)):
            result, _ = self._run()

        self.assertEqual(result, 1)
        self.assertEqual(self._snapshot(), before)
        self._assert_no_temps()

    def test_multiple_rollback_errors_do_not_stop_later_restores(self) -> None:
        headers = []
        for directory in ("a", "b", "c"):
            shader = self._write_shader(directory, directory)
            header = shader.with_name(f"{directory}_wgsl.h")
            header.write_text(f"old {directory}\n")
            headers.append(header)
        registry = g.registry_path()
        registry.write_text("old registry\n")
        real_replace = os.replace
        calls = []

        def fail_commit_and_two_rollbacks(source, destination):
            calls.append(Path(destination))
            if len(calls) in (4, 5, 6):
                raise OSError(f"injected failure on replace {len(calls)}")
            return real_replace(source, destination)

        with mock.patch("os.replace", side_effect=fail_commit_and_two_rollbacks):
            result, output = self._run()

        self.assertEqual(result, 1)
        self.assertEqual(len(calls), 7)
        self.assertEqual(calls[-3:], [headers[1], headers[0], registry])
        self.assertIn(f"cannot roll back {headers[1]}", output)
        self.assertIn(f"cannot roll back {headers[0]}", output)
        self.assertEqual(registry.read_text(), "old registry\n")
        self.assertNotEqual(headers[0].read_text(), "old a\n")
        self.assertNotEqual(headers[1].read_text(), "old b\n")
        self.assertEqual(headers[2].read_text(), "old c\n")
        self._assert_no_temps()

    def test_success_preserves_existing_mode_and_creates_0644(self) -> None:
        shader = self._write_shader("op", "op")
        registry = g.registry_path()
        registry.write_text("old registry\n")
        registry.chmod(0o600)

        result, _ = self._run()

        self.assertEqual(result, 0)
        self.assertEqual(stat.S_IMODE(registry.stat().st_mode), 0o600)
        self.assertEqual(
            stat.S_IMODE(shader.with_name("op_wgsl.h").stat().st_mode), 0o644
        )

    def test_orphans_are_sorted_reported_and_never_deleted(self) -> None:
        self._write_shader("new", "new")
        orphan_z = self.root / "runtime/ops/z/old_z_wgsl.h"
        orphan_a = self.root / "runtime/ops/a/old_a_wgsl.h"
        orphan_z.parent.mkdir(parents=True)
        orphan_a.parent.mkdir(parents=True)
        orphan_z.write_text("// @generated\n")
        orphan_a.write_text("// @generated\n")
        before = self._snapshot()

        check_result, check_output = self._run("--check")
        normal_result, normal_output = self._run()

        self.assertEqual(check_result, 1)
        self.assertEqual(normal_result, 1)
        for output in (check_output, normal_output):
            self.assertIn("Orphan", output)
            self.assertLess(output.index("old_a_wgsl.h"), output.index("old_z_wgsl.h"))
        self.assertEqual(self._snapshot(), before)

    def test_check_fails_read_only_when_outputs_are_only_missing(self) -> None:
        self._write_shader("op", "op")
        before = self._snapshot()

        result, output = self._run("--check")

        self.assertEqual(result, 1)
        self.assertIn("Missing embedded WGSL headers", output)
        self.assertEqual(self._snapshot(), before)

    def test_check_catches_template_syntax_error_without_writing(self) -> None:
        self._write_template(
            "a", "syntax", "$if :\n  " + self._VALID_SHADER, ["syntax"]
        )
        before = self._snapshot()

        result, output = self._run("--check")

        self.assertEqual(result, 1)
        self.assertIn("runtime/ops/a/syntax.wgsl", output)
        self.assertEqual(self._snapshot(), before)

    def test_check_catches_template_name_error_without_writing(self) -> None:
        self._write_template(
            "op", "name", "$if MISSING:\n  " + self._VALID_SHADER, ["name"]
        )
        before = self._snapshot()

        result, output = self._run("--check")

        self.assertEqual(result, 1)
        self.assertIn("runtime/ops/op/name.wgsl", output)
        self.assertEqual(self._snapshot(), before)

    def test_interrupted_commit_is_detected_and_repaired(self) -> None:
        self._write_shader("op", "op")
        before = self._snapshot()
        real_replace = os.replace

        def interrupt_second(source, destination):
            interrupt_second.calls += 1
            if interrupt_second.calls == 2:
                raise KeyboardInterrupt("injected interruption")
            return real_replace(source, destination)

        interrupt_second.calls = 0
        with mock.patch("os.replace", side_effect=interrupt_second):
            with self.assertRaises(KeyboardInterrupt):
                self._run()

        self.assertEqual(self._snapshot(), before)
        self._assert_no_temps()
        check_result, check_output = self._run("--check")
        self.assertEqual(check_result, 1)
        self.assertNotIn("Orphan", check_output)

        normal_result, _ = self._run()
        self.assertEqual(normal_result, 0)
        final_check_result, _ = self._run("--check")
        self.assertEqual(final_check_result, 0)
        self._assert_no_temps()

    def test_interrupt_after_replace_restores_tree(self) -> None:
        self._write_shader("op", "op")
        before = self._snapshot()
        real_replace = os.replace

        def interrupt_after_second(source, destination):
            interrupt_after_second.calls += 1
            result = real_replace(source, destination)
            if interrupt_after_second.calls == 2:
                raise KeyboardInterrupt("injected post-replace interruption")
            return result

        interrupt_after_second.calls = 0
        with mock.patch("os.replace", side_effect=interrupt_after_second):
            with self.assertRaises(KeyboardInterrupt):
                self._run()

        self.assertEqual(self._snapshot(), before)
        self._assert_no_temps()

    def test_generation_renders_once_and_second_run_does_no_io(self) -> None:
        shaders = [
            self._write_shader("a", "first"),
            self._write_shader("b", "second"),
        ]
        render_counts = {shader: 0 for shader in shaders}
        real_headers_for_shader = g.headers_for_shader

        def counted(shader):
            render_counts[shader] += 1
            return real_headers_for_shader(shader)

        with mock.patch.object(g, "headers_for_shader", side_effect=counted):
            first_result, _ = self._run()
        self.assertEqual(first_result, 0)
        self.assertEqual(render_counts, {shader: 1 for shader in shaders})

        with mock.patch(
            "tempfile.mkstemp", wraps=tempfile.mkstemp
        ) as mkstemp, mock.patch("os.replace", wraps=os.replace) as replace:
            second_result, _ = self._run()
        self.assertEqual(second_result, 0)
        mkstemp.assert_not_called()
        replace.assert_not_called()


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
        for name in _SHARED_TRANSPILER_NAMES:
            self.assertEqual(
                _function_source(src_text, name),
                _function_source(gen_text, name),
                f"{name} has drifted from its source of truth "
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
        p = Path(tmp) / f"{name}.yaml"
        p.write_text(yaml.safe_dump(spec_obj))
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
        # UniqueKeyLoader rejects a repeated key anywhere in the spec (this flow
        # mapping is valid YAML with a duplicate key).
        dup = '{"op": {"NAME": 1, "NAME": 2}}'
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "op.yaml"
            p.write_text(dup)
            with self.assertRaises(yaml.YAMLError):
                g.parse_template_spec(p)

    def test_headers_for_shader_top_level_key_must_match_stem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            op_dir = Path(tmp) / "runtime/ops/op"
            op_dir.mkdir(parents=True)
            (op_dir / "op.wgsl").write_text("@workgroup_size(64)\nfn main(){}\n")
            # top-level key "WRONG" != stem "op" -> must raise.
            (op_dir / "op.yaml").write_text(
                '{"WRONG": {"parameter_names_with_default_values": {},'
                ' "shader_variants": [{"NAME": "op"}]}}'
            )
            with self.assertRaises(ValueError):
                list(g.headers_for_shader(op_dir / "op.wgsl"))

    def test_headers_for_shader_templating_without_sidecar_raises(self) -> None:
        # A $if/${ shader with no sibling .yaml spec is a hard error.
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

    def test_to_copy_convert_template_roundtrip_byte_identical(self) -> None:
        to_copy_dir = g.BACKEND_ROOT / "runtime/ops/to_copy"
        template_path = to_copy_dir / "to_copy_convert.wgsl"
        spec = g.parse_template_spec(template_path.with_suffix(".yaml"))
        variants = {params["NAME"]: params for params in spec[template_path.stem]}
        expected = {
            "to_copy_float_to_int": (
                "f32",
                "i32",
                "c331e00e3171eecbe6317ac9df0a5f9cd6d25da26a9a587250f1cc6086dc3c8f",
            ),
            "to_copy_int_to_float": (
                "i32",
                "f32",
                "e18dd733a3838f83eded4977a2a2b21119099c8409b234f12474fae5acc9b195",
            ),
        }
        self.assertEqual(set(variants), set(expected))
        template = template_path.read_text()

        for name, (in_type, out_type, expected_hash) in expected.items():
            params = variants[name]
            self.assertEqual(
                (params["IN_TYPE"], params["OUT_TYPE"]), (in_type, out_type)
            )
            expanded = g.preprocess(template, {**g.WGSL_HELPERS, **params})
            self.assertEqual(g.wgsl_sha256(expanded), expected_hash)

            header = (to_copy_dir / f"{name}_wgsl.h").read_text()
            body = header.split('R"(', 1)[1].split(')";', 1)[0][1:]
            self.assertEqual(body, expanded)
            self.assertEqual(g.embedded_sha256(header), expected_hash)
            self.assertEqual(g.parse_workgroup_size(body), (64, 1, 1))

        entries = {entry.name: entry for entry in g.registry_entries()}
        self.assertEqual(
            entries["to_copy_float_to_int"].include,
            "runtime/ops/to_copy/to_copy_float_to_int_wgsl.h",
        )
        self.assertEqual(
            entries["to_copy_int_to_float"].include,
            "runtime/ops/to_copy/to_copy_int_to_float_wgsl.h",
        )

    def test_extrema_template_roundtrip_byte_identical(self) -> None:
        extrema_dir = g.BACKEND_ROOT / "runtime/ops/extrema"
        template_path = extrema_dir / "extrema.wgsl"
        spec = g.parse_template_spec(template_path.with_suffix(".yaml"))
        variants = {params["NAME"]: params for params in spec[template_path.stem]}
        expected = {
            "amax": (
                "max",
                "35fc059d7c72caa17f9cb1128823ecfd8f75be4ce24b6cd4f9629a97b52f64c0",
            ),
            "amin": (
                "min",
                "8cb6035ae4d34eb2a6cc973d93d9847905722e967239c96033fccfe3a1943cb2",
            ),
        }
        self.assertEqual(set(variants), set(expected))
        template = template_path.read_text()

        for name, (reduce_fn, expected_hash) in expected.items():
            params = variants[name]
            self.assertEqual(params["REDUCE_FN"], reduce_fn)
            expanded = g.preprocess(template, {**g.WGSL_HELPERS, **params})
            self.assertEqual(g.wgsl_sha256(expanded), expected_hash)

            header_path = extrema_dir / f"{name}_wgsl.h"
            header = header_path.read_text()
            body = header.split('R"(', 1)[1].split(')";', 1)[0][1:]
            self.assertEqual(body, expanded)
            self.assertEqual(g.embedded_sha256(header), expected_hash)
            self.assertEqual(g.parse_workgroup_size(body), (256, 1, 1))

        entries = {entry.name: entry for entry in g.registry_entries()}
        for name in expected:
            self.assertEqual(
                entries[name].include,
                f"runtime/ops/extrema/{name}_wgsl.h",
            )
            self.assertEqual(entries[name].symbol, g.symbol_base(name))

        handler_hashes = {
            "amax": "57f929b9f3087dc32403c3587884ce2ed4be2d03c4e80ff7035428b52e7e0e51",
            "amin": "5dc947d4781a67df953b9c5970c4ab2119317b29657f983a9d308dfdf123dede",
        }
        for name, expected_hash in handler_hashes.items():
            handler = g.BACKEND_ROOT / f"runtime/ops/{name}/Reduce.cpp"
            self.assertEqual(
                hashlib.sha256(handler.read_bytes()).hexdigest(), expected_hash
            )

    def test_logical_binary_template_roundtrip_byte_identical(self) -> None:
        logical_dir = g.BACKEND_ROOT / "runtime/ops/logical_binary"
        template_path = logical_dir / "logical_binary.wgsl"
        spec = g.parse_template_spec(template_path.with_suffix(".yaml"))
        variants = {params["NAME"]: params for params in spec[template_path.stem]}
        expected = {
            "logical_and": (
                "&",
                "cf7c1d1dbba94e429120796c9c25a6717786cca03c08f3bd1e291d5627089c20",
            ),
            "logical_or": (
                "|",
                "4ad19ee04e2c7b396b4669cf44f95133d658c3ec2e6f37d7b271bedc0e582ecf",
            ),
        }
        self.assertEqual(set(variants), set(expected))
        template = template_path.read_text()

        for name, (op, expected_hash) in expected.items():
            params = variants[name]
            self.assertEqual(params["OP"], op)
            expanded = g.preprocess(template, {**g.WGSL_HELPERS, **params})
            self.assertEqual(g.wgsl_sha256(expanded), expected_hash)

            header_path = logical_dir / f"{name}_wgsl.h"
            header = header_path.read_text()
            body = header.split('R"(', 1)[1].split(')";', 1)[0][1:]
            self.assertEqual(body, expanded)
            self.assertEqual(g.embedded_sha256(header), expected_hash)
            self.assertEqual(g.parse_workgroup_size(body), (64, 1, 1))

        entries = {entry.name: entry for entry in g.registry_entries()}
        for name in expected:
            self.assertEqual(
                entries[name].include,
                f"runtime/ops/logical_binary/{name}_wgsl.h",
            )
            self.assertEqual(entries[name].symbol, g.symbol_base(name))

        handler_hashes = {
            "logical_and": "eb85a8f97ee7640298a661da49feb08aa79b8c24d3d4458b71d24d3f01bc388d",
            "logical_or": "bda18617f7077fee5a812c21cdc495c89542a1688f7e1ef6739ed01da343a66b",
        }
        for name, expected_hash in handler_hashes.items():
            handler = (
                g.BACKEND_ROOT / f"runtime/ops/{name}/Logical{name[8:].title()}.cpp"
            )
            self.assertEqual(
                hashlib.sha256(handler.read_bytes()).hexdigest(), expected_hash
            )

    def test_binary_family_roundtrip_byte_identical(self) -> None:
        binary_dir = g.BACKEND_ROOT / "runtime/ops/binary_op"
        template_path = binary_dir / "binary_op.wgsl"
        spec = g.parse_template_spec(template_path.with_suffix(".yaml"))
        variants = {params["NAME"]: params for params in spec[template_path.stem]}
        expected = {
            "binary_div": (
                0,
                "e36b560fd623dd5337b9ae57acd8981c9c635b995d6021caf1331c182cd3f0cd",
            ),
            "binary_sub": (
                0,
                "63209ff70422a21fc340d9aadba0945bc259bba89bdf05db018a6507d01c7ae5",
            ),
            "binary_minimum": (
                1,
                "929b7ba85936e3652baea9f4e5e7f049d232c7ae7a74814a536b4c2674897972",
            ),
            "binary_pow": (
                1,
                "a88c161bd3f43d21a72ebd8ca6f8611b6b9b854e3572a8e6b820602091bc464c",
            ),
            "binary_floor_divide": (
                1,
                "baf71d277da79389315a6b96b439e7f0a55842e8288283f2af121f84536b3af3",
            ),
            "binary_mul": (
                1,
                "d248c0f1856b57115a5001a47f4936caa564dd3b787c02ceba504a13ab987812",
            ),
        }
        self.assertEqual(set(variants), set(expected))
        template = template_path.read_text()
        entries = {entry.name: entry for entry in g.registry_entries()}

        for name, (inline, expected_hash) in expected.items():
            params = variants[name]
            self.assertEqual(params["INLINE"], inline)
            expanded = g.preprocess(template, {**g.WGSL_HELPERS, **params})
            self.assertEqual(g.wgsl_sha256(expanded), expected_hash)

            header = (binary_dir / f"{name}_wgsl.h").read_text()
            literal = header.split('R"(', 1)[1].split(')";', 1)[0]
            self.assertEqual(literal, "\n" + expanded)
            self.assertEqual(g.embedded_sha256(header), expected_hash)
            self.assertEqual(g.parse_workgroup_size(expanded), (64, 1, 1))
            self.assertIn(f"k{g.symbol_base(name)}WGSL", header)
            self.assertEqual(
                entries[name].include,
                f"runtime/ops/binary_op/{name}_wgsl.h",
            )

    def test_unary_template_roundtrip_byte_identical(self) -> None:
        unary_dir = g.BACKEND_ROOT / "runtime/ops/unary"
        template_path = unary_dir / "unary.wgsl"
        spec = g.parse_template_spec(template_path.with_suffix(".yaml"))
        variants = {params["NAME"]: params for params in spec[template_path.stem]}
        expected = {
            "abs": "39d3c163fdf6a92286828f4b3217e00294e3ca5634a878ed5fd34e3b1cdf0a27",
            "cos": "9df78873e5fae98d347c26db2a02b047ea3d5d2c93f0761cb9ac6995f9a71ab2",
            "exp": "3171399bc36acf9c1cb2a03c2a31038318203c4c63ab03c4881df7a660346020",
            "hardswish": "c874a15ef6cdaec71187296016cc2a1515f5e7c889b97dfa8fd4b278e6e2c3d5",
            "neg": "8851b9f42d14153f6f04484fee2f8bf67bda26dea892ff48768e09e6ad49cee1",
            "round": "8f3e0edbeb81aa50f35e691c78554e8057fa8d78fe8a86454f4f42e5e8871452",
            "rsqrt": "108765d5a23b87473f34651875d08abf2a5fa8980bd92fc8cbe3617295097747",
            "sin": "e5762804773659d348fddddcef4935807ae6fe7d92c92eb17a2f44aae8f2c5b9",
            "sqrt": "008534ae365969f5c180b42e8d6d0b131df78f181e5435abbcafc3ffb8be8aac",
            "tanh": "5bd7eb1c6411940d84a9b311884f35b39f15b82103b14bab02902290ed6b0339",
        }
        self.assertEqual(set(variants), set(expected))
        template = template_path.read_text()
        entries = {entry.name: entry for entry in g.registry_entries()}
        for name, expected_hash in expected.items():
            expanded = g.preprocess(template, {**g.WGSL_HELPERS, **variants[name]})
            self.assertEqual(g.wgsl_sha256(expanded), expected_hash)
            header = (unary_dir / f"{name}_wgsl.h").read_text()
            body = header.split('R"(', 1)[1].split(')";', 1)[0][1:]
            self.assertEqual(body, expanded)
            self.assertEqual(g.embedded_sha256(header), expected_hash)
            self.assertEqual(g.parse_workgroup_size(body), (256, 1, 1))
            self.assertEqual(entries[name].include, f"runtime/ops/unary/{name}_wgsl.h")
            self.assertEqual(entries[name].symbol, g.symbol_base(name))
        self.assertTrue({"clamp", "pow_scalar"}.isdisjoint(variants))

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
