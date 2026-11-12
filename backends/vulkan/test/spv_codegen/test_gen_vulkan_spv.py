# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import re
import stat
import struct
import tempfile
import unittest
from pathlib import Path

from runtime import gen_vulkan_spv


class TestF16ImageStores(unittest.TestCase):
    def test_writable_rgba16f_images_round_stores(self) -> None:
        for access in ("w", "rw"):
            with self.subTest(access=access):
                declaration = gen_vulkan_spv.layout_declare_image(
                    0, access, "t_out", "half"
                )
                self.assertIn("round_to_f16(texel)", declaration)

    def test_f16_texels_skip_rounding_in_f16_shaders(self) -> None:
        declaration = gen_vulkan_spv.layout_declare_image(0, "w", "t_out", "half")
        self.assertIn(
            "#ifdef HAS_FLOAT16_ACCUMULATION\n"
            "vec4 round_to_f16(const f16vec4 v) { return vec4(v); }",
            declaration,
        )

    def test_other_images_store_unchanged(self) -> None:
        for access, dtype in (("r", "half"), ("w", "float"), ("w", "int32")):
            with self.subTest(access=access, dtype=dtype):
                declaration = gen_vulkan_spv.layout_declare_image(0, access, "t", dtype)
                self.assertNotIn("round_to_f16", declaration)


class TestNestedShaderSources(unittest.TestCase):
    def _write_fake_compiler(
        self, directory: Path, assertions: str = ""
    ) -> tuple[Path, Path]:
        compiler_log = directory / "compiler.log"
        compiler = directory / "fake_glslc.py"
        compiler.write_text(
            "#!/usr/bin/env python3\n"
            "import pathlib\n"
            "import sys\n"
            "log = pathlib.Path(__file__).with_name('compiler.log')\n"
            "with log.open('a') as log_file:\n"
            "    log_file.write('compile\\n')\n"
            "source = pathlib.Path(sys.argv[2])\n"
            + assertions
            + "output = pathlib.Path(sys.argv[sys.argv.index('-o') + 1])\n"
            "output.write_bytes(bytes.fromhex('0302230700000100000000000100000000000000'))\n"
        )
        compiler.chmod(compiler.stat().st_mode | stat.S_IXUSR)
        return compiler, compiler_log

    def _write_shader(self, directory: Path, variant_name: str) -> None:
        directory.mkdir(parents=True)
        (directory / "common.glslh").write_text("const uint value = 1;\n")
        (directory / "kernel.glsl").write_text(
            '#version 450\n#include "common.glslh"\nvoid main() {}\n'
        )
        (directory / "kernel.yaml").write_text(
            "kernel:\n"
            "  parameter_names_with_default_values: {}\n"
            "  shader_variants:\n"
            f"    - NAME: {variant_name}\n"
        )

    def test_repeated_template_names_in_nested_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "src"
            self._write_shader(source_root / "direct", "direct_kernel")
            self._write_shader(source_root / "winograd", "winograd_kernel")
            output_dir = root / "out"
            cache_dir = root / "cache"
            output_dir.mkdir()
            cache_dir.mkdir()

            generator = gen_vulkan_spv.SPVGenerator(
                str(source_root), {}, glslc_path=None
            )
            generator.generateSPV(output_dir, cache_dir, nthreads=1)

            self.assertTrue((output_dir / "direct" / "direct_kernel.glsl").is_file())
            self.assertTrue((output_dir / "direct" / "common.glslh").is_file())
            self.assertTrue(
                (output_dir / "winograd" / "winograd_kernel.glsl").is_file()
            )
            self.assertTrue((output_dir / "winograd" / "common.glslh").is_file())

    def test_multi_dot_source_preserves_legacy_template_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "src"
            source_root.mkdir()
            (source_root / "kernel.special.glsl").write_text(
                "#version 450\nvoid main() {}\n"
            )
            (source_root / "kernel.yaml").write_text(
                "kernel:\n"
                "  parameter_names_with_default_values: {}\n"
                "  shader_variants:\n"
                "    - NAME: generated_kernel\n"
            )
            output_dir = root / "out"
            cache_dir = root / "cache"
            output_dir.mkdir()
            cache_dir.mkdir()

            generator = gen_vulkan_spv.SPVGenerator(
                str(source_root), {}, glslc_path=None
            )
            generator.generateSPV(output_dir, cache_dir, nthreads=1)

            self.assertTrue((output_dir / "generated_kernel.glsl").is_file())

    def test_rejects_generated_shader_outside_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "src"
            self._write_shader(source_root, str(root / "escaped_kernel"))
            output_dir = root / "out"
            cache_dir = root / "cache"
            output_dir.mkdir()
            cache_dir.mkdir()
            compiler, _ = self._write_fake_compiler(root)

            generator = gen_vulkan_spv.SPVGenerator(
                str(source_root), {}, glslc_path=str(compiler)
            )
            with self.assertRaisesRegex(ValueError, "outside output directory"):
                generator.generateSPV(output_dir, cache_dir, nthreads=1)

    def test_local_include_change_recompiles_nested_shader(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "src"
            shader_dir = source_root / "nested"
            self._write_shader(shader_dir, "nested_kernel")
            (source_root / "root_common.glslh").write_text(
                "const uint root_value = 1;\n"
            )
            (shader_dir / "kernel.glsl").write_text(
                '#version 450\n#include "common.glslh"\n'
                '#include "root_common.glslh"\nvoid main() {}\n'
            )
            output_dir = root / "out"
            cache_dir = root / "cache"
            output_dir.mkdir()
            cache_dir.mkdir()

            compiler, compiler_log = self._write_fake_compiler(
                root,
                "include_indices = [i for i, arg in enumerate(sys.argv) if arg == '-I']\n"
                "local_include_dir = pathlib.Path(sys.argv[include_indices[0] + 1])\n"
                "root_include_dir = pathlib.Path(sys.argv[include_indices[1] + 1])\n"
                "assert local_include_dir == source.parent\n"
                "assert (local_include_dir / 'common.glslh').is_file()\n"
                "assert root_include_dir == source.parent.parent\n"
                "assert (root_include_dir / 'root_common.glslh').is_file()\n",
            )

            def generate() -> None:
                generator = gen_vulkan_spv.SPVGenerator(
                    str(source_root), {}, glslc_path=str(compiler)
                )
                generator.generateSPV(output_dir, cache_dir, nthreads=1)

            generate()
            generate()
            self.assertEqual(compiler_log.read_text().splitlines(), ["compile"])

            (shader_dir / "common.glslh").write_text("const uint value = 2;\n")
            generate()

            self.assertEqual(
                compiler_log.read_text().splitlines(), ["compile", "compile"]
            )

            (source_root / "root_common.glslh").write_text(
                "const uint root_value = 2;\n"
            )
            generate()

            self.assertEqual(
                compiler_log.read_text().splitlines(),
                ["compile", "compile", "compile"],
            )

    def test_ancestor_include_change_recompiles_nested_shader(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "src"
            family_dir = source_root / "family"
            shader_dir = family_dir / "route"
            self._write_shader(shader_dir, "route_kernel")
            (family_dir / "family_common.glslh").write_text(
                "const uint family_value = 1;\n"
            )
            (shader_dir / "kernel.glsl").write_text(
                '#version 450\n#include "family_common.glslh"\nvoid main() {}\n'
            )
            output_dir = root / "out"
            cache_dir = root / "cache"
            output_dir.mkdir()
            cache_dir.mkdir()

            compiler, compiler_log = self._write_fake_compiler(
                root,
                "include_dirs = [pathlib.Path(sys.argv[i + 1]) "
                "for i, arg in enumerate(sys.argv) if arg == '-I']\n"
                "assert include_dirs == [source.parent, source.parent.parent, "
                "source.parent.parent.parent]\n",
            )

            def generate() -> None:
                generator = gen_vulkan_spv.SPVGenerator(
                    str(source_root), {}, glslc_path=str(compiler)
                )
                generator.generateSPV(output_dir, cache_dir, nthreads=1)

            generate()
            generate()
            self.assertEqual(compiler_log.read_text().splitlines(), ["compile"])

            (family_dir / "family_common.glslh").write_text(
                "const uint family_value = 2;\n"
            )
            generate()

            self.assertEqual(
                compiler_log.read_text().splitlines(), ["compile", "compile"]
            )

    def test_untracked_include_disables_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "src"
            self._write_shader(source_root, "kernel")
            (source_root / "kernel.glsl").write_text(
                '#version 450\n#include "external.glslh"\nvoid main() {}\n'
            )
            output_dir = root / "out"
            cache_dir = root / "cache"
            output_dir.mkdir()
            cache_dir.mkdir()
            compiler, compiler_log = self._write_fake_compiler(root)

            generator = gen_vulkan_spv.SPVGenerator(
                str(source_root), {}, glslc_path=str(compiler)
            )
            generator.generateSPV(output_dir, cache_dir, nthreads=1)
            generator.generateSPV(output_dir, cache_dir, nthreads=1)

            self.assertEqual(
                compiler_log.read_text().splitlines(), ["compile", "compile"]
            )

    def test_cyclic_includes_use_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "src"
            self._write_shader(source_root, "kernel")
            (source_root / "kernel.glsl").write_text(
                '#version 450\n#include "a.glslh"\nvoid main() {}\n'
            )
            (source_root / "a.glslh").write_text('#include "b.glslh"\n')
            (source_root / "b.glslh").write_text('#include "a.glslh"\n')
            output_dir = root / "out"
            cache_dir = root / "cache"
            output_dir.mkdir()
            cache_dir.mkdir()
            compiler, compiler_log = self._write_fake_compiler(root)

            generator = gen_vulkan_spv.SPVGenerator(
                str(source_root), {}, glslc_path=str(compiler)
            )
            generator.generateSPV(output_dir, cache_dir, nthreads=1)
            generator.generateSPV(output_dir, cache_dir, nthreads=1)

            self.assertEqual(compiler_log.read_text().splitlines(), ["compile"])

    def test_selection_preserves_helpers_but_omits_unselected_glsl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "src"
            self._write_shader(source_root / "keep", "kept")
            self._write_shader(source_root / "drop", "dropped")
            generator = gen_vulkan_spv.SPVGenerator(
                str(source_root),
                {},
                glslc_path=None,
                shader_regexes=[re.compile("kept")],
            )
            self.assertEqual(
                set(generator.output_file_map),
                {"keep/kept", "keep/common", "drop/common"},
            )
            output_dir, cache_dir = root / "out", root / "cache"
            output_dir.mkdir()
            cache_dir.mkdir()
            generator.generateSPV(output_dir, cache_dir, nthreads=1)
            self.assertTrue((output_dir / "keep" / "kept.glsl").is_file())
            self.assertTrue((output_dir / "keep" / "common.glslh").is_file())
            self.assertFalse((output_dir / "drop" / "dropped.glsl").exists())

    def test_selection_does_not_embed_stale_cached_shaders(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "src"
            self._write_shader(source_root / "keep", "kept")
            self._write_shader(source_root / "drop", "dropped")
            output_dir, cache_dir = root / "out", root / "cache"
            output_dir.mkdir()
            cache_dir.mkdir()
            compiler, compiler_log = self._write_fake_compiler(root)
            full_generator = gen_vulkan_spv.SPVGenerator(
                str(source_root), {}, glslc_path=str(compiler)
            )
            full_spv = full_generator.generateSPV(output_dir, cache_dir, nthreads=1)
            self.assertEqual(sum(path is not None for path in full_spv), 2)
            selected_generator = gen_vulkan_spv.SPVGenerator(
                str(source_root),
                {},
                glslc_path=str(compiler),
                shader_regexes=[re.compile("kept")],
            )
            selected_spv = selected_generator.generateSPV(
                output_dir, cache_dir, nthreads=1
            )
            self.assertEqual(sum(path is not None for path in selected_spv), 1)
            self.assertEqual(
                compiler_log.read_text().splitlines(), ["compile", "compile"]
            )
            cpp_path = output_dir / "spv.cpp"
            gen_vulkan_spv.genCppFiles(
                selected_spv, str(output_dir / "spv.h"), str(cpp_path)
            )
            self.assertIn("kept_bin", cpp_path.read_text())
            self.assertNotIn("dropped_bin", cpp_path.read_text())
            self.assertTrue(
                all(Path(path).exists() for path in full_spv if path is not None)
            )

    def test_selection_rejects_empty_unknown_and_header_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_root = Path(temp_dir) / "src"
            self._write_shader(source_root, "kept")
            cases = [
                ([], "must not be empty"),
                ([re.compile("missing")], "match no shaders: missing"),
                (
                    [re.compile("kept"), re.compile("missing")],
                    "match no shaders: missing",
                ),
                ([re.compile("common")], "match no shaders: common"),
                ([re.compile("kep")], "match no shaders: kep"),
            ]
            for patterns, message in cases:
                with self.subTest(patterns=patterns), self.assertRaisesRegex(
                    ValueError, message
                ):
                    gen_vulkan_spv.SPVGenerator(
                        str(source_root), {}, glslc_path=None, shader_regexes=patterns
                    )

    def test_selection_supports_untemplated_shaders(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_root = Path(temp_dir)
            (source_root / "kept.glsl").write_text("#version 450\nvoid main() {}\n")
            (source_root / "dropped.glsl").write_text("#version 450\nvoid main() {}\n")
            generator = gen_vulkan_spv.SPVGenerator(
                str(source_root),
                {},
                glslc_path=None,
                shader_regexes=[re.compile("kept")],
            )
            self.assertEqual(set(generator.output_file_map), {"kept"})
            full_generator = gen_vulkan_spv.SPVGenerator(
                str(source_root), {}, glslc_path=None
            )
            self.assertEqual(set(full_generator.output_file_map), {"kept", "dropped"})

    def test_regex_selects_yaml_expanded_variants(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "src"
            self._write_shader(source_root, "parameterized")
            (source_root / "kernel.yaml").write_text(
                "kernel:\n"
                "  parameter_names_with_default_values: {DTYPE: float}\n"
                "  generate_variant_forall:\n"
                "    DTYPE: [{VALUE: float}, {VALUE: half}]\n"
                "  shader_variants: [{NAME: parameterized}]\n"
            )
            output_dir, cache_dir = root / "out", root / "cache"
            output_dir.mkdir()
            cache_dir.mkdir()
            compiler, compiler_log = self._write_fake_compiler(root)
            full_generator = gen_vulkan_spv.SPVGenerator(
                str(source_root), {}, glslc_path=str(compiler)
            )
            self.assertEqual(
                set(full_generator.output_file_map),
                {"parameterized_float", "parameterized_half", "common"},
            )
            full_spv = full_generator.generateSPV(output_dir, cache_dir, nthreads=1)
            self.assertEqual(sum(path is not None for path in full_spv), 2)
            selected_generator = gen_vulkan_spv.SPVGenerator(
                str(source_root),
                {},
                glslc_path=str(compiler),
                shader_regexes=[re.compile("parameterized_half")],
            )
            self.assertEqual(
                set(selected_generator.output_file_map),
                {"parameterized_half", "common"},
            )
            selected_spv = selected_generator.generateSPV(
                output_dir, cache_dir, nthreads=1
            )
            self.assertEqual(sum(path is not None for path in selected_spv), 1)
            self.assertEqual(
                compiler_log.read_text().splitlines(), ["compile", "compile"]
            )
            cpp_path = output_dir / "spv.cpp"
            gen_vulkan_spv.genCppFiles(
                selected_spv, str(output_dir / "spv.h"), str(cpp_path)
            )
            self.assertIn("parameterized_half_bin", cpp_path.read_text())
            self.assertNotIn("parameterized_float_bin", cpp_path.read_text())

    def test_regex_family_includes_new_variants(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_root = Path(temp_dir) / "src"
            self._write_shader(source_root / "old", "conv2d_old_float")
            self._write_shader(source_root / "drop", "not_conv2d_float")
            patterns = [re.compile("conv2d_.*")]
            generator = gen_vulkan_spv.SPVGenerator(
                str(source_root), {}, glslc_path=None, shader_regexes=patterns
            )
            self.assertIn("old/conv2d_old_float", generator.output_file_map)
            self.assertNotIn("drop/not_conv2d_float", generator.output_file_map)

            self._write_shader(source_root / "new", "conv2d_new_half")
            generator = gen_vulkan_spv.SPVGenerator(
                str(source_root), {}, glslc_path=None, shader_regexes=patterns
            )
            self.assertIn("old/conv2d_old_float", generator.output_file_map)
            self.assertIn("new/conv2d_new_half", generator.output_file_map)
            self.assertNotIn("drop/not_conv2d_float", generator.output_file_map)

    def test_regex_matches_full_expanded_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_root = Path(temp_dir)
            for directory, name in (
                ("keep", "kept"),
                ("prefix", "prefix_kept"),
                ("suffix", "kept_suffix"),
            ):
                self._write_shader(source_root / directory, name)
            generator = gen_vulkan_spv.SPVGenerator(
                str(source_root),
                {},
                glslc_path=None,
                shader_regexes=[re.compile("kept")],
            )
            self.assertIn("keep/kept", generator.output_file_map)
            self.assertNotIn("prefix/prefix_kept", generator.output_file_map)
            self.assertNotIn("suffix/kept_suffix", generator.output_file_map)

    def test_overlapping_regexes_compile_shader_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "src"
            self._write_shader(source_root / "keep", "kept")
            self._write_shader(source_root / "drop", "dropped")
            output_dir, cache_dir = root / "out", root / "cache"
            output_dir.mkdir()
            cache_dir.mkdir()
            compiler, compiler_log = self._write_fake_compiler(root)
            generator = gen_vulkan_spv.SPVGenerator(
                str(source_root),
                {},
                glslc_path=str(compiler),
                shader_regexes=[re.compile("kept"), re.compile("ke.*")],
            )
            spv_files = generator.generateSPV(output_dir, cache_dir, nthreads=1)
            self.assertEqual(sum(path is not None for path in spv_files), 1)
            self.assertEqual(compiler_log.read_text().splitlines(), ["compile"])


class TestShaderAllowlistManifest(unittest.TestCase):
    def test_valid_manifest_accepts_literal_and_family_patterns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "shaders.json"
            path.write_text(
                json.dumps(
                    {"shader_regexes": ["a_float", "b_(float|half)"], "models": []}
                )
            )
            patterns = gen_vulkan_spv.load_shader_regexes(str(path))
            self.assertEqual(
                [pattern.pattern for pattern in patterns], ["a_float", "b_(float|half)"]
            )
            self.assertIsNotNone(patterns[1].fullmatch("b_float"))
            self.assertIsNotNone(patterns[1].fullmatch("b_half"))
            self.assertIsNone(patterns[1].fullmatch("b_half_extra"))

    def test_invalid_manifest_is_rejected(self) -> None:
        cases = [
            [],
            {},
            {"shader_names": ["a_float"]},
            {"shader_regexes": []},
            {"shader_regexes": "a_float"},
            {"shader_regexes": [""]},
            {"shader_regexes": ["a_float", 1]},
            {"shader_regexes": ["a_float", "a_float"]},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "shaders.json"
            for manifest in cases:
                path.write_text(json.dumps(manifest))
                with self.subTest(manifest=manifest), self.assertRaises(
                    argparse.ArgumentTypeError
                ):
                    gen_vulkan_spv.load_shader_regexes(str(path))

    def test_invalid_regex_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "shaders.json"
            path.write_text(json.dumps({"shader_regexes": ["a_float", "("]}))
            with self.assertRaisesRegex(
                argparse.ArgumentTypeError, "Invalid shader regex"
            ):
                gen_vulkan_spv.load_shader_regexes(str(path))

    def test_missing_or_malformed_json_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "shaders.json"
            with self.assertRaises(argparse.ArgumentTypeError):
                gen_vulkan_spv.load_shader_regexes(str(path))
            path.write_text("{")
            with self.assertRaises(argparse.ArgumentTypeError):
                gen_vulkan_spv.load_shader_regexes(str(path))
            path.write_bytes(b"\xff")
            with self.assertRaises(argparse.ArgumentTypeError):
                gen_vulkan_spv.load_shader_regexes(str(path))


class TestShaderCapabilities(unittest.TestCase):
    def _write_spv(self, root: Path, instructions: list[int]) -> Path:
        path = root / "kernel.spv"
        words = [0x07230203, 0x00010000, 0, 1, 0, *instructions]
        path.write_bytes(struct.pack(f"<{len(words)}I", *words))
        return path

    def test_integer_dot_capabilities(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "kernel.glsl"
            source.write_text('#version 450\n#include "dot.glslh"\n')
            for capability in (6016, 6017, 6018, 6019):
                with self.subTest(capability=capability):
                    spv = self._write_spv(root, [(2 << 16) | 17, capability])
                    info = gen_vulkan_spv.getShaderInfo(str(source), str(spv))
                    self.assertTrue(info.requires_integer_dot_product_ext)
                    self.assertFalse(info.requires_shader_int16_ext)
                    self.assertFalse(info.requires_8bit_storage_ext)

    def test_shader_without_integer_dot_capability(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "kernel.glsl"
            source.write_text("#version 450\nvoid main() {}\n")
            spv = self._write_spv(root, [(2 << 16) | 17, 1, (1 << 16) | 0])
            info = gen_vulkan_spv.getShaderInfo(str(source), str(spv))
            self.assertFalse(info.requires_integer_dot_product_ext)

    def test_glsl_extensions_do_not_add_requirements(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "kernel.glsl"
            source.write_text(
                "#extension GL_EXT_integer_dot_product : require\n"
                "#extension GL_EXT_shader_16bit_storage : require\n"
                "#extension GL_EXT_shader_8bit_storage : require\n"
                "#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require\n"
                "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n"
                "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n"
            )
            spv = self._write_spv(root, [(2 << 16) | 17, 1])
            info = gen_vulkan_spv.getShaderInfo(str(source), str(spv))
            self.assertEqual([1], info.required_capabilities)
            self.assertFalse(info.requires_integer_dot_product_ext)
            self.assertFalse(info.requires_16bit_storage_ext)
            self.assertFalse(info.requires_8bit_storage_ext)
            self.assertFalse(info.requires_shader_int16_ext)
            self.assertFalse(info.requires_shader_int64_ext)
            self.assertFalse(info.requires_shader_float64_ext)

    def test_all_existing_feature_flags_come_from_spirv(self) -> None:
        cases = {
            22: "requires_shader_int16_ext",
            11: "requires_shader_int64_ext",
            10: "requires_shader_float64_ext",
            4433: "requires_16bit_storage_ext",
            4434: "requires_16bit_storage_ext",
            4435: "requires_16bit_storage_ext",
            4436: "requires_16bit_storage_ext",
            4448: "requires_8bit_storage_ext",
            4449: "requires_8bit_storage_ext",
            4450: "requires_8bit_storage_ext",
            6016: "requires_integer_dot_product_ext",
            6017: "requires_integer_dot_product_ext",
            6018: "requires_integer_dot_product_ext",
            6019: "requires_integer_dot_product_ext",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "kernel.glsl"
            source.write_text("#version 450\n")
            for capability, flag in cases.items():
                with self.subTest(capability=capability):
                    spv = self._write_spv(root, [(2 << 16) | 17, capability])
                    info = gen_vulkan_spv.getShaderInfo(str(source), str(spv))
                    self.assertEqual([capability], info.required_capabilities)
                    requirements = {
                        "requires_shader_int16_ext": info.requires_shader_int16_ext,
                        "requires_shader_int64_ext": info.requires_shader_int64_ext,
                        "requires_shader_float64_ext": info.requires_shader_float64_ext,
                        "requires_16bit_storage_ext": info.requires_16bit_storage_ext,
                        "requires_8bit_storage_ext": info.requires_8bit_storage_ext,
                        "requires_integer_dot_product_ext": info.requires_integer_dot_product_ext,
                    }
                    self.assertEqual(
                        {flag}, {key for key, value in requirements.items() if value}
                    )

    def test_all_capabilities_are_preserved_in_generated_registration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "kernel.glsl"
            source.write_text("#version 450\n")
            capabilities = [6022, 9, 39, 61, 63, 5430, 65535, 9]
            instructions = [
                word
                for capability in capabilities
                for word in ((2 << 16) | 17, capability)
            ]
            spv = self._write_spv(root, instructions)
            info = gen_vulkan_spv.getShaderInfo(str(source), str(spv))
            expected = [9, 39, 61, 63, 5430, 6022, 65535]
            self.assertEqual(expected, info.required_capabilities)
            cpp = root / "spv.cpp"
            gen_vulkan_spv.genCppFiles(
                {str(spv): str(source)}, str(root / "spv.h"), str(cpp)
            )
            self.assertIn("{9, 39, 61, 63, 5430, 6022, 65535}", cpp.read_text())

    def test_shader_info_requires_compiled_spirv(self) -> None:
        with self.assertRaises(TypeError):
            gen_vulkan_spv.getShaderInfo("unused.glsl")

    def test_generated_registration_requires_header_integer_dot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "kernel.glsl"
            source.write_text('#version 450\n#include "dot.glslh"\n')
            (root / "dot.glslh").write_text(
                "#extension GL_EXT_integer_dot_product : require\n"
            )
            spv = self._write_spv(root, [(2 << 16) | 17, 6018, (2 << 16) | 17, 6019])
            cpp = root / "spv.cpp"
            gen_vulkan_spv.genCppFiles(
                {str(spv): str(source)}, str(root / "spv.h"), str(cpp)
            )
            self.assertRegex(
                cpp.read_text(),
                r'"kernel",\s*kernel_bin,\s*36,\s*\{\},\s*\{1, 1, 1\},'
                r"\s*false,\s*false,\s*false,\s*true,",
            )

    def test_capability_operands_are_not_instructions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "kernel.glsl"
            source.write_text("#version 450\n")
            spv = self._write_spv(root, [(3 << 16) | 330, (2 << 16) | 17, 6018])
            self.assertFalse(
                gen_vulkan_spv.getShaderInfo(
                    str(source), str(spv)
                ).requires_integer_dot_product_ext
            )

    def test_rejects_malformed_spirv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "kernel.glsl"
            source.write_text("#version 450\n")
            for instructions in (
                [17],
                [(2 << 16) | 17],
                [(1 << 16) | 17],
                [(3 << 16) | 17, 6018, 6019],
                [(2 << 16) | 17, 6018, 0],
            ):
                with self.subTest(instructions=instructions):
                    spv = self._write_spv(root, instructions)
                    with self.assertRaisesRegex(ValueError, "SPIR-V"):
                        gen_vulkan_spv.getShaderInfo(str(source), str(spv))
            for data in (b"", b"\x03\x02\x23\x07", b"\x00" * 20, b"\x00" * 21):
                with self.subTest(data=data):
                    spv.write_bytes(data)
                    with self.assertRaisesRegex(ValueError, "SPIR-V"):
                        gen_vulkan_spv.getShaderInfo(str(source), str(spv))
