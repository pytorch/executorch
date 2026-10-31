# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import stat
import struct
import tempfile
import unittest
from pathlib import Path

from runtime import gen_vulkan_spv


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
