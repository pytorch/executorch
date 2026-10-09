# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import stat
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
            "log.write_text(log.read_text() + 'compile\\n' if log.exists() else 'compile\\n')\n"
            "source = pathlib.Path(sys.argv[2])\n"
            + assertions
            + "output = pathlib.Path(sys.argv[sys.argv.index('-o') + 1])\n"
            "output.write_bytes(b'\\x03\\x02\\x23\\x07')\n"
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
