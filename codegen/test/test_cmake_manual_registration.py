# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "codegen/test/cmake_manual_registration"


class TestCMakeManualRegistration(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def run_command(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(args, text=True, capture_output=True, check=False)

    def check_command(self, *args: str) -> None:
        result = self.run_command(*args)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def configure(
        self, build: Path, **options: str
    ) -> subprocess.CompletedProcess[str]:
        options.update(
            EXECUTORCH_ROOT=str(ROOT),
            PYTHON_EXECUTABLE=sys.executable,
        )
        if os.environ.get("EXECUTORCH_TEST_CODEGEN_CMAKE"):
            options["CODEGEN_CMAKE"] = os.environ["EXECUTORCH_TEST_CODEGEN_CMAKE"]
        return self.run_command(
            "cmake",
            "-S",
            str(FIXTURE),
            "-B",
            str(build),
            *(f"-D{key}={value}" for key, value in options.items()),
        )

    def test_invalid_names(self) -> None:
        for function in ("generate_bindings_for_kernels", "gen_operators_lib"):
            for name in (None, ""):
                with self.subTest(function=function, name=name):
                    result = self.configure(
                        self.directory
                        / f"case_{function}_{len(list(self.directory.iterdir()))}",
                        TEST_CASE="invalid",
                        TEST_FUNCTION=function,
                        OMIT_LIB_NAME="ON" if name is None else "OFF",
                        TEST_LIB_NAME=name or "",
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("Manual registration LIB_NAME", result.stderr)

        for name in ("all", "foo-bar", "foo.bar", "1foo", "foo/bar"):
            with self.subTest(registration_name=name):
                result = self.configure(
                    self.directory
                    / f"registration_name_{len(list(self.directory.iterdir()))}",
                    TEST_CASE="invalid",
                    TEST_FUNCTION="generate_bindings_for_kernels",
                    TEST_LIB_NAME="valid_target",
                    TEST_REGISTRATION_NAME=name,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Manual registration REGISTRATION_NAME", result.stderr)

    def test_cpp_name_does_not_depend_on_target_name(self) -> None:
        result = self.configure(
            self.directory / "renamed-target",
            BINDINGS_MANUAL="ON",
            OPERATORS_MANUAL="ON",
            TEST_LIB_NAME="renamed-target",
            TEST_REGISTRATION_NAME="stable_api",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_cross_directory_pairing(self) -> None:
        include_root = self.directory / "cross_directory_include"
        shutil.copytree(ROOT / "runtime", include_root / "executorch/runtime")
        build = self.directory / "cross_directory_build"
        result = self.configure(
            build,
            TEST_CASE="cross_directory",
            SOURCE_INCLUDE_ROOT=str(include_root),
            TORCH_INCLUDE_ROOT=str(Path(torch.__file__).parent / "include"),
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.check_command("cmake", "--build", str(build), "--parallel", "4")
        self.check_command(
            str(build / "cross_directory/consumer/cross_directory_consumer")
        )
        mismatch = self.configure(
            self.directory / "cross_directory_mismatch",
            TEST_CASE="cross_directory",
            CROSS_OPERATOR_MANUAL="OFF",
        )
        self.assertNotEqual(mismatch.returncode, 0)
        self.assertIn("matching MANUAL_REGISTRATION", mismatch.stderr)

    def test_registration_modes(self) -> None:
        for bindings in ("ON", "OFF"):
            for operators in ("ON", "OFF"):
                with self.subTest(bindings=bindings, operators=operators):
                    result = self.configure(
                        self.directory / f"mode_{bindings}_{operators}",
                        BINDINGS_MANUAL=bindings,
                        OPERATORS_MANUAL=operators,
                    )
                    if bindings == operators:
                        self.assertEqual(
                            result.returncode, 0, result.stdout + result.stderr
                        )
                    else:
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn("matching MANUAL_REGISTRATION", result.stderr)

    def test_shared_option(self) -> None:
        for manual in ("ON", "OFF"):
            with self.subTest(manual=manual):
                result = self.configure(
                    self.directory / manual,
                    BINDINGS_MANUAL=manual,
                    OPERATORS_MANUAL=manual,
                    SHARED_LIBRARY="ON",
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_empty_shared_library_consumer(self) -> None:
        build = self.directory / "empty_shared_build"
        install = self.directory / "empty_shared_install"
        include_root = self.directory / "source_include"
        shutil.copytree(ROOT / "runtime", include_root / "executorch/runtime")
        result = self.configure(
            build,
            TEST_CASE="empty_shared",
            SOURCE_INCLUDE_ROOT=str(include_root),
            TORCH_INCLUDE_ROOT=str(Path(torch.__file__).parent / "include"),
            CMAKE_INSTALL_PREFIX=str(install),
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.check_command("cmake", "--build", str(build), "--parallel", "4")
        self.check_command(str(build / "empty_shared_consumer"))
        self.check_command("cmake", "--install", str(build))
        self.assertTrue(
            (install / "include/executorch/empty_ops/RegisterKernels.h").is_file()
        )
        relocated = self.directory / "empty_shared_relocated"
        install.rename(relocated)
        build.rename(self.directory / "empty_shared_hidden_build")
        consumer = self.directory / "empty_installed_consumer"
        result = self.configure(
            consumer,
            TEST_CASE="empty_installed_consumer",
            SOURCE_INCLUDE_ROOT=str(include_root),
            TORCH_INCLUDE_ROOT=str(Path(torch.__file__).parent / "include"),
            MANUAL_TARGETS=str(
                relocated / "lib/cmake/EmptyRegistration/EmptyRegistrationTargets.cmake"
            ),
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.check_command("cmake", "--build", str(consumer), "--parallel", "4")
        self.check_command(str(consumer / "empty_shared_consumer"))

    @unittest.skipUnless(
        os.environ.get("EXECUTORCH_TEST_INSTALL_PREFIX"),
        "requires an installed ExecuTorch CMake package",
    )
    def test_build_install_and_collision(self) -> None:
        runtime_prefix = os.environ["EXECUTORCH_TEST_INSTALL_PREFIX"]
        build = self.directory / "build"
        install = self.directory / "install"
        result = self.configure(
            build,
            TEST_CASE="integration",
            CMAKE_PREFIX_PATH=runtime_prefix,
            CMAKE_INSTALL_PREFIX=str(install),
            CMAKE_INSTALL_LIBDIR="lib",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.check_command("cmake", "--build", str(build), "--parallel", "4")
        executable = build / "named_manual_registration_test"
        self.check_command(str(executable))
        self.check_command(str(build / "automatic_consumer"))
        self.check_command("cmake", "--install", str(build))
        for name in ("Functions.h", "NativeFunctions.h"):
            self.assertTrue((install / "include/legacy" / name).is_file())
        for lib in ("manual_ops_1_lib", "manual_ops_2_lib"):
            header = install / "include/executorch" / lib / "RegisterKernels.h"
            self.assertIn(f"register_{lib}_kernels", header.read_text())
        self.assertFalse(
            (install / "include/executorch/manual_ops_overlap_lib").exists()
        )
        # Relocation and removing the producer build tree catch leaked paths.
        relocated = self.directory / "relocated"
        install.rename(relocated)
        build.rename(self.directory / "hidden_build")
        consumer = self.directory / "consumer"
        result = self.configure(
            consumer,
            TEST_CASE="consumer",
            CMAKE_PREFIX_PATH=runtime_prefix,
            MANUAL_TARGETS=str(
                relocated
                / "lib/cmake/ManualRegistration/ManualRegistrationTargets.cmake"
            ),
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.check_command("cmake", "--build", str(consumer), "--parallel", "4")
        self.check_command(str(consumer / "consumer"))
        overlap = self.run_command(
            str(self.directory / "hidden_build/named_manual_registration_test"),
            "--overlap",
        )
        self.assertEqual(
            overlap.returncode, -signal.SIGABRT, overlap.stdout + overlap.stderr
        )
        self.assertIn(
            "Re-registering my_ops::mul3.out", overlap.stdout + overlap.stderr
        )


if __name__ == "__main__":
    unittest.main()
