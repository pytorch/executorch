# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1]


class MuseGlimmerBatchingWorkloadTest(unittest.TestCase):
    @unittest.skipIf(
        sys.platform == "win32", "Builds with a GCC or Clang style c++ driver"
    )
    def test_workload_without_a_model(self):
        compiler = shutil.which("c++")
        self.assertIsNotNone(compiler, "CI workload tests require a C++ compiler")
        root = SCRIPTS.parents[1]
        project = root if (root / "examples").is_dir() else root.parent
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            binary = temporary / "workload_test"
            # Only the unused multimodal/tensor boundary is stubbed. Compile the
            # real serving declarations, executor and private workload helper.
            multimodal = (
                temporary / "executorch/extension/llm/runner/multimodal_input.h"
            )
            multimodal.parent.mkdir(parents=True)
            multimodal.write_text(
                "#pragma once\n#include <cstdint>\n#include <vector>\n"
                "namespace executorch::extension::llm {\n"
                "struct MultimodalInput {\n"
                "  explicit MultimodalInput(std::vector<std::uint64_t>) {}\n"
                "};\n}\n"
            )
            result = subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-Wno-deprecated-declarations",
                    "-pthread",
                    "-I",
                    str(temporary),
                    "-I",
                    str(project.parent),
                    str(SCRIPTS / "tests/muse_glimmer_batching_workload_test.cpp"),
                    "-o",
                    str(binary),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("tests passed", result.stdout)

    def test_build_enables_smoke_and_propagates_failures(self):
        for failure in ("none", "configure", "build"):
            with self.subTest(
                failure=failure
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                log = root / "commands"
                cmake = root / "cmake"
                cmake.write_text(
                    '#!/bin/bash\nprintf "%s\\n" "$*" >> "$COMMAND_LOG"\n'
                    'if [[ "$1" == "--build" ]]; then\n'
                    '  [[ "$FAIL_AT" != "build" ]]\n'
                    'else\n  [[ "$FAIL_AT" != "configure" ]]\nfi\n'
                )
                cmake.chmod(0o755)
                build_dir = root / "build with spaces"
                result = subprocess.run(
                    [
                        "bash",
                        str(SCRIPTS / "build_muse_glimmer_batching_smoke.sh"),
                        str(build_dir),
                    ],
                    env={
                        **os.environ,
                        "PATH": f"{root}:{os.environ['PATH']}",
                        "COMMAND_LOG": str(log),
                        "FAIL_AT": failure,
                    },
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    result.returncode == 0, failure == "none", result.stderr
                )
                commands = log.read_text().splitlines()
                self.assertIn("-DMUSE_GLIMMER_BUILD_CI_SMOKE=ON", commands[0])
                self.assertIn("-DMUSE_GLIMMER_BUILD_BATCHING=ON", commands[0])
                self.assertIn(str(build_dir), commands[0])
                self.assertNotIn("BENCHMARK", "\n".join(commands))
                self.assertEqual(len(commands), 1 if failure == "configure" else 2)
                if len(commands) == 2:
                    self.assertIn("--target muse_glimmer_batching_smoke", commands[1])
