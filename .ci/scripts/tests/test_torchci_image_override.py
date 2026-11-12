# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
IMAGE = "executorch-ubuntu-22.04-clang12"
PACKAGES = {
    "pytorch": "TORCH_VERSION",
    "torchvision": "TORCHVISION_VERSION",
    "torchaudio": "TORCHAUDIO_VERSION",
}


class TorchciImageOverrideTest(unittest.TestCase):
    def run_build(self, image=IMAGE, invalid=None):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp)
            docker = fixture / ".ci/docker"
            pins = docker / "ci_commit_pins"
            pins.mkdir(parents=True)
            (pins / "pytorch.txt").write_text("shared-pin\n")
            (pins / "buck2.txt").write_text("buck-pin\n")
            for package in PACKAGES:
                source = (
                    ROOT / ".ci/docker/ci_commit_pins" / f"{package}-linux-clang12.txt"
                )
                value = invalid if invalid is not None else source.read_text()
                (pins / source.name).write_text(value)
            (fixture / "requirements-lintrunner.txt").write_text("")
            arm = fixture / "backends/arm/scripts"
            arm.mkdir(parents=True)
            for name in ("utils", "fvp_utils", "toolchain_utils"):
                (arm / f"{name}.sh").write_text("")
            shutil.copyfile(ROOT / ".ci/docker/build.sh", docker / "build.sh")
            binary = fixture / "bin"
            binary.mkdir()
            mock = binary / "docker"
            mock.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            mock.chmod(0o755)
            return subprocess.run(
                ["bash", "build.sh", image],
                cwd=docker,
                env={
                    **os.environ,
                    "PATH": f"{binary}:{os.environ['PATH']}",
                    "REMOTE_BUILDKIT": "",
                },
                text=True,
                capture_output=True,
            )

    def test_clang_image_uses_all_published_pins(self):
        result = self.run_build()
        self.assertEqual(result.returncode, 0, result.stderr)
        for package, variable in PACKAGES.items():
            pin = (
                (ROOT / ".ci/docker/ci_commit_pins" / f"{package}-linux-clang12.txt")
                .read_text()
                .strip()
            )
            self.assertRegex(pin, r"^[0-9a-f]{40}$")
            self.assertIn(f"{variable}={pin}", result.stdout.splitlines())
        self.assertIn("TORCHCI_REPORTS=1", result.stdout.splitlines())

    def test_other_images_keep_the_shared_pin(self):
        result = self.run_build("executorch-ubuntu-22.04-gcc11", invalid="invalid")
        self.assertEqual(result.returncode, 0, result.stderr)
        for value in (
            "TORCH_VERSION=shared-pin",
            "TORCHVISION_VERSION=",
            "TORCHAUDIO_VERSION=",
            "TORCHCI_REPORTS=",
        ):
            self.assertIn(value, result.stdout.splitlines())

    def test_missing_or_mutable_pin_stops_before_docker(self):
        for pin in ("", "main", "a" * 39, "A" * 40):
            with self.subTest(pin=pin):
                result = self.run_build(invalid=pin)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("--no-cache", result.stdout.splitlines())

    def test_docker_forwards_the_domain_pins_and_checks_installed_torch(self):
        dockerfile = (ROOT / ".ci/docker/ubuntu/Dockerfile").read_text()
        script = (ROOT / ".ci/docker/common/install_pytorch.sh").read_text()
        for variable in (*PACKAGES.values(), "TORCHCI_REPORTS"):
            self.assertIn(f"ARG {variable}\n", dockerfile)
        self.assertIn("TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION:-release/2.11}", script)
        self.assertIn(
            "TORCHVISION_VERSION=${TORCHVISION_VERSION:-release/0.29}", script
        )
        self.assertLess(
            script.index("  install_domains\n"),
            script.index("if torch.version.git_version"),
        )
        self.assertIn('report.SCHEMA_VERSION != "1.0"', script)
        self.assertIn("environment.capture()", script)
