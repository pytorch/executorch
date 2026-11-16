# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bridge Nova's legacy bdist_wheel command to the standalone PEP 517 build."""

import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel
from wheel_version import dynamic_metadata


class NativeWheel(bdist_wheel):
    def run(self) -> None:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--outdir",
                str(Path(self.dist_dir).resolve()),
                str(Path(__file__).resolve().parent),
            ],
            check=True,
        )


setup(
    version=dynamic_metadata("version", {}),
    packages=[],
    cmdclass={"bdist_wheel": NativeWheel},
)
