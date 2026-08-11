#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import site
import subprocess
import sys
import tempfile
from pathlib import Path

import test_base
from examples.models import Backend, Model


def test_native_library_rpaths() -> None:
    package_root = next(
        Path(root) / "executorch"
        for root in site.getsitepackages()
        if (Path(root) / "executorch").is_dir()
    )
    absolute_rpaths = []
    for binary in [*package_root.rglob("*.so"), *package_root.rglob("*.dylib")]:
        output = subprocess.run(
            ["otool", "-l", binary],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        rpaths = re.findall(
            r"cmd LC_RPATH\s+cmdsize \d+\s+path (.+?) \(offset",
            output,
        )
        absolute_rpaths.extend(
            f"{binary}: {rpath}"
            for rpath in rpaths
            if not rpath.startswith("@loader_path")
        )

    assert not absolute_rpaths, "Absolute RPATHs found:\n" + "\n".join(absolute_rpaths)

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    with tempfile.TemporaryDirectory() as temp_dir:
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; assert 'torch' not in sys.modules; "
                "from executorch.extension.pybindings._portable_lib import "
                "_load_for_executorch",
            ],
            check=True,
            cwd=temp_dir,
            env=env,
        )


if __name__ == "__main__":
    test_native_library_rpaths()
    test_base.test_cmsis_nn_install()

    model_tests = [
        test_base.ModelTest(
            model=Model.Mv3,
            backend=Backend.XnnpackQuantizationDelegation,
        ),
    ]
    if sys.version_info < (3, 14):
        model_tests.append(
            test_base.ModelTest(
                model=Model.Mv3,
                backend=Backend.CoreMlExportAndTest,
            )
        )
    else:
        print("Skipping Core ML test: coremltools 9.0 does not support Python 3.14")

    test_base.run_tests(model_tests=model_tests)
