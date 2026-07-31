#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
from pathlib import Path

import test_base
import test_cpp_sdk
from examples.models import Backend, Model

if __name__ == "__main__":
    # coremltools does not support linux aarch64 yet and install from the source fails on runtime
    # https://github.com/apple/coremltools/issues/1254
    # https://github.com/apple/coremltools/issues/2195

    from executorch.extension.pybindings.portable_lib import (
        _get_registered_backend_names,
    )

    registered = _get_registered_backend_names()

    # OpenVINO backend uses dlopen (no build-time SDK dependency), so it
    # is compiled into the wheel on all Linux architectures.
    assert (
        "OpenvinoBackend" in registered
    ), f"OpenvinoBackend not found in registered backends: {registered}"
    print("✓ OpenvinoBackend is registered")

    # The wheel ships a prebuilt C++ runtime and a CMake package config, so check
    # that a standalone application can actually link and run against them, and
    # that the process still has a single backend registry.
    with tempfile.TemporaryDirectory() as work_dir:
        test_cpp_sdk.run_tests(Path(work_dir))

    test_base.run_tests(
        model_tests=[
            test_base.ModelTest(
                model=Model.Mv3,
                backend=Backend.XnnpackQuantizationDelegation,
            ),
        ]
    )
