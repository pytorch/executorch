#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import platform
import sys
import tempfile
from pathlib import Path

import test_base
import test_clean_install
import test_cpp_sdk
import test_shared_libraries
from examples.models import Backend, Model

if __name__ == "__main__":
    # Before anything else, because this is the check that fails the way a user
    # fails: with only the dependencies the wheel declares.
    with tempfile.TemporaryDirectory() as work_dir:
        test_clean_install.run_tests(Path(work_dir))

    if platform.system() == "Linux":
        from executorch.extension.pybindings.portable_lib import (
            _get_registered_backend_names,
        )

        registered = _get_registered_backend_names()

        # QNN backend is only available on x86_64.
        if platform.machine() in ("x86_64", "amd64"):
            assert (
                "QnnBackend" in registered
            ), f"QnnBackend not found in registered backends: {registered}"
            print("✓ QnnBackend is registered")

        # OpenVINO backend is available on all Linux architectures.
        assert (
            "OpenvinoBackend" in registered
        ), f"OpenvinoBackend not found in registered backends: {registered}"
        print("✓ OpenvinoBackend is registered")

        # Vulkan backend is optional: only present when the wheel was built with
        # EXECUTORCH_BUILD_VULKAN=1 and the Vulkan SDK (glslc) was available.
        if "VulkanBackend" in registered:
            print("✓ VulkanBackend is registered")
        else:
            print("⚠ VulkanBackend not registered (expected for the default wheel)")

        test_base.test_cmsis_nn_install()

        # Registration above proves the delegate loaded, not that it computes. This runs a
        # model through it where the OpenVINO runtime is available and says why when it is not.
        test_base.test_a_model_runs_through_the_openvino_delegate()

        # The wheel ships the runtime, the kernels, the delegate, the thread
        # pool and the profiler as separate shared libraries now, so check that
        # each has exactly one owner and that all of them are loadable.
        with tempfile.TemporaryDirectory() as work_dir:
            test_shared_libraries.run_tests(Path(work_dir))

        # And that a C++ application outside the wheel can actually use them.
        # Nothing above covers this: the Python extension links those libraries
        # itself, so it passes whether or not the package config names them or the
        # shipped headers are complete.
        with tempfile.TemporaryDirectory() as work_dir:
            test_cpp_sdk.run_tests(Path(work_dir))

    model_tests = [
        test_base.ModelTest(
            model=Model.Mv3,
            backend=Backend.XnnpackQuantizationDelegation,
        ),
    ]
    # The wheel declares coremltools only below 3.14, because that release publishes no build for
    # it, so on 3.14 this case would fail on the missing import rather than exercise Core ML.
    if sys.version_info < (3, 14):
        model_tests.append(
            test_base.ModelTest(
                model=Model.Mv3,
                backend=Backend.CoreMlExportOnly,
            )
        )
    else:
        print("Skipping the Core ML case: coremltools has no Python 3.14 build")

    test_base.run_tests(model_tests=model_tests)
