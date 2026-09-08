#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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

    test_base.test_cmsis_nn_install()

    # The wheel ships the runtime, the kernels, the delegate, the thread pool and
    # the profiler as separate libraries here too, so check that each has exactly
    # one owner and that all of them are loadable.
    with tempfile.TemporaryDirectory() as work_dir:
        test_shared_libraries.run_tests(Path(work_dir))

    # And that a C++ application outside the wheel can actually use them. Nothing
    # else covers this: the Python extension links those libraries itself, so it
    # passes whether or not the package config names them or the shipped headers
    # are complete.
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
                backend=Backend.CoreMlExportAndTest,
            )
        )
    else:
        print("Skipping the Core ML case: coremltools has no Python 3.14 build")

    test_base.run_tests(model_tests=model_tests)
