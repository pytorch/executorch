#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys

import test_base
from examples.models import Backend, Model

if __name__ == "__main__":
    test_base.test_cmsis_nn_install()

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
