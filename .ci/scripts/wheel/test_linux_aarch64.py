#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import test_base
from examples.models import Backend, Model

if __name__ == "__main__":
    # coremltools does not support linux aarch64 yet and install from the source fails on runtime
    # https://github.com/apple/coremltools/issues/1254
    # https://github.com/apple/coremltools/issues/2195
    test_base.run_tests(
        model_tests=[
            test_base.ModelTest(
                model=Model.Mv3,
                backend=Backend.XnnpackQuantizationDelegation,
            ),
        ]
    )
