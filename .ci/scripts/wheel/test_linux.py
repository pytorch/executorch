#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from test_base import run_tests, ModelTest, Model, Backend

if __name__ == "__main__":
    run_tests(model_tests=[
        ModelTest(
            model=Model.Mv3,
            backend=Backend.XnnpackQuantizationDelegation,
        )
    ])
