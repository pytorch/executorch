#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import test_base
from examples.models import Backend, Model

if __name__ == "__main__":
    test_base.run_tests(
        model_tests=[
            test_base.ModelTest(
                model=Model.Mv3,
                backend=Backend.XnnpackQuantizationDelegation,
            ),
            test_base.ModelTest(
                model=Model.Mv3,
                backend=Backend.CoreMlExportAndTest,
            ),
        ]
    )
