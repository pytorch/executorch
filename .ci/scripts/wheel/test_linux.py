#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import platform

import test_base
from examples.models import Backend, Model

if __name__ == "__main__":
    # On Linux x86_64 the wheel is built with the Qualcomm backend.
    # Verify that it was registered correctly.
    if platform.system() == "Linux" and platform.machine() in ("x86_64", "amd64"):
        from executorch.extension.pybindings.portable_lib import (
            _get_registered_backend_names,
        )

        registered = _get_registered_backend_names()
        assert (
            "QnnBackend" in registered
        ), f"QnnBackend not found in registered backends: {registered}"
        print("✓ QnnBackend is registered")

    test_base.run_tests(
        model_tests=[
            test_base.ModelTest(
                model=Model.Mv3,
                backend=Backend.XnnpackQuantizationDelegation,
            ),
            test_base.ModelTest(
                model=Model.Mv3,
                backend=Backend.CoreMlExportOnly,
            ),
        ]
    )
