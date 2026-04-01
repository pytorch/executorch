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
