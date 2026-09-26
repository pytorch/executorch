# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pytest conftest for test_qnn_delegate.py.

When test_qnn_delegate.py is invoked via pytest (instead of as __main__),
setup_environment() is never called, so TestQNN class attributes remain at
their defaults (None / empty string / False). This conftest bridges the gap
by calling setup_environment() with a synthetic argv built from environment
variables, keeping a single source of truth for attribute assignment.

Required env vars for x86 simulator execution:
    QNN_SOC_MODEL       - e.g. "SM8650"
    QNN_SDK_ROOT        - path to QNN SDK (also used by verify_output)

Optional env vars (with defaults shown):
    QNN_BACKEND         - "htp"
    QNN_BUILD_FOLDER    - "build-x86"
    QNN_ENABLE_X86_64   - "1" to enable x86 simulator mode
    EXECUTORCH_ROOT     - cwd
    QNN_ARTIFACT_DIR    - "/tmp/qnn_test_artifacts"
"""

import os
import sys


def pytest_configure(config):
    if os.environ.get("QNN_DELEGATE_TEST") != "1":
        return

    argv = [
        "test_qnn_delegate.py",
        "--soc_model",
        os.environ.get("QNN_SOC_MODEL", "SM8650"),
        "--backend",
        os.environ.get("QNN_BACKEND", "htp"),
        "--build_folder",
        os.environ.get("QNN_BUILD_FOLDER", "build-x86"),
        "--executorch_root",
        os.environ.get("EXECUTORCH_ROOT", os.getcwd()),
        "--artifact_dir",
        os.environ.get("QNN_ARTIFACT_DIR", "/tmp/qnn_test_artifacts"),
    ]

    if os.environ.get("QNN_ENABLE_X86_64", "1") == "1":
        argv.append("--enable_x86_64")

    original_argv = sys.argv
    try:
        sys.argv = argv
        from executorch.backends.qualcomm.tests.test_qnn_delegate import (
            setup_environment,
        )

        setup_environment()
    finally:
        sys.argv = original_argv
