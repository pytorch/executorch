# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import platform
import re

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManagerAdaptor


def get_qnn_lib_name(base: str) -> str:
    """Returns the platform-specific shared library filename for a QNN library."""
    if platform.system().lower() == "windows":
        return f"{base}.dll"
    return f"lib{base}.so"


def _get_qnn_host_lib_dir_name() -> str:
    """Returns the QNN SDK library subdirectory name for the current x86-64 host OS."""
    if platform.system().lower() == "windows":
        return "x86_64-windows-msvc"
    return "x86_64-linux-clang"


def get_sdk_build_id():
    htp_library_path = os.path.join(
        os.environ.get("QNN_SDK_ROOT", None),
        "lib",
        _get_qnn_host_lib_dir_name(),
        get_qnn_lib_name("QnnHtp"),
    )
    # The GetQnnSdkBuildId API can be used without needing to create a backend first, so it works regardless of which backend is used.
    sdk_build_id = PyQnnManagerAdaptor.GetQnnSdkBuildId(htp_library_path)
    return sdk_build_id


def is_qnn_sdk_version_less_than(target_version):
    current_version = get_sdk_build_id()

    match = re.search(r"v(\d+)\.(\d+)", current_version)
    if match:
        current_major, current_minor = map(int, match.groups()[:2])
    else:
        raise ValueError(
            f"Failed to get current major and minor version from QNN SDK Build id {current_version}"
        )

    target_major, target_minor = map(int, target_version.split(".")[:2])

    return current_major == target_major and current_minor < target_minor


def is_qnn_sdk_version_greater_than(target_version):
    current_version = get_sdk_build_id()

    match = re.search(r"v(\d+)\.(\d+)", current_version)
    if match:
        current_major, current_minor = map(int, match.groups()[:2])
    else:
        raise ValueError(
            f"Failed to get current major and minor version from QNN SDK Build id {current_version}"
        )

    target_major, target_minor = map(int, target_version.split(".")[:2])

    return current_major == target_major and current_minor > target_minor
