# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
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


@functools.cache
def _get_sdk_build_id(qnn_sdk_root: str):
    htp_library_path = os.path.join(
        qnn_sdk_root,
        "lib",
        _get_qnn_host_lib_dir_name(),
        get_qnn_lib_name("QnnHtp"),
    )
    return PyQnnManagerAdaptor.GetQnnSdkBuildId(htp_library_path)


class QnnSdkRootNotSet(EnvironmentError):
    """No SDK path is configured, so there is no version to read.

    Its own type because the version helpers below fall back when they see it, while any other
    failure means the SDK is there but unreadable, which they must not hide. EnvironmentError is
    an alias for OSError, so catching that would swallow both.
    """


def get_sdk_build_id():
    qnn_sdk_root = os.environ.get("QNN_SDK_ROOT")
    if not qnn_sdk_root:
        raise QnnSdkRootNotSet(
            "QNN_SDK_ROOT must be set to query the QNN SDK build id."
        )
    return _get_sdk_build_id(qnn_sdk_root)


def is_qnn_sdk_version_less_than(target_version):
    try:
        current_version = get_sdk_build_id()
    except QnnSdkRootNotSet:
        # No SDK path set, so there is no version to compare. Treated as older than any target,
        # which is what the callers want when they gate a newer feature. Any other failure means
        # the SDK is there but unreadable, and that must not be reported as an old version.
        return True

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
    try:
        current_version = get_sdk_build_id()
    except QnnSdkRootNotSet:
        # No SDK path set, so there is no version to compare. Treated as not newer than any
        # target, the conservative answer. Any other failure is left to propagate.
        return False

    match = re.search(r"v(\d+)\.(\d+)", current_version)
    if match:
        current_major, current_minor = map(int, match.groups()[:2])
    else:
        raise ValueError(
            f"Failed to get current major and minor version from QNN SDK Build id {current_version}"
        )

    target_major, target_minor = map(int, target_version.split(".")[:2])

    return current_major == target_major and current_minor > target_minor
