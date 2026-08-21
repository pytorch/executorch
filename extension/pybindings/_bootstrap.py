# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Prepares the process so the pybindings extension module can be imported.

Two of these have to happen before the extension is imported: torch is made resident so the
extension resolves libtorch symbols at load time, and on Windows the extension's directory is
added to the DLL search path. The third, pointing the OpenVINO backend at its C library, is
only read when that backend is first used, but it is set here so a single import prepares
everything.

Private. Import the extension through `executorch.runtime` instead.
"""

import glob
import importlib.util
import logging
import os
import sys

# Importing torch first loads libtorch, so the extension resolves against the same copy the
# rest of the process uses. The wheel also records an rpath to torch/lib (see
# _python_extension_rpath in CMakeLists.txt), so this is not the only way libtorch is found,
# but it keeps the resident copy authoritative. Kept out of the sorted block because the
# order is load-bearing.
import torch as _torch  # noqa: F401  # usort: skip

logger = logging.getLogger(__name__)


def _discover_openvino_library() -> None:
    """Point the OpenVINO backend at the C library inside the pip-installed package.

    The backend calls dlopen("libopenvino_c.so"), which only resolves if the library is on
    the loader path. Finding it here means a user who pip-installed openvino does not have
    to set LD_LIBRARY_PATH or OPENVINO_LIB_PATH by hand.
    """
    if os.environ.get("OPENVINO_LIB_PATH"):
        return

    try:
        spec = importlib.util.find_spec("openvino")
        if spec is None or not spec.submodule_search_locations:
            return

        directory = spec.submodule_search_locations[0]
        libraries = sorted(
            glob.glob(os.path.join(directory, "libs", "libopenvino_c.so*"))
        )
        if libraries:
            os.environ["OPENVINO_LIB_PATH"] = libraries[0]
        else:
            logger.warning(
                "OpenVINO package found but libopenvino_c.so not in %s; "
                "set OPENVINO_LIB_PATH manually if needed",
                os.path.join(directory, "libs"),
            )
    except Exception as e:
        logger.debug("OpenVINO auto-discovery failed: %s", e)


def _add_extension_directory_to_dll_path() -> None:
    """Let Windows find the extension's own DLLs, which sit next to this file."""
    if sys.platform != "win32":
        return

    try:
        os.add_dll_directory(os.path.dirname(os.path.abspath(__file__)))
    except Exception as e:
        logger.error(
            "Failed to add the pybinding extension DLL to the search path. "
            "The extension may not work: %s",
            e,
        )


_discover_openvino_library()
_add_extension_directory_to_dll_path()
