# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""API for loading and executing ExecuTorch PTE files using the C++ runtime.

.. warning::

    This API is experimental and subject to change without notice.
"""

import logging
import os
import sys
import warnings as _warnings

import executorch.exir._warnings as _exir_warnings

_warnings.warn(
    "This API is experimental and subject to change without notice.",
    _exir_warnings.ExperimentalWarning,
)

# When installed as a pip wheel, we must import `torch` before trying to import
# the pybindings shared library extension. This will load libtorch.so and
# related libs, ensuring that the pybindings lib can resolve those runtime
# dependencies.
import torch as _torch

logger = logging.getLogger(__name__)

# Auto-discover the OpenVINO C library path from the pip-installed openvino
# package so the C++ backend's dlopen("libopenvino_c.so") works without the
# user having to set LD_LIBRARY_PATH or OPENVINO_LIB_PATH manually.
if not os.environ.get("OPENVINO_LIB_PATH"):
    try:
        import glob
        import importlib.util

        spec = importlib.util.find_spec("openvino")
        if spec is not None and spec.submodule_search_locations:
            _ov_dir = spec.submodule_search_locations[0]
            _ov_libs = sorted(
                glob.glob(os.path.join(_ov_dir, "libs", "libopenvino_c.so*"))
            )
            if _ov_libs:
                os.environ["OPENVINO_LIB_PATH"] = _ov_libs[0]
            else:
                logger.warning(
                    "OpenVINO package found but libopenvino_c.so not in %s; "
                    "set OPENVINO_LIB_PATH manually if needed",
                    os.path.join(_ov_dir, "libs"),
                )
            del _ov_libs, _ov_dir, spec
    except Exception as e:
        logger.debug("OpenVINO auto-discovery failed: %s", e)

# Update the DLL search path on Windows. This is the recommended way to handle native
# extensions.
if sys.platform == "win32":
    try:
        # The extension DLL should be in the same directory as this file.
        pybindings_dir = os.path.dirname(os.path.abspath(__file__))
        os.add_dll_directory(pybindings_dir)
    except Exception as e:
        logger.error(
            "Failed to add the pybinding extension DLL to the search path. The extension may not work.",
            e,
        )

# Let users import everything from the C++ _portable_lib extension as if this
# python file defined them. Although we could import these dynamically, it
# wouldn't preserve the static type annotations.
#
# Note that all of these are experimental, and subject to change without notice.
from executorch.extension.pybindings._portable_lib import (  # noqa: F401
    # Disable "imported but unused" (F401) checks.
    _create_profile_block,  # noqa: F401
    _dump_profile_results,  # noqa: F401
    _get_operator_names,  # noqa: F401
    _get_registered_backend_names,  # noqa: F401
    _is_available,  # noqa: F401
    _load_bundled_program_from_buffer,  # noqa: F401
    _load_flat_tensor_data_map,  # noqa: F401
    _load_flat_tensor_data_map_from_buffer,  # noqa: F401
    _load_for_executorch,  # noqa: F401
    _load_for_executorch_from_buffer,  # noqa: F401
    _load_for_executorch_from_bundled_program,  # noqa: F401
    _load_program,  # noqa: F401
    _load_program_from_buffer,  # noqa: F401
    _reset_profile_results,  # noqa: F401
    _threadpool_get_thread_count,  # noqa: F401
    _unsafe_reset_threadpool,  # noqa: F401
    BundledModule,  # noqa: F401
    ExecuTorchMethod,  # noqa: F401
    ExecuTorchModule,  # noqa: F401
    ExecuTorchProgram,  # noqa: F401
    FlatTensorDataMap,  # noqa: F401
    MethodMeta,  # noqa: F401
    Verification,  # noqa: F401
)

# Clean up so that `dir(portable_lib)` is the same as `dir(_portable_lib)`
# (apart from some __dunder__ names).
del _torch
del _exir_warnings
del _warnings
