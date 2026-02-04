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

# Set dlopen flags to RTLD_GLOBAL to ensure that the symbols in _portable_lib can
# be found by another shared library (for example, in AOTI where we want to load
# an AOTI compiled .so file with needed symbols defined in _portable_lib).
prev = sys.getdlopenflags()
sys.setdlopenflags(prev | os.RTLD_GLOBAL)
from executorch.extension.pybindings._portable_lib import (  # noqa: F401
    # Disable "imported but unused" (F401) checks.
    _create_profile_block,  # noqa: F401
    _dump_profile_results,  # noqa: F401
    _get_operator_names,  # noqa: F401
    _get_registered_backend_names,  # noqa: F401
    _is_available,  # noqa: F401
    _load_bundled_program_from_buffer,  # noqa: F401
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
    MethodMeta,  # noqa: F401
    Verification,  # noqa: F401
)

sys.setdlopenflags(prev)

# Clean up so that `dir(portable_lib)` is the same as `dir(_portable_lib)`
# (apart from some __dunder__ names).
del _torch
del _exir_warnings
del _warnings
