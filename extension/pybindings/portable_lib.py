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
    _load_bundled_program_from_buffer,  # noqa: F401
    _load_for_executorch,  # noqa: F401
    _load_for_executorch_from_buffer,  # noqa: F401
    _load_for_executorch_from_bundled_program,  # noqa: F401
    _reset_profile_results,  # noqa: F401
    BundledModule,  # noqa: F401
    ExecuTorchModule,  # noqa: F401
    MethodMeta,  # noqa: F401
    Verification,  # noqa: F401
)

# Clean up so that `dir(portable_lib)` is the same as `dir(_portable_lib)`
# (apart from some __dunder__ names).
del _torch
del _exir_warnings
del _warnings
