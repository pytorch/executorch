# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Compatibility re-export.

``dim_maps`` moved to ``executorch.backends.transforms`` so backends other than
Arm can use the view/permute algebra without depending on the Arm backend. The
module has no Arm or TOSA imports. Prefer the new location; this shim keeps
in-flight Arm work building.

"""

from executorch.backends.transforms.dim_maps import *  # noqa: F401,F403
from executorch.backends.transforms.dim_maps import (  # noqa: F401
    _dim_equals,
    _is_permutation,
    _normalize_dim,
    _normalize_dims,
    normalize_view_shape,
    PermuteMap,
    same_numel,
    ViewMap,
)
