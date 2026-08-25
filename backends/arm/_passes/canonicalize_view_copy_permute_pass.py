# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compatibility re-export; the pass moved to executorch.backends.transforms."""

from executorch.backends.transforms.canonicalize_view_copy_permute_pass import (  # noqa: F401
    CanonicalizeViewCopyPermutePass,
)
