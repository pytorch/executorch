# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.transforms.canonicalize_view_copy_permute_pass import (
    CanonicalizeViewCopyPermutePass as _CanonicalizeViewCopyPermutePass,
)


class CanonicalizeViewCopyPermutePass(_CanonicalizeViewCopyPermutePass, ArmPass):
    """Arm configuration of the shared view/permute canonicalizer.

    Canonicalization retraces the graph, and on an Arm graph that retrace has to
    go through ArmPass: a quantized bmm or leaky_relu has no fake kernel of its
    own, so ExportPass would dispatch it to the float one and report a dtype
    mismatch.

    """
