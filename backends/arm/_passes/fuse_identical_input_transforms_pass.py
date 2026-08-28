# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

from executorch.backends.arm._passes.arm_pass import ArmOpTargetedPass
from executorch.backends.transforms.fuse_identical_input_transforms_pass import (
    FuseIdenticalInputTransformsPass as _FuseIdenticalInputTransformsPass,
)


class FuseIdenticalInputTransformsPass(
    _FuseIdenticalInputTransformsPass, ArmOpTargetedPass
):
    """Arm configuration of the shared input-transform fusion.

    The fusion retraces the graph, and on an Arm graph that retrace has to go
    through ArmPass: a quantized bmm or leaky_relu has no fake kernel of its
    own, so ExportPass would dispatch it to the float one and report a dtype
    mismatch.

    """

    # Both bases declare this; restate it once so the two agree.
    target_ops: set[Any] = _FuseIdenticalInputTransformsPass.target_ops
