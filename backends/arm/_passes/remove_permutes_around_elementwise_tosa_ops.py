# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.transforms.remove_permutes_around_elementwise_ops import (
    RemovePermutesAroundElementwiseOps,
)
from executorch.exir.dialects._ops import ops as exir_ops


class RemovePermutesAroundElementwiseTosaOps(RemovePermutesAroundElementwiseOps):
    permutable_ops = {
        *RemovePermutesAroundElementwiseOps.permutable_ops,
        exir_ops.backend.tosa.RESCALE.default,
        exir_ops.backend.tosa.TABLE.default,
    }
