# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.nxp.aten_passes.simulated_linear_bn_fusion_passes.add_simulated_linear_bn_fusion_qat_pass import (
    AddSimulatedLinearBatchNormFusionQATPass,
)
from executorch.backends.nxp.aten_passes.simulated_linear_bn_fusion_passes.remove_simulated_linear_bn_fusion_qat_pass import (
    RemoveSimulatedLinearBatchNormFusionQATPass,
)

__all__ = [
    "AddSimulatedLinearBatchNormFusionQATPass",
    "RemoveSimulatedLinearBatchNormFusionQATPass",
]
