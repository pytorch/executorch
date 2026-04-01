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
