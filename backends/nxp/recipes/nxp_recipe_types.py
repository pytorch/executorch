# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.export import RecipeType


NXP_BACKEND: str = "nxp"


class NXPRecipeType(RecipeType):
    """NXP-specific recipe types for Neutron NPU export.

    Choose the recipe that matches your intended export configuration:
      - INT8_PTQ_NEUTRON: standard post-training quantization, delegates to Neutron NPU.
      - INT8_QAT_NEUTRON: quantization-aware training flow, delegates to Neutron NPU.
      - INT8_PTQ_NO_DELEGATE: PTQ without NPU delegation (useful for debugging or CPU-only deployment).
    """

    # INT8 static PTQ (weights + activations). Calibration dataset required.
    # Applicable operators are delegated to the Neutron NPU.
    INT8_PTQ_NEUTRON = "nxp_int8_ptq_neutron"

    # INT8 QAT flow. Requires train_fn in NeutronRecipeConfig.
    # Applicable operators are delegated to the Neutron NPU.
    INT8_QAT_NEUTRON = "nxp_int8_qat_neutron"

    # INT8 PTQ without NPU delegation. Produces a quantized graph that runs on CPU.
    # Useful for accuracy evaluation or debugging before enabling delegation.
    INT8_PTQ_NO_DELEGATE = "nxp_int8_ptq_no_delegate"

    @classmethod
    def get_backend_name(cls) -> str:
        return NXP_BACKEND
