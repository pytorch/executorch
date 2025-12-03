# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .activation_fusion_pass import ActivationFusionPass  # noqa
from .clamp_hardswish_pass import ClampHardswishPass  # noqa
from .convert_to_cortex_m_pass import ConvertToCortexMPass  # noqa
from .decompose_hardswish_pass import DecomposeHardswishPass  # noqa
from .quantized_op_fusion_pass import QuantizedOpFusionPass  # noqa
from .replace_quant_nodes_pass import ReplaceQuantNodesPass  # noqa
from .cortex_m_pass_manager import CortexMPassManager  # noqa  # usort: skip
