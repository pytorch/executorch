# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.arm.ao_ext.mxfp import MXFPOpConfig
from executorch.backends.arm.ao_ext.ops.mxfp_linear_op import transform_linear_to_mxfp
from torchao.quantization.transform_module import register_quantize_module_handler


@register_quantize_module_handler(MXFPOpConfig)  # type: ignore[misc]
def _transform_to_mxfp(
    module: torch.nn.Module,
    config: MXFPOpConfig,
) -> torch.nn.Module:
    """Transforms a given module to use MXFP operations based on the provided
    MXFPOpConfig configuration.
    """
    if isinstance(module, torch.nn.Linear):
        return transform_linear_to_mxfp(module, config)
    else:
        return module
