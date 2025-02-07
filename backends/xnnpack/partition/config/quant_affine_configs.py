# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch
from executorch.backends.xnnpack.partition.config.xnnpack_config import (
    ConfigPrecisionType,
    XNNPartitionerConfig,
)
from torch.export import ExportedProgram


class QDQAffineConfigs(XNNPartitionerConfig):
    def check_constraints(self, node: torch.fx.Node, ep: ExportedProgram) -> bool:
        return True

    def get_node_and_deps(
        self, node: torch.fx.Node, ep: ExportedProgram
    ) -> List[torch.fx.Node]:
        # Do not return anything from this because we only use this to
        # preserve the decomposition
        return []

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.DYNAMIC_QUANT]


class QuantizeAffineConfig(QDQAffineConfigs):
    target_name = "quantize_affine.default"

    def get_original_aten(self) -> Optional[torch._ops.OpOverload]:
        try:
            import torchao.quantization.quant_primitives  # noqa

            return torch.ops.quant.quantize_affine.default
        except:
            return None


class DeQuantizeAffineConfig(QDQAffineConfigs):
    target_name = "dequantize_affine.default"

    def get_original_aten(self) -> Optional[torch._ops.OpOverload]:
        try:
            import torchao.quantization.quant_primitives  # noqa

            return torch.ops.quant.dequantize_affine.default
        except:
            return None


class ChooseQParamsAffineConfig(QDQAffineConfigs):
    target_name = "choose_qparams_affine.default"

    def get_original_aten(self) -> Optional[torch._ops.OpOverload]:
        try:
            import torchao.quantization.quant_primitives  # noqa

            return torch.ops.quant.choose_qparams_affine.default
        except:
            return None
