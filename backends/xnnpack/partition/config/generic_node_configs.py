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
from executorch.backends.xnnpack.utils.quant_utils import is_dequant, is_quant
from executorch.exir.backend.canonical_partitioners.config_partitioner import (
    format_target_name,
)
from torch.export import ExportedProgram


class GenericNodePartitionerConfig(XNNPartitionerConfig):
    def __init__(self, fused_act: Optional[List[str]] = None):
        """
        fused_act is a list of node target names that can be fused with this
        node under quantization
        """
        self.fused_acts = fused_act or []
        super().__init__()

    def check_constraints(self, node: torch.fx.Node, ep: ExportedProgram) -> bool:
        return self.check_common_constraints(node, ep)

    def get_node_and_deps(
        self, node: torch.fx.Node, ep: ExportedProgram
    ) -> List[torch.fx.Node]:
        deps = [node]
        quantized_deps = []
        if ConfigPrecisionType.STATIC_QUANT in self.enabled_precision_types:
            # try to partition dequant inputs and quant outputs if static quant is enabled
            if [(is_dequant(dq_input)) for dq_input in node.all_input_nodes].count(
                False
            ):
                # if not all inputs are dequant nodes then it isn't quantized
                return deps

            quantized_deps.extend(node.all_input_nodes)

            # check if quantized pattern has fused activation
            if len(node.users) != 1:
                return deps

            node_output = list(node.users)[0]
            if (
                node_output.op == "call_function"
                and format_target_name(node_output.target.__name__) in self.fused_acts
            ):
                quantized_deps.append(node_output)
                fused_out_users = list(node_output.users.keys())
                if len(fused_out_users) == 1:
                    node_output = fused_out_users[0]

            if not is_quant(node_output):
                # Expected node --> fused_act (optional) --> dequant
                return deps

            quantized_deps.append(node_output)

        return deps + quantized_deps


class QuantizedPerTensorConfig(GenericNodePartitionerConfig):
    target_name = "quantize_per_tensor.default"

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.STATIC_QUANT]


class DeQuantizedPerTensorConfig(GenericNodePartitionerConfig):
    target_name = "dequantize_per_tensor.default"

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.STATIC_QUANT]


class HardtanhConfig(GenericNodePartitionerConfig):
    target_name = "hardtanh.default"

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32, ConfigPrecisionType.STATIC_QUANT]


class AddConfig(GenericNodePartitionerConfig):
    target_name = "add.Tensor"

    def __init__(self):
        super().__init__(fused_act=["relu.default"])

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32, ConfigPrecisionType.STATIC_QUANT]


class ReLUConfig(GenericNodePartitionerConfig):
    target_name = "relu.default"

    def supported_precision_types(self) -> List[ConfigPrecisionType]:
        return [ConfigPrecisionType.FP32, ConfigPrecisionType.STATIC_QUANT]
