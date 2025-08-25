# Copyright 2024-2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from executorch.backends.nxp.backend.ir.converter.node_converter import (
    CustomDelegationOptions,
    NodeConverter,
)
from executorch.backends.nxp.backend.ir.converter.quantization_utils import (
    set_quantization_parameters_to_tensor,
)
from torch.fx import Node
from torch.nn import Parameter


class QDQQuantizeConverter(NodeConverter):

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        if "cluster" not in node.meta or node.args[5] != torch.int8:
            return False

        return True

    def convert(self, node: Node):
        self.assert_convertible(node)

        from_tensor = self.builder.tensor_for_name(node.name)
        to_tensor = self.builder.tensor_for_name(node.args[0].name)

        scale = np.array(node.args[1], dtype=np.float32)
        zero_point = np.array(node.args[2], dtype=np.int8)

        set_quantization_parameters_to_tensor(to_tensor, scale, zero_point, 0)

        # Change type so we pass check tensor similarity check when redirecting
        to_tensor.type = from_tensor.type
        self.builder.redirect_tensor(from_tensor, to_tensor)
