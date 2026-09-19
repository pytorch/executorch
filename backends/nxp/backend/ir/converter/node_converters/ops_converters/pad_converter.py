# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Collection

import numpy as np
import torch
from executorch.backends.nxp.backend.custom_delegation_options import (
    CustomDelegationOptions,
)

from executorch.backends.nxp.backend.ir.converter.conversion.translator import (
    apply_permutation_to,
    create_channels_first_to_channels_last_permutation,
)
from executorch.backends.nxp.backend.ir.converter.node_converter import NodeConverter
from executorch.backends.nxp.backend.ir.lib.tflite.MirrorPadMode import MirrorPadMode
from executorch.backends.nxp.backend.ir.tflite_generator import tflite_model
from executorch.backends.nxp.backend.ir.tflite_generator.builtin_options import (
    mirror_pad_options,
)
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from torch.fx import Node
from torch.nn import Parameter


class PadConverter(NodeConverter):
    @staticmethod
    def _get_mode(node: Node) -> str:
        return node.args[2] if len(node.args) > 2 else "constant"

    @staticmethod
    def _is_supported_in_IR(
        node: Node,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        mode = PadConverter._get_mode(node)
        # `constant` mode is decomposed to `aten.constant_pad_nd`.
        # Conversion of `replicate`/`circular` Torch padding to TFLite
        # is more complicated and is skipped for now.
        if mode != "reflect":
            return False

        return True

    @staticmethod
    def _is_supported_on_target(
        node: Node,
        neutron_target_spec: NeutronTargetSpec,
        parameters_mapping: dict[str, Parameter],
        custom_delegation_options: CustomDelegationOptions,
    ) -> bool:
        if not NodeConverter.uses_quantization_type_for_io(
            node,
            supported_types=[torch.int8, torch.uint8],
            input_indices=[0],
            output_indices=[0],
        ):
            return False

        return True

    @staticmethod
    def _convert_paddings_to_tflite(
        paddings: Collection[int], input_tensor: tflite_model.Tensor
    ) -> list[int]:
        # Group `padding` by two elements per list.
        paddings_grouped = np.array(paddings).reshape(-1, 2)

        # In TFLite, `padding` order is reversed.
        paddings_reversed = list(reversed(paddings_grouped))

        # Add complementary zero pairs to `padding` to match input tensor rank.
        zero_pair_compl = [[0, 0]] * (input_tensor.rank - len(paddings_reversed))
        padding_tfl = zero_pair_compl + paddings_reversed

        if input_tensor.tensor_format.is_channels_last():
            # Permute padding to match tensor format.
            to_tflite_perm = create_channels_first_to_channels_last_permutation(
                input_tensor.rank
            )
            padding_tfl = apply_permutation_to(padding_tfl, to_tflite_perm)

        return padding_tfl

    def convert(self, node: Node):
        """Convert `aten.pad.default` to a TFLite padding operator.

        The ExecuTorch schema is:
            aten::pad(
                Tensor self,
                SymInt[] pad,
                str mode="constant",
                float? value=None
            ) -> Tensor

        Only mode "reflect" is handled in this converter.
        Mode "constant" is decomposed to `aten.constant_pad_nd` and handled in its own converter.
        Modes "replicate"/"circular" are not handled for now, such conversion to TFLite is not trivial.
        """
        self.assert_convertible(node)

        t_op = self._create_tflite_op_with_io_tensors(node)
        x = t_op.tmp_inputs[0]

        paddings = self._convert_paddings_to_tflite(node.args[1], x)

        paddings_tensor = self.builder.create_tensor_for_data(
            np.asarray(paddings, "int32"), "paddings"
        )

        t_op.builtin_options = mirror_pad_options.MirrorPad(MirrorPadMode.REFLECT)
        t_op.tmp_inputs = [x, paddings_tensor]

        self.builder.append_operators([t_op])
