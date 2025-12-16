# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import cast, Sequence, Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.quant_args import QuantArgs

from executorch.backends.arm.tosa.specification import get_context_spec
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class DecomposeConvWithInt16ActivationPass(ArmPass):
    """
    This pass decomposes a convolution with input dtype int16 and bias
    into a convolution without bias followed by an addition of the bias.
    We also reshape the 1D bias to [1, C, 1, …] so it broadcasts along the channel
    dimension. Since the TOSA op requires the bias to be int48 which is hard to represent
    in torch. Instead rescale the int48 output to int16 and add the bias in int16.
    """

    def __init__(self) -> None:
        super().__init__()

    _passes_required_after: Set[Type[ExportPass]] = set()

    def bias_view_shape(
        self, bias: torch.Tensor, activation_rank: int
    ) -> Sequence[int]:
        # reshape bias to match convolution output rank so addition broadcasts over channels
        return [1, bias.shape[0], *([1] * (activation_rank - 2))]

    def call_operator(self, op, args, kwargs, meta):
        if op != exir_ops.edge.aten.convolution.default:
            return super().call_operator(op, args, kwargs, meta)

        tosa_spec = get_context_spec()
        if not tosa_spec.support_integer():
            return super().call_operator(op, args, kwargs, meta)

        # return if no bias
        if args[2] is None:
            return super().call_operator(op, args, kwargs, meta)

        activation_tensor = args[0].data
        activation_rank = activation_tensor.dim()

        if activation_rank not in (4, 5) or activation_tensor.dtype != torch.int16:
            return super().call_operator(op, args, kwargs, meta)

        if not tosa_spec.support_extension("int16"):
            raise ValueError(
                "int16 activation for convolution requires TOSA int16 extension"
            )

        # convolution with bias and activation is int16 (expected activation rank enforced above)
        # The bias is assumed to be quantized with the same quantization parameters as
        # the output of the convolution
        bias_arg = args[2]
        bias_data = bias_arg.data

        no_bias_args = list(args)
        no_bias_args[2] = None
        # split up to convolution + bias
        convolution = super().call_operator(op, tuple(no_bias_args), kwargs, meta)

        # create a copy of the meta without the qparams, to be used with the new nodes
        new_meta = meta.copy()
        new_meta.data.pop("output_qparams", None)
        new_meta.data.pop("input_qparams", None)

        # reshape the tensor to the same rank as the convolution output to add the bias to the channels
        channel_bias = super().call_operator(
            exir_ops.edge.aten.view_copy.default,
            (bias_arg, self.bias_view_shape(bias_data, activation_rank)),
            {},
            new_meta,
        )

        output_dtype = meta.data["output_qparams"][0].dtype

        if output_dtype == torch.int16:
            # The conv will get the output int48 scaled to int32 in serialization step.
            # To be able to add the bias we need to first scale (cast?) the output to int32.
            # The resulting i32 sum will then need to be scaled back to the output dtype.
            output_qparams = cast(QuantArgs, meta.data["output_qparams"][0])
            conv_output_scale = output_qparams.scale

            bias_qparams = cast(QuantArgs, meta.data["input_qparams"][2])
            per_channel_quant = bias_qparams.per_channel

            if per_channel_quant:
                bias_scale = bias_qparams.get_scale_per_channel()
            else:
                bias_scale = [bias_qparams.get_scale_per_tensor()]

            conv_rescale_factors = [1.0] * len(bias_scale)
            final_output_scale = [b / conv_output_scale for b in bias_scale]

            conv_output = super().call_operator(
                exir_ops.backend.tosa.RESCALE.default,
                (convolution, torch.int32, conv_rescale_factors, 0, 0),
                {},
                new_meta,
            )

            add = super().call_operator(
                exir_ops.edge.aten.add.Tensor,
                (conv_output, channel_bias),
                {},
                new_meta,
            )

            res_rescale = super().call_operator(
                exir_ops.backend.tosa.RESCALE.default,
                (
                    add,
                    output_dtype,
                    final_output_scale,
                    0,
                    0,
                ),
                {},
                new_meta,
            )

        else:
            raise NotImplementedError(
                f"Decomposition to conv+add only implemented for activation of int16 type, not for {output_dtype}"
            )

        return res_rescale
