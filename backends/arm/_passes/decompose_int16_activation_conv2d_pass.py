# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import cast

import torch
from executorch.backends.arm._passes.quant_args import QuantArgs

from executorch.backends.arm.tosa.specification import get_context_spec, Tosa_1_00
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class DecomposeConv2dWithInt16ActivationPass(ExportPass):
    """
    This pass decomposes a convolution with input dtype int16 and bias
    into a convolution without bias followed by an addition of the bias
    since the TOSA op requires the bias to be int48 which is hard to represent
    in torch. Instead rescale the int48 output to int16 and add the bias in int16.
    """

    def call_operator(self, op, args, kwargs, meta):
        if op != exir_ops.edge.aten.convolution.default:
            return super().call_operator(op, args, kwargs, meta)

        tosa_spec = get_context_spec()
        if not tosa_spec.support_integer():
            return super().call_operator(op, args, kwargs, meta)

        # return if no bias
        if args[2] is None:
            return super().call_operator(op, args, kwargs, meta)

        if args[0].data.dtype == torch.int8:
            return super().call_operator(op, args, kwargs, meta)
        elif args[0].data.dtype == torch.int16:
            if isinstance(tosa_spec, Tosa_1_00) and not tosa_spec.support_extension(
                "int16"
            ):
                raise ValueError(
                    "int16 activation for convolution requires TOSA int16 extension"
                )
        else:
            raise NotImplementedError(
                "Decomposition to conv+add only implemented for activation of int16 type"
            )

        # convolution with bias and activation is int16
        # The bias is assumed to be quantized with the same quantization parameters as
        # as the output of the convolution
        bias = args[2]
        assert (
            meta.data["output_qparams"][0].dtype == bias.data.dtype
        ), "Bias needs to have same type as quantized output type"
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
            (bias, [1, len(bias.data), 1, 1]),
            {},
            new_meta,
        )

        output_dtype = meta.data["output_qparams"][0].dtype

        if output_dtype == torch.int16:
            # The conv will get the output int48 scaled to int32 in serialization step.
            # To be able to add the bias we need to first scale (cast?) the output to int32.
            # The resulting i32 sum will then need to be scaled back to the output dtype.

            # calculate common rescale factor from convolution output and bias quantization
            output_qparams = cast(QuantArgs, meta.data["output_qparams"][0])
            conv_output_scale = output_qparams.scale
            bias_qparams = cast(QuantArgs, meta.data["input_qparams"][2])
            bias_scale = bias_qparams.scale

            common_scale = max(bias_scale, conv_output_scale)

            # calculate how we can rescale bias and conv to a common scale and maximize the output range
            bias_rescale_factor = bias_scale / common_scale
            conv_rescale_factor = conv_output_scale / common_scale

            # Either of conv output or bias now covers the full int16 range and the other one a smaller range.
            # Since we are upscaling to int32 we have 16 additional bits to work with to maximize the output range.
            # Worst case here is that both bias and conv output covers the full int16 range so we leave one bit
            # and then one for the sign bit.
            bits_left_to_shift = 14

            # update rescale factors
            bias_rescale_factor *= 1 << bits_left_to_shift
            conv_rescale_factor *= 1 << bits_left_to_shift

            conv_output = super().call_operator(
                exir_ops.backend.tosa.RESCALE.default,
                (convolution, torch.int32, [conv_rescale_factor], 0, 0),
                {},
                new_meta,
            )

            bias_rescaled = super().call_operator(
                exir_ops.backend.tosa.RESCALE.default,
                (channel_bias, torch.int32, [bias_rescale_factor], 0, 0),
                {},
                new_meta,
            )

            add = super().call_operator(
                exir_ops.edge.aten.add.Tensor,
                (conv_output, bias_rescaled),
                {},
                new_meta,
            )

            res_rescale = super().call_operator(
                exir_ops.backend.tosa.RESCALE.default,
                (
                    add,
                    output_dtype,
                    [(common_scale / (conv_output_scale * (1 << bits_left_to_shift)))],
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
