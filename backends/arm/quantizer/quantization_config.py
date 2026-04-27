# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Provide quantization configuration helpers for the Arm backend.

Define a small dataclass to carry activation/weight/bias specs and helper
accessors that validate specs before use. Use this module to build and validate
quantization specs consumed by the annotator.

"""


from dataclasses import dataclass
from typing import Any, Callable, cast, Optional

import torch
from torch.fx import Node
from torchao.quantization.pt2e import ObserverOrFakeQuantize

from torchao.quantization.pt2e.quantizer import (
    DerivedQuantizationSpec,
    QuantizationSpec,
    QuantizationSpecBase,
    SharedQuantizationSpec,
)


@dataclass(eq=True, frozen=True)
class QuantizationConfig:
    """Provide a container for quantization specs.

    Hold optional specs for input/output activations, weights, and bias, and
    expose validated accessors.

    Attributes:
        input_activation (Optional[QuantizationSpec]): Spec for input activations.
        output_activation (Optional[QuantizationSpec]): Spec for output activations.
        weight (Optional[QuantizationSpec]): Spec for weights.
        bias (Optional[QuantizationSpec]): Spec for bias values.

    """

    input_activation: Optional[QuantizationSpecBase]
    output_activation: Optional[QuantizationSpecBase]
    weight: Optional[QuantizationSpecBase]
    bias: Optional[QuantizationSpecBase] | Callable[[Any], Any]
    label: Optional[str] = None  # Optional label for debugging/visualization purposes

    def get_input_act_qspec(
        self, node: Optional[Node] = None, input_node: Optional[Node] = None
    ) -> Optional[QuantizationSpecBase]:
        """Get the validated input activation spec.

        Validate that the input activation qscheme is supported before
        returning the spec.

        Returns:
            Optional[QuantizationSpecBase]: Input activation spec, or ``None`` when
                unset. The ``node`` and ``input_node`` arguments are used by subclasses.

        Raises:
            ValueError: If the qscheme is not per-tensor affine or symmetric.

        """
        if self.input_activation is None:
            return None
        # Validate that input_activation uses a supported qscheme
        if not hasattr(
            self.input_activation, "qscheme"
        ) or self.input_activation.qscheme not in [
            torch.per_tensor_affine,
            torch.per_tensor_symmetric,
        ]:
            raise ValueError(
                f"Unsupported quantization_spec {self.input_activation} for input_activation."
            )
        return self.input_activation

    def get_output_act_qspec(
        self, node: Optional[Node] = None
    ) -> Optional[QuantizationSpecBase]:
        """Get the validated output activation spec.

        Validate that the output activation qscheme is supported before
        returning the spec.

        Returns:
            Optional[QuantizationSpecBase]: Output activation spec, or ``None`` when
                unset. The ``node`` argument is currently unused and kept for
                API parity.

        Raises:
            ValueError: If the qscheme is not per-tensor affine or symmetric.

        """
        if self.output_activation is None:
            return None
        # Validate that output_activation uses a supported qscheme
        if not hasattr(
            self.output_activation, "qscheme"
        ) or self.output_activation.qscheme not in [
            torch.per_tensor_affine,
            torch.per_tensor_symmetric,
        ]:
            raise ValueError(
                f"Unsupported quantization_spec {self.output_activation} for output_activation."
            )
        return self.output_activation

    def get_weight_qspec(
        self, node: Optional[Node] = None
    ) -> Optional[QuantizationSpecBase]:
        """Get the validated weight spec.

        Validate that the weight qscheme is supported (per-tensor or
        per-channel symmetric) before returning the spec.

        Returns:
            Optional[QuantizationSpecBase]: Weight spec, or ``None`` when unset.

        Raises:
            ValueError: If the qscheme is not a supported symmetric scheme.

        """
        if self.weight is None:
            return None
        # Validate that weight uses a supported qscheme
        if not hasattr(self.weight, "qscheme") or self.weight.qscheme not in [
            torch.per_tensor_symmetric,
            torch.per_channel_symmetric,
        ]:
            raise ValueError(f"Unsupported quantization_spec {self.weight} for weight")
        return self.weight

    def get_bias_qspec(
        self, node: Optional[Node] = None
    ) -> Optional[QuantizationSpecBase] | Callable[[Any], Any]:
        """Get the derived or validated bias spec.

        For conv/linear ops, derive bias qparams from the input/weight observers.
        Otherwise, validate a user-provided floating-point bias spec.

        Args:
            node (Optional[Node]): Node whose bias spec is requested.

        Returns:
            Optional[QuantizationSpecBase]: Derived or provided bias spec, or
                ``None`` when unset.

        Raises:
            ValueError: If deriving qparams sees an unexpected number of
                observers/fake-quantizers, or if a provided bias dtype is not
                floating-point.

        """

        if self.bias is None or node is None:
            return None

        def _derive_qparams_fn(
            obs_or_fqs: list[ObserverOrFakeQuantize],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Compute bias scale/zero-point from activation/weight observers.

            Expect two observers or fake-quantize modules: one for the input
            activation and one for the weight. The bias scale is the product of
            input and weight scales, and the zero-point is a tensor of zeros.

            Args:
                obs_or_fqs (list[ObserverOrFakeQuantize]): Observers/fake-quant
                    in order ``[act, weight]``.

            Returns:
                Tuple[torch.Tensor, torch.Tensor]: Bias scale tensor and
                    integer zero-point tensor.

            Raises:
                ValueError: If the list does not contain exactly two items.

            """
            # Validate expected number of observers/fake-quantizes
            if len(obs_or_fqs) != 2:
                raise ValueError(
                    f"Expecting two obs/fqs, one for activation and one for weight, got: {len(obs_or_fqs)}"
                )
            act_obs_or_fq = obs_or_fqs[0]
            weight_obs_or_fq = obs_or_fqs[1]
            act_scale, _ = act_obs_or_fq.calculate_qparams()
            weight_scale, _ = weight_obs_or_fq.calculate_qparams()
            return torch.tensor(act_scale * weight_scale).to(
                torch.float32
            ), torch.full_like(weight_scale, fill_value=0, dtype=torch.int32)

        if node.target in [
            torch.ops.aten.conv1d.default,
            torch.ops.aten.conv2d.default,
            torch.ops.aten.linear.default,
            torch.ops.aten.conv2d.padding,
            torch.ops.aten.conv_transpose2d.input,
            torch.ops.aten.conv3d.default,
            torch.ops.aten.conv3d.padding,
        ]:
            if self.input_activation is None or self.weight is None:
                raise ValueError(
                    "Input activation and weight QuantizationConfig must be specified."
                )
            if not isinstance(
                self.input_activation, QuantizationSpec
            ) or not isinstance(self.weight, QuantizationSpec):
                raise ValueError(
                    "QuantizationConfig input_activation and weight must be instances of QuantizationSpec."
                )

            if (self.input_activation.dtype == self.weight.dtype == torch.int8) or (
                self.input_activation.dtype == torch.int16
                and self.weight.dtype == torch.int8
            ):
                input_act = node.args[0]
                weight = node.args[1]

                # If the weights are quantized per_tensor, do the same with bias
                qscheme = (
                    torch.per_tensor_symmetric
                    if self.weight is None
                    else self.weight.qscheme
                )
                ch_axis = None
                if self.weight is not None:
                    if qscheme == torch.per_channel_symmetric:
                        ch_axis = self.weight.ch_axis
                        if (
                            node.target == torch.ops.aten.conv_transpose2d.input
                            and ch_axis is not None
                        ):
                            # Bias is 1-D so channel axis is always 0 even for transpose conv
                            ch_axis = 0

                quantization_spec = DerivedQuantizationSpec(
                    derived_from=((input_act, node), (weight, node)),  # type: ignore[arg-type]
                    derive_qparams_fn=_derive_qparams_fn,
                    dtype=torch.int32,
                    quant_min=torch.iinfo(torch.int32).min + 1,
                    quant_max=torch.iinfo(torch.int32).max,
                    qscheme=qscheme,
                    ch_axis=ch_axis,
                )
                return quantization_spec  # type: ignore[return-value]
            else:
                raise NotImplementedError(
                    f"Bias quantization of types: i:{self.input_activation.dtype}, w:{self.weight.dtype} not implemented"
                )

        return self.bias


class TOSAQuantizationConfig(QuantizationConfig):
    """Configures quantization, while enforcing TOSA specific constraints."""

    SHARED_OUTPUT_ACT_QSPEC_PATTERNS = {
        torch.ops.aten.adaptive_avg_pool2d.default,
        torch.ops.aten.upsample_bilinear2d.vec,
        torch.ops.aten.upsample_nearest2d.vec,
        torch.ops.aten.avg_pool2d.default,
        torch.ops.aten.max_pool2d.default,
        torch.ops.aten.mean.default,
        torch.ops.aten.mean.dim,
        torch.ops.aten.silu.default,
        torch.ops.aten.silu_.default,
    }

    SHARED_INPUT_ACT_QSPEC_PATTERNS = {
        torch.ops.aten.lt.Tensor,
        torch.ops.aten.le.Tensor,
        torch.ops.aten.gt.Tensor,
        torch.ops.aten.ge.Tensor,
        torch.ops.aten.eq.Tensor,
        torch.ops.aten.ne.Tensor,
    }

    def get_input_act_qspec(self, node=None, input_node=None):
        """Return the configured input quantization spec.

        For comparison operators, make sure that both inputs share the same
        quantization spec, by returning a SharedQuantizationSpec that ties the
        quantization of both inputs together. For other operators, return the
        default input activation spec.

        """
        if node is None or input_node is None:
            return super().get_input_act_qspec(node, input_node)

        if node.target in self.SHARED_INPUT_ACT_QSPEC_PATTERNS:
            if input_node == node.args[0]:
                return super().get_input_act_qspec(node, input_node)
            else:
                return SharedQuantizationSpec((node.args[0], node))

        return super().get_input_act_qspec(node, input_node)

    def get_weight_qspec(
        self, node: Optional[Node] = None
    ) -> Optional[QuantizationSpecBase]:
        """Return the configured weight quantization spec.

        For conv transpose, return the per-channel quantization spec with
        `ch_axis=1` to match the IOHW weight format used by TOSA, instead of
        the default `ch_axis=0`. If no weight spec is configured, return
        ``None``.

        """
        weight_qspec = super().get_weight_qspec()
        if (
            node is not None
            and weight_qspec is not None
            and isinstance(weight_qspec, QuantizationSpec)
            and weight_qspec.qscheme == torch.per_channel_symmetric
            and node.target == torch.ops.aten.conv_transpose2d.input
        ):
            # MLETORCH-1853: Fix lazy import when moving files around
            from executorch.backends.arm.quantizer.quantization_annotator import (
                _adjust_weight_qspec_for_conv_transpose,
            )

            weight_qspec = _adjust_weight_qspec_for_conv_transpose(node, weight_qspec)

        return weight_qspec

    def get_output_act_qspec(
        self, node: Optional[Node] = None
    ) -> Optional[QuantizationSpecBase]:
        """Return the configured output activation quantization spec.

        If node is a pooling or upsample operator, returns a shared quantization spec.
        If no weight spec is configured, return ``None``.

        """

        if node is None:
            return super().get_output_act_qspec()
        if node.target not in self.SHARED_OUTPUT_ACT_QSPEC_PATTERNS:
            return super().get_output_act_qspec()
        if len(node.args) == 0:
            return super().get_output_act_qspec()
        return SharedQuantizationSpec((cast(Node, node.args[0]), node))
