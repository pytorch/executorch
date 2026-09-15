# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Provide quantization configuration helpers for the Arm backend.

Define a small dataclass to carry activation/weight/bias specs and helper
accessors that validate specs before use. Use this module to build and validate
quantization specs consumed by the quantizer.

"""

import functools
from dataclasses import dataclass, replace
from typing import Any, Callable, cast, NamedTuple, Optional

import torch
from torch.fx import Node
from torchao.quantization.pt2e import (
    FakeQuantize,
    FusedMovingAvgObsFakeQuantize,
    MovingAveragePerChannelMinMaxObserver,
    ObserverOrFakeQuantize,
    PartialWrapper,
)

from torchao.quantization.pt2e.quantizer import (
    DerivedQuantizationSpec,
    FixedQParamsQuantizationSpec,
    QuantizationSpec,
    QuantizationSpecBase,
    SharedQuantizationSpec,
)


def _is_fused_moving_avg_obs_fake_quant_ctor(func: object) -> bool:
    """Return True when ``func`` is the fused fake-quant class or a subclass."""

    return isinstance(func, type) and issubclass(func, FusedMovingAvgObsFakeQuantize)


class _QParams(NamedTuple):
    scale: float
    zero_point: int
    quant_min: int | None = None
    quant_max: int | None = None


def _adjust_weight_qspec_for_conv_transpose(
    node: Node, weight_qspec: QuantizationSpec | None
) -> QuantizationSpec | None:
    """Adjust weight qspec axis/ctor for conv_transpose2d per-channel
    quantization.

    Use axis 1 for ungrouped ConvTranspose2d weights because the weight layout is
    (in_channels, out_channels / groups, kH, kW). Grouped transpose conv keeps axis 0.

    If the weight qspec contains a TorchAO QAT fake-quant/observer constructor
    (e.g. PartialWrapper(partial(...)) or a with_args-based constructor), the
    constructor is rebuilt with the corrected axis. For fused per-channel
    FakeQuantize, which only supports axis 0, the constructor is replaced with
    a non-fused FakeQuantize + MovingAveragePerChannelMinMaxObserver when the
    required axis is not 0.

    Return the qspec unchanged when weights are unset.

    """

    if (
        node.target != torch.ops.aten.conv_transpose2d.input
        or weight_qspec is None
        or weight_qspec.qscheme != torch.per_channel_symmetric
    ):
        return weight_qspec

    # For now skip axis adjustment for a8w4 per-channel configs (int4 weights).
    if weight_qspec.quant_min == -7 and weight_qspec.quant_max == 7:
        return weight_qspec

    groups = 1
    if len(node.args) > 6 and isinstance(node.args[6], int):
        groups = node.args[6]
    expected_axis = 0 if groups != 1 else 1

    observer_or_fake_quant_ctr = weight_qspec.observer_or_fake_quant_ctr
    observer_or_fake_quant_ctr_changed = False
    # QAT FakeQuantize uses PartialWrapper; rebuild its partial to update ch_axis
    # without breaking TorchAO introspection.
    if isinstance(observer_or_fake_quant_ctr, PartialWrapper):
        original_callable_args = dict(observer_or_fake_quant_ctr.callable_args)
        base_partial = observer_or_fake_quant_ctr.p
        if isinstance(base_partial, functools.partial):
            base_keywords = dict(base_partial.keywords or {})
            base_keywords["ch_axis"] = expected_axis
            if (
                _is_fused_moving_avg_obs_fake_quant_ctor(base_partial.func)
                and expected_axis != 0
            ):
                # Fused per-channel FakeQuant only supports axis 0; for other axes,
                # fall back to FakeQuantize with a per-channel observer.
                base_keywords["observer"] = MovingAveragePerChannelMinMaxObserver
                observer_or_fake_quant_ctr = PartialWrapper(
                    functools.partial(FakeQuantize, **base_keywords)
                )
            else:
                observer_or_fake_quant_ctr = PartialWrapper(
                    functools.partial(base_partial.func, **base_keywords)
                )
            observer_or_fake_quant_ctr.callable_args = original_callable_args
            observer_or_fake_quant_ctr_changed = True
    # Non-QAT observer/fake-quant ctrs can be updated via with_args.
    elif hasattr(observer_or_fake_quant_ctr, "with_args"):
        observer_or_fake_quant_ctr = observer_or_fake_quant_ctr.with_args(
            ch_axis=expected_axis
        )
        observer_or_fake_quant_ctr_changed = True

    if weight_qspec.ch_axis == expected_axis and not observer_or_fake_quant_ctr_changed:
        return weight_qspec

    return QuantizationSpec(
        dtype=weight_qspec.dtype,
        observer_or_fake_quant_ctr=observer_or_fake_quant_ctr,
        quant_min=weight_qspec.quant_min,
        quant_max=weight_qspec.quant_max,
        qscheme=weight_qspec.qscheme,
        ch_axis=expected_axis,
        is_dynamic=weight_qspec.is_dynamic,
    )


# For these ops, we use fixed qspecs, meaning that quantization params for
# these are statically defined. This is to prevent issues with out-of-range
# values when using dynamic quantization.
#
# Dict of operator to a dict of num_bits to qparams for that operator.
_fixed_input_qspec_ops: dict[Any, dict[int, _QParams]] = {
    # acos has a valid range of [-1, 1]
    torch.ops.aten.acos.default: {
        8: _QParams((1.0 - (-1.0)) / (1 << 8), 0),
        16: _QParams((1.0 - (-1.0)) / (1 << 16), 0),
    },
    # asin has a valid range of [-1, 1]
    torch.ops.aten.asin.default: {
        8: _QParams((1.0 - (-1.0)) / (1 << 8), 0),
        16: _QParams((1.0 - (-1.0)) / (1 << 16), 0),
    },
    # atanh has a valid range of (-1, 1) (excluding -1 and 1).
    torch.ops.aten.atanh.default: {
        8: _QParams((0.999 - (-0.999)) / (1 << 8), 0),
        16: _QParams((0.99999 - (-0.99999)) / (1 << 16), 0),
    },
    # grid_sampler image input/output use SNORM-compatible qparams. Input 1
    # follows the standard activation qspec, but the supported VGF lowering
    # modes are still only:
    # - float image / float grid / float output
    # - int8 image / int8 grid / int8 output
    # Mixed int8-image / float-grid shader lowering is not supported.
    torch.ops.aten.grid_sampler.default: {
        8: _QParams(1.0 / 127.0, 0, -127, 127),
    },
}


_fixed_output_qspec_ops: dict[Any, dict[int, _QParams]] = {
    torch.ops.aten.grid_sampler.default: {
        8: _QParams(1.0 / 127.0, 0, -127, 127),
    },
}


def _get_fixed_qparams_qspec(
    node_target: Any,
    qparams_table: dict[Any, dict[int, _QParams]],
    input_act_qspec: QuantizationSpecBase,
) -> FixedQParamsQuantizationSpec | None:
    if not isinstance(input_act_qspec, QuantizationSpec):
        raise ValueError("Fixed qparams require a QuantizationSpec input.")

    num_bits = torch.iinfo(input_act_qspec.dtype).bits
    qparams = qparams_table[node_target].get(num_bits)
    if qparams is None:
        return None

    return FixedQParamsQuantizationSpec(
        dtype=input_act_qspec.dtype,
        scale=qparams.scale,
        zero_point=qparams.zero_point,
        quant_min=(
            input_act_qspec.quant_min
            if qparams.quant_min is None
            else qparams.quant_min
        ),
        quant_max=(
            input_act_qspec.quant_max
            if qparams.quant_max is None
            else qparams.quant_max
        ),
        qscheme=input_act_qspec.qscheme,
        is_dynamic=input_act_qspec.is_dynamic,
    )


def _is_canonical_flow_offset_grid_sampler(node: Node) -> bool:  # noqa: C901
    if node.target != torch.ops.aten.grid_sampler.default or len(node.args) < 5:
        return False
    if tuple(node.args[2:5]) != (0, 1, True):
        return False
    image = node.args[0]
    if not isinstance(image, Node):
        return False
    image_value = image.meta.get("val")
    output_value = node.meta.get("val")
    if (
        not isinstance(image_value, torch.Tensor)
        or len(image_value.shape) != 4
        or int(image_value.shape[0]) != 1
        or int(image_value.shape[1]) not in (3, 4)
        or not isinstance(output_value, torch.Tensor)
        or tuple(output_value.shape) != tuple(image_value.shape)
    ):
        return False
    grid = node.args[1]
    if not isinstance(grid, Node) or grid.target != torch.ops.aten.permute.default:
        return False
    permutation = grid.args[1]
    if not isinstance(permutation, (list, tuple)) or list(permutation) != [0, 2, 3, 1]:
        return False
    add = grid.args[0]
    if not isinstance(add, Node) or add.target != torch.ops.aten.add.Tensor:
        return False
    if len(add.args) > 2 and add.args[2] != 1:
        return False
    if add.kwargs.get("alpha", 1) != 1:
        return False
    for base_grid, flow_slice in (
        (add.args[0], add.args[1]),
        (add.args[1], add.args[0]),
    ):
        if not isinstance(flow_slice, Node) or flow_slice.target not in (
            torch.ops.aten.slice.Tensor,
            torch.ops.aten.slice_copy.Tensor,
        ):
            continue
        flow, dim, start, end, *step = flow_slice.args
        if dim != 1 or (start, end) not in ((0, 2), (2, 4)):
            continue
        if (
            step not in ([], [1])
            or flow_slice.kwargs.get("step", 1) != 1
            or not isinstance(flow, Node)
        ):
            continue
        flow_value = flow.meta.get("val")
        while isinstance(base_grid, Node) and base_grid.target in (
            torch.ops.aten.to.dtype,
            torch.ops.aten.to.dtype_layout,
            torch.ops.aten.alias.default,
        ):
            base_grid = base_grid.args[0]
        if not isinstance(base_grid, Node):
            continue
        base_value = base_grid.meta.get("val")
        if (
            not isinstance(flow_value, torch.Tensor)
            or len(flow_value.shape) != 4
            or tuple(flow_value.shape[:2]) != (1, 4)
        ):
            continue
        height, width = map(int, flow_value.shape[2:])
        if height <= 1 or width <= 1:
            continue
        if tuple(image_value.shape[2:]) != (height, width):
            continue
        if not isinstance(base_value, torch.Tensor) or tuple(base_value.shape) != (
            1,
            2,
            height,
            width,
        ):
            continue
        if base_grid.target != torch.ops.aten.cat.default or base_grid.args[1] != 1:
            continue
        axes = base_grid.args[0]
        if not isinstance(axes, (list, tuple)) or len(axes) != 2:
            continue

        def matches_axis(axis_node, axis):
            if not isinstance(axis_node, Node) or axis_node.target not in (
                torch.ops.aten.expand.default,
                torch.ops.aten.expand_copy.default,
            ):
                return False
            view = axis_node.args[0]
            if not isinstance(view, Node) or view.target not in (
                torch.ops.aten.view.default,
                torch.ops.aten.view_copy.default,
            ):
                return False
            linspace = view.args[0]
            if (
                not isinstance(linspace, Node)
                or linspace.target != torch.ops.aten.linspace.default
            ):
                return False
            if linspace.kwargs.get("dtype") not in (None, torch.float32):
                return False
            extent = width if axis == 3 else height  # noqa: B023
            expected_view = [1, 1, 1, width] if axis == 3 else [1, 1, height, 1]  # noqa: B023  # fmt: skip
            expected_expand = [1, -1, height, -1] if axis == 3 else [1, -1, -1, width]  # noqa: B023  # fmt: skip
            return (
                tuple(linspace.args[:3]) == (-1.0, 1.0, extent)
                and list(view.args[1]) == expected_view
                and list(axis_node.args[1]) == expected_expand
            )

        if matches_axis(axes[0], 3) and matches_axis(axes[1], 2):
            return True
    return False


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
            return (act_scale * weight_scale).to(torch.float32), torch.full_like(
                weight_scale, fill_value=0, dtype=torch.int32
            )

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

    quantize_grid_sampler_grid = False

    # Ops whose output range matches their input's, so reusing the input qspec
    # costs no resolution. Ops that compress the value range do not belong here.
    SHARED_OUTPUT_ACT_QSPEC_PATTERNS = {
        torch.ops.aten.adaptive_avg_pool2d.default,
        torch.ops.aten.upsample_bilinear2d.vec,
        torch.ops.aten.upsample_nearest2d.vec,
        torch.ops.aten.avg_pool2d.default,
        torch.ops.aten.max_pool2d.default,
        torch.ops.aten.mean.default,
        torch.ops.aten.mean.dim,
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
        quantization of both inputs together.

        For trigonometric ops, ensure that input spec has fixed qparams.

        For other operators, return the default input activation spec.

        """
        if node is None or input_node is None:
            return super().get_input_act_qspec(node, input_node)

        if node.target in self.SHARED_INPUT_ACT_QSPEC_PATTERNS:
            if input_node == node.args[0]:
                return super().get_input_act_qspec(node, input_node)
            else:
                return SharedQuantizationSpec((node.args[0], node))
        elif node.target == torch.ops.aten.grid_sampler.default:
            if input_node == node.args[0]:
                input_act_qspec = super().get_input_act_qspec(node, input_node)
                return _get_fixed_qparams_qspec(
                    node.target, _fixed_input_qspec_ops, input_act_qspec
                )
            if not self.quantize_grid_sampler_grid:
                return None
            return super().get_input_act_qspec(node, input_node)
        elif node.target in _fixed_input_qspec_ops:
            input_act_qspec = super().get_input_act_qspec(node, input_node)
            return _get_fixed_qparams_qspec(
                node.target, _fixed_input_qspec_ops, input_act_qspec
            )

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
            weight_qspec = _adjust_weight_qspec_for_conv_transpose(node, weight_qspec)

        return weight_qspec

    def get_output_act_qspec(
        self, node: Optional[Node] = None
    ) -> Optional[QuantizationSpecBase]:
        """Return the configured output activation quantization spec.

        If node is a pooling or upsample operator, returns a shared quantization spec.
        If no weight spec is configured, return ``None``.
        If node is a `to.dtype` operator, returns a fixed quantization spec if the input is integer and the output is float32.

        """
        if node is None:
            return super().get_output_act_qspec()
        if node.target in _fixed_output_qspec_ops:
            output_act_qspec = super().get_output_act_qspec(node)
            if output_act_qspec is None:
                return None
            return _get_fixed_qparams_qspec(
                node.target, _fixed_output_qspec_ops, output_act_qspec
            )
        if node.target == torch.ops.aten.to.dtype:
            from executorch.backends.arm.quantizer.quantizer_support import CastCheck

            input_node = node.all_input_nodes[0]
            input_val = input_node.meta.get("val", None)
            output_val = node.meta.get("val", None)
            if (
                isinstance(input_val, torch.Tensor)
                and isinstance(output_val, torch.Tensor)
                and CastCheck.is_integer_to_integer(input_val.dtype, output_val.dtype)
            ):
                return None
            if (
                isinstance(input_val, torch.Tensor)
                and isinstance(output_val, torch.Tensor)
                and CastCheck.is_integer_to_float(input_val.dtype, output_val.dtype)
                and output_val.dtype == torch.float32
            ):
                return FixedQParamsQuantizationSpec(
                    dtype=input_val.dtype,
                    scale=1.0,
                    zero_point=0,
                    quant_min=torch.iinfo(input_val.dtype).min,
                    quant_max=torch.iinfo(input_val.dtype).max,
                    qscheme=torch.per_tensor_symmetric,
                    is_dynamic=False,
                )
            if (
                isinstance(input_val, torch.Tensor)
                and isinstance(output_val, torch.Tensor)
                and CastCheck.is_float_identity(input_val.dtype, output_val.dtype)
            ):
                return SharedQuantizationSpec((input_node, node))
            return super().get_output_act_qspec(node)
        if node.target not in self.SHARED_OUTPUT_ACT_QSPEC_PATTERNS:
            return super().get_output_act_qspec()
        if len(node.args) == 0:
            return super().get_output_act_qspec()
        return SharedQuantizationSpec((cast(Node, node.args[0]), node))


class VGFQuantizationConfig(TOSAQuantizationConfig):
    """Configure quantization for VGF-specific lowering paths."""

    quantize_grid_sampler_grid = True

    @staticmethod
    def _snorm_compatible_qspec(
        qspec: Optional[QuantizationSpecBase],
    ) -> Optional[QuantizationSpecBase]:
        if qspec is None:
            return None
        if not isinstance(qspec, QuantizationSpec):
            raise ValueError("SNORM-compatible qparams require a QuantizationSpec.")
        if qspec.quant_min == -127 and qspec.quant_max == 127:
            return qspec
        return replace(qspec, quant_min=-127, quant_max=127)

    def get_input_act_qspec(self, node=None, input_node=None):
        """Return the input activation spec for a VGF graph node.

        Args:
            node (Optional[Node]): Node whose input spec is requested.
            input_node (Optional[Node]): Input node whose spec is requested.

        Returns:
            Optional[QuantizationSpecBase]: Input activation spec, or ``None``.

        """
        if (
            node is not None
            and input_node is not None
            and _is_canonical_flow_offset_grid_sampler(node)
            and input_node == node.args[0]
        ):
            qspec = QuantizationConfig.get_input_act_qspec(self, node, input_node)
            if isinstance(qspec, QuantizationSpec) and qspec.dtype == torch.int8:
                return self._snorm_compatible_qspec(qspec)
        return super().get_input_act_qspec(node, input_node)

    def get_output_act_qspec(self, node=None):
        """Return the output activation spec for a VGF graph node.

        Args:
            node (Optional[Node]): Node whose output spec is requested.

        Returns:
            Optional[QuantizationSpecBase]: Output activation spec, or ``None``.

        """
        if node is not None and _is_canonical_flow_offset_grid_sampler(node):
            qspec = QuantizationConfig.get_output_act_qspec(self, node)
            if isinstance(qspec, QuantizationSpec) and qspec.dtype == torch.int8:
                return self._snorm_compatible_qspec(qspec)
        return super().get_output_act_qspec(node)
