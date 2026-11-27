# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# pyre-strict
"""Fold exact quantized DyT affine maps into following convolutions."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import cast, Set, Type

import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import (
    get_constant_placeholder_kind,
    get_param_tensor,
    is_persistent_buffer,
)
from executorch.backends.arm._passes.fold_dyt_alpha_into_lut_pass import (
    _apply_tosa_rescale,
    _RescaleParams,
)
from executorch.backends.arm._passes.quant_args import QuantArgs
from executorch.backends.transforms.utils import (
    create_constant_placeholder,
    delete_constant_placeholder,
)
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import Graph, GraphModule, Node


@dataclass(frozen=True)
class _Operand:
    source: Node
    rescale: _RescaleParams
    constant: torch.Tensor | None
    view_shape: tuple[int, ...] | None


@dataclass(frozen=True)
class _DyTAffineMatch:
    table: Node
    table_values: torch.Tensor
    table_qargs: QuantArgs
    gamma_activation_rescale: _RescaleParams
    gamma: Node
    gamma_codes: torch.Tensor
    gamma_rescale: _RescaleParams
    gamma_output: Node
    gamma_output_rescale: _RescaleParams
    add_activation_rescale: _RescaleParams
    beta: Node
    beta_codes: torch.Tensor
    beta_rescale: _RescaleParams
    add_output: Node
    add_output_rescale: _RescaleParams
    conv: Node
    layout_chain: tuple[Node, ...]


@dataclass(frozen=True)
class _AddChainMatch:
    output: Node
    output_rescale: _RescaleParams
    activation_operand: _Operand
    beta_operand: _Operand
    layout_chain: tuple[Node, ...]


@dataclass(frozen=True)
class _GammaChainMatch:
    output: Node
    output_rescale: _RescaleParams
    table_operand: _Operand
    gamma_operand: _Operand


@dataclass(frozen=True)
class _AffineCodes:
    """Validated INT8 constants backing one DyT affine map."""

    table_values: torch.Tensor
    table_qargs: QuantArgs
    gamma_codes: torch.Tensor
    beta_codes: torch.Tensor


@dataclass(frozen=True)
class _ConvOperands:
    """Weight/bias placeholders and quantization args of a foldable conv."""

    weight_node: Node
    bias_node: Node
    weight: torch.Tensor
    bias: torch.Tensor
    input_qparams: dict[int, QuantArgs]
    activation_qargs: QuantArgs
    weight_qargs: QuantArgs


@dataclass(frozen=True)
class _ConvConstants:
    """A conv whose constants are shape-compatible with the affine map."""

    operands: _ConvOperands
    groups: int
    out_channels: int
    in_channels_per_group: int
    weight_zero_points: torch.Tensor


class FoldDyTAffineIntoConvPass(ArmPass):
    """Fold exact integer DyT gamma/beta maps into a following convolution.

    INT8 requantization makes generic floating-point affine folding inexact.

    The pass evaluates the 256-entry TOSA TABLE and rewrites only exact affine
    maps.

    Position-dependent padding keeps beta; gamma folds only for an exact
    identity.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    _VIEW_TARGETS: Set[object] = {
        exir_ops.edge.aten.view_copy.default,
    }
    _LAYOUT_TARGETS: Set[object] = {
        exir_ops.edge.aten.permute_copy.default,
        exir_ops.edge.aten.slice_copy.Tensor,
    }

    def __init__(self, exported_program: ExportedProgram) -> None:
        super().__init__()
        self.exported_program = exported_program

    @staticmethod
    def _single_qargs(node: Node, key: str) -> QuantArgs | None:
        qparams = cast(dict[int, QuantArgs], node.meta.get(key, {}))
        if len(qparams) != 1:
            return None
        qargs = next(iter(qparams.values()))
        if qargs.per_channel:
            return None
        return qargs

    @staticmethod
    def _tensor_shape(node: Node) -> tuple[int, ...] | None:
        value = node.meta.get("val")
        if not isinstance(value, torch.Tensor) or not all(
            type(dim) is int for dim in value.shape
        ):
            return None
        return cast(tuple[int, ...], tuple(value.shape))

    def _constant(self, node: Node) -> torch.Tensor | None:
        try:
            return get_param_tensor(self.exported_program, node)
        except RuntimeError:
            return None

    def _unwrap_views(self, node: Node) -> tuple[Node, tuple[int, ...] | None] | None:
        view_shape = None
        while node.target in self._VIEW_TARGETS:
            if (
                len(node.args) < 2
                or not isinstance(node.args[0], Node)
                or len(node.users) != 1
            ):
                return None
            shape = node.args[1]
            if (
                view_shape is not None
                or not isinstance(shape, (list, tuple))
                or not all(type(value) is int for value in shape)
            ):
                return None
            view_shape = tuple(shape)
            node = node.args[0]
        return node, view_shape

    def _operand(self, node: Node) -> _Operand | None:
        unwrapped = self._unwrap_views(node)
        if unwrapped is None:
            return None
        rescale_node, view_shape = unwrapped
        rescale = _RescaleParams.from_node(rescale_node)
        if (
            rescale is None
            or rescale.output_dtype != torch.int32
            or not isinstance(rescale_node.args[0], Node)
        ):
            return None
        source = rescale_node.args[0]
        return _Operand(
            source=source,
            rescale=rescale,
            constant=self._constant(source),
            view_shape=view_shape,
        )

    def _binary_operands(self, node: Node) -> tuple[_Operand, _Operand] | None:
        if len(node.args) < 2:
            return None
        lhs, rhs = node.args[:2]
        if not isinstance(lhs, Node) or not isinstance(rhs, Node):
            return None
        lhs_operand = self._operand(lhs)
        rhs_operand = self._operand(rhs)
        if lhs_operand is None or rhs_operand is None:
            return None
        return lhs_operand, rhs_operand

    def _trace_layout_source(self, node: Node) -> tuple[Node, tuple[Node, ...]] | None:
        """Walk back through layout ops to the affine site.

        Exactly one ``permute_copy`` is required and pinned to the NHWC->NCHW
        dim order this pass is written against. ``slice_copy`` is deliberately not inspected: a slice
        that changes which channels the conv consumes leaves the site's
        per-channel slope/offset count disagreeing with the conv weight's input
        channels, and ``_fold_conv_constants`` refuses the fold on that
        mismatch. Slices on the batch or spatial dims cannot invalidate a
        per-channel affine map. Both paths are pinned by
        ``test_channel_narrowing_slice_is_rejected`` and
        ``test_identity_affine_behind_channel_slice_leaves_conv_constants``.

        """
        chain = []
        permute_count = 0
        while node.target in self._LAYOUT_TARGETS:
            if len(node.args) == 0 or not isinstance(node.args[0], Node):
                return None
            if node.target == exir_ops.edge.aten.permute_copy.default:
                dims = node.args[1] if len(node.args) > 1 else None
                if (
                    permute_count != 0
                    or not isinstance(dims, (list, tuple))
                    or tuple(dims) != (0, 3, 1, 2)
                ):
                    return None
                permute_count += 1
            chain.append(node)
            node = node.args[0]
        if permute_count != 1:
            return None
        return node, tuple(chain)

    def _match_add_chain(self, conv: Node) -> _AddChainMatch | None:
        if (
            conv.op != "call_function"
            or conv.target != exir_ops.edge.aten.convolution.default
            or len(conv.args) < 9
            or bool(conv.args[6])
            or not isinstance(conv.args[0], Node)
        ):
            return None

        traced = self._trace_layout_source(conv.args[0])
        if traced is None:
            return None
        add_output, layout_chain = traced
        add_output_rescale = _RescaleParams.from_node(add_output)
        if (
            add_output_rescale is None
            or add_output_rescale.output_dtype != torch.int8
            or not isinstance(add_output.args[0], Node)
        ):
            return None

        add = add_output.args[0]
        if add.target != exir_ops.edge.aten.add.Tensor or len(add.users) != 1:
            return None
        add_operands = self._binary_operands(add)
        if add_operands is None:
            return None
        beta_operands = [
            operand for operand in add_operands if operand.constant is not None
        ]
        if len(beta_operands) != 1:
            return None
        beta_operand = beta_operands[0]
        activation_operand = next(
            operand for operand in add_operands if operand is not beta_operand
        )
        return _AddChainMatch(
            output=add_output,
            output_rescale=add_output_rescale,
            activation_operand=activation_operand,
            beta_operand=beta_operand,
            layout_chain=layout_chain,
        )

    def _match_gamma_chain(self, gamma_output: Node) -> _GammaChainMatch | None:
        gamma_output_rescale = _RescaleParams.from_node(gamma_output)
        if (
            gamma_output_rescale is None
            or gamma_output_rescale.output_dtype != torch.int8
            or len(gamma_output.users) != 1
            or not isinstance(gamma_output.args[0], Node)
        ):
            return None

        mul = gamma_output.args[0]
        if mul.target != exir_ops.edge.aten.mul.Tensor or len(mul.users) != 1:
            return None
        mul_operands = self._binary_operands(mul)
        if mul_operands is None:
            return None
        gamma_operands = [
            operand for operand in mul_operands if operand.constant is not None
        ]
        if len(gamma_operands) != 1:
            return None
        gamma_operand = gamma_operands[0]
        table_operand = next(
            operand for operand in mul_operands if operand is not gamma_operand
        )
        return _GammaChainMatch(
            output=gamma_output,
            output_rescale=gamma_output_rescale,
            table_operand=table_operand,
            gamma_operand=gamma_operand,
        )

    @staticmethod
    def _table_values_node(table: Node) -> Node | None:
        """Return the node holding a TOSA TABLE's lookup values.

        Returns ``None`` when ``table`` is not a TABLE or does not carry its
        values as a node, which also narrows the operand for the caller.

        """
        if (
            table.target != exir_ops.backend.tosa.TABLE.default
            or len(table.args) < 2
            or not isinstance(table.args[1], Node)
        ):
            return None
        return table.args[1]

    def _affine_codes(
        self,
        table: Node,
        table_values_node: Node,
        gamma_match: _GammaChainMatch,
        add_match: _AddChainMatch,
    ) -> _AffineCodes | None:
        """Collect the INT8 table, gamma and beta constants of an affine map.

        Returns ``None`` unless every constant is present, INT8, and sized as
        the fold requires: a 256-entry table and matching gamma/beta lengths.

        """
        table_values = self._constant(table_values_node)
        table_qargs = self._single_qargs(table, "output_qparams")
        gamma_codes = gamma_match.gamma_operand.constant
        beta_codes = add_match.beta_operand.constant
        if (
            table_values is None
            or table_values.dtype != torch.int8
            or table_values.numel() != 256
            or table_qargs is None
            or table_qargs.dtype != torch.int8
            or gamma_codes is None
            or gamma_codes.dtype != torch.int8
            or beta_codes is None
            or beta_codes.dtype != torch.int8
            or gamma_codes.numel() != beta_codes.numel()
        ):
            return None
        return _AffineCodes(
            table_values=table_values,
            table_qargs=table_qargs,
            gamma_codes=gamma_codes,
            beta_codes=beta_codes,
        )

    @staticmethod
    def _views_are_channel_broadcasts(
        gamma_match: _GammaChainMatch,
        add_match: _AddChainMatch,
        channels: int,
    ) -> bool:
        """Return True when only gamma/beta carry the channel broadcast view."""
        channel_view_shape = (1, 1, 1, channels)
        return (
            gamma_match.table_operand.view_shape is None
            and add_match.activation_operand.view_shape is None
            and gamma_match.gamma_operand.view_shape == channel_view_shape
            and add_match.beta_operand.view_shape == channel_view_shape
        )

    @staticmethod
    def _table_shape_matches(
        table_shape: tuple[int, ...] | None,
        add_output_shape: tuple[int, ...] | None,
        channels: int,
    ) -> bool:
        """Return True when the TABLE is NHWC and channel-aligned with gamma."""
        return (
            table_shape is not None
            and table_shape == add_output_shape
            and len(table_shape) == 4
            and table_shape[-1] == channels
        )

    def _match(self, conv: Node) -> _DyTAffineMatch | None:
        add_match = self._match_add_chain(conv)
        if add_match is None:
            return None
        gamma_match = self._match_gamma_chain(add_match.activation_operand.source)
        if gamma_match is None:
            return None

        table = gamma_match.table_operand.source
        table_values_node = self._table_values_node(table)
        if table_values_node is None:
            return None

        codes = self._affine_codes(table, table_values_node, gamma_match, add_match)
        if codes is None:
            return None
        channels = codes.gamma_codes.numel()
        if not self._views_are_channel_broadcasts(gamma_match, add_match, channels):
            return None
        if not self._table_shape_matches(
            self._tensor_shape(table),
            self._tensor_shape(add_match.output),
            channels,
        ):
            return None

        return _DyTAffineMatch(
            table=table,
            table_values=codes.table_values,
            table_qargs=codes.table_qargs,
            gamma_activation_rescale=gamma_match.table_operand.rescale,
            gamma=gamma_match.gamma_operand.source,
            gamma_codes=codes.gamma_codes,
            gamma_rescale=gamma_match.gamma_operand.rescale,
            gamma_output=gamma_match.output,
            gamma_output_rescale=gamma_match.output_rescale,
            add_activation_rescale=add_match.activation_operand.rescale,
            beta=add_match.beta_operand.source,
            beta_codes=codes.beta_codes,
            beta_rescale=add_match.beta_operand.rescale,
            add_output=add_match.output,
            add_output_rescale=add_match.output_rescale,
            conv=conv,
            layout_chain=add_match.layout_chain,
        )

    @staticmethod
    def _checked_int32(values: torch.Tensor) -> torch.Tensor | None:
        limits = torch.iinfo(torch.int32)
        if values.numel() and (
            int(values.min()) < limits.min or int(values.max()) > limits.max
        ):
            return None
        return values.to(torch.int32)

    def _gamma_outputs(self, match: _DyTAffineMatch) -> torch.Tensor | None:
        table_codes = match.table_values.reshape(-1, 1)
        activation_i32 = _apply_tosa_rescale(
            table_codes,
            match.gamma_activation_rescale,
        )
        gamma_i32 = _apply_tosa_rescale(
            match.gamma_codes.reshape(1, -1),
            match.gamma_rescale,
        )
        product = self._checked_int32(
            activation_i32.to(torch.int64) * gamma_i32.to(torch.int64)
        )
        if product is None:
            return None
        return _apply_tosa_rescale(product, match.gamma_output_rescale)

    def _affine_outputs(
        self,
        match: _DyTAffineMatch,
        gamma_outputs: torch.Tensor,
    ) -> torch.Tensor | None:
        activation_i32 = _apply_tosa_rescale(
            gamma_outputs,
            match.add_activation_rescale,
        )
        beta_i32 = _apply_tosa_rescale(
            match.beta_codes.reshape(1, -1),
            match.beta_rescale,
        )
        summed = self._checked_int32(
            activation_i32.to(torch.int64) + beta_i32.to(torch.int64)
        )
        if summed is None:
            return None
        return _apply_tosa_rescale(summed, match.add_output_rescale)

    @staticmethod
    def _fit_integer_affine(
        input_codes: torch.Tensor,
        output_codes: torch.Tensor,
        *,
        input_zp: int,
        output_zp: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        centered_inputs = input_codes.to(torch.int64).reshape(-1) - input_zp
        centered_outputs = output_codes.to(torch.int64) - output_zp
        slopes = []
        offsets = []

        for channel in range(centered_outputs.shape[1]):
            mapping: dict[int, int] = {}
            for row in range(centered_inputs.numel()):
                x = int(centered_inputs[row].item())
                y = int(centered_outputs[row, channel].item())
                previous = mapping.get(x)
                if previous is not None and previous != y:
                    return None
                mapping[x] = y

            points = sorted(mapping.items())
            if len(points) == 1:
                slope = 0
                offset = points[0][1]
            else:
                x0, y0 = points[0]
                x1, y1 = points[1]
                dx = x1 - x0
                dy = y1 - y0
                if dy % dx != 0:
                    return None
                slope = dy // dx
                offset = y0 - slope * x0

            if any(y != slope * x + offset for x, y in points):
                return None
            slopes.append(slope)
            offsets.append(offset)

        return (
            torch.tensor(slopes, dtype=torch.int64),
            torch.tensor(offsets, dtype=torch.int64),
        )

    @staticmethod
    def _has_padding(conv: Node) -> bool:
        padding = conv.args[4]
        if not isinstance(padding, (list, tuple)):
            return True
        for value in padding:
            if not isinstance(value, int) or value != 0:
                return True
        return False

    @staticmethod
    def _exclusive_conv_input(match: _DyTAffineMatch) -> bool:
        expected_user = match.conv
        for node in match.layout_chain:
            if set(node.users) != {expected_user}:
                return False
            expected_user = node
        return set(match.add_output.users) == {expected_user}

    @staticmethod
    def _weight_zero_points(
        weight_qargs: QuantArgs,
        out_channels: int,
        weight_dim: int,
    ) -> torch.Tensor | None:
        if weight_qargs.per_channel:
            if weight_qargs.axis != 0:
                return None
            zero_points = weight_qargs.get_zp_per_channel()
            if len(zero_points) != out_channels:
                return None
            return torch.tensor(zero_points, dtype=torch.int64).reshape(
                (out_channels,) + (1,) * (weight_dim - 1)
            )
        return torch.tensor(
            weight_qargs.get_zp_per_tensor(),
            dtype=torch.int64,
        )

    def _create_constant(
        self,
        graph: Graph,
        original: Node,
        *,
        name: str,
        data: torch.Tensor,
    ) -> Node:
        kind = get_constant_placeholder_kind(self.exported_program, original)
        persistent_buffer = is_persistent_buffer(self.exported_program, original)
        with graph.inserting_before(original):
            return create_constant_placeholder(
                self.exported_program,
                graph=graph,
                name=name,
                kind=kind,
                data=data,
                persistent_buffer=persistent_buffer,
            )

    def _conv_operands(self, match: _DyTAffineMatch) -> _ConvOperands | None:
        """Read the conv's weight/bias constants and quantization arguments.

        Returns ``None`` unless the conv carries INT8 weights, an INT32 bias,
        and per-tensor INT8 activation qparams whose zero point already agrees
        with the affine output rescale.

        """
        conv = match.conv
        if not isinstance(conv.args[1], Node) or not isinstance(conv.args[2], Node):
            return None
        weight_node = conv.args[1]
        bias_node = conv.args[2]
        weight = self._constant(weight_node)
        bias = self._constant(bias_node)
        input_qparams = cast(dict[int, QuantArgs], conv.meta.get("input_qparams", {}))
        activation_qargs = input_qparams.get(0)
        weight_qargs = input_qparams.get(1)
        if (
            weight is None
            or weight.dtype != torch.int8
            or weight.dim() != 4
            or bias is None
            or bias.dtype != torch.int32
            or bias.dim() != 1
        ):
            return None
        if (
            activation_qargs is None
            or activation_qargs.per_channel
            or activation_qargs.dtype != torch.int8
            or activation_qargs.get_zp_per_tensor()
            != match.add_output_rescale.output_zp
            or weight_qargs is None
            or weight_qargs.dtype != torch.int8
        ):
            return None
        return _ConvOperands(
            weight_node=weight_node,
            bias_node=bias_node,
            weight=weight,
            bias=bias,
            input_qparams=input_qparams,
            activation_qargs=activation_qargs,
            weight_qargs=weight_qargs,
        )

    def _validated_conv_constants(
        self,
        match: _DyTAffineMatch,
        slopes: torch.Tensor,
        offsets: torch.Tensor,
    ) -> _ConvConstants | None:
        """Check the conv grouping and channel counts against the affine map.

        Returns ``None`` when the group layout is unusable or when the per
        channel slopes/offsets do not line up with the conv's input channels.

        """
        operands = self._conv_operands(match)
        if operands is None:
            return None
        groups = match.conv.args[8]
        if not isinstance(groups, int) or groups <= 0:
            return None
        weight = operands.weight
        out_channels = weight.shape[0]
        in_channels_per_group = weight.shape[1]
        in_channels = in_channels_per_group * groups
        if (
            slopes.numel() != in_channels
            or offsets.numel() != in_channels
            or out_channels % groups != 0
            or operands.bias.numel() != out_channels
        ):
            return None
        weight_zero_points = self._weight_zero_points(
            operands.weight_qargs,
            out_channels,
            weight.dim(),
        )
        if weight_zero_points is None:
            return None
        return _ConvConstants(
            operands=operands,
            groups=groups,
            out_channels=out_channels,
            in_channels_per_group=in_channels_per_group,
            weight_zero_points=weight_zero_points,
        )

    def _fold_conv_constants(
        self,
        graph: Graph,
        match: _DyTAffineMatch,
        slopes: torch.Tensor,
        offsets: torch.Tensor,
    ) -> bool:
        conv = match.conv
        constants = self._validated_conv_constants(match, slopes, offsets)
        if constants is None:
            return False
        operands = constants.operands
        weight_node = operands.weight_node
        bias_node = operands.bias_node
        weight = operands.weight
        bias = operands.bias
        input_qparams = operands.input_qparams
        activation_qargs = operands.activation_qargs
        weight_qargs = operands.weight_qargs
        groups = constants.groups
        out_channels = constants.out_channels
        in_channels_per_group = constants.in_channels_per_group
        weight_zero_points = constants.weight_zero_points

        out_channels_per_group = out_channels // groups
        output_groups = torch.arange(out_channels, dtype=torch.int64).div(
            out_channels_per_group,
            rounding_mode="floor",
        )
        local_inputs = torch.arange(in_channels_per_group, dtype=torch.int64)
        global_inputs = output_groups.reshape(
            -1, 1
        ) * in_channels_per_group + local_inputs.reshape(1, -1)
        broadcast_shape = (
            out_channels,
            in_channels_per_group,
            *([1] * (weight.dim() - 2)),
        )
        channel_slopes = slopes[global_inputs].reshape(broadcast_shape)
        channel_offsets = offsets[global_inputs].reshape(broadcast_shape)

        centered_weight = weight.to(torch.int64) - weight_zero_points
        folded_centered_weight = centered_weight * channel_slopes
        folded_weight_i64 = folded_centered_weight + weight_zero_points
        if folded_weight_i64.numel() and (
            int(folded_weight_i64.min()) < weight_qargs.qmin
            or int(folded_weight_i64.max()) > weight_qargs.qmax
        ):
            return False

        correction_dims = tuple(range(1, centered_weight.dim()))
        bias_correction = (centered_weight * channel_offsets).sum(dim=correction_dims)
        folded_bias_i64 = bias.to(torch.int64) + bias_correction
        int32_limits = torch.iinfo(torch.int32)
        if folded_bias_i64.numel() and (
            int(folded_bias_i64.min()) < int32_limits.min
            or int(folded_bias_i64.max()) > int32_limits.max
        ):
            return False

        folded_weight = folded_weight_i64.to(torch.int8)
        folded_bias = folded_bias_i64.to(torch.int32)
        new_weight_node = weight_node
        new_bias_node = bias_node
        if not torch.equal(folded_weight, weight):
            new_weight_node = self._create_constant(
                graph,
                weight_node,
                name=f"{weight_node.name}_{conv.name}_dyt_affine_folded",
                data=folded_weight,
            )
        if not torch.equal(folded_bias, bias):
            new_bias_node = self._create_constant(
                graph,
                bias_node,
                name=f"{bias_node.name}_{conv.name}_dyt_affine_folded",
                data=folded_bias,
            )

        conv.args = (
            conv.args[0],
            new_weight_node,
            new_bias_node,
            *conv.args[3:],
        )
        for original, replacement in (
            (weight_node, new_weight_node),
            (bias_node, new_bias_node),
        ):
            if original is not replacement and len(original.users) == 0:
                delete_constant_placeholder(self.exported_program, original)
        updated_qparams = copy(input_qparams)
        # Keep the original convolution activation scale: the folded integer
        # weights and bias preserve that accumulator domain. Only the source
        # zero point changes when the input is rewired to the TABLE.
        updated_qparams[0] = QuantArgs(
            scale=activation_qargs.scale,
            zp=match.table_qargs.get_zp_per_tensor(),
            qmin=activation_qargs.qmin,
            qmax=activation_qargs.qmax,
            dtype=activation_qargs.dtype,
            axis=activation_qargs.axis,
            per_channel=False,
        )
        conv.meta["input_qparams"] = updated_qparams
        return True

    def _fold_unpadded(
        self,
        graph: Graph,
        match: _DyTAffineMatch,
        affine_outputs: torch.Tensor,
    ) -> bool:
        if not self._exclusive_conv_input(match):
            return False
        fitted = self._fit_integer_affine(
            match.table_values,
            affine_outputs,
            input_zp=match.table_qargs.get_zp_per_tensor(),
            output_zp=match.add_output_rescale.output_zp,
        )
        if fitted is None:
            return False
        slopes, offsets = fitted
        if not self._fold_conv_constants(graph, match, slopes, offsets):
            return False
        match.add_output.replace_all_uses_with(match.table)
        return True

    @staticmethod
    def _affine_is_identity(
        match: _DyTAffineMatch,
        affine_outputs: torch.Tensor,
    ) -> bool:
        expected = match.table_values.reshape(-1, 1).expand_as(affine_outputs)
        return torch.equal(affine_outputs, expected)

    @staticmethod
    def _gamma_is_identity(
        match: _DyTAffineMatch,
        gamma_outputs: torch.Tensor,
    ) -> bool:
        expected = match.table_values.reshape(-1, 1).expand_as(gamma_outputs)
        return torch.equal(gamma_outputs, expected)

    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False
        constants_to_delete: set[Node] = set()
        for node in list(graph.nodes):
            match = self._match(node)
            if match is None:
                continue
            gamma_outputs = self._gamma_outputs(match)
            if gamma_outputs is None:
                continue

            affine_outputs = self._affine_outputs(match, gamma_outputs)
            if affine_outputs is None:
                continue

            folded = False
            if not self._has_padding(match.conv):
                folded = self._fold_unpadded(graph, match, affine_outputs)
            if folded:
                constants_to_delete.update((match.gamma, match.beta))
            # Identity here means equality of the emitted INT8 codes. Keep the
            # consumers' original zero points: changing them to the TABLE zero
            # point would reinterpret the same bytes and alter convolution
            # padding or downstream RESCALE arithmetic.
            elif self._affine_is_identity(match, affine_outputs):
                match.add_output.replace_all_uses_with(match.table)
                constants_to_delete.update((match.gamma, match.beta))
                folded = True
            elif self._gamma_is_identity(match, gamma_outputs):
                match.gamma_output.replace_all_uses_with(match.table)
                constants_to_delete.add(match.gamma)
                folded = True
            modified = modified or folded

        if modified:
            graph.eliminate_dead_code()
            for constant in constants_to_delete:
                if constant.op == "placeholder" and len(constant.users) == 0:
                    delete_constant_placeholder(self.exported_program, constant)
            graph.lint()
            graph_module.recompile()
        return PassResult(graph_module, modified)
