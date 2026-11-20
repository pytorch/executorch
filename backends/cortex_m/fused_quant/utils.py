# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared pieces of the Cortex-M fused_quant lowering."""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast, Optional

import torch
from executorch.backends.cortex_m.target_config import CortexMTargetConfig
from executorch.backends.cortex_m.utils import quantize_multiplier_aot
from executorch.backends.transforms.fused_quant.graph_utils import get_qparams_from_node
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.backends.transforms.utils import (
    add_constant,
    compute_meta_val,
    get_constant,
)
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.passes import make_alloc_node
from torch import fx
from torch._export.utils import _detect_fake_mode_from_gm
from torch._subclasses.fake_tensor import FakeTensor
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind
from torch.fx.node import Argument

INT8_MIN, INT8_MAX = -128, 127


@dataclass(frozen=True)
class QParams:
    """The per-tensor qparams of one fused op operand."""

    scale: float
    zero_point: int
    quant_min: int
    quant_max: int
    dtype: torch.dtype = torch.int8

    @property
    def is_full_range(self) -> bool:
        """Whether no activation is folded into the range, which kernels without
        an activation range need: they saturate to the dtype's range.

        A symmetric range counts, though the kernel can then write the dtype's
        most negative value, one step past it, for a value beyond the
        calibrated range.
        """
        info = torch.iinfo(self.dtype)
        return self.quant_max == info.max and self.quant_min in (info.min, info.min + 1)

    @property
    def grid(self) -> tuple[float, int]:
        return self.scale, self.zero_point

    def quantize(self, value: float) -> int:
        """``value`` on this grid, saturated to the range, rounded as the fused
        ops round."""
        if math.isinf(value):
            return self.quant_min if value < 0 else self.quant_max
        return int(
            torch.ops.torchao.quantize_affine(
                torch.tensor([value]),
                [1],
                torch.tensor([self.scale]),
                torch.tensor([self.zero_point]),
                self.dtype,
                self.quant_min,
                self.quant_max,
            ).item()
        )

    def dequantize(self, value: int) -> float:
        return (value - self.zero_point) * self.scale


@dataclass(frozen=True)
class Context:
    exported_program: ExportedProgram
    target_config: CortexMTargetConfig

    def constant(self, arg: object) -> Optional[torch.Tensor]:
        if not isinstance(arg, fx.Node):
            return None
        return get_constant(self.exported_program, arg)

    def qparams(
        self,
        node: fx.Node,
        prefix: str,
        dtypes: tuple[torch.dtype, ...] = (torch.int8,),
    ) -> Optional[QParams]:
        """``prefix``'s qparams if they are per-tensor constants of one of
        ``dtypes``, else None.

        An input's qparams carry its dequantized dtype, so its dtype is checked
        on the tensor itself; the output's carry the quantized one.
        """
        qparams = get_qparams_from_node(node, prefix)
        if qparams is None or not qparams.is_per_tensor():
            return None
        dtype = (
            qparams.dtype
            if prefix == "out"
            else cast(fx.Node, get_arg(node, prefix)).meta["val"].dtype
        )
        if dtype not in dtypes:
            return None
        scale = self.constant(qparams.scale)
        zero_point = self.constant(qparams.zero_point)
        if scale is None or zero_point is None:
            return None
        return QParams(
            float(scale.item()),
            int(zero_point.item()),
            qparams.quant_min,
            qparams.quant_max,
            dtype,
        )

    def weight_scales(
        self, node: fx.Node, axis: int, channels: int
    ) -> Optional[torch.Tensor]:
        """The per-output-channel scales of a symmetric int8 constant weight.

        A per-tensor scale is repeated for every channel. None if the weight is
        not a constant or not int8 symmetric per tensor or per ``axis``.
        """
        qparams = get_qparams_from_node(node, "weight")
        weight = get_arg(node, "weight", fx.Node)
        constant = self.constant(weight)
        if qparams is None or constant is None or constant.dtype != torch.int8:
            return None
        scale = self.constant(qparams.scale)
        zero_point = self.constant(qparams.zero_point)
        if scale is None or zero_point is None or bool(torch.any(zero_point != 0)):
            return None
        if scale.numel() == 1:
            return scale.double().reshape(1).expand(channels)
        if (
            not qparams.is_per_channel(weight.meta["val"].shape)
            or qparams.channel_axis() != axis
        ):
            return None
        return scale.double().reshape(-1)

    def int32_bias(
        self, node: fx.Node, input_scale: float, weight_scales: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """The bias, quantized to int32 at the accumulator scale.

        A bias already quantized at that scale comes back unchanged.
        """
        bias = self.constant(get_arg(node, "bias"))
        if bias is None:
            return None
        qparams = get_qparams_from_node(node, "bias")
        value = bias.double()
        if qparams is not None:
            scale = self.constant(qparams.scale)
            zero_point = self.constant(qparams.zero_point)
            assert scale is not None and zero_point is not None
            value = (value - zero_point.double()) * scale.double()
        int32 = torch.iinfo(torch.int32)
        return (
            torch.round(value / (input_scale * weight_scales))
            .clamp(int32.min, int32.max)
            .to(torch.int32)
        )

    def has_constant_bias(self, node: fx.Node) -> bool:
        bias: Argument = get_arg(node, "bias")
        return bias is None or self.constant(bias) is not None

    def add_constant(self, node: fx.Node, suffix: str, tensor: torch.Tensor) -> fx.Node:
        return add_constant(
            self.exported_program,
            f"{node.name}_{suffix}",
            tensor,
            node,
            InputKind.PARAMETER,
        )

    def scratch(self, node: fx.Node) -> fx.Node:
        """An empty scratch buffer; ``InitializeScratchBuffersPass`` sizes it."""
        graph_module = self.exported_program.graph_module
        fake_mode = _detect_fake_mode_from_gm(graph_module)
        assert fake_mode is not None, "graph has no fake tensor mode"
        with fake_mode:
            val = cast(FakeTensor, torch.empty(0, dtype=torch.uint8))
        with graph_module.graph.inserting_before(node):
            return make_alloc_node(graph_module, val, None)

    def call(
        self, node: fx.Node, target: EdgeOpOverload, args: Sequence[Argument]
    ) -> fx.Node:
        """Insert ``target(*args)`` ahead of ``node``, keeping node's meta."""
        with node.graph.inserting_before(node):
            new = node.graph.call_function(target, tuple(args))
        new.meta = {
            key: value for key, value in node.meta.items() if key != "tensor_meta"
        }
        new.meta["val"] = compute_meta_val(new)
        return new

    def replace(
        self, node: fx.Node, target: EdgeOpOverload, args: Sequence[Argument]
    ) -> None:
        node.replace_all_uses_with(self.call(node, target, args))
        node.graph.erase_node(node)


@dataclass(frozen=True)
class Rule:
    """Whether CMSIS-NN can run a fused_quant node, and how to rewrite it."""

    can_lower: Callable[[Context, fx.Node], bool]
    lower: Callable[[Context, fx.Node], None]


Rules = dict[EdgeOpOverload, Rule]


def shape(arg: object) -> torch.Size:
    assert isinstance(arg, fx.Node)
    return arg.meta["val"].shape


def requantize_params(
    input_scale: float, weight_scales: torch.Tensor, output_scale: float
) -> tuple[list[int], list[int]]:
    """The fixed-point multiplier and shift of each output channel."""
    multipliers: list[int] = []
    shifts: list[int] = []
    for weight_scale in weight_scales.tolist():
        multiplier, shift = quantize_multiplier_aot(
            input_scale * weight_scale / output_scale
        )
        multipliers.append(multiplier)
        shifts.append(shift)
    return multipliers, shifts


def requantize_table(inp: QParams, out: QParams) -> torch.Tensor:
    """The 256-entry ``quantized_activation`` table moving int8 from inp's grid
    to out's, exactly as ``fused_quant.requantize`` rounds."""

    def block(qparams: QParams, dtype: torch.dtype) -> tuple[object, ...]:
        return (
            torch.tensor(qparams.scale),
            torch.tensor(qparams.zero_point),
            dtype,
            qparams.quant_min,
            qparams.quant_max,
        )

    return torch.ops.fused_quant.requantize(
        torch.arange(INT8_MIN, INT8_MAX + 1).to(torch.int8),
        *block(inp, torch.float32),
        *block(out, torch.int8),
    )
