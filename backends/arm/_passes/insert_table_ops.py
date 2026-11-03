# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from itertools import chain
from typing import Callable, cast, Dict, Iterator, Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import (
    create_node,
    POW_LOG_POSITIVE_LOWER_BOUND_META,
)
from executorch.backends.arm._passes.quant_args import QuantArgs
from executorch.backends.transforms.fuse_elementwise_to_lut import (
    FuseElementwiseToLUTPass,
    FusionRegion,
)

from executorch.exir import ExportedProgram

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload

from executorch.exir.pass_base import ExportPass
from torch.fx.node import map_arg, Node


class TableOps:
    """Helper class for finding the corresponding table operator for a given
    Node.
    """

    # Targets that follow a straigtforward one-to-one mapping to their table op
    unary_table_ops: Dict[EdgeOpOverload, Callable[[torch.Tensor], torch.Tensor]] = {
        exir_ops.edge.aten.ceil.default: torch.ceil,
        exir_ops.edge.aten.erf.default: torch.erf,
        exir_ops.edge.aten.exp.default: torch.exp,
        exir_ops.edge.aten.expm1.default: torch.expm1,
        exir_ops.edge.aten.erfinv.default: torch.erfinv,
        exir_ops.edge.aten.floor.default: torch.floor,
        exir_ops.edge.aten.log.default: torch.log,
        exir_ops.edge.aten.log1p.default: torch.log1p,
        exir_ops.edge.aten.log10.default: torch.log10,
        exir_ops.edge.aten.reciprocal.default: torch.reciprocal,
        exir_ops.edge.aten.rsqrt.default: torch.rsqrt,
        exir_ops.edge.aten.sigmoid.default: torch.sigmoid,
        exir_ops.edge.aten.cos.default: torch.cos,
        exir_ops.edge.aten.sin.default: torch.sin,
        exir_ops.edge.aten.tanh.default: torch.tanh,
        exir_ops.edge.aten.atan.default: torch.atan,
        exir_ops.edge.aten.atanh.default: torch.atanh,
        exir_ops.edge.aten.hardsigmoid.default: torch.nn.functional.hardsigmoid,
        exir_ops.edge.aten.hardswish.default: torch.nn.functional.hardswish,
        exir_ops.edge.aten.sinh.default: torch.sinh,
        exir_ops.edge.aten.acosh.default: torch.acosh,
        exir_ops.edge.aten.asin.default: torch.asin,
        exir_ops.edge.aten.asinh.default: torch.asinh,
        exir_ops.edge.aten.cosh.default: torch.cosh,
        exir_ops.edge.aten.acos.default: torch.acos,
        exir_ops.edge.aten.tan.default: torch.tan,
        exir_ops.edge.aten.silu.default: torch.nn.functional.silu,
        exir_ops.edge.aten.round.default: torch.round,
    }

    # Targets that must be treated explicitly
    special_table_ops: Set[EdgeOpOverload] = {
        exir_ops.edge.aten.softplus.default,
        exir_ops.edge.aten.pow.Tensor_Scalar,
        exir_ops.edge.aten.gelu.default,
        exir_ops.edge.aten.elu.default,
        exir_ops.edge.aten.leaky_relu.default,
        exir_ops.edge.aten.remainder.Scalar,
    }

    def __init__(self, exported_program: ExportedProgram):
        self.exported_program = exported_program

    def __contains__(self, node: Node) -> bool:
        return (
            node.target in self.unary_table_ops or node.target in self.special_table_ops
        )

    def __getitem__(self, node: Node):
        target = cast(EdgeOpOverload, node.target)
        if target in self.unary_table_ops:
            return self.unary_table_ops[target]
        elif target in self.special_table_ops:
            match target:
                case exir_ops.edge.aten.softplus.default:
                    beta = cast(
                        float,
                        (
                            node.args[1]
                            if len(node.args) > 1
                            else node.kwargs.get("beta", 1)
                        ),
                    )
                    threshold = cast(
                        float,
                        (
                            node.args[2]
                            if len(node.args) > 2
                            else node.kwargs.get("threshold", 20)
                        ),
                    )
                    return lambda x: torch.nn.functional.softplus(
                        x, beta=beta, threshold=threshold
                    ).flatten()
                case exir_ops.edge.aten.pow.Tensor_Scalar:
                    # Exponent is a constant. Embed it into a lambda.
                    exp = cast(int, node.args[1])
                    return lambda x: torch.pow(x, exp).flatten()
                case exir_ops.edge.aten.gelu.default:
                    # If kwargs not present it is default "none"
                    approximate = cast(
                        str,
                        (
                            node.kwargs["approximate"]
                            if "approximate" in node.kwargs
                            else "none"
                        ),
                    )
                    return lambda x: torch.nn.functional.gelu(
                        x, approximate=approximate
                    ).flatten()
                case exir_ops.edge.aten.elu.default:
                    input_alpha = cast(float, node.meta["float_alpha"])
                    input_scale = cast(float, node.meta.get("float_input_scale", 1.0))
                    scale = cast(float, node.meta.get("float_scale", 1.0))
                    return lambda x: torch.ops.aten.elu.default(
                        x, input_alpha, scale, input_scale
                    ).flatten()
                case exir_ops.edge.aten.leaky_relu.default:
                    negative_slope = cast(
                        float,
                        (
                            node.args[1]
                            if len(node.args) > 1
                            else node.kwargs.get("negative_slope", 0.01)
                        ),
                    )
                    return lambda x: torch.nn.functional.leaky_relu(
                        x, negative_slope=negative_slope
                    ).flatten()
                case exir_ops.edge.aten.remainder.Scalar:
                    divisor = cast(float | int, node.args[1])
                    return lambda x: torch.remainder(x, divisor).flatten()
                case _:
                    # Op must be handled if it's inside self.special_ops
                    raise AssertionError("Unhandled table operation")
        else:
            raise KeyError(f"Table op for {target} does not exist")

    @staticmethod
    def included_ops() -> Iterator[EdgeOpOverload]:
        return chain(TableOps.unary_table_ops, TableOps.special_table_ops)


class InsertTableOpsPass(FuseElementwiseToLUTPass, ArmPass):
    """Lower quantized pointwise regions directly to INT8 or INT16 TOSA
    TABLE.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()
    quantize_op = exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
    dequantize_op = exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default

    arithmetic_ops = {
        exir_ops.edge.aten.add.Tensor,
        exir_ops.edge.aten.sub.Tensor,
        exir_ops.edge.aten.mul.Tensor,
        exir_ops.edge.aten.div.Tensor,
        exir_ops.edge.aten.clamp.default,
        exir_ops.edge.aten.minimum.default,
        exir_ops.edge.aten.maximum.default,
    }

    def __init__(self, exported_program: ExportedProgram) -> None:
        super().__init__(
            exported_program,
            exir_ops.backend.tosa.TABLE.default,
            {
                *TableOps.included_ops(),
                *self.arithmetic_ops,
                self.quantize_op,
                self.dequantize_op,
            },
            supported_dtypes=(
                torch.int8,
                torch.int16,
                torch.int32,
                torch.float32,
                torch.float64,
            ),
            min_region_size=1,
        )
        self.table_ops = TableOps(exported_program)
        self.table_targets = set(TableOps.included_ops())

    def _is_supported_node(self, node: Node) -> bool:
        if node.target == self.table_op or not super()._is_supported_node(node):
            return False
        if node.target in (self.quantize_op, self.dequantize_op):
            if node.target == self.quantize_op:
                source = node.args[0]
                if (
                    source.target == self.dequantize_op
                    and source.args[5] != node.args[5]
                ):
                    # Keep dtype-changing requantization outside the LUT domain.
                    return False
            return not any(isinstance(arg, Node) for arg in node.args[1:]) and (
                node.target != self.quantize_op or self._is_supported_node(node.args[0])
            )
        inputs = node.meta.get("input_qparams", {})
        outputs = node.meta.get("output_qparams", {})
        if not inputs and not outputs:
            return node.meta["val"].is_floating_point()
        return (
            bool(inputs)
            and set(outputs) == {0}
            and all(
                not qparams.per_channel
                for qparams in (*inputs.values(), *outputs.values())
            )
            and all(
                index in inputs or self._constant(arg) is not None
                for index, arg in enumerate(node.args)
                if isinstance(arg, Node)
            )
        )

    def _region_input_qparams(self, region: FusionRegion) -> list[QuantArgs]:
        qparams = []
        for node in region.nodes:
            for index, arg in enumerate(node.args):
                if arg is not region.input_node:
                    continue
                if node.target == self.dequantize_op:
                    qparams.append(QuantArgs.from_operator(node.target, node.args))
                elif index in node.meta.get("input_qparams", {}):
                    qparams.append(node.meta["input_qparams"][index])
                else:
                    return []
        return qparams

    def _region_output_qparams(self, region: FusionRegion) -> dict[int, QuantArgs]:
        sink = region.nodes[-1]
        if sink.target == self.quantize_op:
            return {0: QuantArgs.from_operator(sink.target, sink.args)}
        return sink.meta.get("output_qparams", {})

    def _find_fusion_region(self, sink: Node) -> FusionRegion | None:
        region = super()._find_fusion_region(sink)
        if region is None or (
            len(region.nodes) == 1 and sink.target not in self.table_targets
        ):
            return None
        pointwise_ops = sum(
            node.target not in (self.quantize_op, self.dequantize_op)
            for node in region.nodes
        )
        if pointwise_ops == 0:
            return None
        if pointwise_ops == 1:
            table_index = next(
                (
                    index
                    for index, node in enumerate(region.nodes)
                    if node.target in self.table_targets
                ),
                None,
            )
            if table_index is not None:
                for index in range(table_index - 1, -1, -1):
                    if region.nodes[index].target == self.quantize_op:
                        # A lone TABLE already lowers the operator; keep its
                        # input requantization visible near singularities.
                        region = FusionRegion(
                            region.nodes[index + 1 :], region.nodes[index]
                        )
                        break
        qparams = self._region_input_qparams(region)
        if (
            not qparams
            or region.input_node.meta["val"].dtype != qparams[0].dtype
            or any(value != qparams[0] for value in qparams)
        ):
            return None
        outputs = self._region_output_qparams(region)
        if not outputs or (qparams[0].dtype, outputs[0].dtype) not in (
            (torch.int8, torch.int8),
            (torch.int16, torch.int16),
            (torch.int16, torch.int32),
        ):
            return None
        return region

    def _evaluate_region(
        self, region: FusionRegion, value: torch.Tensor
    ) -> torch.Tensor:
        values = {region.input_node: value.to(torch.float32)}
        for node in region.nodes:
            args = []
            for index, arg in enumerate(node.args):
                if not isinstance(arg, Node):
                    args.append(arg)
                    continue
                constant = self._constant(arg)
                if constant is None:
                    args.append(values[arg])
                    continue
                tensor = constant.reshape(()).to(torch.float32)
                qparams = node.meta.get("input_qparams", {}).get(index)
                if node.target == self.dequantize_op:
                    qparams = QuantArgs.from_operator(node.target, node.args)
                args.append(
                    (tensor - qparams.get_zp_per_tensor())
                    * qparams.get_scale_per_tensor()
                    if qparams
                    else tensor
                )
            # Region inputs are already dequantized by the table generator.
            # Keep internal edges in float until it quantizes the final result.
            values[node] = (
                args[0]
                if node.target in (self.quantize_op, self.dequantize_op)
                else self._evaluate_node(
                    node, args, map_arg(node.kwargs, values.__getitem__)
                )
            )
        return values[region.nodes[-1]]

    def _evaluate_node(self, node: Node, args: list, kwargs: dict) -> torch.Tensor:
        if node in self.table_ops:
            if node.meta.get("input_qparams"):
                self._validate_pow_log_positive_lower_bound(
                    node, node.meta["input_qparams"][0]
                )
            return self.table_ops[node](args[0])
        if node.target == exir_ops.edge.aten.clamp.default and node.meta.get(
            "output_qparams"
        ):
            # QuantizeClampArgumentsPass has already encoded the bounds.
            qparams = node.meta["output_qparams"][0]
            for index in range(1, len(args)):
                if args[index] is not None:
                    args[index] = (
                        args[index] - qparams.get_zp_per_tensor()
                    ) * qparams.get_scale_per_tensor()
            kwargs = dict(kwargs)
            for key in ("min", "max"):
                if kwargs.get(key) is not None:
                    kwargs[key] = (
                        kwargs[key] - qparams.get_zp_per_tensor()
                    ) * qparams.get_scale_per_tensor()
        return node.target(*args, **kwargs)

    @staticmethod
    def _validate_pow_log_positive_lower_bound(
        node: Node, input_qargs: QuantArgs
    ) -> None:
        """Reject POW-generated LOG when quantization destroys positivity.

        The POW decomposition proves positivity in the source floating-point
        graph before observers run. Once the LOG input qparams are available,
        verify that the proven lower bound is still strictly positive after the
        exact QDQ mapping used by the backend.

        """
        custom_meta = node.meta.get("custom", {})
        lower_bound = custom_meta.get(
            POW_LOG_POSITIVE_LOWER_BOUND_META,
            node.meta.get(POW_LOG_POSITIVE_LOWER_BOUND_META),
        )
        if lower_bound is None:
            return

        if isinstance(lower_bound, bool) or not isinstance(lower_bound, (int, float)):
            raise RuntimeError(
                "Invalid POW LOG positive lower-bound metadata: " f"{lower_bound!r}"
            )

        if input_qargs.per_channel:
            raise RuntimeError(
                "Tensor/Tensor POW LOG positivity validation requires "
                "per-tensor activation quantization."
            )

        lower_bound_tensor = torch.tensor([float(lower_bound)], dtype=torch.float32)
        quantized = input_qargs.quantize_value(lower_bound_tensor)
        dequantized = input_qargs.dequantize_value(quantized)

        if bool(torch.all(torch.isfinite(dequantized) & (dequantized > 0)).item()):
            return

        dequantized_lower_bound = float(dequantized.reshape(-1)[0].item())
        raise RuntimeError(
            "Unsafe Tensor/Tensor POW INT decomposition: the structurally "
            f"proven positive base lower bound {float(lower_bound)} becomes "
            f"{dequantized_lower_bound} after LOG input quantization "
            f"(scale={input_qargs.get_scale_per_tensor()}, "
            f"zero_point={input_qargs.get_zp_per_tensor()}). "
            "LOG requires a strictly positive input."
        )

    @staticmethod
    def _get_8bit_table_domain() -> torch.Tensor:
        """Return the canonical 8-bit TOSA TABLE input domain."""
        int8_info = torch.iinfo(torch.int8)
        # torch.arange excludes the end value, so use max + 1 to include 127.
        return torch.arange(
            int8_info.min,
            int8_info.max + 1,
            dtype=torch.int8,
        )

    def generate_8bit_table_values(
        self,
        torch_op: Callable[[torch.Tensor], torch.Tensor],
        in_quantargs: QuantArgs,
        out_quantargs: QuantArgs,
    ) -> tuple[torch.Tensor, int]:
        """Compute LUT values for a INT8 TOSA.TABLE.

        Also returns 0 since no shifting is required after 8bit table. The INT8
        table is a simple 256 value 1-1 LUT.

        """

        def f(x: torch.Tensor) -> torch.Tensor:
            x = in_quantargs.dequantize_value(x)
            x = torch_op(x)
            return out_quantargs.quantize_value(x)

        effective_codes = self._get_8bit_table_domain().clamp(
            in_quantargs.qmin, in_quantargs.qmax
        )
        return (f(effective_codes).to(dtype=torch.int8), 0)

    @staticmethod
    def generate_16_bit_table_values(
        torch_op: Callable[[torch.Tensor], torch.Tensor],
        in_quantargs: QuantArgs,
        out_quantargs: QuantArgs,
    ) -> tuple[torch.Tensor, int]:
        """Compute LUT values for a INT16 TOSA.TABLE with 32 bit output.
        In practice the output is 23 bits that should be interpreted as 16 'whole' bits and 7 fractional bits, see
        the specification: https://github.com/arm/tosa-specification/blob/main/chapters/ewise_binary.adoc#table. This means that the output
        will interpreted as 2**7=128 times too large unless accounted for by rescaling down the table output.

        Quantization can be either int16 or int32 which means that the op output could be larger than the 23 bits from
        the TOSA.TABLE output. In that case, we need to rescale up the output.

        To handle this we need to:
        1) Make sure that our table values fit within 16 bits.
        2) Insert a rescale after the table to handle the x128 from the fractional bits and match the quantization.

        The function returns rescale_lshift which says how much to rescale after the table. This value can negative.
        """

        def f(x: torch.Tensor) -> torch.Tensor:
            int16_max = torch.iinfo(torch.int16).max
            has_full_int16_upper_range = (
                in_quantargs.dtype == torch.int16 and in_quantargs.qmax == int16_max
            )
            x = x.clamp(in_quantargs.qmin, in_quantargs.qmax).to(
                dtype=in_quantargs.dtype
            )
            # Dont use the 7 LSBs.
            x = in_quantargs.dequantize_value((x & ~0x7F))
            if has_full_int16_upper_range:
                # TOSA uses table[512] as the virtual right endpoint when
                # interpolating raw INT16 input codes [32640, 32767]. Evaluate
                # its real value at affine code 32768 without casting to int16.
                x[-1] = (
                    int16_max + 1 - in_quantargs.get_zp_per_tensor()
                ) * in_quantargs.get_scale_per_tensor()
            x = torch_op(x)
            return out_quantargs.quantize_value(x)

        lut_values = f(
            torch.linspace(
                start=in_quantargs.qmin,
                end=in_quantargs.qmax + 1,
                steps=513,
                # use torch.int32 to avoid overflow for end=in_quantargs.qmax + 1.
                dtype=torch.int32,
            )
        )
        # Calculate how much we need to shift table values to fit in 16 signed bits
        # ceil(log2(max absolute table value)) + 1 bit for signedness - 16
        # Example:
        #       Max value in the table is 70 000. We want to fit it in 16 signed bits.
        #       70 000=0b10001000101110000 (17 digits) has ceil(log2(70 000)) = ceil(16.095) = 17 bits.
        #       If we shift it 17-16=1 bit, we do get 16 bits (0b1000100010111000),
        #       but due to signedness this is a negative number! So we need to shift it one more bit.
        # Note: for out_quantargs.dtype=torch.int16, rshift == 0 and rescale_lshift = -7.
        rshift = int(torch.ceil(torch.log2(lut_values.abs().max()))) + 1 - 16
        # When the table values use fewer than 16 bits (e.g. a sigmoid output
        # quantized with a small scale, so the max table value is well below
        # 2**15), the formula above yields a negative rshift. The values already
        # fit in signed int16, and a negative right-shift is undefined (on host it
        # masks the shift count and zeroes the table, giving a degenerate
        # step-function LUT on device). Clamp to 0 so no shift is applied; this is
        # the documented int16 case (rshift == 0, rescale_lshift == -7) and keeps
        # rescale_lshift consistent with the shift actually performed below.
        rshift = max(rshift, 0)
        # The 7 fractional bits are equivalent to a lshift of 7, so subtract 7 from the lshift we do.
        rescale_lshift = rshift - 7
        lut_values = lut_values >> rshift
        return lut_values.to(dtype=torch.int16), rescale_lshift

    def generate_table_values(
        self,
        torch_op: Callable[[torch.Tensor], torch.Tensor],
        in_quantargs: QuantArgs,
        out_quantargs: QuantArgs,
    ) -> tuple[torch.Tensor, int]:
        match out_quantargs.dtype:
            case torch.int8:
                return self.generate_8bit_table_values(
                    torch_op, in_quantargs, out_quantargs
                )
            case torch.int16 | torch.int32:
                return self.generate_16_bit_table_values(
                    torch_op, in_quantargs, out_quantargs
                )
            case _:
                raise ValueError(
                    f"Unsupported output dtype for table: {out_quantargs.dtype}"
                )

    def _fuse_region(self, region: FusionRegion) -> None:
        sink = region.nodes[-1]
        input_qparams = {0: self._region_input_qparams(region)[0]}
        output_qparams = self._region_output_qparams(region)
        table, lshift = self.generate_table_values(
            lambda value: self._evaluate_region(region, value),
            input_qparams[0],
            output_qparams[0],
        )
        table_node = self._insert_table_constant(region, table)
        sink.target = self.table_op
        sink.args = (region.input_node, table_node)
        sink.kwargs = {}
        sink.meta["input_qparams"] = input_qparams
        sink.meta["output_qparams"] = output_qparams
        sink.meta["val"] = torch.empty_like(
            sink.meta["val"], dtype=output_qparams[0].dtype
        )
        if lshift != 0:
            with sink.graph.inserting_before(sink):
                table_op = create_node(sink.graph, self.table_op, args=sink.args)
            table_op.meta["input_qparams"] = input_qparams
            table_op.meta["output_qparams"] = output_qparams
            table_op.meta["val"] = torch.empty_like(
                region.input_node.meta["val"], dtype=torch.int32
            )
            sink.target = exir_ops.backend.tosa.RESCALE.default
            sink.args = (table_op, output_qparams[0].dtype, [2.0**lshift], 0, 0)
            sink.meta.pop("input_qparams")
            sink.meta.pop("output_qparams")
