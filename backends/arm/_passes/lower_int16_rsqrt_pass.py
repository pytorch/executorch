# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import create_node
from executorch.backends.arm._passes.quant_args import QuantArgs
from executorch.backends.arm._passes.tosa_lowering_utils import create_const
from executorch.backends.arm.tosa.specification import get_context_spec
from executorch.backends.transforms.utils import create_constant_placeholder
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.export.graph_signature import InputKind
from torch.fx import GraphModule
from torch.fx.node import Node


class LowerInt16RsqrtPass(ArmPass):
    """Lower quantized int16 rsqrt via range reduction instead of a full LUT.

    The generic int16 TOSA TABLE lowering samples uniformly over the whole
    calibrated activation range. That is numerically poor for rsqrt near zero.
    This pass rewrites the quantized int16 rsqrt path so the TABLE only sees a
    normalized positive code range [8192, 32767]. A per-element power-of-two
    factor is then reapplied with integer shifts.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    _RSQRT = exir_ops.edge.aten.rsqrt.default
    _CAST = exir_ops.backend.tosa.CAST.default
    _CLZ = exir_ops.backend.tosa.CLZ.default
    _CLAMP = exir_ops.backend.tosa.CLAMP.default
    _SUB = exir_ops.backend.tosa.SUB.default
    _ARITHMETIC_RIGHT_SHIFT = exir_ops.backend.tosa.ARITHMETIC_RIGHT_SHIFT.default
    _ADD = exir_ops.backend.tosa.ADD.default
    _MUL = exir_ops.backend.tosa.MUL.default
    _LOGICAL_LEFT_SHIFT = exir_ops.backend.tosa.LOGICAL_LEFT_SHIFT.default
    _TABLE = exir_ops.backend.tosa.TABLE.default
    _RESCALE = exir_ops.backend.tosa.RESCALE.default

    def __init__(
        self,
        exported_program: ExportedProgram,
    ) -> None:
        super().__init__()
        self.exported_program = exported_program

    @staticmethod
    def _is_supported_qargs(
        input_qargs: QuantArgs,
        output_qargs: QuantArgs,
    ) -> bool:
        if input_qargs.per_channel or output_qargs.per_channel:
            return False
        if input_qargs.dtype != torch.int16:
            return False
        if output_qargs.dtype not in (torch.int16, torch.int32):
            return False
        if input_qargs.get_zp_per_tensor() != 0:
            return False
        if output_qargs.get_zp_per_tensor() != 0:
            return False
        if input_qargs.qmin != -32767 or input_qargs.qmax != 32767:
            return False
        return True

    @staticmethod
    def _generate_normalized_rsqrt_table() -> torch.Tensor:
        """Represent rsqrt(n / 2**14) in Q14 over the normalized domain."""
        codes = torch.arange(-32768, 32769, 128, dtype=torch.float64)
        normalized_values = codes.clamp_min(8192) / 2**14
        table_values = torch.round(torch.rsqrt(normalized_values) * 2**14)
        return table_values.to(dtype=torch.int16)

    def _create_scalar_constant(
        self,
        graph: torch.fx.Graph,
        insert_pos: Node,
        node_name: str,
        suffix: str,
        value: int,
        dtype: torch.dtype,
    ) -> Node:
        with graph.inserting_before(insert_pos):
            return create_constant_placeholder(
                exp_program=self.exported_program,
                graph=graph,
                kind=InputKind.BUFFER,
                name=f"b_{node_name}_{suffix}",
                data=torch.tensor(value, dtype=dtype),
                persistent_buffer=True,
            )

    def _create_tensor_constant(
        self,
        graph: torch.fx.Graph,
        insert_pos: Node,
        node_name: str,
        suffix: str,
        value: torch.Tensor,
    ) -> Node:
        with graph.inserting_before(insert_pos):
            return create_constant_placeholder(
                exp_program=self.exported_program,
                graph=graph,
                kind=InputKind.BUFFER,
                name=f"b_{node_name}_{suffix}",
                data=value,
                persistent_buffer=True,
            )

    def _clamped_i32(
        self, graph: torch.fx.Graph, node: Node, low: int, high: int
    ) -> Node:
        # Clamp in int16, before widening: int32 elementwise MAXIMUM /
        # MINIMUM around CLZ deadlocks Ethos-U55 silicon.
        clamped = create_node(
            graph, self._CLAMP, args=(node.args[0], low, high), from_node=node
        )
        return create_node(
            graph, self._CAST, args=(clamped, torch.int32), from_node=node
        )

    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        graph = graph_module.graph
        insert_pos = list(graph.nodes)[0]

        for node in list(graph.nodes):
            if node.op != "call_function" or node.target != self._RSQRT:
                continue

            input_qparams = node.meta.get("input_qparams", {})
            output_qparams = node.meta.get("output_qparams", {})
            if len(input_qparams) != 1 or len(output_qparams) != 1:
                continue

            input_qargs = input_qparams[0]
            output_qargs = output_qparams[0]
            if not self._is_supported_qargs(input_qargs, output_qargs):
                continue

            table_values = self._generate_normalized_rsqrt_table()
            # Q14 rsqrt(n / 2**14) contributes 21 bits; TABLE adds 7 more.
            rescale_scale = 2.0**-28 / (
                math.sqrt(input_qargs.get_scale_per_tensor())
                * output_qargs.get_scale_per_tensor()
            )
            one_i32 = self._create_scalar_constant(
                graph, insert_pos, node.name, "one_i32", 1, torch.int32
            )
            # floor((clz - 17) / 2) as a 1-bit right shift. U55 only has the
            # rounding shift, floor((v + 1) / 2), so it shifts clz - 18 instead.
            round_shift = get_context_spec().is_U55_subset
            clz_offset_i32 = self._create_scalar_constant(
                graph,
                insert_pos,
                node.name,
                "clz_offset_i32",
                18 if round_shift else 17,
                torch.int32,
            )
            qmax_output_i32 = self._create_scalar_constant(
                graph,
                insert_pos,
                node.name,
                "qmax_output_i32",
                output_qargs.qmax,
                torch.int32,
            )
            table_const = self._create_tensor_constant(
                graph,
                insert_pos,
                node.name,
                "rsqrt_table_constant",
                table_values,
            )

            with graph.inserting_before(node):
                x_nonneg = self._clamped_i32(graph, node, 0, input_qargs.qmax)
                x_pos = self._clamped_i32(graph, node, 1, input_qargs.qmax)
                # Normalize each positive input into the stable [8192, 32767]
                # range before table lookup.
                clz = create_node(
                    graph,
                    self._CLZ,
                    args=(x_nonneg,),
                    from_node=node,
                )
                clz_biased = create_node(
                    graph,
                    self._SUB,
                    args=(clz, clz_offset_i32),
                    from_node=node,
                )
                half_shift = create_node(
                    graph,
                    self._ARITHMETIC_RIGHT_SHIFT,
                    args=(clz_biased, one_i32),
                    kwargs={"round": round_shift},
                    from_node=node,
                )
                shift = create_node(
                    graph,
                    self._ADD,
                    args=(half_shift, half_shift),
                    from_node=node,
                )
                normalized = create_node(
                    graph,
                    self._LOGICAL_LEFT_SHIFT,
                    args=(x_pos, shift),
                    from_node=node,
                )
                normalized_i16 = create_node(
                    graph,
                    self._CAST,
                    args=(normalized, torch.int16),
                    from_node=node,
                )
                # The LUT approximates rsqrt on the normalized domain only.
                table = create_node(
                    graph,
                    self._TABLE,
                    args=(normalized_i16, table_const),
                    from_node=node,
                )
                # Reapply the power-of-two factor removed during normalization.
                shifted = create_node(
                    graph,
                    self._LOGICAL_LEFT_SHIFT,
                    args=(table, half_shift),
                    from_node=node,
                )
                output_node = create_node(
                    graph,
                    self._RESCALE,
                    args=(
                        shifted,
                        output_qargs.dtype,
                        [rescale_scale],
                        0,
                        output_qargs.get_zp_per_tensor(),
                    ),
                    from_node=node,
                )
                # Arithmetic, not int8 TABLEs: TABLE ops next to int32 elementwise
                # ops deadlock Ethos-U55 silicon (the FVP does not model it).
                nonzero_mask = self._clamped_i32(graph, node, 0, 1)
                output_i32 = output_node
                if output_qargs.dtype != torch.int32:
                    output_i32 = create_node(
                        graph,
                        self._CAST,
                        args=(output_node, torch.int32),
                        from_node=node,
                    )
                # qmax + nonzero * (output - qmax): saturate rsqrt(0) to qmax.
                mul_shift = create_const(graph, 0, torch.int8, node)
                output_minus_qmax = create_node(
                    graph,
                    self._SUB,
                    args=(output_i32, qmax_output_i32),
                    from_node=node,
                )
                masked = create_node(
                    graph,
                    self._MUL,
                    args=(output_minus_qmax, nonzero_mask, mul_shift),
                    from_node=node,
                )
                output_node = create_node(
                    graph,
                    self._ADD,
                    args=(masked, qmax_output_i32),
                    from_node=node,
                )
                if output_qargs.dtype != torch.int32:
                    output_node = create_node(
                        graph,
                        self._CAST,
                        args=(output_node, output_qargs.dtype),
                        from_node=node,
                    )

                node.replace_all_uses_with(output_node)

            graph.erase_node(node)
            modified = True

        if modified:
            graph.eliminate_dead_code()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, modified)
