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
    _MAXIMUM = exir_ops.backend.tosa.MAXIMUM.default
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

    @staticmethod
    def _generate_clz_half_shift_table() -> torch.Tensor:
        """Map raw int8 CLZ codes to floor((clz - 17) / 2).

        The valid CLZ range for positive int16 inputs is [17, 31]. We use an
        INT8 TOSA TABLE here rather than ARITHMETIC_RIGHT_SHIFT because U55
        serializes that shift with round=True, which would turn odd positive
        values into ceil(x / 2) instead of the floor division needed for exact
        range reduction.
        """
        half_shift_table = torch.zeros(256, dtype=torch.int8)
        for clz in range(17, 32):
            half_shift_table[clz + 128] = (clz - 17) // 2
        return half_shift_table

    @staticmethod
    def _generate_clz_zero_mask_table() -> torch.Tensor:
        """Map non-negative int32 CLZ codes to a unit zero mask."""
        zero_mask_table = torch.zeros(256, dtype=torch.int8)
        zero_mask_table[32 + 128] = 1
        return zero_mask_table

    @staticmethod
    def _generate_clz_nonzero_mask_table() -> torch.Tensor:
        """Map non-negative int32 CLZ codes to a unit non-zero mask."""
        nonzero_mask_table = torch.zeros(256, dtype=torch.int8)
        for clz in range(17, 32):
            nonzero_mask_table[clz + 128] = 1
        return nonzero_mask_table

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
            zero_i32 = self._create_scalar_constant(
                graph, insert_pos, node.name, "zero_i32", 0, torch.int32
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
            half_shift_table_const = self._create_tensor_constant(
                graph,
                insert_pos,
                node.name,
                "clz_half_shift_table_constant",
                self._generate_clz_half_shift_table(),
            )
            zero_mask_table_const = self._create_tensor_constant(
                graph,
                insert_pos,
                node.name,
                "clz_zero_mask_table_constant",
                self._generate_clz_zero_mask_table(),
            )
            nonzero_mask_table_const = self._create_tensor_constant(
                graph,
                insert_pos,
                node.name,
                "clz_nonzero_mask_table_constant",
                self._generate_clz_nonzero_mask_table(),
            )

            with graph.inserting_before(node):
                x_i32 = create_node(
                    graph,
                    self._CAST,
                    args=(node.args[0], torch.int32),
                    from_node=node,
                )
                x_nonneg = create_node(
                    graph,
                    self._MAXIMUM,
                    args=(x_i32, zero_i32),
                    from_node=node,
                )
                x_pos = create_node(
                    graph,
                    self._MAXIMUM,
                    args=(x_nonneg, one_i32),
                    from_node=node,
                )
                # Normalize each positive input into the stable [8192, 32767]
                # range before table lookup.
                clz = create_node(
                    graph,
                    self._CLZ,
                    args=(x_nonneg,),
                    from_node=node,
                )
                clz_i8 = create_node(
                    graph,
                    self._CAST,
                    args=(clz, torch.int8),
                    from_node=node,
                )
                half_shift_i8 = create_node(
                    graph,
                    self._TABLE,
                    args=(clz_i8, half_shift_table_const),
                    from_node=node,
                )
                half_shift = create_node(
                    graph,
                    self._CAST,
                    args=(half_shift_i8, torch.int32),
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
                zero_mask_i8 = create_node(
                    graph,
                    self._TABLE,
                    args=(clz_i8, zero_mask_table_const),
                    from_node=node,
                )
                nonzero_mask_i8 = create_node(
                    graph,
                    self._TABLE,
                    args=(clz_i8, nonzero_mask_table_const),
                    from_node=node,
                )
                zero_mask = create_node(
                    graph,
                    self._CAST,
                    args=(zero_mask_i8, torch.int32),
                    from_node=node,
                )
                nonzero_mask = create_node(
                    graph,
                    self._CAST,
                    args=(nonzero_mask_i8, torch.int32),
                    from_node=node,
                )
                output_i32 = output_node
                if output_qargs.dtype != torch.int32:
                    output_i32 = create_node(
                        graph,
                        self._CAST,
                        args=(output_node, torch.int32),
                        from_node=node,
                    )
                saturated_zero = create_node(
                    graph,
                    self._MUL,
                    args=(qmax_output_i32, zero_mask),
                    from_node=node,
                )
                masked_output = create_node(
                    graph,
                    self._MUL,
                    args=(output_i32, nonzero_mask),
                    from_node=node,
                )
                output_node = create_node(
                    graph,
                    self._ADD,
                    args=(masked_output, saturated_zero),
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
