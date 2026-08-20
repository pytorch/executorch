# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from functools import reduce
from typing import Any, cast, Sequence, Set, Type

import torch
from executorch.backends.arm._passes import ArmOpTargetedPass
from executorch.backends.arm._passes.arm_pass_utils import (
    create_node,
    get_first_fake_tensor,
)
from executorch.backends.arm.ao_ext.mxfp import (
    mxfp_dtype_to_str,
    mxfp_str_to_dtype,
    MXFPDType,
)
from executorch.backends.arm.tosa.mapping import TosaSpecialDtype
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torchao.prototype.mx_formats.mx_tensor import DTYPE_FP6_E2M3, DTYPE_FP6_E3M2


def _get_weights_payload_dtype(
    qdata_node: torch.fx.Node,
    dtype: str = "",
) -> MXFPDType:
    if dtype:
        return mxfp_str_to_dtype(dtype)
    qdata = get_first_fake_tensor(qdata_node)
    if qdata.dtype == torch.uint8:
        return torch.float4_e2m1fn_x2
    return qdata.dtype


def _mark_mxfp_payload(node: torch.fx.Node, payload_dtype: MXFPDType) -> None:
    """Annotate uint8-backed MXFP payload nodes with their TOSA dtype.

    PyTorch represents sub-byte MXFP payloads as ``torch.uint8`` tensors, so
    the tensor dtype alone cannot distinguish FP4E2M1, FP6E2M3, and FP6E3M2.
    Store the logical TOSA dtype in node metadata so later lowering and
    serialization treat the payload as MXFP data rather than ordinary uint8.
    FP8 payloads have native PyTorch dtypes and do not need this metadata.

    """
    if payload_dtype == torch.float4_e2m1fn_x2:
        node.meta[TosaSpecialDtype.meta_key()] = TosaSpecialDtype.FP4E2M1
    elif payload_dtype == DTYPE_FP6_E2M3:
        node.meta[TosaSpecialDtype.meta_key()] = TosaSpecialDtype.FP6E2M3
    elif payload_dtype == DTYPE_FP6_E3M2:
        node.meta[TosaSpecialDtype.meta_key()] = TosaSpecialDtype.FP6E3M2


class RewriteMXFPLinearPass(ArmOpTargetedPass):
    """Rewrite ``tosa_mxfp.linear`` into explicit TOSA MXFP operators.

    For each MXFP linear custom op, the pass:
    1. Reshapes activations and precomputed weight tensors to the rank expected
       by the block-scaled TOSA ops.
    2. Inserts ``tosa.CAST_TO_BLOCK_SCALED`` for the activation input.
    3. Inserts ``tosa.MATMUL_T_BLOCK_SCALED`` using the cast activations and the
       MXFP weight data/scale tensors.
    4. Restores the original output shape.
    5. Re-applies bias, reshaping it first to match the output rank when
       needed.

    """

    target_ops = {
        torch.ops.tosa_mxfp.linear.default,
        exir_ops.edge.tosa_mxfp.linear.default,
    }
    _passes_required_after: Set[Type[ExportPass]] = set()

    def __init__(self, exported_program: torch.export.ExportedProgram, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exported_program = exported_program

    def _get_linear_args(self, node: torch.fx.Node) -> tuple[
        torch.fx.Node,
        torch.fx.Node,
        torch.fx.Node,
        torch.fx.Node | None,
        int,
        MXFPDType,
    ]:
        """Extract the MXFP linear operands from a custom-op node."""
        input_node = cast(torch.fx.Node, node.args[0])
        weight_qdata_node = cast(torch.fx.Node, node.args[1])
        weight_scale_node = cast(torch.fx.Node, node.args[2])
        bias_node = cast(
            torch.fx.Node | None,
            node.args[3] if len(node.args) > 3 else node.kwargs.get("bias"),
        )
        block_size = cast(
            int,
            node.args[4] if len(node.args) > 4 else node.kwargs.get("block_size", 32),
        )
        payload_dtype_str = cast(
            str,
            (
                node.args[5]
                if len(node.args) > 5
                else node.kwargs.get(
                    "weight_payload_dtype",
                    node.kwargs.get("weight_dtype", ""),
                )
            ),
        )
        payload_dtype = _get_weights_payload_dtype(weight_qdata_node, payload_dtype_str)
        return (
            input_node,
            weight_qdata_node,
            weight_scale_node,
            bias_node,
            block_size,
            payload_dtype,
        )

    def _reshape_with_view(
        self,
        graph_module: torch.fx.GraphModule,
        input_node: torch.fx.Node,
        shape: Sequence[int | torch.SymInt],
        from_node: torch.fx.Node,
    ) -> torch.fx.Node:
        """Insert a ``view_copy`` node and update its fake-tensor metadata."""
        reshaped = create_node(
            graph=graph_module.graph,
            op_target=exir_ops.edge.aten.view_copy.default,
            args=(input_node, shape),
            kwargs={},
            from_node=from_node,
        )
        reshaped.meta["val"] = exir_ops.edge.aten.view_copy.default(
            get_first_fake_tensor(input_node),
            shape,
        )
        return reshaped

    def _create_block_scaled_inputs(
        self,
        graph_module: torch.fx.GraphModule,
        mxfp_linear_node: torch.fx.Node,
        input_node: torch.fx.Node,
        weight_qdata_node: torch.fx.Node,
        weight_scale_node: torch.fx.Node,
        block_size: int,
        payload_dtype: MXFPDType,
    ) -> tuple[torch.fx.Node, torch.fx.Node]:
        """Create rank-3 inputs for the block-scaled cast and matmul ops."""
        graph = graph_module.graph
        input_fake = get_first_fake_tensor(input_node)
        weight_qdata_fake = get_first_fake_tensor(weight_qdata_node)
        weight_scale_fake = get_first_fake_tensor(weight_scale_node)
        payload_dtype_str = mxfp_dtype_to_str(payload_dtype)
        _mark_mxfp_payload(weight_qdata_node, payload_dtype)

        batches = reduce(operator.mul, input_fake.shape[:-1], 1)
        input_reshape_shape = [1, batches, input_fake.shape[-1]]

        input_reshaped = self._reshape_with_view(
            graph_module,
            input_node,
            input_reshape_shape,
            mxfp_linear_node,
        )
        if weight_qdata_fake.ndim != 3 or weight_scale_fake.ndim != 3:
            raise RuntimeError(
                "Expected pre-reshaped rank-3 MXFP weight placeholders in rewrite pass"
            )

        cast_node = create_node(
            graph=graph,
            op_target=exir_ops.backend.tosa.CAST_TO_BLOCK_SCALED.default,
            args=(input_reshaped, block_size),
            kwargs={"output_dtype": payload_dtype_str},
            from_node=mxfp_linear_node,
        )
        cast_node.meta["val"] = exir_ops.backend.tosa.CAST_TO_BLOCK_SCALED.default(
            get_first_fake_tensor(input_reshaped),
            block_size,
            output_dtype=payload_dtype_str,
        )

        input_qdata_node = create_node(
            graph=graph,
            op_target=cast(Any, operator.getitem),
            args=(cast_node, 0),
            kwargs={},
            from_node=mxfp_linear_node,
        )
        input_qdata_node.meta["val"] = cast_node.meta["val"][0]
        _mark_mxfp_payload(input_qdata_node, payload_dtype)

        input_scale_node = create_node(
            graph=graph,
            op_target=cast(Any, operator.getitem),
            args=(cast_node, 1),
            kwargs={},
            from_node=mxfp_linear_node,
        )
        input_scale_node.meta["val"] = cast_node.meta["val"][1]

        return (
            input_qdata_node,
            input_scale_node,
        )

    def _create_matmul_node(
        self,
        graph_module: torch.fx.GraphModule,
        mxfp_linear_node: torch.fx.Node,
        input_qdata_node: torch.fx.Node,
        input_scale_node: torch.fx.Node,
        weight_qdata_node: torch.fx.Node,
        weight_scale_node: torch.fx.Node,
        block_size: int,
        payload_dtype: MXFPDType,
    ) -> torch.fx.Node:
        """Insert ``MATMUL_T_BLOCK_SCALED`` with updated fake metadata."""
        payload_dtype_str = mxfp_dtype_to_str(payload_dtype)
        matmul_node = create_node(
            graph=graph_module.graph,
            op_target=exir_ops.backend.tosa.MATMUL_T_BLOCK_SCALED.default,
            args=(
                input_qdata_node,
                input_scale_node,
                weight_qdata_node,
                weight_scale_node,
                block_size,
            ),
            kwargs={"payload_dtype": payload_dtype_str},
            from_node=mxfp_linear_node,
        )
        matmul_node.meta["val"] = exir_ops.backend.tosa.MATMUL_T_BLOCK_SCALED.default(
            get_first_fake_tensor(input_qdata_node),
            get_first_fake_tensor(input_scale_node),
            get_first_fake_tensor(weight_qdata_node),
            get_first_fake_tensor(weight_scale_node),
            block_size,
            payload_dtype=payload_dtype_str,
        )
        return matmul_node

    def _create_output_view(
        self,
        graph_module: torch.fx.GraphModule,
        mxfp_linear_node: torch.fx.Node,
        matmul_node: torch.fx.Node,
    ) -> torch.fx.Node:
        """Restore the original linear output shape after block matmul."""
        output_fake = get_first_fake_tensor(mxfp_linear_node)
        output_node = create_node(
            graph=graph_module.graph,
            op_target=exir_ops.edge.aten.view_copy.default,
            args=(matmul_node, list(output_fake.shape)),
            kwargs={},
            from_node=mxfp_linear_node,
        )
        output_node.meta["val"] = exir_ops.edge.aten.view_copy.default(
            get_first_fake_tensor(matmul_node),
            list(output_fake.shape),
        )
        return output_node

    def _create_bias_add(
        self,
        graph_module: torch.fx.GraphModule,
        mxfp_linear_node: torch.fx.Node,
        output_node: torch.fx.Node,
        bias_node: torch.fx.Node,
    ) -> torch.fx.Node:
        """Reshape bias to match output rank and append the final add node."""
        output_fake = get_first_fake_tensor(mxfp_linear_node)
        bias_fake = get_first_fake_tensor(bias_node)
        bias_shape = [1] * (output_fake.dim() - 1) + [output_fake.shape[-1]]
        bias_arg = bias_node

        if tuple(bias_fake.shape) != tuple(bias_shape):
            # Match ranks by prepending singleton dimensions.
            with graph_module.graph.inserting_after(output_node):
                bias_arg = self._reshape_with_view(
                    graph_module,
                    bias_node,
                    bias_shape,
                    mxfp_linear_node,
                )
            with graph_module.graph.inserting_after(bias_arg):
                add_node = create_node(
                    graph=graph_module.graph,
                    op_target=exir_ops.edge.aten.add.Tensor,
                    args=(output_node, bias_arg),
                    kwargs={},
                    from_node=mxfp_linear_node,
                )
        else:
            # Bias already has the right shape, so add it directly.
            with graph_module.graph.inserting_after(output_node):
                add_node = create_node(
                    graph=graph_module.graph,
                    op_target=exir_ops.edge.aten.add.Tensor,
                    args=(output_node, bias_arg),
                    kwargs={},
                    from_node=mxfp_linear_node,
                )
        add_node.meta["val"] = exir_ops.edge.aten.add.Tensor(
            get_first_fake_tensor(output_node),
            get_first_fake_tensor(bias_arg),
        )

        return add_node

    def _rewrite_mxfp_linear_node(
        self,
        graph_module: torch.fx.GraphModule,
        mxfp_linear_node: torch.fx.Node,
    ) -> torch.fx.Node:
        """Rewrite one MXFP linear node to explicit TOSA MXFP ops."""
        graph = graph_module.graph
        (
            input_node,
            weight_qdata_node,
            weight_scale_node,
            bias_node,
            block_size,
            payload_dtype,
        ) = self._get_linear_args(mxfp_linear_node)

        with graph.inserting_before(mxfp_linear_node):
            (
                input_qdata_node,
                input_scale_node,
            ) = self._create_block_scaled_inputs(
                graph_module,
                mxfp_linear_node,
                input_node,
                weight_qdata_node,
                weight_scale_node,
                block_size,
                payload_dtype,
            )
            matmul_node = self._create_matmul_node(
                graph_module,
                mxfp_linear_node,
                input_qdata_node,
                input_scale_node,
                weight_qdata_node,
                weight_scale_node,
                block_size,
                payload_dtype,
            )

        with graph.inserting_after(matmul_node):
            output_node = self._create_output_view(
                graph_module, mxfp_linear_node, matmul_node
            )

        if bias_node is None:
            return output_node

        return self._create_bias_add(
            graph_module,
            mxfp_linear_node,
            output_node,
            bias_node,
        )

    def call(self, graph_module: torch.fx.GraphModule):
        modified = False
        graph = graph_module.graph

        for node in list(graph.nodes):
            if node.op != "call_function" or node.target not in self.target_ops:
                continue

            modified = True
            replacement = self._rewrite_mxfp_linear_node(graph_module, node)
            node.replace_all_uses_with(replacement)
            graph.erase_node(node)

        if modified:
            graph.eliminate_dead_code()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)
