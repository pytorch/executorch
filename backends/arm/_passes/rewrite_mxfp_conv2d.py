# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import operator
from typing import cast, Set, Type

import torch
from executorch.backends.arm._passes import ArmOpTargetedPass
from executorch.backends.arm._passes.arm_pass_utils import (
    create_node,
    create_shape_node,
    get_first_fake_tensor,
)
from executorch.backends.arm.ao_ext.mxfp import (
    mxfp_dtype_to_str,
    mxfp_str_to_dtype,
    MXFPDType,
)
from executorch.backends.arm.constants import NHWC_INVERSE_ORDER, NHWC_ORDER
from executorch.backends.arm.tosa.mapping import TosaSpecialDtype
from executorch.backends.transforms.utils import create_constant_placeholder
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.export.graph_signature import InputKind
from torch.fx import GraphModule, Node
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
    if payload_dtype == torch.float4_e2m1fn_x2:
        node.meta[TosaSpecialDtype.meta_key()] = TosaSpecialDtype.FP4E2M1
    elif payload_dtype == DTYPE_FP6_E2M3:
        node.meta[TosaSpecialDtype.meta_key()] = TosaSpecialDtype.FP6E2M3
    elif payload_dtype == DTYPE_FP6_E3M2:
        node.meta[TosaSpecialDtype.meta_key()] = TosaSpecialDtype.FP6E3M2


def _set_tensor_meta(node: Node, fake_tensor: torch.Tensor) -> None:
    node.meta["val"] = fake_tensor


def _set_shape_meta(node: Node, values: list[int]) -> None:
    node.meta["val"] = values
    node.meta[TosaSpecialDtype.meta_key()] = TosaSpecialDtype.SHAPE


class RewriteMXFPConv2dPass(ArmOpTargetedPass):
    """Rewrite the MXFP Conv2d custom op to TOSA block-scaled ops."""

    target_ops = {
        torch.ops.tosa_mxfp.conv2d.default,
        exir_ops.edge.tosa_mxfp.conv2d.default,
    }
    _passes_required_after: Set[Type[ExportPass]] = set()

    def __init__(self, exported_program: torch.export.ExportedProgram, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exported_program = exported_program

    def _permute_with_meta(
        self,
        graph_module: GraphModule,
        input_node: Node,
        dims: list[int],
        from_node: Node,
    ) -> Node:
        """Insert a permute node and attach FakeTensor metadata.

        Args:
            graph_module (torch.fx.GraphModule): Graph being rewritten.
            input_node (torch.fx.Node): Input node to permute.
            dims (list[int]): Permutation order to apply.
            from_node (torch.fx.Node): Source node used to seed metadata.

        Returns:
            torch.fx.Node: The inserted permute node with updated metadata.

        """
        permuted = create_node(
            graph=graph_module.graph,
            op_target=exir_ops.edge.aten.permute_copy.default,
            args=(input_node, dims),
            kwargs={},
            from_node=from_node,
        )
        _set_tensor_meta(
            permuted,
            exir_ops.edge.aten.permute_copy.default(
                get_first_fake_tensor(input_node),
                dims,
            ),
        )
        return permuted

    def _create_zero_bias(
        self,
        graph_module: GraphModule,
        node: Node,
        weight_qdata_node: Node,
    ) -> Node:
        """Create a zero bias placeholder for bias-free MXFP Conv2d.

        Args:
            graph_module (torch.fx.GraphModule): Graph being rewritten.
            node (torch.fx.Node): Original MXFP Conv2d node.
            weight_qdata_node (torch.fx.Node): Quantized weight payload node.

        Returns:
            torch.fx.Node: Placeholder node for the inserted zero bias buffer.

        """
        out_channels = get_first_fake_tensor(weight_qdata_node).shape[0]
        bias_data = torch.zeros((out_channels,), dtype=torch.float32)
        with graph_module.graph.inserting_after(weight_qdata_node):
            return create_constant_placeholder(
                self.exported_program,
                graph_module.graph,
                name=f"{node.name}_bias",
                kind=InputKind.BUFFER,
                data=bias_data,
                persistent_buffer=True,
            )

    def _create_const_shape(
        self,
        graph_module: GraphModule,
        node: Node,
        name: str,
        values: list[int],
    ) -> Node:
        const_shape = create_shape_node(
            graph=graph_module.graph,
            op_target=exir_ops.backend.tosa.CONST_SHAPE.default,
            args=(values,),
            kwargs={},
            from_node=node,
        )
        const_shape.name = f"{node.name}_{name}"
        _set_shape_meta(const_shape, values)
        return const_shape

    def _get_conv2d_args(
        self,
        node: torch.fx.Node,
    ) -> tuple[
        Node,
        Node,
        Node,
        Node | None,
        list[int] | tuple[int, int],
        list[int] | tuple[int, int],
        list[int] | tuple[int, int],
        int,
        int,
        MXFPDType,
    ]:
        input_node = cast(Node, node.args[0])
        weight_qdata_node = cast(Node, node.args[1])
        weight_scale_node = cast(Node, node.args[2])
        bias_node = cast(
            Node | None,
            node.args[3] if len(node.args) > 3 else node.kwargs.get("bias"),
        )
        stride = cast(
            list[int] | tuple[int, int],
            node.args[4] if len(node.args) > 4 else node.kwargs["stride"],
        )
        padding = cast(
            list[int] | tuple[int, int],
            node.args[5] if len(node.args) > 5 else node.kwargs["padding"],
        )
        dilation = cast(
            list[int] | tuple[int, int],
            node.args[6] if len(node.args) > 6 else node.kwargs["dilation"],
        )
        groups = cast(
            int,
            node.args[7] if len(node.args) > 7 else node.kwargs.get("groups", 1),
        )
        if groups != 1:
            raise ValueError(f"Only groups=1 is supported, got {groups}")
        block_size = cast(
            int,
            node.args[8] if len(node.args) > 8 else node.kwargs.get("block_size", 32),
        )
        payload_dtype_str = cast(
            str,
            (
                node.args[9]
                if len(node.args) > 9
                else node.kwargs.get(
                    "weight_payload_dtype",
                    node.kwargs.get("weight_dtype", ""),
                )
            ),
        )
        payload_dtype = _get_weights_payload_dtype(
            weight_qdata_node,
            payload_dtype_str,
        )
        return (
            input_node,
            weight_qdata_node,
            weight_scale_node,
            bias_node,
            stride,
            padding,
            dilation,
            groups,
            block_size,
            payload_dtype,
        )

    def call(self, graph_module: GraphModule):
        """Rewrite MXFP Conv2d custom ops into TOSA MXFP ops.

        The source custom op uses NCHW activations and standard Conv2d
        arguments. This pass rewrites each matching node into:

        1. ``permute_copy`` from NCHW to NHWC.
        2. ``tosa.CAST_TO_BLOCK_SCALED`` to quantize the activation tensor.
        3. ``operator.getitem`` nodes to extract quantized data and scales.
        4. ``tosa.CONV2D_BLOCK_SCALED`` using NHWC activations and OHWI
           weights.
        5. ``permute_copy`` from NHWC back to NCHW.

        If the source op has no bias, the pass inserts a persistent zero bias
        buffer because the TOSA op always expects one. The pass also expands
        2-D Conv2d padding into the 4-element TOSA padding format. The pass
        Args:
            graph_module (torch.fx.GraphModule): Graph module to rewrite.

        Returns:
            PassResult: The rewritten graph module and whether it changed.

        """
        modified = False
        graph = graph_module.graph

        for node in list(graph.nodes):
            if node.op != "call_function" or node.target not in self.target_ops:
                continue

            modified = True

            (
                input_node,
                weight_qdata_node,
                weight_scale_node,
                bias_node,
                stride,
                padding,
                dilation,
                _groups,
                block_size,
                payload_dtype,
            ) = self._get_conv2d_args(node)
            payload_dtype_str = mxfp_dtype_to_str(payload_dtype)
            _mark_mxfp_payload(weight_qdata_node, payload_dtype)

            output_fake = get_first_fake_tensor(node)
            stride_list = list(stride)
            dilation_list = list(dilation)
            pad_list = [padding[0], padding[0], padding[1], padding[1]]

            with graph.inserting_before(node):
                if bias_node is None:
                    bias_node = self._create_zero_bias(
                        graph_module,
                        node,
                        weight_qdata_node,
                    )

                input_nhwc = self._permute_with_meta(
                    graph_module,
                    input_node,
                    list(NHWC_ORDER),
                    node,
                )

                cast_node = create_node(
                    graph=graph,
                    op_target=exir_ops.backend.tosa.CAST_TO_BLOCK_SCALED.default,
                    args=(input_nhwc, block_size),
                    kwargs={"output_dtype": payload_dtype_str},
                    from_node=node,
                )
                cast_node.meta["val"] = (
                    exir_ops.backend.tosa.CAST_TO_BLOCK_SCALED.default(
                        get_first_fake_tensor(input_nhwc),
                        block_size,
                        output_dtype=payload_dtype_str,
                    )
                )

                input_qdata_node = create_node(
                    graph=graph,
                    op_target=operator.getitem,  # type: ignore[arg-type]
                    args=(cast_node, 0),
                    kwargs={},
                    from_node=node,
                )
                _set_tensor_meta(input_qdata_node, cast_node.meta["val"][0])
                _mark_mxfp_payload(input_qdata_node, payload_dtype)

                input_scale_node = create_node(
                    graph=graph,
                    op_target=operator.getitem,  # type: ignore[arg-type]
                    args=(cast_node, 1),
                    kwargs={},
                    from_node=node,
                )
                _set_tensor_meta(input_scale_node, cast_node.meta["val"][1])

                stride_node = self._create_const_shape(
                    graph_module,
                    node,
                    "stride",
                    stride_list,
                )
                pad_node = self._create_const_shape(
                    graph_module,
                    node,
                    "pad",
                    pad_list,
                )
                dilation_node = self._create_const_shape(
                    graph_module,
                    node,
                    "dilation",
                    dilation_list,
                )

                conv_node = create_node(
                    graph=graph,
                    op_target=exir_ops.backend.tosa.CONV2D_BLOCK_SCALED.default,
                    args=(
                        input_qdata_node,
                        input_scale_node,
                        weight_qdata_node,
                        weight_scale_node,
                        bias_node,
                        stride_node,
                        pad_node,
                        dilation_node,
                        block_size,
                    ),
                    kwargs={"payload_dtype": payload_dtype_str},
                    from_node=node,
                )
                _set_tensor_meta(
                    conv_node,
                    exir_ops.backend.tosa.CONV2D_BLOCK_SCALED.default(
                        get_first_fake_tensor(input_qdata_node),
                        get_first_fake_tensor(input_scale_node),
                        get_first_fake_tensor(weight_qdata_node),
                        get_first_fake_tensor(weight_scale_node),
                        get_first_fake_tensor(bias_node),
                        stride_list,
                        pad_list,
                        dilation_list,
                        block_size,
                        payload_dtype=payload_dtype_str,
                    ),
                )

            with graph.inserting_after(conv_node):
                output_node = self._permute_with_meta(
                    graph_module,
                    conv_node,
                    list(NHWC_INVERSE_ORDER),
                    node,
                )

            if tuple(get_first_fake_tensor(output_node).shape) != tuple(
                output_fake.shape
            ):
                raise ValueError(
                    f"Expected rewritten Conv2d output shape {tuple(output_fake.shape)}, "
                    f"got {tuple(get_first_fake_tensor(output_node).shape)}"
                )

            node.replace_all_uses_with(output_node)
            graph.erase_node(node)

        if modified:
            graph.eliminate_dead_code()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)
