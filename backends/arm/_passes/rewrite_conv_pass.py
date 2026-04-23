# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import itertools
from typing import Any, Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import (
    create_node,
    expand_around_channel,
    get_constant_placeholder_kind,
    get_first_fake_tensor,
    get_param_tensor,
    is_persistent_buffer,
)
from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    get_input_qparams,
    get_output_qparams,
)
from executorch.backends.arm.constants import (
    HWCM_ORDER,
    NHWC_INVERSE_ORDER,
    NHWC_ORDER,
    ODHWI_ORDER,
    OHWI_ORDER,
)
from executorch.backends.arm.tosa.mapping import TosaSpecialDtype
from executorch.backends.arm.tosa.specification import get_context_shape_env
from executorch.backends.transforms.utils import create_constant_placeholder
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

from torch.export.graph_signature import InputKind


class RewriteConvPass(ArmPass):
    """Rewrites aten.convolution to TOSA conv ops
    (CONV2D/DEPTHWISE/TRANSPOSE/CONV3D).
    """

    def __init__(self, exported_program: torch.export.ExportedProgram, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exported_program = exported_program

    _passes_required_after: Set[Type[ExportPass]] = set()
    _OUTPUT_DIM_ORDER_META_KEY = "arm_output_dim_order"
    _NDHWC_ORDER = (0, 2, 3, 4, 1)
    _NDHWC_INVERSE_ORDER = (0, 4, 1, 2, 3)

    # torch.nn.Conv2d does not require the result of
    # `(input + 2 * pad - dilation * (weight - 1) - 1) / stride`
    # to be an integer, but tosa currently strictly require this property.
    # This function adjusts the pad value to meet the requirement.
    def _adjust_pad_if_needed(
        self,
        input_len: int | torch.SymInt,
        input_weight: int,
        stride: int,
        pad: int | torch.SymInt,
        dilation: int,
    ) -> int | torch.SymInt:
        """Adjust padding to satisfy TOSA's integer output-size requirement.

        Torch ``Conv2d`` does not require the result of
        ``(input + 2 * pad - dilation * (weight - 1) - 1) / stride`` to be an
        integer, but TOSA does. This helper reduces the provided padding so
        that the expression becomes divisible by ``stride``.

        Args:
            input_size (int): Spatial input size along the dimension (H or W).
            input_weight (int): Kernel size along the same dimension.
            stride (int): Stride along the same dimension.
            pad (int): Padding value to adjust (bottom or right after duplication).
            dilation (int): Dilation along the same dimension.

        Returns:
            int: Adjusted padding value that yields an integer output size.

        Raises:
            RuntimeError: If the required adjustment exceeds the provided
                padding, which should be handled by the ``SizeAdjustInputPass``
                pass instead.

        """
        mod_remainder = (
            input_len + 2 * pad - dilation * (input_weight - 1) - 1
        ) % stride

        if isinstance(mod_remainder, torch.SymInt):
            shape_env = get_context_shape_env()
            value_ranges = shape_env.bound_sympy(mod_remainder.node.expr)
            mod_remainder_upper = int(value_ranges.upper)
            if mod_remainder_upper == 0:
                mod_remainder = 0
        else:
            mod_remainder_upper = mod_remainder

        if mod_remainder_upper > pad:
            raise RuntimeError(
                "This case should be handled by the SizeAdjustInputPass, is it enabled?"
            )
        return pad - mod_remainder

    def _is_depthwise_conv2d(self, node: torch.fx.Node) -> bool:
        if (
            node.op != "call_function"
            or node.target != exir_ops.edge.aten.convolution.default
        ):
            return False
        input_tensor = get_first_fake_tensor(node.all_input_nodes[0])
        if len(input_tensor.shape) != 4:
            return False
        groups = node.args[-1]
        in_channels = input_tensor.shape[1]
        out_channels = get_first_fake_tensor(node).shape[1]
        return (in_channels == groups) and (out_channels % in_channels) == 0

    def _is_conv3d(self, rank, groups) -> bool:
        if rank == 5:
            # A Conv3D is considered depthwise if Group == InChannels and
            # Group * N == OutChannels, where N is a possitive integer.
            # Currently we do not support depthwise or grouped conv3d.
            # @TODO Add grouped/depthwise conv3d support or reject in partitioner.
            if groups != 1:
                raise RuntimeError(
                    "CONV3D with groups != 1 is not supported in the Arm backend."
                )
            return True
        return False

    def _add_bias(
        self,
        graph_module: torch.fx.GraphModule,
        node: torch.fx.Node,
        weight_node: torch.fx.Node,
    ) -> torch.fx.Node:
        output_channels = get_first_fake_tensor(node).shape[1]
        # add a node containing zeros if quantized, use int32, otherwise use float32
        if self._is_quantized_conv(node):
            bias_data = torch.zeros(size=(output_channels,), dtype=torch.int32)
        else:
            output_dtype = node.meta["val"].dtype
            bias_data = torch.zeros(size=(output_channels,), dtype=output_dtype)

        with graph_module.graph.inserting_after(weight_node):
            bias_node = create_constant_placeholder(
                self.exported_program,
                graph=graph_module.graph,
                kind=InputKind.PARAMETER,
                data=bias_data,
                persistent_buffer=True,
                name=f"{node.name}_bias",
            )
            self._mark_bias_as_int48_if_needed(node, bias_node)
        node.update_arg(2, bias_node)
        return bias_node

    def _rewrite_weight(
        self,
        graph_module: torch.fx.GraphModule,
        weight_node: torch.fx.Node,
        conv_node: torch.fx.Node,
        permute_dims: tuple[int, ...],
        name_suffix: str,
        reshape_dims: tuple[int, ...] | None = None,
    ) -> torch.fx.Node:
        """Create a convolution-local rewritten weight placeholder."""
        weight_tensor = get_param_tensor(self.exported_program, weight_node)  # type: ignore[arg-type]
        if weight_tensor is None:
            raise RuntimeError(
                f"Weight node {weight_node.name} is not a parameter or buffer"
            )

        rewritten_weight = weight_tensor.permute(permute_dims)
        if reshape_dims is not None:
            rewritten_weight = rewritten_weight.reshape(*reshape_dims)
        rewritten_weight = rewritten_weight.contiguous()
        kind = get_constant_placeholder_kind(self.exported_program, weight_node)
        persistent_buffer = is_persistent_buffer(self.exported_program, weight_node)

        with graph_module.graph.inserting_after(weight_node):
            rewritten_weight_node = create_constant_placeholder(
                self.exported_program,
                graph=graph_module.graph,
                name=f"{conv_node.name}_weight_{name_suffix}",
                kind=kind,
                data=rewritten_weight,
                persistent_buffer=persistent_buffer,
            )
        if special_dtype := weight_node.meta.get(TosaSpecialDtype.meta_key()):
            rewritten_weight_node.meta[TosaSpecialDtype.meta_key()] = special_dtype
        return rewritten_weight_node

    def _is_quantized_conv(self, node: torch.fx.Node) -> bool:
        return bool(node.meta.get("input_qparams", {}))

    def _is_int16_activation_conv(self, node: torch.fx.Node) -> bool:
        input_qparams = node.meta.get("input_qparams", {})
        if 0 in input_qparams:
            return input_qparams[0].dtype == torch.int16
        return get_first_fake_tensor(node.all_input_nodes[0]).dtype == torch.int16

    def _mark_bias_as_int48_if_needed(
        self, node: torch.fx.Node, bias_node: torch.fx.Node
    ) -> None:
        if self._is_int16_activation_conv(node):
            bias_node.meta[TosaSpecialDtype.meta_key()] = TosaSpecialDtype.INT48

    def _get_effective_output_qparams(self, node: torch.fx.Node):
        """Return the quantized output domain for a conv node.

        Quantization annotation may place output qparams on a following
        activation instead of on the conv itself. If that activation is not
        fuseable, it survives as a quantized ``clamp`` and still owns the
        branch output qparams needed for the conv output rescale.

        """
        output_qparams = node.meta.get("output_qparams", {})
        if output_qparams:
            return output_qparams

        users = list(node.users)
        if len(users) != 1:
            raise ValueError(
                f"RewriteConvPass: No output quantization parameter found in node {node}\n"
                f"original_aten={node.meta.get('original_aten', 'None')}"
            )

        activation = users[0]
        if activation.target == exir_ops.edge.aten.clamp.default:
            activation_output_qparams = activation.meta.get("output_qparams", {})
            if activation_output_qparams:
                return activation_output_qparams

        return get_output_qparams(node)

    def insert_output_rescale(
        self,
        graph_module,
        source_node,
        conv_node,
        conv_fake_tensor: torch.Tensor,
    ):
        input_qparams = get_input_qparams(source_node)
        output_qparams = self._get_effective_output_qparams(source_node)[0]
        weight_qparams = input_qparams[1]
        input_qparams = input_qparams[0]
        is_per_channel = weight_qparams.per_channel
        if is_per_channel:
            weight_scale = weight_qparams.get_scale_per_channel()
        else:
            weight_scale = [weight_qparams.get_scale_per_tensor()]
        input_scale = input_qparams.get_scale_per_tensor()
        post_conv2d_scale = [
            (inp * w) / out
            for inp, w, out in zip(
                itertools.cycle([input_scale]),
                weight_scale,
                itertools.cycle([output_qparams.get_scale_per_tensor()]),
            )
        ]
        with graph_module.graph.inserting_after(conv_node):
            rescale_node = create_node(
                graph=graph_module.graph,
                op_target=exir_ops.backend.tosa.RESCALE.default,
                args=(
                    conv_node,
                    output_qparams.dtype,
                    post_conv2d_scale,
                    0,
                    output_qparams.get_zp_per_tensor(),
                ),
                from_node=source_node,
            )
        rescale_fake_tensor = exir_ops.backend.tosa.RESCALE.default(
            conv_fake_tensor,
            output_qparams.dtype,
            post_conv2d_scale,
            0,
            output_qparams.get_zp_per_tensor(),
        )
        return rescale_node, rescale_fake_tensor

    def insert_identity_int32_rescale(
        self,
        graph_module,
        source_node,
        conv_node,
        conv_fake_tensor: torch.Tensor,
    ):
        with graph_module.graph.inserting_after(conv_node):
            rescale_node = create_node(
                graph=graph_module.graph,
                op_target=exir_ops.backend.tosa.RESCALE.default,
                args=(
                    conv_node,
                    torch.int32,
                    [1.0],
                    0,
                    0,
                ),
                from_node=source_node,
            )
        rescale_fake_tensor = exir_ops.backend.tosa.RESCALE.default(
            conv_fake_tensor,
            torch.int32,
            [1.0],
            0,
            0,
        )
        return rescale_node, rescale_fake_tensor

    def _has_int32_rescale_user(self, node: torch.fx.Node) -> bool:
        for user in node.users:
            if (
                user.op == "call_function"
                and user.target == exir_ops.backend.tosa.RESCALE.default
                and len(user.args) > 1
                and user.args[1] == torch.int32
            ):
                return True
            if (
                user.op == "call_function"
                and user.target == exir_ops.edge.aten.permute_copy.default
            ):
                for inner_user in user.users:
                    if (
                        inner_user.op == "call_function"
                        and inner_user.target == exir_ops.backend.tosa.RESCALE.default
                        and inner_user.args[1] == torch.int32
                    ):
                        return True
        return False

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:  # noqa: C901
        modified = False
        for node in graph_module.graph.nodes:
            if (
                node.op != "call_function"
                or node.target != exir_ops.edge.aten.convolution.default
            ):
                continue

            modified = True

            (
                x,
                weight,
                bias,
                stride,
                pad,
                dilation,
                transposed,
                output_padding,
                group,
            ) = node.args

            input_fake_tensor = get_first_fake_tensor(x)
            weight_fake_tensor = get_first_fake_tensor(weight)
            input_shape = input_fake_tensor.shape
            weight_shape = weight_fake_tensor.shape
            spatial_rank = len(input_shape) - 2
            stride_list = expand_around_channel(stride, spatial_rank)
            dilation_list = expand_around_channel(dilation, spatial_rank)
            pad_list = expand_around_channel(pad, spatial_rank)

            stride = tuple(stride_list)

            has_bias = bias is not None
            if not has_bias:
                bias = self._add_bias(graph_module, node, weight)
            elif isinstance(bias, torch.fx.Node):
                self._mark_bias_as_int48_if_needed(node, bias)

            conv_args: tuple[Any, ...]
            input_tensor_for_tosa_fake: torch.Tensor = input_fake_tensor
            pre_permute_dims: tuple[int, ...]
            post_permute_dims: tuple[int, ...]
            if transposed:
                if spatial_rank != 2:
                    raise RuntimeError(
                        "Only 2D transpose convolutions are supported in the Arm backend."
                    )
                if any(d != 1 for d in dilation_list):
                    raise RuntimeError(
                        "Transpose convolutions with dilation are not supported in the Arm backend."
                    )
                output_padding_list = expand_around_channel(
                    output_padding, spatial_rank
                )
                out_pad = [
                    -pad_list[0],
                    -pad_list[0] + output_padding_list[0],
                    -pad_list[1],
                    -pad_list[1] + output_padding_list[1],
                ]
                target_op = exir_ops.backend.tosa.TRANSPOSE_CONV2D.default
                pre_permute_dims = NHWC_ORDER
                post_permute_dims = NHWC_INVERSE_ORDER
                with graph_module.graph.inserting_before(node):
                    x = create_node(
                        graph=graph_module.graph,
                        op_target=exir_ops.edge.aten.permute_copy.default,
                        args=(x, list(pre_permute_dims)),
                        from_node=node,
                    )
                x.meta["val"] = exir_ops.edge.aten.permute_copy.default(
                    input_fake_tensor, list(pre_permute_dims)
                )
                weight = self._rewrite_weight(
                    graph_module,
                    weight,
                    node,
                    permute_dims=OHWI_ORDER,
                    name_suffix="ohwi",
                )
                input_tensor_for_tosa_fake = input_fake_tensor.permute(pre_permute_dims)
                weight_fake_tensor = get_first_fake_tensor(weight)
                conv_args = (
                    x,
                    weight,
                    bias,
                    out_pad,
                    stride,
                )
            else:
                pad_attr: list[int | torch.SymInt] = []
                for value in pad_list:
                    pad_attr.extend(
                        [value, value]
                    )  # duplicate pad before/after per axis

                for axis_index in range(spatial_rank):
                    pad_index = axis_index * 2 + 1  # adjust trailing pad entry
                    pad_attr[pad_index] = self._adjust_pad_if_needed(
                        input_shape[axis_index + 2],
                        weight_shape[axis_index + 2],
                        stride_list[axis_index],
                        pad_attr[pad_index],
                        dilation_list[axis_index],
                    )

                dilation = tuple(dilation_list)
                pad = pad_attr

                if self._is_conv3d(len(input_shape), group):
                    target_op = exir_ops.backend.tosa.CONV3D.default
                    pre_permute_dims = self._NDHWC_ORDER
                    post_permute_dims = self._NDHWC_INVERSE_ORDER
                    with graph_module.graph.inserting_before(node):
                        x = create_node(
                            graph=graph_module.graph,
                            op_target=exir_ops.edge.aten.permute_copy.default,
                            args=(x, list(pre_permute_dims)),
                            from_node=node,
                        )
                    x.meta["val"] = exir_ops.edge.aten.permute_copy.default(
                        input_fake_tensor, list(pre_permute_dims)
                    )
                    weight = self._rewrite_weight(
                        graph_module,
                        weight,
                        node,
                        permute_dims=ODHWI_ORDER,
                        name_suffix="odhwi",
                    )
                    input_tensor_for_tosa_fake = input_fake_tensor.permute(
                        pre_permute_dims
                    )
                    weight_fake_tensor = get_first_fake_tensor(weight)
                elif self._is_depthwise_conv2d(node):
                    target_op = exir_ops.backend.tosa.DEPTHWISE_CONV2D.default
                    pre_permute_dims = NHWC_ORDER
                    post_permute_dims = NHWC_INVERSE_ORDER
                    with graph_module.graph.inserting_before(node):
                        x = create_node(
                            graph=graph_module.graph,
                            op_target=exir_ops.edge.aten.permute_copy.default,
                            args=(x, list(pre_permute_dims)),
                            from_node=node,
                        )
                    x.meta["val"] = exir_ops.edge.aten.permute_copy.default(
                        input_fake_tensor, list(pre_permute_dims)
                    )
                    kh, kw = weight_shape[2], weight_shape[3]
                    in_channels = input_fake_tensor.shape[1]
                    m_length = weight_shape[0] // in_channels
                    weight = self._rewrite_weight(
                        graph_module,
                        weight,
                        node,
                        permute_dims=HWCM_ORDER,
                        name_suffix="hwicm",
                        reshape_dims=(kh, kw, in_channels, m_length),
                    )
                    input_tensor_for_tosa_fake = input_fake_tensor.permute(
                        pre_permute_dims
                    )
                    weight_fake_tensor = get_first_fake_tensor(weight)
                else:
                    target_op = exir_ops.backend.tosa.CONV2D.default
                    pre_permute_dims = NHWC_ORDER
                    post_permute_dims = NHWC_INVERSE_ORDER
                    with graph_module.graph.inserting_before(node):
                        x = create_node(
                            graph=graph_module.graph,
                            op_target=exir_ops.edge.aten.permute_copy.default,
                            args=(x, list(pre_permute_dims)),
                            from_node=node,
                        )
                    x.meta["val"] = exir_ops.edge.aten.permute_copy.default(
                        input_fake_tensor, list(pre_permute_dims)
                    )
                    weight = self._rewrite_weight(
                        graph_module,
                        weight,
                        node,
                        permute_dims=NHWC_ORDER,
                        name_suffix="ohwi",
                    )
                    input_tensor_for_tosa_fake = input_fake_tensor.permute(
                        pre_permute_dims
                    )
                    weight_fake_tensor = get_first_fake_tensor(weight)

                conv_args = (
                    x,
                    weight,
                    bias,
                    stride,
                    pad,
                    dilation,
                )

            with graph_module.graph.inserting_after(node):
                tosa_op = create_node(
                    graph=graph_module.graph,
                    op_target=target_op,
                    args=conv_args,
                    from_node=node,
                    inherit_qparams=True,
                )
            bias_fake_tensor = get_first_fake_tensor(bias) if bias else None
            tosa_node_fake_tensor = target_op(
                input_tensor_for_tosa_fake,
                weight_fake_tensor,
                bias_fake_tensor,
                *conv_args[3:],
            )
            tosa_op.meta["val"] = tosa_node_fake_tensor

            node_replacement: torch.fx.Node = tosa_op
            node_replacement_fake_tensor = tosa_node_fake_tensor
            if (
                tosa_node_fake_tensor.dtype == torch.int32
                and input_fake_tensor.dtype == torch.int8
            ):
                output_rescale, output_rescale_fake = self.insert_output_rescale(
                    graph_module, node, tosa_op, tosa_node_fake_tensor
                )
                node_replacement = output_rescale
                node_replacement_fake_tensor = output_rescale_fake
            elif (
                tosa_node_fake_tensor.dtype == torch.int32
                and input_fake_tensor.dtype == torch.int16
            ):
                # Explicit layout paths require a post-conv permute, which does
                # not support INT48. Always rescale before post-permute.
                if self._has_int32_rescale_user(node):
                    output_rescale, output_rescale_fake = (
                        self.insert_identity_int32_rescale(
                            graph_module, node, tosa_op, tosa_node_fake_tensor
                        )
                    )
                else:
                    output_rescale, output_rescale_fake = self.insert_output_rescale(
                        graph_module, node, tosa_op, tosa_node_fake_tensor
                    )
                node_replacement = output_rescale
                node_replacement_fake_tensor = output_rescale_fake

                tosa_op.meta[TosaSpecialDtype.meta_key()] = TosaSpecialDtype.INT48

            if post_permute_dims is None:
                raise RuntimeError("Expected post permute dims for explicit layout")
            post_permute_input = node_replacement
            with graph_module.graph.inserting_after(node_replacement):
                node_replacement = create_node(
                    graph=graph_module.graph,
                    op_target=exir_ops.edge.aten.permute_copy.default,
                    args=(node_replacement, list(post_permute_dims)),
                    from_node=node,
                )
            if special_dtype := post_permute_input.meta.get(
                TosaSpecialDtype.meta_key()
            ):
                node_replacement.meta[TosaSpecialDtype.meta_key()] = special_dtype
            original_output_fake = node.meta.get("val")
            if isinstance(original_output_fake, torch.Tensor):
                node_replacement.meta[self._OUTPUT_DIM_ORDER_META_KEY] = tuple(
                    original_output_fake.dim_order()
                )
            node_replacement.meta["val"] = exir_ops.edge.aten.permute_copy.default(
                node_replacement_fake_tensor, list(post_permute_dims)
            )

            node.replace_all_uses_with(node_replacement)

            graph_module.graph.erase_node(node)

        if modified:
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, modified)
