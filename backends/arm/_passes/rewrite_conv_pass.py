# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import itertools
from typing import Any, cast, Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import (
    create_node,
    expand_around_channel,
    get_first_fake_tensor,
    get_param_tensor,
    is_buffer,
    is_param,
)
from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    get_input_qparams,
    get_output_qparams,
)
from executorch.backends.arm._passes.quant_args import QuantArgs
from executorch.backends.arm.constants import HWCM_ORDER, NHWC_INVERSE_ORDER
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

    @staticmethod
    def _nchw_to_nhwc_perm(rank: int) -> list[int]:
        if rank == 4:
            return [0, 2, 3, 1]
        if rank == 5:
            return [0, 2, 3, 4, 1]
        return list(range(rank))

    @staticmethod
    def _nhwc_to_nchw_perm(rank: int) -> list[int]:
        if rank == 4:
            return [0, 3, 1, 2]
        if rank == 5:
            return [0, 4, 1, 2, 3]
        return list(range(rank))

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

    def _reshape_weights(self, weight_node: torch.fx.Node, in_channels: int) -> None:
        """Reshape the weights for depthwise convolution such that when
        serialized to TOSA, the weights are in the format [H, W, in_channels,
        m_length] where m_length is the number of output channels per input
        channel.
        """
        weight_tensor = get_param_tensor(self.exported_program, weight_node)  # type: ignore[arg-type]
        if weight_tensor is None:
            raise RuntimeError(
                f"Weight node {weight_node.name} is not a parameter or buffer"
            )

        reshaped_weight_tensor = (
            weight_tensor.permute(HWCM_ORDER)
            .reshape(
                weight_tensor.shape[2],
                weight_tensor.shape[3],
                in_channels,
                weight_tensor.shape[0] // in_channels,
            )
            .permute(NHWC_INVERSE_ORDER)
        )

        if is_buffer(self.exported_program, weight_node):
            param_name = self.exported_program.graph_signature.inputs_to_buffers[
                weight_node.name
            ]
            reshaped_weight_tensor = torch.nn.Buffer(reshaped_weight_tensor)
        elif is_param(self.exported_program, weight_node):
            param_name = self.exported_program.graph_signature.inputs_to_parameters[
                weight_node.name
            ]
            reshaped_weight_tensor = torch.nn.Parameter(
                reshaped_weight_tensor, requires_grad=False
            )
        else:
            raise RuntimeError(
                f"Weight node {weight_node.name} is neither a parameter nor a buffer"
            )

        self.exported_program.state_dict[param_name] = reshaped_weight_tensor
        weight_node.meta["val"] = weight_node.meta["val"].reshape(
            weight_tensor.shape[2],
            weight_tensor.shape[0] // in_channels,
            weight_tensor.shape[3],
            in_channels,
        )

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
            if node.all_input_nodes[0].meta["val"].dtype == torch.int16:
                bias_node.meta[TosaSpecialDtype.meta_key()] = TosaSpecialDtype.INT48
        node.update_arg(2, bias_node)
        return bias_node

    def _is_quantized_conv(self, node: torch.fx.Node) -> bool:
        return bool(node.meta.get("input_qparams", {}))

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

    def insert_output_rescale(self, graph_module, source_node, conv_node):
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
        return rescale_node

    def _insert_permute(self, graph_module, anchor_node, input_node, perm, before=True):
        ctx = (
            graph_module.graph.inserting_before(anchor_node)
            if before
            else graph_module.graph.inserting_after(anchor_node)
        )
        with ctx:
            return create_node(
                graph=graph_module.graph,
                op_target=exir_ops.edge.aten.permute_copy.default,
                args=(input_node, perm),
                from_node=input_node,
            )

    def _is_grouped_conv(self, node: torch.fx.Node) -> bool:
        """Return True for grouped convolutions that need decomposition.

        Depthwise convolutions (groups == in_channels) are handled natively
        by TOSA and are *not* considered grouped here.
        """
        groups = node.args[-1]
        if groups <= 1:
            return False
        input_tensor = get_first_fake_tensor(node.all_input_nodes[0])
        if len(input_tensor.shape) != 4:
            return False
        return not self._is_depthwise_conv2d(node)

    def _handle_grouped_conv(  # noqa: C901
        self,
        graph_module: torch.fx.GraphModule,
        node: torch.fx.Node,
        x: torch.fx.Node,
        weight: torch.fx.Node,
        bias: torch.fx.Node | None,
        stride_list: list[int],
        pad_list: list[int],
        dilation_list: list[int],
        group: int,
        input_shape: torch.Size,
        weight_shape: torch.Size,
        spatial_rank: int,
        rank: int,
    ) -> torch.fx.Node:
        """Decompose a grouped conv into per-group TOSA.CONV2D ops in NHWC.

        Produces a single input permute (NCHW→NHWC) and a single output
        permute (NHWC→NCHW), with the per-group slice / conv / cat operating
        entirely in NHWC.  This avoids the problematic pattern of one permute
        pair per sub-conv that downstream optimisation passes can mishandle.
        """
        nchw_to_nhwc = self._nchw_to_nhwc_perm(rank)
        nhwc_to_nchw = self._nhwc_to_nchw_perm(rank)
        nhwc_channel_dim = rank - 1

        in_channels = input_shape[1]
        out_channels = get_first_fake_tensor(node).shape[1]
        input_slice_size = in_channels // group
        output_slice_size = out_channels // group

        # Compute TOSA pad attribute (same logic as the non-grouped path).
        pad_attr: list[int] = []
        for value in pad_list:
            pad_attr.extend([value, value])
        for axis_index in range(spatial_rank):
            pad_index = axis_index * 2 + 1
            pad_attr[pad_index] = self._adjust_pad_if_needed(
                input_shape[axis_index + 2],
                weight_shape[axis_index + 2],
                stride_list[axis_index],
                pad_attr[pad_index],
                dilation_list[axis_index],
            )
        stride_tuple = tuple(stride_list)
        dilation_tuple = tuple(dilation_list)

        weight_perm = self._nchw_to_nhwc_perm(len(weight_shape))

        # ---- Quantisation info ------------------------------------------
        input_dtype = get_first_fake_tensor(x).dtype
        is_quantized = self._is_quantized_conv(node)
        has_qparam_bias = (
            is_quantized and len(node.meta.get("input_qparams", {})) > 2
        )
        is_int8 = is_quantized and input_dtype == torch.int8
        is_int16_with_bias = (
            is_quantized and input_dtype == torch.int16 and has_qparam_bias
        )
        is_int16_no_bias = (
            is_quantized and input_dtype == torch.int16 and not has_qparam_bias
        )

        original_bias = bias  # Keep for INT16+bias decomposition

        # Pre-compute rescale factors for INT8 / INT16-no-bias paths.
        full_weight_scale: list[float] = []
        input_scale = 0.0
        output_scale = 0.0
        output_zp = 0
        rescale_dtype = torch.int8
        if is_int8 or is_int16_no_bias:
            iq = get_input_qparams(node)
            oq = self._get_effective_output_qparams(node)[0]
            wq = iq[1]
            if wq.per_channel:
                full_weight_scale = wq.get_scale_per_channel()
            else:
                full_weight_scale = [wq.get_scale_per_tensor()]
            input_scale = iq[0].get_scale_per_tensor()
            output_scale = oq.get_scale_per_tensor()
            output_zp = oq.get_zp_per_tensor()
            rescale_dtype = oq.dtype

        # ---- ONE input permute NCHW→NHWC --------------------------------
        x_permuted = self._insert_permute(
            graph_module, node, x, nchw_to_nhwc, before=True
        )

        group_outputs: list[torch.fx.Node] = []
        cursor = node

        for g in range(group):
            # Slice NHWC input along channel dim
            with graph_module.graph.inserting_before(node):
                sliced_input = create_node(
                    graph=graph_module.graph,
                    op_target=exir_ops.edge.aten.slice_copy.Tensor,
                    args=(
                        x_permuted,
                        nhwc_channel_dim,
                        g * input_slice_size,
                        (g + 1) * input_slice_size,
                    ),
                    from_node=x,
                )

            # Slice weight along output-channel dim (dim 0 in OIHW)
            with graph_module.graph.inserting_before(node):
                sliced_weight = create_node(
                    graph=graph_module.graph,
                    op_target=exir_ops.edge.aten.slice_copy.Tensor,
                    args=(
                        weight,
                        0,
                        g * output_slice_size,
                        (g + 1) * output_slice_size,
                    ),
                    from_node=weight,
                )

            # Permute weight OIHW→OHWI
            sliced_weight_permuted = self._insert_permute(
                graph_module, node, sliced_weight, weight_perm, before=True
            )

            # ---- Per-group bias -----------------------------------------
            if is_int16_with_bias or is_int16_no_bias:
                # INT16: TOSA conv always needs an INT48-tagged zero bias.
                # Create a fresh per-group constant so the tag survives
                # constant-folding passes.
                zb = torch.zeros(size=(output_slice_size,), dtype=torch.int32)
                with graph_module.graph.inserting_after(weight):
                    group_bias = create_constant_placeholder(
                        self.exported_program,
                        graph=graph_module.graph,
                        kind=InputKind.PARAMETER,
                        data=zb,
                        persistent_buffer=True,
                        name=f"{node.name}_g{g}_zero_bias",
                    )
                    group_bias.meta[
                        TosaSpecialDtype.meta_key()
                    ] = TosaSpecialDtype.INT48
            elif bias is not None:
                with graph_module.graph.inserting_before(node):
                    group_bias = create_node(
                        graph=graph_module.graph,
                        op_target=exir_ops.edge.aten.slice_copy.Tensor,
                        args=(
                            bias,
                            0,
                            g * output_slice_size,
                            (g + 1) * output_slice_size,
                        ),
                        from_node=bias,
                    )
                # Propagate INT48 tag from parent bias if present.
                if bias.meta.get(TosaSpecialDtype.meta_key()) == TosaSpecialDtype.INT48:
                    group_bias.meta[
                        TosaSpecialDtype.meta_key()
                    ] = TosaSpecialDtype.INT48
            else:
                dtype = torch.int32 if is_quantized else node.meta["val"].dtype
                zb = torch.zeros(size=(output_slice_size,), dtype=dtype)
                with graph_module.graph.inserting_after(weight):
                    group_bias = create_constant_placeholder(
                        self.exported_program,
                        graph=graph_module.graph,
                        kind=InputKind.PARAMETER,
                        data=zb,
                        persistent_buffer=True,
                        name=f"{node.name}_g{g}_bias",
                    )
                    if input_dtype == torch.int16:
                        group_bias.meta[
                            TosaSpecialDtype.meta_key()
                        ] = TosaSpecialDtype.INT48

            # ---- TOSA.CONV2D --------------------------------------------
            conv_args = (
                sliced_input,
                sliced_weight_permuted,
                group_bias,
                stride_tuple,
                pad_attr,
                dilation_tuple,
            )
            with graph_module.graph.inserting_after(cursor):
                tosa_op = create_node(
                    graph=graph_module.graph,
                    op_target=exir_ops.backend.tosa.CONV2D.default,
                    args=conv_args,
                    from_node=node,
                    inherit_qparams=True,
                )
            cursor = tosa_op

            # ---- Per-group quantised output -----------------------------
            if is_int8 or is_int16_no_bias:
                if len(full_weight_scale) > 1:  # per-channel
                    gws = full_weight_scale[
                        g * output_slice_size : (g + 1) * output_slice_size
                    ]
                else:
                    gws = full_weight_scale
                gscale = [(input_scale * w) / output_scale for w in gws]
                with graph_module.graph.inserting_after(cursor):
                    rescale = create_node(
                        graph=graph_module.graph,
                        op_target=exir_ops.backend.tosa.RESCALE.default,
                        args=(tosa_op, rescale_dtype, gscale, 0, output_zp),
                        from_node=node,
                    )
                if is_int16_no_bias:
                    tosa_op.meta[
                        TosaSpecialDtype.meta_key()
                    ] = TosaSpecialDtype.INT48
                cursor = rescale
                group_outputs.append(rescale)
            elif is_int16_with_bias:
                # Full per-group INT16+bias decomposition so that each
                # group is self-contained (required by U55 Vela).
                output_qparams = cast(
                    QuantArgs, node.meta["output_qparams"][0]
                )
                bias_qparams = cast(
                    QuantArgs, node.meta["input_qparams"][2]
                )
                if bias_qparams.per_channel:
                    full_bias_scale = bias_qparams.get_scale_per_channel()
                else:
                    full_bias_scale = [bias_qparams.get_scale_per_tensor()]

                tosa_op.meta[
                    TosaSpecialDtype.meta_key()
                ] = TosaSpecialDtype.INT48

                # 1. RESCALE INT48 → INT32 (identity)
                with graph_module.graph.inserting_after(cursor):
                    int48_rescale = create_node(
                        graph=graph_module.graph,
                        op_target=exir_ops.backend.tosa.RESCALE.default,
                        args=(
                            tosa_op,
                            torch.int32,
                            [1.0] * output_slice_size,
                            0,
                            0,
                        ),
                        from_node=node,
                    )
                cursor = int48_rescale

                # 2. Slice original bias for this group
                with graph_module.graph.inserting_before(node):
                    group_bias_slice = create_node(
                        graph=graph_module.graph,
                        op_target=exir_ops.edge.aten.slice_copy.Tensor,
                        args=(
                            original_bias,
                            0,
                            g * output_slice_size,
                            (g + 1) * output_slice_size,
                        ),
                        from_node=original_bias,
                    )

                # 3. Reshape sliced bias to NHWC: [1, 1, ..., 1, C_group]
                group_bias_view_shape = [
                    1,
                    *([1] * (rank - 2)),
                    output_slice_size,
                ]
                with graph_module.graph.inserting_after(cursor):
                    group_bias_view = create_node(
                        graph=graph_module.graph,
                        op_target=exir_ops.edge.aten.view_copy.default,
                        args=(group_bias_slice, group_bias_view_shape),
                        from_node=original_bias,
                    )
                cursor = group_bias_view

                # 4. ADD bias (INT32, NHWC)
                with graph_module.graph.inserting_after(cursor):
                    group_bias_add = create_node(
                        graph=graph_module.graph,
                        op_target=exir_ops.edge.aten.add.Tensor,
                        args=(int48_rescale, group_bias_view),
                        from_node=node,
                    )
                cursor = group_bias_add

                # 5. RESCALE INT32 → INT16 with group-specific scale
                if len(full_bias_scale) > 1:  # per-channel
                    gbs = full_bias_scale[
                        g * output_slice_size : (g + 1) * output_slice_size
                    ]
                else:
                    gbs = full_bias_scale
                group_final_scale = [
                    b / output_qparams.scale for b in gbs
                ]
                with graph_module.graph.inserting_after(cursor):
                    group_final_rescale = create_node(
                        graph=graph_module.graph,
                        op_target=exir_ops.backend.tosa.RESCALE.default,
                        args=(
                            group_bias_add,
                            output_qparams.dtype,
                            group_final_scale,
                            0,
                            0,
                        ),
                        from_node=node,
                    )
                cursor = group_final_rescale
                group_outputs.append(group_final_rescale)
            else:
                group_outputs.append(tosa_op)

        # ---- Cat along NHWC channel dim ---------------------------------
        with graph_module.graph.inserting_after(cursor):
            cat_node = create_node(
                graph=graph_module.graph,
                op_target=exir_ops.edge.aten.cat.default,
                args=(group_outputs, nhwc_channel_dim),
                from_node=node,
            )
        cursor = cat_node

        # ---- ONE output permute NHWC→NCHW -------------------------------
        output_permute = self._insert_permute(
            graph_module, cursor, cursor, nhwc_to_nchw, before=False
        )
        return output_permute

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

            # Insert activation permute: NCHW → NHWC
            rank = len(input_shape)
            nchw_to_nhwc = self._nchw_to_nhwc_perm(rank)
            nhwc_to_nchw = self._nhwc_to_nchw_perm(rank)

            # Grouped conv (not depthwise): decompose in NHWC with a single
            # input/output permute pair so downstream permute-optimisation
            # passes cannot break the output layout.
            if not transposed and self._is_grouped_conv(node):
                result = self._handle_grouped_conv(
                    graph_module, node, x, weight, bias,
                    stride_list, pad_list, dilation_list,
                    group, input_shape, weight_shape, spatial_rank, rank,
                )
                node.replace_all_uses_with(result)
                graph_module.graph.erase_node(node)
                continue

            x_permuted = self._insert_permute(
                graph_module, node, x, nchw_to_nhwc, before=True
            )

            conv_args: tuple[Any, ...]
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
                # Weight permute: IOHW → OHWI
                weight_perm = [1, 2, 3, 0]
                weight_permuted = self._insert_permute(
                    graph_module, node, weight, weight_perm, before=True
                )
                conv_args = (
                    x_permuted,
                    weight_permuted,
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
                elif self._is_depthwise_conv2d(node):
                    target_op = exir_ops.backend.tosa.DEPTHWISE_CONV2D.default
                    # If there are any TOSA.DEPTHWISE_CONV2D nodes using
                    # the weights (possibly via a permute_copy), we've
                    # already reshaped them.
                    already_reshaped = any(
                        user.target == target_op
                        or (
                            user.target
                            == exir_ops.edge.aten.permute_copy.default
                            and any(
                                u2.target == target_op for u2 in user.users
                            )
                        )
                        for user in weight.users
                    )
                    if not already_reshaped:
                        self._reshape_weights(weight, input_fake_tensor.shape[1])
                    weight_fake_tensor = get_first_fake_tensor(weight)
                else:
                    target_op = exir_ops.backend.tosa.CONV2D.default

                # Weight permute: OIHW → OHWI (or reshaped depthwise equivalent)
                weight_perm = self._nchw_to_nhwc_perm(
                    len(weight_fake_tensor.shape)
                )
                weight_permuted = self._insert_permute(
                    graph_module, node, weight, weight_perm, before=True
                )
                conv_args = (
                    x_permuted,
                    weight_permuted,
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
            input_fake_nhwc = input_fake_tensor.permute(nchw_to_nhwc)
            weight_fake_permuted = weight_fake_tensor.permute(weight_perm)

            tosa_node_fake_tensor = target_op(
                input_fake_nhwc,
                weight_fake_permuted,
                bias_fake_tensor,
                *conv_args[3:],
            )

            # Insert output permute: NHWC → NCHW
            output_permute = self._insert_permute(
                graph_module, tosa_op, tosa_op, nhwc_to_nchw, before=False
            )

            if (
                tosa_node_fake_tensor.dtype == torch.int32
                and input_fake_tensor.dtype == torch.int8
            ):
                output_rescale = self.insert_output_rescale(graph_module, node, tosa_op)
                output_permute_after_rescale = self._insert_permute(
                    graph_module, output_rescale, output_rescale, nhwc_to_nchw, before=False
                )
                output_permute.replace_all_uses_with(tosa_op)
                graph_module.graph.erase_node(output_permute)
                node.replace_all_uses_with(output_permute_after_rescale)
            elif (
                tosa_node_fake_tensor.dtype == torch.int32
                and input_fake_tensor.dtype == torch.int16
            ):
                has_bias = len(node.meta["input_qparams"]) > 2
                if not has_bias:
                    output_rescale = self.insert_output_rescale(
                        graph_module, node, tosa_op
                    )
                    output_permute_after_rescale = self._insert_permute(
                        graph_module, output_rescale, output_rescale, nhwc_to_nchw, before=False
                    )
                    output_permute.replace_all_uses_with(tosa_op)
                    graph_module.graph.erase_node(output_permute)
                    node.replace_all_uses_with(output_permute_after_rescale)
                else:
                    # INT16 conv with bias: the TOSA conv produces INT48
                    # output. We handle the full bias decomposition here
                    # entirely in NHWC layout, with the output permute placed
                    # AFTER the final RESCALE.
                    #
                    # Graph produced:
                    #   tosa.CONV2D(input, weight, zero_bias_INT48) → INT48, NHWC
                    #   → RESCALE(INT48 → INT32, scale=1.0, NHWC)
                    #   → ADD(bias reshaped to [1,1,...,1,C] for NHWC broadcast)
                    #   → RESCALE(INT32 → INT16, final_scale, NHWC)
                    #   → permute(NHWC → NCHW)

                    # Save original bias before replacing with zero bias
                    original_bias_node = node.args[2]

                    # Create a zero bias tagged INT48 for the TOSA conv
                    output_channels = get_first_fake_tensor(node).shape[1]
                    zero_bias_data = torch.zeros(
                        size=(output_channels,), dtype=torch.int32
                    )
                    with graph_module.graph.inserting_after(weight):
                        zero_bias_node = create_constant_placeholder(
                            self.exported_program,
                            graph=graph_module.graph,
                            kind=InputKind.PARAMETER,
                            data=zero_bias_data,
                            persistent_buffer=True,
                            name=f"{node.name}_zero_bias",
                        )
                        zero_bias_node.meta[
                            TosaSpecialDtype.meta_key()
                        ] = TosaSpecialDtype.INT48
                    # Replace original bias with zero bias in tosa conv
                    bias_arg_index = list(tosa_op.args).index(bias)
                    tosa_op.update_arg(bias_arg_index, zero_bias_node)

                    output_qparams = cast(
                        QuantArgs, node.meta["output_qparams"][0]
                    )
                    bias_qparams = cast(
                        QuantArgs, node.meta["input_qparams"][2]
                    )
                    if bias_qparams.per_channel:
                        bias_scale = bias_qparams.get_scale_per_channel()
                    else:
                        bias_scale = [bias_qparams.get_scale_per_tensor()]

                    # Remove the original output permute first — we'll add
                    # a new one at the very end of the chain.
                    output_permute.replace_all_uses_with(tosa_op)
                    graph_module.graph.erase_node(output_permute)

                    # Build the chain sequentially, each node after the
                    # previous, so graph ordering matches logical ordering.
                    # Use a cursor variable to track insertion point.
                    cursor = tosa_op

                    # 1. RESCALE INT48 → INT32 (NHWC)
                    conv_rescale_factors = [1.0] * len(bias_scale)
                    with graph_module.graph.inserting_after(cursor):
                        int48_rescale = create_node(
                            graph=graph_module.graph,
                            op_target=exir_ops.backend.tosa.RESCALE.default,
                            args=(tosa_op, torch.int32, conv_rescale_factors, 0, 0),
                            from_node=node,
                        )
                    cursor = int48_rescale

                    # 2. Reshape bias to NHWC: [1, 1, ..., 1, C]
                    bias_data = get_first_fake_tensor(original_bias_node)
                    bias_view_shape = [1, *([1] * (rank - 2)), bias_data.shape[0]]
                    with graph_module.graph.inserting_after(cursor):
                        bias_view = create_node(
                            graph=graph_module.graph,
                            op_target=exir_ops.edge.aten.view_copy.default,
                            args=(original_bias_node, bias_view_shape),
                            from_node=original_bias_node,
                        )
                    cursor = bias_view

                    # 3. ADD bias (NHWC)
                    with graph_module.graph.inserting_after(cursor):
                        bias_add = create_node(
                            graph=graph_module.graph,
                            op_target=exir_ops.edge.aten.add.Tensor,
                            args=(int48_rescale, bias_view),
                            from_node=node,
                        )
                    cursor = bias_add

                    # 4. RESCALE INT32 → output dtype (NHWC)
                    final_output_scale = [
                        b / output_qparams.scale for b in bias_scale
                    ]
                    with graph_module.graph.inserting_after(cursor):
                        final_rescale = create_node(
                            graph=graph_module.graph,
                            op_target=exir_ops.backend.tosa.RESCALE.default,
                            args=(bias_add, output_qparams.dtype, final_output_scale, 0, 0),
                            from_node=node,
                        )
                    cursor = final_rescale

                    # 5. Output permute NHWC → NCHW (LAST in chain)
                    output_permute_after_bias = self._insert_permute(
                        graph_module, cursor, cursor, nhwc_to_nchw, before=False
                    )

                    node.replace_all_uses_with(output_permute_after_bias)

                tosa_op.meta[TosaSpecialDtype.meta_key()] = TosaSpecialDtype.INT48
            else:
                node.replace_all_uses_with(output_permute)

            graph_module.graph.erase_node(node)

        if modified:
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, modified)
