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
    get_constant_placeholder_kind,
    get_first_fake_tensor,
    get_param_tensor,
    is_persistent_buffer,
    permute_fake_tensor_metadata,
)
from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    get_input_qparams,
    get_output_qparams,
)
from executorch.backends.arm._passes.symbolic_value_range import (
    evaluate_symbolic_expr_values,
)
from executorch.backends.arm.constants import (
    HWCM_ORDER,
    NHWC_INVERSE_ORDER,
    NHWC_ORDER,
    ODHWI_INVERSE_ORDER,
    ODHWI_ORDER,
    OHWI_ORDER,
)
from executorch.backends.arm.tosa.mapping import TosaSpecialDtype
from executorch.backends.arm.tosa.specification import get_context_shape_env
from executorch.backends.transforms.utils import create_constant_placeholder
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

from torch._subclasses.fake_tensor import FakeTensor
from torch.export.graph_signature import InputKind


class RewriteConvPass(ArmPass):
    """Rewrites aten.convolution to TOSA conv ops
    (CONV2D/DEPTHWISE/TRANSPOSE/CONV3D).
    """

    _FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

    def __init__(self, exported_program: torch.export.ExportedProgram, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exported_program = exported_program

    _passes_required_after: Set[Type[ExportPass]] = set()

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
            exact_values = evaluate_symbolic_expr_values(mod_remainder, shape_env)
            if exact_values is not None:
                mod_remainder_upper = max(exact_values)
                if len(exact_values) == 1:
                    mod_remainder = int(next(iter(exact_values)))
                elif mod_remainder_upper == 0:
                    mod_remainder = 0
                else:
                    return pad - mod_remainder
            else:
                # SizeAdjustInputPass already trims symbolic remainder classes
                # that would force negative padding. Keep the symbolic
                # expression here instead of asking ShapeEnv to normalize it.
                return pad - mod_remainder
        if mod_remainder > pad:
            raise RuntimeError(
                "This case should be handled by SizeAdjustInputPass, is it enabled?\n"
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
            # Both grouped and depthwise Conv3D are decomposed into groups==1
            # convolutions by DecomposeGroupedConvPass before reaching here.
            # This guard is defense-in-depth for paths that bypass that pass.
            if groups != 1:
                raise RuntimeError(
                    "CONV3D with groups != 1 reached unexpectedly; "
                    "DecomposeGroupedConvPass should have decomposed it first."
                )
            return True
        return False

    def _add_bias(
        self,
        graph_module: torch.fx.GraphModule,
        node: torch.fx.Node,
        weight_node: torch.fx.Node,
        input_fake_tensor: torch.Tensor,
    ) -> torch.fx.Node:
        output_channels = get_first_fake_tensor(node).shape[1]
        # Add a zero bias with the dtype TOSA expects: int32 for
        # quantized conv, fp16 for FP8 conv, and the output dtype otherwise.
        if self._is_quantized_conv(node):
            bias_data = torch.zeros(size=(output_channels,), dtype=torch.int32)
        elif input_fake_tensor.dtype in self._FP8_DTYPES:
            bias_data = torch.zeros(size=(output_channels,), dtype=torch.float16)
        else:
            output_dtype = node.meta["val"].dtype
            bias_data = torch.zeros(size=(output_channels,), dtype=output_dtype)

        # Constant placeholders must appear before user-input placeholders in
        # the graph. Insert the synthetic bias at the first placeholder slot
        # instead of near the conv node.
        first_placeholder = next(
            n for n in graph_module.graph.nodes if n.op == "placeholder"
        )
        with graph_module.graph.inserting_before(first_placeholder):
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

    def _rewrite_fp8_bias(
        self,
        graph_module: torch.fx.GraphModule,
        node: torch.fx.Node,
        bias_node: torch.fx.Node,
    ) -> torch.fx.Node:
        bias_tensor = get_param_tensor(  # type: ignore[arg-type]
            self.exported_program, bias_node
        )
        if bias_tensor is None:
            raise RuntimeError(
                f"Bias node {bias_node.name} is not a parameter or buffer"
            )

        kind = get_constant_placeholder_kind(self.exported_program, bias_node)
        persistent_buffer = is_persistent_buffer(self.exported_program, bias_node)
        with graph_module.graph.inserting_after(bias_node):
            return create_constant_placeholder(
                self.exported_program,
                graph=graph_module.graph,
                name=f"{node.name}_bias_fp16",
                kind=kind,
                data=bias_tensor.to(torch.float16),
                persistent_buffer=persistent_buffer,
            )

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

    @staticmethod
    def _combine_rescale_scales(
        lhs_scales: list[float], rhs_scales: list[float]
    ) -> list[float]:
        """Multiply scalar or equally sized per-channel rescale factors."""
        if not lhs_scales or not rhs_scales:
            raise ValueError("Cannot combine empty rescale factors.")
        if (
            len(lhs_scales) != 1
            and len(rhs_scales) != 1
            and len(lhs_scales) != len(rhs_scales)
        ):
            raise ValueError(
                "Cannot combine rescales with incompatible scale counts: "
                f"{len(lhs_scales)} and {len(rhs_scales)}."
            )
        scale_count = max(len(lhs_scales), len(rhs_scales))
        return [
            lhs_scales[index % len(lhs_scales)] * rhs_scales[index % len(rhs_scales)]
            for index in range(scale_count)
        ]

    @staticmethod
    def _is_direct_int32_rescale(node: torch.fx.Node) -> bool:
        """Return whether a node directly rescales its input to INT32."""
        return (
            node.op == "call_function"
            and node.target == exir_ops.backend.tosa.RESCALE.default
            and len(node.args) > 1
            and node.args[1] == torch.int32
        )

    def _get_direct_int32_rescale_users(
        self, node: torch.fx.Node
    ) -> list[torch.fx.Node]:
        """Return consumers that directly request an INT32 value."""
        return [user for user in node.users if self._is_direct_int32_rescale(user)]

    def _get_permute_int32_rescale_users(
        self, node: torch.fx.Node
    ) -> dict[torch.fx.Node, list[torch.fx.Node]]:
        """Return permutations and their direct INT32 rescale consumers."""
        result: dict[torch.fx.Node, list[torch.fx.Node]] = {}
        for user in node.users:
            if (
                user.op == "call_function"
                and user.target == exir_ops.edge.aten.permute_copy.default
            ):
                if int32_users := self._get_direct_int32_rescale_users(user):
                    result[user] = int32_users
        return result

    @staticmethod
    def _insert_layout_permute(
        graph_module: torch.fx.GraphModule,
        source_node: torch.fx.Node,
        input_node: torch.fx.Node,
        input_fake_tensor: FakeTensor,
        dims: tuple[int, ...],
    ) -> tuple[torch.fx.Node, FakeTensor]:
        """Insert the mandatory TOSA-to-Edge output layout permutation."""
        with graph_module.graph.inserting_after(input_node):
            output = create_node(
                graph=graph_module.graph,
                op_target=exir_ops.edge.aten.permute_copy.default,
                args=(input_node, list(dims)),
                from_node=source_node,
            )
        output_fake_tensor = permute_fake_tensor_metadata(input_fake_tensor, dims)
        output.meta["val"] = output_fake_tensor
        return output, output_fake_tensor

    def _insert_a16w8_output_branches(
        self,
        graph_module: torch.fx.GraphModule,
        node: torch.fx.Node,
        tosa_op: torch.fx.Node,
        tosa_node_fake_tensor: torch.Tensor,
        default_rescale: torch.fx.Node,
        post_permute_dims: tuple[int, ...],
    ) -> None:
        """Route A16W8 convolution users through the required output types.

        Keep INT32 consumers on a widened path from the INT48 accumulator, even
        though the convolution's declared output is INT16. This preserves
        numerical range by avoiding unnecessary INT16 rounding and clamping.

        """
        direct_int32_users = self._get_direct_int32_rescale_users(node)
        permute_int32_users = self._get_permute_int32_rescale_users(node)
        if not direct_int32_users and not permute_int32_users:
            return

        conv_scales = cast(list[float], default_rescale.args[2])
        conv_output_zp = cast(int, default_rescale.args[4])

        # The layout permutation cannot consume the INT48 accumulator. Fork
        # before narrowing. A direct INT32 rescale can be combined with the
        # convolution rescale:
        #
        #                         CONV (INT48)
        #                               |
        #                    +----------+---------+
        #                    |                    |
        #          RESCALE (INT16)     RESCALE (INT32, combined)
        #                    |                    |
        #                 PERMUTE              PERMUTE
        #                    |                    |
        #             INT16 consumer        INT32 consumer
        #
        # Direct consumers have no axis-changing operation between the
        # convolution and rescale, so both scale factors can be combined.
        for int32_user in direct_int32_users:
            # Retarget the existing consumer rescale to the TOSA convolution;
            # its previous consumers will be moved after the layout permute.
            user_scales = cast(list[float], int32_user.args[2])
            user_input_zp = int32_user.args[3]
            user_output_zp = cast(int, int32_user.args[4])
            if conv_output_zp != user_input_zp:
                raise ValueError(
                    "Cannot combine convolution and INT32 rescales with "
                    f"different intermediate zero points: {conv_output_zp} "
                    f"and {user_input_zp}."
                )

            # The intermediate zero points cancel, so the direct path can
            # combine both scales without rounding through INT16.
            combined_scales = self._combine_rescale_scales(conv_scales, user_scales)
            # Capture the old consumers before adding the layout permute. A
            # blanket replace afterward would also rewrite the new permute's
            # input and create a cycle.
            previous_users = list(int32_user.users)
            int32_user.args = (
                tosa_op,
                torch.int32,
                combined_scales,
                0,
                user_output_zp,
            )
            int32_fake_tensor = exir_ops.backend.tosa.RESCALE.default(
                tosa_node_fake_tensor,
                torch.int32,
                combined_scales,
                0,
                user_output_zp,
            )
            int32_user.meta["val"] = int32_fake_tensor
            int32_output, _ = self._insert_layout_permute(
                graph_module,
                node,
                int32_user,
                int32_fake_tensor,
                post_permute_dims,
            )
            for user in previous_users:
                user.replace_input_with(int32_user, int32_output)

        # A source permutation may change the per-channel axis, so keep both
        # rescales on a separate widened branch instead of combining them:
        #
        #              CONV (INT48)
        #                   |
        #           RESCALE (INT32)
        #                   |
        #           layout PERMUTE
        #                   |
        #           source PERMUTE
        #                   |
        #           RESCALE (INT32)
        #                   |
        #           INT32 consumer
        #
        # This also avoids an intermediate INT16 rounding step.
        for permute, int32_users in permute_int32_users.items():
            # Convert the accumulator to the declared convolution domain, but
            # use INT32 storage to avoid clamping values to the INT16 range.
            with graph_module.graph.inserting_after(tosa_op):
                widened_rescale = create_node(
                    graph=graph_module.graph,
                    op_target=exir_ops.backend.tosa.RESCALE.default,
                    args=(
                        tosa_op,
                        torch.int32,
                        conv_scales,
                        0,
                        conv_output_zp,
                    ),
                    from_node=node,
                )
            widened_fake_tensor = exir_ops.backend.tosa.RESCALE.default(
                tosa_node_fake_tensor,
                torch.int32,
                conv_scales,
                0,
                conv_output_zp,
            )
            widened_rescale.meta["val"] = widened_fake_tensor
            widened_output, widened_output_fake_tensor = self._insert_layout_permute(
                graph_module,
                node,
                widened_rescale,
                widened_fake_tensor,
                post_permute_dims,
            )

            # Clone the source permutation for the widened branch. The
            # original may still have sibling consumers that require INT16.
            source_permute_dims = cast(list[int], permute.args[1])
            with graph_module.graph.inserting_after(widened_output):
                widened_permute = create_node(
                    graph=graph_module.graph,
                    op_target=exir_ops.edge.aten.permute_copy.default,
                    args=(widened_output, source_permute_dims),
                    from_node=permute,
                )
            widened_permute.meta["val"] = permute_fake_tensor_metadata(
                widened_output_fake_tensor,
                tuple(source_permute_dims),
            )
            for int32_user in int32_users:
                int32_user.replace_input_with(permute, widened_permute)

            if not permute.users:
                graph_module.graph.erase_node(permute)

    def _insert_output_conversion(
        self,
        graph_module: torch.fx.GraphModule,
        node: torch.fx.Node,
        tosa_op: torch.fx.Node,
        input_fake_tensor: torch.Tensor,
        tosa_node_fake_tensor: torch.Tensor,
    ) -> tuple[torch.fx.Node, FakeTensor]:
        # Convolutions that match none of the special cases below require no
        # output conversion and keep the original TOSA node.
        node_replacement: torch.fx.Node = tosa_op
        node_replacement_fake_tensor = tosa_node_fake_tensor
        is_a8w8_conv = (
            tosa_node_fake_tensor.dtype == torch.int32
            and input_fake_tensor.dtype == torch.int8
        )
        is_a16w8_conv = (
            tosa_node_fake_tensor.dtype == torch.int32
            and input_fake_tensor.dtype == torch.int16
        )
        is_fp8_conv = (
            tosa_node_fake_tensor.dtype == torch.float16
            and input_fake_tensor.dtype in self._FP8_DTYPES
        )
        if is_a8w8_conv:
            node_replacement, node_replacement_fake_tensor = self.insert_output_rescale(
                graph_module, node, tosa_op, tosa_node_fake_tensor
            )
        elif is_a16w8_conv:
            # Explicit layout paths require a post-conv permute, which cannot
            # consume INT48. Return the declared INT16 rescale so the caller
            # can create its standard layout permute first, then add parallel
            # INT32 branches with their own layout permutations. If every
            # consumer moves to INT32, the caller removes the unused INT16
            # rescale and permutation.
            node_replacement, node_replacement_fake_tensor = self.insert_output_rescale(
                graph_module, node, tosa_op, tosa_node_fake_tensor
            )
            tosa_op.meta[TosaSpecialDtype.meta_key()] = TosaSpecialDtype.INT48
        elif is_fp8_conv:
            node_output_fake_tensor = get_first_fake_tensor(node)
            # TOSA FP8 conv widens the output. Cast back to the exported
            # graph dtype before the post-layout permute.
            node_replacement_fake_tensor = (
                exir_ops.edge.dim_order_ops._to_dim_order_copy.default(
                    tosa_node_fake_tensor,
                    dtype=node_output_fake_tensor.dtype,
                )
            )
            with graph_module.graph.inserting_after(tosa_op):
                node_replacement = create_node(
                    graph=graph_module.graph,
                    op_target=exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
                    args=(tosa_op,),
                    kwargs={"dtype": node_output_fake_tensor.dtype},
                    from_node=tosa_op,
                )
            node_replacement.meta["val"] = node_replacement_fake_tensor

        return node_replacement, cast(FakeTensor, node_replacement_fake_tensor)

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
                bias = self._add_bias(graph_module, node, weight, input_fake_tensor)
            elif isinstance(bias, torch.fx.Node):
                if input_fake_tensor.dtype in self._FP8_DTYPES:
                    bias = self._rewrite_fp8_bias(graph_module, node, bias)
                else:
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
                input_tensor_for_tosa_fake = permute_fake_tensor_metadata(
                    input_fake_tensor, pre_permute_dims
                )
                x.meta["val"] = input_tensor_for_tosa_fake
                weight = self._rewrite_weight(
                    graph_module,
                    weight,
                    node,
                    permute_dims=OHWI_ORDER,
                    name_suffix="ohwi",
                )
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
                    pre_permute_dims = ODHWI_ORDER
                    post_permute_dims = ODHWI_INVERSE_ORDER
                    with graph_module.graph.inserting_before(node):
                        x = create_node(
                            graph=graph_module.graph,
                            op_target=exir_ops.edge.aten.permute_copy.default,
                            args=(x, list(pre_permute_dims)),
                            from_node=node,
                        )
                    input_tensor_for_tosa_fake = permute_fake_tensor_metadata(
                        input_fake_tensor, pre_permute_dims
                    )
                    x.meta["val"] = input_tensor_for_tosa_fake
                    weight = self._rewrite_weight(
                        graph_module,
                        weight,
                        node,
                        permute_dims=ODHWI_ORDER,
                        name_suffix="odhwi",
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
                    input_tensor_for_tosa_fake = permute_fake_tensor_metadata(
                        input_fake_tensor, pre_permute_dims
                    )
                    x.meta["val"] = input_tensor_for_tosa_fake
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
                    input_tensor_for_tosa_fake = permute_fake_tensor_metadata(
                        input_fake_tensor, pre_permute_dims
                    )
                    x.meta["val"] = input_tensor_for_tosa_fake
                    weight = self._rewrite_weight(
                        graph_module,
                        weight,
                        node,
                        permute_dims=NHWC_ORDER,
                        name_suffix="ohwi",
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

            # Compute fake tensor BEFORE materializing SymInts into FX nodes,
            # since the underlying op expects ints/SymInts (not FX Nodes).
            bias_fake_tensor = get_first_fake_tensor(bias) if bias else None
            tosa_node_fake_tensor = target_op(
                input_tensor_for_tosa_fake,
                weight_fake_tensor,
                bias_fake_tensor,
                *conv_args[3:],
            )

            # ``Graph.create_node`` rejects raw SymInts in call_function args.
            # If ``pad`` contains symbolic entries, materialize them into FX
            # nodes so the TOSA conv node references the producing graph
            # subgraph instead of holding raw SymInts.
            if isinstance(pad, (list, tuple)) and any(
                isinstance(p, torch.SymInt) for p in pad
            ):
                with graph_module.graph.inserting_before(node):
                    materialized_pad = graph_module.graph.materialize_symints(pad)
                new_conv_args = list(conv_args)
                new_conv_args[4] = materialized_pad
                conv_args = tuple(new_conv_args)

            with graph_module.graph.inserting_after(node):
                tosa_op = create_node(
                    graph=graph_module.graph,
                    op_target=target_op,
                    args=conv_args,
                    from_node=node,
                    inherit_qparams=True,
                )
            tosa_op.meta["val"] = tosa_node_fake_tensor

            node_replacement, node_replacement_fake_tensor = (
                self._insert_output_conversion(
                    graph_module,
                    node,
                    tosa_op,
                    input_fake_tensor,
                    tosa_node_fake_tensor,
                )
            )

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
            node_replacement.meta["val"] = permute_fake_tensor_metadata(
                node_replacement_fake_tensor, post_permute_dims
            )

            is_a16w8_conv = (
                tosa_node_fake_tensor.dtype == torch.int32
                and input_fake_tensor.dtype == torch.int16
            )
            if is_a16w8_conv:
                # Keep values in INT32 whenever a consumer supports it, even
                # though the declared output is INT16, by branching from the
                # accumulator before narrowing.
                self._insert_a16w8_output_branches(
                    graph_module,
                    node,
                    tosa_op,
                    tosa_node_fake_tensor,
                    post_permute_input,
                    post_permute_dims,
                )
                # Only users not moved to widened branches remain on the
                # source node. Route those through the declared INT16 output,
                # or remove that branch when every consumer was moved.
                if node.users:
                    node.replace_all_uses_with(node_replacement)
                else:
                    graph_module.graph.erase_node(node_replacement)
                    graph_module.graph.erase_node(post_permute_input)
            else:
                node.replace_all_uses_with(node_replacement)

            graph_module.graph.erase_node(node)

        if modified:
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, modified)
