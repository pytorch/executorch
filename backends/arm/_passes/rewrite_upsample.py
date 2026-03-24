# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import sympy  # type: ignore

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import (
    create_node,
    create_shape_node,
    get_first_fake_tensor,
)
from executorch.backends.arm.tosa.mapping import TosaSpecialDtype
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class RewriteUpsamplePass(ArmPass):
    """Rewrite upsample2d nodes to TOSA.RESIZE nodes with appropriate
    parameters.

    For constant parameters, CONST_SHAPE nodes are inserted for the scale,
    offset, and border values. For symbolic parameters, the parameters are
    directly passed to the TOSA.RESIZE node, and we rely on subsequent passes to
    handle them correctly once symbolic shapes are delegated by the TOSA
    backend.

    """

    targeted_ops = (
        exir_ops.edge.aten.upsample_nearest2d.vec,
        exir_ops.edge.aten.upsample_bilinear2d.vec,
    )

    _passes_required_after: Set[Type[ExportPass]] = set()

    @staticmethod
    def get_resize_parameters_1d(
        input_size: int | torch.SymInt,
        output_size: int | torch.SymInt,
        align_corners: bool,
    ):
        """Compute resize coefficients for a single spatial dimension.

        Args:
            input_size (int | torch.SymInt): Input size for the axis, possibly
                symbolic.
            output_size (int | torch.SymInt): Output size for the axis, possibly
                symbolic.
            align_corners (bool): Whether the resize should align the corner
                pixels.

        Returns:
            tuple[int, int, int, int]: Numerator, denominator, offset, and border
                terms encoded as integers.

        Raises:
            RuntimeError: If symbolic shapes are used with ``align_corners`` or if
                the computed ratio or border is not constant.

        """
        # We don't support align_corners for symbolic shapes, because handling the edge case where size == 1 is tricky.
        if align_corners:
            if (not isinstance(input_size, int)) or (not isinstance(output_size, int)):
                raise RuntimeError(
                    "We do not support align_corners=True for symbolic shapes."
                )

        # SymInt seems to not actually work for symbolic expressions, so use the underlying sympy objects instead
        input_size = (
            input_size.node._expr
            if isinstance(input_size, torch.SymInt)
            else input_size
        )
        output_size = (
            output_size.node._expr
            if isinstance(output_size, torch.SymInt)
            else output_size
        )
        if align_corners and input_size > 1 and output_size > 1:
            scale_n = output_size - 1
        else:
            scale_n = output_size
        if align_corners and input_size > 1 and output_size > 1:
            scale_d = input_size - 1
        else:
            scale_d = input_size
        ratio = scale_n / scale_d
        if not sympy.sympify(ratio).is_constant():
            raise RuntimeError(
                "Resize requires a constant ratio: " + str(ratio) + " is not constant!"
            )
        gcd = sympy.gcd(scale_n, scale_d)
        scale_n = 2 * scale_n // gcd
        scale_d = 2 * scale_d // gcd
        # These should always be whole integers, based on the above calculations
        scale_n = int(scale_n.evalf())
        scale_d = int(scale_d.evalf())

        if align_corners:
            offset = 0
        else:
            # Half pixel centers so input and output sampling positions are offset by 1/2 pixel.
            offset = scale_d // 2 - scale_n // 2

        # Calculate border to maintain the correct the output size.
        # Note that this should always result in a constant value, as the ratio is constant.
        border = scale_d * (output_size - 1) - scale_n * (input_size - 1) + offset

        if not sympy.sympify(border).is_constant():
            raise RuntimeError(
                "Resize requires a constant border: "
                + str(border)
                + " is not constant!"
            )

        border = int(sympy.sympify(border).evalf())
        return scale_n, scale_d, offset, border

    def call(self, graph_module):
        modified = False
        for node in graph_module.graph.nodes:
            if node.op != "call_function" or node.target not in self.targeted_ops:
                continue
            modified = True

            if node.target == exir_ops.edge.aten.upsample_bilinear2d.vec:
                x, output_size, align_corners, scale_factors = node.args
                resize_mode = "bilinear"
            else:
                x, output_size, scale_factors = node.args
                # As per https://docs.pytorch.org/docs/stable/generated/torch.nn.Upsample.html
                # align_corners is not valid for nearest mode. Default to False.
                align_corners = False
                resize_mode = "nearest"

            input_size_yx = node.args[0].meta["val"].shape[2:]
            output_size_yx = node.meta["val"].shape[2:]

            scale_y_n, scale_y_d, offset_y, border_y = (
                RewriteUpsamplePass.get_resize_parameters_1d(
                    input_size_yx[0], output_size_yx[0], align_corners
                )
            )
            scale_x_n, scale_x_d, offset_x, border_x = (
                RewriteUpsamplePass.get_resize_parameters_1d(
                    input_size_yx[1], output_size_yx[1], align_corners
                )
            )

            scales = [
                scale_y_n,
                scale_y_d,
                scale_x_n,
                scale_x_d,
            ]
            with graph_module.graph.inserting_before(node):
                if all(isinstance(s, int) for s in scales):
                    scale = create_shape_node(
                        graph_module.graph,
                        op_target=exir_ops.backend.tosa.CONST_SHAPE.default,
                        args=(scales,),
                        kwargs={},
                        from_node=node,
                    )
                else:
                    scale = scales
                offset = [offset_y, offset_x]
                if all(isinstance(o, int) for o in offset):
                    offset = create_shape_node(
                        graph_module.graph,
                        op_target=exir_ops.backend.tosa.CONST_SHAPE.default,
                        args=(offset,),
                        kwargs={},
                        from_node=node,
                    )
                border = [border_y, border_x]
                if all(isinstance(b, int) for b in border):
                    border = create_shape_node(
                        graph_module.graph,
                        op_target=exir_ops.backend.tosa.CONST_SHAPE.default,
                        args=(border,),
                        kwargs={},
                        from_node=node,
                    )

                tosa_resize_node = create_node(
                    graph_module.graph,
                    op_target=exir_ops.backend.tosa.RESIZE.default,
                    args=(x, scale, offset, border),
                    kwargs={"resize_mode": resize_mode},
                    from_node=node,
                    inherit_qparams=True,
                )
                node.replace_all_uses_with(tosa_resize_node)
                graph_module.graph.erase_node(node)
            input_dtype = get_first_fake_tensor(x).dtype
            if (
                input_dtype == torch.int8 or input_dtype == torch.int16
            ) and resize_mode == "bilinear":
                output_dtype = get_first_fake_tensor(node).dtype
                output_scale = float(1 / (scale_y_n * scale_x_n))
                with graph_module.graph.inserting_after(tosa_resize_node):
                    rescale_node = create_node(
                        graph_module.graph,
                        exir_ops.backend.tosa.RESCALE.default,
                    )
                    tosa_resize_node.replace_all_uses_with(rescale_node)
                    if input_dtype == torch.int16:
                        tosa_resize_node.meta[TosaSpecialDtype.meta_key()] = (
                            TosaSpecialDtype.INT48
                        )

                    rescale_node.args = (
                        tosa_resize_node,
                        output_dtype,
                        [output_scale],
                        0,  # zero point
                        0,  # zero point
                    )

        if modified:
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, modified)
