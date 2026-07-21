# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import (
    create_node,
    get_first_fake_tensor,
)
from executorch.backends.arm._passes.rewrite_avg_pool2d_pass import RewriteAvgPool2dPass
from executorch.backends.arm._passes.rewrite_upsample import RewriteUpsamplePass
from executorch.backends.arm.common.type import ensure_type
from executorch.backends.arm.tosa.resize_utils import (
    is_exact_tosa_resize_boundary_downscale_case,
)
from executorch.backends.arm.tosa.specification import TosaSpecification
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx.passes.shape_prop import _extract_tensor_metadata

_EXACT_BOUNDARY_DOWNSCALE = 16
_EQUIVALENT_AVG_POOL_KERNEL = 2
_EQUIVALENT_AVG_POOL_OFFSET = _EXACT_BOUNDARY_DOWNSCALE // 2 - 1


def is_exact_tosa_boundary_bilinear_downscale(
    node: torch.fx.Node,
    tosa_spec: TosaSpecification,
) -> bool:
    if (
        node.op != "call_function"
        or node.target != exir_ops.edge.aten.upsample_bilinear2d.vec
    ):
        return False

    input_node = ensure_type(torch.fx.Node, node.args[0])
    align_corners = ensure_type(bool, node.args[2])
    if align_corners:
        return False
    input_size_yx = get_first_fake_tensor(input_node).shape[2:]
    output_size_yx = get_first_fake_tensor(node).shape[2:]

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
    return is_exact_tosa_resize_boundary_downscale_case(
        input_hw=input_size_yx,
        output_hw=output_size_yx,
        scale=[scale_y_n, scale_y_d, scale_x_n, scale_x_d],
        offset=[offset_y, offset_x],
        border=[border_y, border_x],
        tosa_spec=tosa_spec,
    )


class DecomposeUnsupportedBilinearResizePass(ArmPass):
    targeted_ops = (exir_ops.edge.aten.upsample_bilinear2d.vec,)
    _passes_required_after: Set[Type[ExportPass]] = {RewriteAvgPool2dPass}

    def __init__(self, tosa_spec: TosaSpecification) -> None:
        super().__init__()
        self.tosa_spec = tosa_spec

    @staticmethod
    def _set_fake_tensor_meta(node: torch.fx.Node, value: torch.Tensor) -> None:
        node.meta["val"] = value
        node.meta["tensor_meta"] = _extract_tensor_metadata(value)

    def call(self, graph_module):
        graph = graph_module.graph
        modified = False
        for node in list(graph.nodes):
            if not is_exact_tosa_boundary_bilinear_downscale(node, self.tosa_spec):
                continue

            input_node = ensure_type(torch.fx.Node, node.args[0])
            input_val = input_node.meta["val"]
            input_hw = [int(dim) for dim in get_first_fake_tensor(input_node).shape[2:]]

            with graph.inserting_before(node):
                cropped_h = create_node(
                    graph,
                    op_target=exir_ops.edge.aten.slice_copy.Tensor,
                    args=(
                        input_node,
                        2,
                        _EQUIVALENT_AVG_POOL_OFFSET,
                        input_hw[0] - _EQUIVALENT_AVG_POOL_OFFSET,
                        1,
                    ),
                    from_node=node,
                )
                cropped_h_val = exir_ops.edge.aten.slice_copy.Tensor(
                    input_val,
                    2,
                    _EQUIVALENT_AVG_POOL_OFFSET,
                    input_hw[0] - _EQUIVALENT_AVG_POOL_OFFSET,
                    1,
                )
                self._set_fake_tensor_meta(cropped_h, cropped_h_val)

                cropped_hw = create_node(
                    graph,
                    op_target=exir_ops.edge.aten.slice_copy.Tensor,
                    args=(
                        cropped_h,
                        3,
                        _EQUIVALENT_AVG_POOL_OFFSET,
                        input_hw[1] - _EQUIVALENT_AVG_POOL_OFFSET,
                        1,
                    ),
                    from_node=node,
                )
                cropped_hw_val = exir_ops.edge.aten.slice_copy.Tensor(
                    cropped_h_val,
                    3,
                    _EQUIVALENT_AVG_POOL_OFFSET,
                    input_hw[1] - _EQUIVALENT_AVG_POOL_OFFSET,
                    1,
                )
                self._set_fake_tensor_meta(cropped_hw, cropped_hw_val)

                pooled = create_node(
                    graph,
                    op_target=exir_ops.edge.aten.avg_pool2d.default,
                    args=(
                        cropped_hw,
                        [_EQUIVALENT_AVG_POOL_KERNEL, _EQUIVALENT_AVG_POOL_KERNEL],
                        [_EXACT_BOUNDARY_DOWNSCALE, _EXACT_BOUNDARY_DOWNSCALE],
                    ),
                    from_node=node,
                )
                pooled_val = torch.nn.functional.avg_pool2d(
                    cropped_hw_val,
                    kernel_size=_EQUIVALENT_AVG_POOL_KERNEL,
                    stride=_EXACT_BOUNDARY_DOWNSCALE,
                )
                self._set_fake_tensor_meta(pooled, pooled_val)

            node.replace_all_uses_with(pooled)
            graph.erase_node(node)
            modified = True

        if modified:
            graph.lint()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, modified)
