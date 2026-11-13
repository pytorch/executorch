# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, List, Optional, Tuple

import executorch.backends.xnnpack.custom_ops_lib  # noqa: F401  registers et_xnnpack ops
import torch
from executorch.backends.transforms.utils import (
    create_constant_placeholder,
    delete_constant_placeholder,
    get_param_tensor,
    is_param_node,
)
from executorch.backends.xnnpack._passes.xnnpack_pass import XNNPACKPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassResult


class DecomposeConvTransposePass(XNNPACKPass):
    """
    Lower a dense (groups == 1) 1-D or 2-D transposed conv whose kernel is a
    multiple of its stride s > 1 as one stride-1 conv that computes every
    output phase, a depth to space that interleaves the phases, and a slice
    for the padding.

    With k = m * s taps per axis, output phase j of a stride-s transposed conv
    is an m-tap correlation of the input (before the padding p is cropped):

        y[s*q + j] = sum_{u < m} x[q + u - (m - 1)] * w[s*(m - 1 - u) + j]

    The conv has padding m - 1 and s * out_channels outputs per axis, ordered
    phase-major so that the 2-D interleave is the block-major
    depth_to_space_2d. XNNPACK runs it on its convolution kernels, which are
    much faster than its deconvolution; k = 2 * s is the usual upsampler of
    audio decoders.

    Depthwise transposed convs are left to DecomposeDepthwiseConvTransposePass.
    Grouped ones and those with dilation, output_padding > padding, unequal
    2-D strides, a dequantized weight or dynamic shapes stay on XNNPACK's own
    node.
    """

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        # Plan every conv before rewriting any: the nodes created by
        # _decompose have no meta["val"] until the retrace below.
        plans = []
        for node in graph.nodes:
            if (
                node.op == "call_function"
                and node.target == exir_ops.edge.aten.convolution.default
            ):
                plan = self._plan(node)
                if plan is not None:
                    plans.append((node, plan))
        for node, plan in plans:
            self._decompose(graph, node, *plan)

        modified = bool(plans)
        if modified:
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, modified)

    def _plan(
        self, node: torch.fx.Node
    ) -> Optional[Tuple[int, List[int], List[int], List[int]]]:
        (
            x,
            weight,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        ) = node.args
        if not transposed or groups != 1:
            return None
        if not is_param_node(self.exported_program, weight):
            return None
        if bias is not None and not is_param_node(self.exported_program, bias):
            return None
        x_shape = list(x.meta["val"].shape)
        out_shape = list(node.meta["val"].shape)
        if not all(isinstance(d, int) for d in x_shape + out_shape):
            return None
        kernel = list(weight.meta["val"].shape[2:])
        dims = len(kernel)
        if dims not in (1, 2):
            return None
        # ATen broadcasts a single spatial value over every spatial dim.
        stride, padding, dilation, output_padding = (
            p * dims if len(p) == 1 else p
            for p in cast(List[List[int]], [stride, padding, dilation, output_padding])
        )
        s = stride[0]
        if (
            s < 2
            or any(v != s for v in stride)
            or any(k % s for k in kernel)
            or any(d != 1 for d in dilation)
            or any(o > p for o, p in zip(output_padding, padding))
        ):
            return None
        return s, padding, x_shape, out_shape

    def _decompose(
        self,
        graph: torch.fx.Graph,
        node: torch.fx.Node,
        s: int,
        padding: List[int],
        x_shape: List[int],
        out_shape: List[int],
    ) -> None:
        x, weight_node, bias_node = node.args[0], node.args[1], node.args[2]
        weight = get_param_tensor(self.exported_program, weight_node)
        assert weight is not None
        c_in, c_out, *kernel = weight.shape
        dims = len(kernel)
        taps = [k // s for k in kernel]
        phases = s**dims

        # w[ci, co, s*t + j] per axis -> phase_weight[(j, co), ci, m - 1 - t]
        tap_axes = [2 + 2 * i for i in range(dims)]
        phase_axes = [3 + 2 * i for i in range(dims)]
        phase_weight = (
            weight.reshape(c_in, c_out, *[d for m in taps for d in (m, s)])
            .flip(tap_axes)
            .permute(*phase_axes, 1, 0, *tap_axes)
            .reshape(phases * c_out, c_in, *taps)
            .contiguous()
        )
        phase_weight_node = self._constant_like(
            graph, weight_node, f"_s{s}_phases", phase_weight
        )
        phase_bias_node = None
        if bias_node is not None:
            bias = get_param_tensor(self.exported_program, bias_node)
            assert bias is not None
            phase_bias_node = self._constant_like(
                graph, bias_node, f"_x{phases}", bias.repeat(phases)
            )

        def call(target, *args) -> torch.fx.Node:
            return graph.create_node("call_function", target, args)

        with graph.inserting_before(node):
            y = call(
                exir_ops.edge.aten.convolution.default,
                x,
                phase_weight_node,
                phase_bias_node,
                [1] * dims,
                [m - 1 for m in taps],
                [1] * dims,
                False,
                [0] * dims,
                1,
            )
            if dims == 2:
                y = call(exir_ops.edge.et_xnnpack.depth_to_space_2d.default, y, s)
            else:
                batch, steps = x_shape[0], x_shape[2] + taps[0] - 1
                y = call(
                    exir_ops.edge.aten.view_copy.default, y, [batch, s, c_out, steps]
                )
                y = call(exir_ops.edge.aten.permute_copy.default, y, [0, 2, 3, 1])
                y = call(
                    exir_ops.edge.aten.view_copy.default, y, [batch, c_out, steps * s]
                )
            for axis, p in enumerate(padding):
                if p:
                    y = call(
                        exir_ops.edge.aten.slice_copy.Tensor,
                        y,
                        2 + axis,
                        p,
                        p + out_shape[2 + axis],
                        1,
                    )
        node.replace_all_uses_with(y)
        graph.erase_node(node)
        for constant in (weight_node, bias_node):
            if constant is not None and len(constant.users) == 0:
                delete_constant_placeholder(self.exported_program, constant)

    def _constant_like(
        self,
        graph: torch.fx.Graph,
        like: torch.fx.Node,
        suffix: str,
        data: torch.Tensor,
    ) -> torch.fx.Node:
        spec = next(
            spec
            for spec in self.exported_program.graph_signature.input_specs
            if spec.arg.name == like.name
        )
        with graph.inserting_before(like):
            node = create_constant_placeholder(
                exp_program=self.exported_program,
                graph=graph,
                kind=spec.kind,
                persistent_buffer=spec.persistent,
                name=like.name + suffix,
                data=data,
            )
        # create_constant_placeholder only sets meta["val"]; keep the
        # external-constant tag so the new constant follows the original into
        # its .ptd file.
        if "custom" in like.meta:
            node.meta["custom"] = dict(like.meta["custom"])
        return node
