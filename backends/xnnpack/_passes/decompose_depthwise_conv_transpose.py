# Copyright (c) 2026 Software Mansion
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
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


class DecomposeDepthwiseConvTransposePass(XNNPACKPass):
    """
    Lower a depthwise transposed conv (groups == in_channels == out_channels)
    with stride s > 1 into s*s depthwise convs on the low-resolution input, a
    channel concatenation and a block-major depth_to_space_2d.

    XNNPACK runs transposed convs through its subconvolution IGEMM path, one
    group at a time; with one channel per group that is far slower than its
    depthwise conv microkernels. Polyphase decomposition: output phase a of a
    stride-s transposed conv is an ordinary stride-1 correlation of the input
    with every s-th tap of the kernel,

        y[s*m + a] = sum_t x[m - t] * w[s*t + a + p],   0 <= s*t + a + p < k,

    per axis. The s*s phase outputs concatenated along channels in (ay, ax)
    order are exactly the block-major input depth_to_space_2d expects. Each
    phase reads the input through a (possibly asymmetric) zero pad.

    Only applied when the transposed conv is a clean x s upscale, i.e.
    (H - 1)*s - 2p + k + output_padding == s*H on both axes with the same s,
    dilation is 1 and every phase's taps stay inside the input; any other
    transposed conv is left to XNNPACK's own node.
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

    # (t_min, t_max) per phase for one axis, None when the axis
    # does not decompose.
    @staticmethod
    def _axis_phases(
        size: int, kernel: int, stride: int, padding: int, output_padding: int
    ) -> Optional[List[Tuple[int, int]]]:
        if stride < 2:
            return None
        if (size - 1) * stride - 2 * padding + kernel + output_padding != size * stride:
            return None
        phases = []
        for a in range(stride):
            # integer t with 0 <= stride*t + a + padding <= kernel - 1
            t_min = math.ceil(-(a + padding) / stride)
            t_max = math.floor((kernel - 1 - a - padding) / stride)
            if t_min > 0 or t_max < 0:
                return None
            phases.append((t_min, t_max))
        return phases

    def _plan(self, node: torch.fx.Node):
        (
            x,
            weight,
            _bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        ) = node.args
        if not transposed:
            return None
        if not is_param_node(self.exported_program, weight):
            return None
        x_shape = list(x.meta["val"].shape)
        w_shape = list(weight.meta["val"].shape)
        if len(x_shape) != 4 or len(w_shape) != 4:
            return None
        channels = x_shape[1]
        # transposed conv weight is (in, out / groups, kh, kw)
        if groups != channels or w_shape[0] != channels or w_shape[1] != 1:
            return None
        if any(d != 1 for d in dilation):
            return None
        # ATen broadcasts a single spatial value over both spatial dims.
        stride, padding, output_padding = (
            p * 2 if len(p) == 1 else p
            for p in cast(List[List[int]], [stride, padding, output_padding])
        )
        if stride[0] != stride[1]:
            return None
        phases_y = self._axis_phases(
            x_shape[2], w_shape[2], stride[0], padding[0], output_padding[0]
        )
        phases_x = self._axis_phases(
            x_shape[3], w_shape[3], stride[1], padding[1], output_padding[1]
        )
        if phases_y is None or phases_x is None:
            return None
        return stride[0], padding, phases_y, phases_x

    def _decompose(
        self,
        graph: torch.fx.Graph,
        node: torch.fx.Node,
        s: int,
        padding: List[int],
        phases_y: List[Tuple[int, int]],
        phases_x: List[Tuple[int, int]],
    ) -> None:
        x, weight_node, bias_node = node.args[0], node.args[1], node.args[2]
        groups = node.args[8]
        weight = get_param_tensor(self.exported_program, weight_node)
        assert weight is not None
        weight_spec = next(
            spec
            for spec in self.exported_program.graph_signature.input_specs
            if spec.arg.name == weight_node.name
        )

        phase_convs = []
        for ay, (ty_min, ty_max) in enumerate(phases_y):
            for ax, (tx_min, tx_max) in enumerate(phases_x):
                # taps of phase (ay, ax) as a stride-1 correlation kernel:
                # W[q] = w[s * (t_max - q) + a + p]
                iy = torch.tensor(
                    [
                        s * (ty_max - q) + ay + padding[0]
                        for q in range(ty_max - ty_min + 1)
                    ]
                )
                ix = torch.tensor(
                    [
                        s * (tx_max - q) + ax + padding[1]
                        for q in range(tx_max - tx_min + 1)
                    ]
                )
                phase_weight = weight[:, :, iy][:, :, :, ix].contiguous()
                with graph.inserting_before(weight_node):
                    phase_weight_node = create_constant_placeholder(
                        exp_program=self.exported_program,
                        graph=graph,
                        kind=weight_spec.kind,
                        persistent_buffer=weight_spec.persistent,
                        name=f"{weight_node.name}_phase_{ay}_{ax}",
                        data=phase_weight,
                    )
                # (left, right, top, bottom)
                pad = [tx_max, -tx_min, ty_max, -ty_min]
                with graph.inserting_before(node):
                    conv_input = x
                    if any(pad):
                        conv_input = graph.create_node(
                            "call_function",
                            exir_ops.edge.aten.constant_pad_nd.default,
                            (x, pad, 0.0),
                        )
                    phase_convs.append(
                        graph.create_node(
                            "call_function",
                            exir_ops.edge.aten.convolution.default,
                            (
                                conv_input,
                                phase_weight_node,
                                bias_node,
                                [1, 1],
                                [0, 0],
                                [1, 1],
                                False,
                                [0, 0],
                                groups,
                            ),
                        )
                    )

        with graph.inserting_before(node):
            cat = graph.create_node(
                "call_function", exir_ops.edge.aten.cat.default, (phase_convs, 1)
            )
            d2s = graph.create_node(
                "call_function",
                exir_ops.edge.et_xnnpack.depth_to_space_2d.default,
                (cat, s),
            )
        node.replace_all_uses_with(d2s)
        graph.erase_node(node)
        if len(weight_node.users) == 0:
            delete_constant_placeholder(self.exported_program, weight_node)
