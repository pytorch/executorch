# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 Software Mansion
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import math
from dataclasses import dataclass
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


@dataclass
class _Axis:
    # Per phase: the kernel indices of its stride-1 correlation and its
    # (before, after) zero pad of the input.
    taps: List[List[int]]
    pads: List[Tuple[int, int]]
    # Each phase computes size + extra outputs; crop slices the interleaved
    # axis to the transposed conv's output.
    extra: int
    crop: Optional[Tuple[int, int]]


@dataclass
class _Plan:
    stride: int
    padding: List[int]
    depthwise: bool
    out_channels: int
    x_shape: List
    axes: List[_Axis]


def _call(graph: torch.fx.Graph, target, *args) -> torch.fx.Node:
    return graph.create_node("call_function", target, args)


def _conv(graph: torch.fx.Graph, x, w, b, padding: List[int], groups: int):
    ones, zeros = [1] * len(padding), [0] * len(padding)
    return _call(
        graph,
        exir_ops.edge.aten.convolution.default,
        *(x, w, b, ones, padding, ones, False, zeros, groups),
    )


class DecomposeConvTransposePass(XNNPACKPass):
    """
    Lower a 1-D or 2-D transposed conv with stride s > 1 as stride-1 convs
    that compute its output phases, an interleave of the phases and, where
    needed, a slice. XNNPACK runs transposed convs through its deconvolution,
    which is much slower than its convolution kernels.

    Polyphase decomposition: output phase a of a stride-s transposed conv is an
    ordinary correlation of the input with every s-th tap of the kernel,

        y[s*q + a] = sum_t x[q - t] * w[s*t + a + p],   0 <= s*t + a + p < k,

    per axis. The phases concatenated along channels in phase-major order are
    the block-major input of depth_to_space_2d; in 1-D the interleave is a
    view, a permute and a view.

    A dense transposed conv (groups == 1) computes all phases in one conv with
    s * out_channels outputs (s**2 * out_channels in 2-D). That needs the same
    taps in every phase, so it computes the output before the padding p is
    cropped, where that holds whenever the kernel is a multiple of the stride,
    and a slice drops the padding.

    A depthwise transposed conv (groups == in_channels == out_channels)
    computes each phase in its own depthwise conv, since XNNPACK only runs its
    depthwise microkernels with one output channel per group. Each phase reads
    the input through its own zero pad, so only an output size that is not a
    multiple of s needs a slice.

    Grouped transposed convs, dilation, unequal 2-D strides, a dequantized
    weight, and phases without taps or that would need a negative pad stay on
    XNNPACK's own node.
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
            self._decompose(graph, node, plan)

        modified = bool(plans)
        if modified:
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, modified)

    @staticmethod
    def _axis(
        size, kernel: int, s: int, padding: int, output_padding: int, dense: bool
    ) -> Optional[_Axis]:
        p, op = (0, 0) if dense else (padding, output_padding)
        # The output has (size - 1) * s + shift elements.
        shift = kernel - 2 * p + op
        extra = math.ceil(shift / s) - 1
        taps, pads = [], []
        for a in range(s):
            # integer t with 0 <= s*t + a + p <= kernel - 1, as a stride-1
            # correlation kernel W[q] = w[s * (t_max - q) + a + p]
            t_min = math.ceil(-(a + p) / s)
            t_max = math.floor((kernel - 1 - a - p) / s)
            pad = (t_max, extra - t_min)
            if t_min > t_max or min(pad) < 0:
                return None
            taps.append([s * (t_max - q) + a + p for q in range(t_max - t_min + 1)])
            pads.append(pad)
        if dense:
            if len(set(pads)) != 1 or output_padding > padding:
                return None
            out = (size - 1) * s - 2 * padding + kernel + output_padding
            crop = (padding, padding + out) if padding else None
        else:
            crop = (0, (size - 1) * s + shift) if shift % s else None
        return _Axis(taps, pads, extra, crop)

    def _plan(self, node: torch.fx.Node) -> Optional[_Plan]:
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
        if not transposed:
            return None
        if not is_param_node(self.exported_program, weight):
            return None
        if bias is not None and not is_param_node(self.exported_program, bias):
            return None
        w_shape = list(weight.meta["val"].shape)
        dims = len(w_shape) - 2
        if dims not in (1, 2):
            return None
        # transposed conv weight is (in, out / groups, *kernel)
        depthwise = groups > 1 and groups == w_shape[0] and w_shape[1] == 1
        if groups != 1 and not depthwise:
            return None
        # ATen broadcasts a single spatial value over every spatial dim.
        stride, padding, dilation, output_padding = (
            p * dims if len(p) == 1 else p
            for p in cast(List[List[int]], [stride, padding, dilation, output_padding])
        )
        s = stride[0]
        if s < 2 or any(v != s for v in stride) or any(d != 1 for d in dilation):
            return None
        x_shape = list(x.meta["val"].shape)
        axes = []
        for i in range(dims):
            axis = self._axis(
                x_shape[2 + i],
                w_shape[2 + i],
                s,
                padding[i],
                output_padding[i],
                not depthwise,
            )
            if axis is None:
                return None
            axes.append(axis)
        # The 1-D interleave and the slice take explicit sizes.
        needs_static = dims == 1 or any(axis.crop for axis in axes)
        if needs_static and not all(isinstance(d, int) for d in x_shape):
            return None
        out_channels = w_shape[0] if depthwise else w_shape[1]
        return _Plan(s, padding, depthwise, out_channels, x_shape, axes)

    def _decompose(self, graph: torch.fx.Graph, node: torch.fx.Node, plan: _Plan):
        x, weight_node, bias_node = node.args[0], node.args[1], node.args[2]
        with graph.inserting_before(node):
            phases = (self._depthwise_phases if plan.depthwise else self._dense_phases)(
                graph, x, weight_node, bias_node, plan
            )
            y = self._interleave(graph, phases, plan)
        node.replace_all_uses_with(y)
        graph.erase_node(node)
        for constant in (weight_node, bias_node):
            if constant is not None and len(constant.users) == 0:
                delete_constant_placeholder(self.exported_program, constant)

    def _phase_weights(
        self, weight_node: torch.fx.Node, plan: _Plan
    ) -> List[Tuple[Tuple[int, ...], torch.Tensor]]:
        weight = get_param_tensor(self.exported_program, weight_node)
        assert weight is not None
        phase_weights = []
        for phase in itertools.product(range(plan.stride), repeat=len(plan.axes)):
            w = weight
            for i, a in enumerate(phase):
                w = w.index_select(2 + i, torch.tensor(plan.axes[i].taps[a]))
            phase_weights.append((phase, w))
        return phase_weights

    def _dense_phases(self, graph, x, weight_node, bias_node, plan) -> torch.fx.Node:
        phase_weights = self._phase_weights(weight_node, plan)
        w = torch.cat([w.transpose(0, 1) for _, w in phase_weights])
        w_node = self._constant_like(
            graph, weight_node, f"_s{plan.stride}_phases", w.contiguous()
        )
        b_node = None
        if bias_node is not None:
            bias = get_param_tensor(self.exported_program, bias_node)
            assert bias is not None
            b_node = self._constant_like(
                graph,
                bias_node,
                f"_x{len(phase_weights)}",
                bias.repeat(len(phase_weights)),
            )
        padding = [axis.pads[0][0] for axis in plan.axes]
        return _conv(graph, x, w_node, b_node, padding, 1)

    def _depthwise_phases(
        self, graph, x, weight_node, bias_node, plan
    ) -> torch.fx.Node:
        # In 1-D the phases run as 2-D convs on one unsqueezed input and are
        # concatenated before one squeeze, so that XNNPACK converts the layout
        # once rather than around every phase.
        dims = len(plan.axes)
        if dims == 1:
            x = _call(graph, exir_ops.edge.aten.unsqueeze_copy.default, x, -1)
        phase_convs = []
        for phase, w in self._phase_weights(weight_node, plan):
            pads = [plan.axes[i].pads[a] for i, a in enumerate(phase)]
            if dims == 1:
                w, pads = w.unsqueeze(-1), pads + [(0, 0)]
            # The taps depend on s and padding too, so a weight shared by
            # differently configured deconvs must not reuse the placeholders
            # create_constant_placeholder finds by name.
            w_node = self._constant_like(
                graph,
                weight_node,
                f"_s{plan.stride}_p{'x'.join(map(str, plan.padding))}"
                f"_phase_{'_'.join(map(str, phase))}",
                w.contiguous(),
            )
            conv_input, conv_padding = x, [before for before, _ in pads]
            if any(before != after for before, after in pads):
                # constant_pad_nd lists the last dim first.
                conv_input = _call(
                    graph,
                    exir_ops.edge.aten.constant_pad_nd.default,
                    x,
                    [v for pad in reversed(pads) for v in pad],
                    0.0,
                )
                conv_padding = [0, 0]
            phase_convs.append(
                _conv(
                    graph,
                    conv_input,
                    w_node,
                    bias_node,
                    conv_padding,
                    plan.out_channels,
                )
            )
        y = _call(graph, exir_ops.edge.aten.cat.default, phase_convs, 1)
        if dims == 1:
            y = _call(graph, exir_ops.edge.aten.squeeze_copy.dim, y, -1)
        return y

    @staticmethod
    def _interleave(graph: torch.fx.Graph, y: torch.fx.Node, plan: _Plan):
        s, axes = plan.stride, plan.axes
        if len(axes) == 2:
            y = _call(graph, exir_ops.edge.et_xnnpack.depth_to_space_2d.default, y, s)
        else:
            batch, c = plan.x_shape[0], plan.out_channels
            steps = plan.x_shape[2] + axes[0].extra
            view = exir_ops.edge.aten.view_copy.default
            y = _call(graph, view, y, [batch, s, c, steps])
            y = _call(graph, exir_ops.edge.aten.permute_copy.default, y, [0, 2, 3, 1])
            y = _call(graph, view, y, [batch, c, steps * s])
        for i, axis in enumerate(axes):
            if axis.crop is not None:
                y = _call(
                    graph, exir_ops.edge.aten.slice_copy.Tensor, y, 2 + i, *axis.crop, 1
                )
        return y

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
