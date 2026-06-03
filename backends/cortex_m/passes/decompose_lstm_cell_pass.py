# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

import torch
from executorch.backends.arm._passes.arm_pass_utils import create_node
from executorch.exir.pass_base import ExportPass, PassResult


class DecomposeLSTMCellPass(ExportPass):
    """Decompose `aten.lstm_cell` into quantizable primitives, pre-annotation.

    `nn.LSTMCell` is captured as a single opaque `aten.lstm_cell.default` op,
    so the quantizer can't observe its internal gates and `to_edge` only
    decomposes it (into `addmm` etc.) after quantization. Running this in the
    quantizer's `transform_for_annotation` phase rewrites the cell into
    `aten.linear` + slice + sigmoid/tanh + mul/add, which the cortex_m
    quantizer annotates and the existing passes lower (linear via kernel-sum,
    sigmoid/tanh via the activation LUT, add/mul via quantized_add/mul).

    For a single timestep, with gates packed [i, f, g, o] in the weight rows:
        gates = linear(x, W_ih, b_ih) + linear(h, W_hh, b_hh)
        i = sigmoid(gates[:, 0:H]);   f = sigmoid(gates[:, H:2H])
        g = tanh(gates[:, 2H:3H]);    o = sigmoid(gates[:, 3H:4H])
        c' = f * c + i * g
        h' = o * tanh(c')
    `lstm_cell` returns `(h', c')`.
    """

    _TARGET = torch.ops.aten.lstm_cell.default

    _linear = torch.ops.aten.linear.default
    _add = torch.ops.aten.add.Tensor
    _mul = torch.ops.aten.mul.Tensor
    _sigmoid = torch.ops.aten.sigmoid.default
    _tanh = torch.ops.aten.tanh.default
    _slice = torch.ops.aten.slice_copy.Tensor

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False

        for node in list(graph.nodes):
            if node.op != "call_function" or node.target != self._TARGET:
                continue

            x = node.args[0]
            h_prev, c_prev = node.args[1]
            w_ih = node.args[2]
            w_hh = node.args[3]
            b_ih = node.args[4] if len(node.args) > 4 else None
            b_hh = node.args[5] if len(node.args) > 5 else None

            H = h_prev.meta["val"].shape[-1]

            with graph.inserting_before(node):
                gates_x = create_node(
                    graph, self._linear, args=(x, w_ih, b_ih), from_node=node
                )
                gates_h = create_node(
                    graph, self._linear, args=(h_prev, w_hh, b_hh), from_node=node
                )
                gates = create_node(
                    graph, self._add, args=(gates_x, gates_h), from_node=node
                )

                # Gates are packed [i, f, g, o] along the feature dim.
                i_pre = create_node(
                    graph, self._slice, args=(gates, 1, 0, H), from_node=node
                )
                f_pre = create_node(
                    graph, self._slice, args=(gates, 1, H, 2 * H), from_node=node
                )
                g_pre = create_node(
                    graph, self._slice, args=(gates, 1, 2 * H, 3 * H), from_node=node
                )
                o_pre = create_node(
                    graph, self._slice, args=(gates, 1, 3 * H, 4 * H), from_node=node
                )

                i_t = create_node(graph, self._sigmoid, args=(i_pre,), from_node=node)
                f_t = create_node(graph, self._sigmoid, args=(f_pre,), from_node=node)
                g_t = create_node(graph, self._tanh, args=(g_pre,), from_node=node)
                o_t = create_node(graph, self._sigmoid, args=(o_pre,), from_node=node)

                f_c = create_node(graph, self._mul, args=(f_t, c_prev), from_node=node)
                i_g = create_node(graph, self._mul, args=(i_t, g_t), from_node=node)
                c_t = create_node(graph, self._add, args=(f_c, i_g), from_node=node)

                tanh_c = create_node(graph, self._tanh, args=(c_t,), from_node=node)
                h_t = create_node(graph, self._mul, args=(o_t, tanh_c), from_node=node)

            getitems = []
            for user in list(node.users.keys()):
                if user.target == operator.getitem:
                    idx = user.args[1]
                    user.replace_all_uses_with(h_t if idx == 0 else c_t)
                    getitems.append(user)
            for gi in getitems:
                graph.erase_node(gi)
            graph.erase_node(node)
            modified = True

        if not modified:
            return PassResult(graph_module, False)

        graph_module.recompile()
        graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, True)
