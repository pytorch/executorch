# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from typing import List, Set, Tuple, Type

import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import create_node
from executorch.backends.arm._passes.insert_table_ops import InsertTableOpsPass
from executorch.exir.pass_base import ExportPass, PassResult


class DecomposeRnnPass(ArmPass):
    """Decomposes aten.rnn_tanh.input and aten.rnn_relu.input into
    elementary ops supported by TOSA.

    RNN cell equation per timestep:
        h_t = activation(x_t @ W_ih.T + b_ih + h_{t-1} @ W_hh.T + b_hh)

    where activation is tanh (rnn_tanh) or relu (rnn_relu).

    Supports multi-layer, bidirectional, with/without bias, and batch_first.
    """

    _passes_required_after: Set[Type[ExportPass]] = {InsertTableOpsPass}

    _TARGETS = {
        torch.ops.aten.rnn_tanh.input,
        torch.ops.aten.rnn_relu.input,
    }

    _mm = torch.ops.aten.mm.default
    _t = torch.ops.aten.t.default
    _add = torch.ops.aten.add.Tensor
    _tanh = torch.ops.aten.tanh.default
    _relu = torch.ops.aten.relu.default
    _unsqueeze = torch.ops.aten.unsqueeze.default
    _cat = torch.ops.aten.cat.default
    _select = torch.ops.aten.select_copy.int

    def _build_direction(
        self,
        graph: torch.fx.Graph,
        node: torch.fx.Node,
        current_input: torch.fx.Node,
        h_prev: torch.fx.Node,
        weight_ih: torch.fx.Node,
        weight_hh: torch.fx.Node,
        bias_ih,
        bias_hh,
        seq_len: int,
        time_dim: int,
        reverse: bool,
        activation,
    ) -> Tuple[List[torch.fx.Node], torch.fx.Node]:
        """Build RNN cell computation for one direction.

        Returns (timestep_outputs, h_final) where timestep_outputs are
        unsqueezed hidden states in forward time order.
        """
        w_ih_t = create_node(graph, self._t, args=(weight_ih,), from_node=node)
        w_hh_t = create_node(graph, self._t, args=(weight_hh,), from_node=node)

        time_indices = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
        timestep_outputs = []

        for t_idx in time_indices:
            x_t = create_node(
                graph,
                self._select,
                args=(current_input, time_dim, t_idx),
                from_node=node,
            )

            out_ih = create_node(graph, self._mm, args=(x_t, w_ih_t), from_node=node)
            out_hh = create_node(graph, self._mm, args=(h_prev, w_hh_t), from_node=node)

            if bias_ih is not None:
                out_ih = create_node(
                    graph, self._add, args=(out_ih, bias_ih), from_node=node
                )
            if bias_hh is not None:
                out_hh = create_node(
                    graph, self._add, args=(out_hh, bias_hh), from_node=node
                )

            pre_act = create_node(
                graph, self._add, args=(out_ih, out_hh), from_node=node
            )
            h_t = create_node(graph, activation, args=(pre_act,), from_node=node)
            h_prev = h_t

            h_t_expanded = create_node(
                graph, self._unsqueeze, args=(h_t, time_dim), from_node=node
            )
            timestep_outputs.append(h_t_expanded)

        if reverse:
            timestep_outputs.reverse()

        return timestep_outputs, h_prev

    def call(self, graph_module: torch.fx.GraphModule):  # noqa: C901
        graph = graph_module.graph
        made_changes = False

        for node in list(graph.nodes):
            if (
                node.op != "call_function"
                or node.target not in self._TARGETS
                or not self.allowed_to_transform(node.meta)
            ):
                continue

            is_relu = node.target == torch.ops.aten.rnn_relu.input
            activation = self._relu if is_relu else self._tanh

            args = node.args
            input_node = args[0]
            hx_node = args[1]
            params = args[2]
            has_biases = args[3]
            num_layers = args[4]
            # dropout (args[5]) and train (args[6]) are unused at inference
            bidirectional = args[7]
            batch_first = args[8]

            input_val = input_node.meta["val"]

            if batch_first:
                seq_len = input_val.shape[1]
                time_dim = 1
            else:
                seq_len = input_val.shape[0]
                time_dim = 0

            num_directions = 2 if bidirectional else 1
            # Params per layer: (w_ih, w_hh[, b_ih, b_hh]) * num_directions
            dir_step = 4 if has_biases else 2
            layer_step = dir_step * num_directions

            with graph.inserting_before(node):
                current_input = input_node
                layer_final_hiddens = []

                for layer_idx in range(num_layers):
                    layer_offset = layer_idx * layer_step

                    # Forward direction
                    fw_off = layer_offset
                    fw_w_ih = params[fw_off]
                    fw_w_hh = params[fw_off + 1]
                    fw_b_ih = params[fw_off + 2] if has_biases else None
                    fw_b_hh = params[fw_off + 3] if has_biases else None

                    fw_h0 = create_node(
                        graph,
                        self._select,
                        args=(hx_node, 0, num_directions * layer_idx),
                        from_node=node,
                    )

                    fw_outputs, fw_h_final = self._build_direction(
                        graph,
                        node,
                        current_input,
                        fw_h0,
                        fw_w_ih,
                        fw_w_hh,
                        fw_b_ih,
                        fw_b_hh,
                        seq_len,
                        time_dim,
                        reverse=False,
                        activation=activation,
                    )

                    if bidirectional:
                        bw_off = layer_offset + dir_step
                        bw_w_ih = params[bw_off]
                        bw_w_hh = params[bw_off + 1]
                        bw_b_ih = params[bw_off + 2] if has_biases else None
                        bw_b_hh = params[bw_off + 3] if has_biases else None

                        bw_h0 = create_node(
                            graph,
                            self._select,
                            args=(hx_node, 0, 2 * layer_idx + 1),
                            from_node=node,
                        )

                        bw_outputs, bw_h_final = self._build_direction(
                            graph,
                            node,
                            current_input,
                            bw_h0,
                            bw_w_ih,
                            bw_w_hh,
                            bw_b_ih,
                            bw_b_hh,
                            seq_len,
                            time_dim,
                            reverse=True,
                            activation=activation,
                        )

                        merged = []
                        for fw_out, bw_out in zip(fw_outputs, bw_outputs):
                            merged.append(
                                create_node(
                                    graph,
                                    self._cat,
                                    args=([fw_out, bw_out], -1),
                                    from_node=node,
                                )
                            )

                        layer_output = create_node(
                            graph,
                            self._cat,
                            args=(merged, time_dim),
                            from_node=node,
                        )

                        layer_final_hiddens.append(
                            create_node(
                                graph,
                                self._unsqueeze,
                                args=(fw_h_final, 0),
                                from_node=node,
                            )
                        )
                        layer_final_hiddens.append(
                            create_node(
                                graph,
                                self._unsqueeze,
                                args=(bw_h_final, 0),
                                from_node=node,
                            )
                        )
                    else:
                        layer_output = create_node(
                            graph,
                            self._cat,
                            args=(fw_outputs, time_dim),
                            from_node=node,
                        )

                        layer_final_hiddens.append(
                            create_node(
                                graph,
                                self._unsqueeze,
                                args=(fw_h_final, 0),
                                from_node=node,
                            )
                        )

                    current_input = layer_output

                # Build h_n
                if len(layer_final_hiddens) == 1:
                    h_n = layer_final_hiddens[0]
                else:
                    h_n = create_node(
                        graph,
                        self._cat,
                        args=(layer_final_hiddens, 0),
                        from_node=node,
                    )

                output_node = current_input

            # Replace getitem users: RNN returns (output, h_n)
            getitem_nodes = []
            for user in list(node.users.keys()):
                if user.target == operator.getitem:
                    idx = user.args[1]
                    if idx == 0:
                        user.replace_all_uses_with(output_node)
                    elif idx == 1:
                        user.replace_all_uses_with(h_n)
                    getitem_nodes.append(user)

            for gi in getitem_nodes:
                graph.erase_node(gi)
            graph.erase_node(node)
            made_changes = True

        if not made_changes:
            return PassResult(graph_module, False)

        graph_module.recompile()
        return PassResult(graph_module, True)
