# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass
from torch.fx import GraphModule


def map_hardtan_relux(tanhnode: torch.fx.node.Node) -> Optional[str]:
    assert (
        tanhnode.target == exir_ops.edge.aten.hardtanh.default
    ), "Must be a hardtanh node"
    if not tanhnode.args[1] == 0.0:
        return None
    if tanhnode.args[2] == 6.0:
        return "RELU6"
    return None


class FuseActivationPass(ExportPass):
    TARGET_ACTS_MAP = {
        exir_ops.edge.aten.relu.default: (lambda x: "RELU"),
        exir_ops.edge.aten.relu_.default: (lambda x: "RELU"),
        exir_ops.edge.aten.relu6.default: (lambda x: "RELU6"),
        exir_ops.edge.aten.relu6_.default: (lambda x: "RELU6"),
        exir_ops.edge.aten.hardtanh.default: map_hardtan_relux,
        exir_ops.edge.aten.hardtanh_.default: map_hardtan_relux,
    }
    TARGET_SOURCE_NODES = {
        exir_ops.edge.aten.convolution.default,
        exir_ops.edge.aten.linear.default,
    }

    def _fuse(
        self,
        graph_module: GraphModule,
    ):
        for target_src, target_act in self.get_target_src_act(graph_module):
            assert (
                act_name := self.TARGET_ACTS_MAP.get(target_act.target)(target_act)
            ), f"Not supported {target_act.name} now."
            target_src.meta["activation"] = act_name
            if "quantize_attrs" in target_act.meta:
                target_src.meta["quantize_attrs"] = target_act.meta["quantize_attrs"]
            else:
                continue
            for user in [user for user in target_act.users.keys()]:  # noqa: C416
                user.replace_input_with(target_act, target_src)
            graph_module.graph.erase_node(target_act)

    def get_target_src_act(self, graph_module: GraphModule):
        for node in graph_module.graph.nodes:
            if node.target not in self.TARGET_SOURCE_NODES:
                continue
            if len(node.users) != 1:
                # Such cases couldn't be src + act
                continue
            act_node = list(node.users.keys())[0]
            if act_node.target not in self.TARGET_ACTS_MAP:
                continue
            if "quantize_attrs" in node.meta:
                # If we merge the real out activation to source, the source should be the real out
                continue
            yield node, act_node

    def call(self, graph_module: GraphModule):
        self._fuse(graph_module)
        graph_module.recompile()
        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)
