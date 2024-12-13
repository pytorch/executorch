# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.exir.pass_base import ExportPass, NodeMetadata, ProxyValue
from executorch.exir.passes.spec_prop_pass import make_spec

class InitMutableBufferPass(ExportPass):
    def __init__(self) -> None:
        super().__init__()

    def update_placeholder_tensor_specs(
        self,
        exported_program: torch.export.ExportedProgram,
        graph_module: torch.fx.GraphModule,
    ) -> None:
        """
        Update the tensor specs for all placeholder nodes such that
        placeholders that are parameters are marked as constant.
        """
        for node in graph_module.graph.nodes:
            if node.op != "placeholder":
                continue
            if "spec" not in node.meta:
                raise RuntimeError(f"Placeholder node {node} missing meta['spec']")
            # print(node)
            spec = node.meta["spec"]
            if (isinstance(node.target, str) and
                node.target in exported_program.graph_signature.inputs_to_buffers and exported_program.graph_signature.inputs_to_buffers[node.target] in exported_program.state_dict):
                # print(f"Setting {node.target}.const = True")
                # breakpoint()
                # print(exported_program.state_dict[exported_program.graph_signature.inputs_to_buffers[node.target]])
                spec.const = True

    # pyre-ignore
    def placeholder(self, name: str, arg, meta):
        # print(name)
        meta["spec"] = make_spec(arg, const=meta.data['spec'].const)
        # if name == "b_kv_cache_cache_pos":
        #     print("breakpoint")
        #     breakpoint()
        
        return super().placeholder(name, arg, meta)
