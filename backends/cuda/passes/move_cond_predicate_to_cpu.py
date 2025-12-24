# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.export import ExportedProgram


class MoveCondPredicateToCpuPass:
    """
    A pass that moves the predicate of torch.cond to CPU if the predicate is a constantbuffer.
    This is useful for models that use the predicate as a constant buffer, such as an `initialized` flag for cross attention kv cache.

    This saves ~50us per torch.cond call on RTX 5080.

    Example:
    ```
    class CrossAttentionWithCache(torch.nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.k_proj = torch.nn.Linear(hidden_size, hidden_size)
            self.v_proj = torch.nn.Linear(hidden_size, hidden_size)
            self.q_proj = torch.nn.Linear(hidden_size, hidden_size)
            self.out_proj = torch.nn.Linear(hidden_size, hidden_size)
            # Buffer used as predicate for torch.cond
            self.register_buffer("initialized", torch.tensor([False]), persistent=False)
            self.register_buffer("k_cache", torch.zeros(1, 10, hidden_size), persistent=False)
            self.register_buffer("v_cache", torch.zeros(1, 10, hidden_size), persistent=False)

        def compute_kv(self, encoder_hidden_states):
            k = self.k_proj(encoder_hidden_states)
            v = self.v_proj(encoder_hidden_states)
            self.k_cache.copy_(k)
            self.v_cache.copy_(v)
            self.initialized.fill_(True)
            return k, v

        def use_cached_kv(self, encoder_hidden_states):
            return self.k_cache.clone(), self.v_cache.clone()

        def forward(self, hidden_states, encoder_hidden_states):
            q = self.q_proj(hidden_states)
            # Use torch.cond with initialized buffer as predicate
            k, v = torch.cond(
                self.initialized,
                self.use_cached_kv,
                self.compute_kv,
                (encoder_hidden_states,),
            )
            attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            return self.out_proj(attn_output)
    ```
    In this example if we keep `self.initialized` on GPU, we will need to copy it to CPU for every forward pass.
    We move the predicate to CPU to avoid device to host copies.
    This pass is only applicable to models that use torch.cond and its predicate is a constant buffer.
    """

    requires_exported_program = True

    def __call__(self, exported_program: ExportedProgram):
        graph_module = exported_program.graph_module

        # Map input names to buffer names
        inputs_to_buffers = exported_program.graph_signature.inputs_to_buffers

        for node in graph_module.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.higher_order.cond
            ):
                pred_node = node.args[0]
                if (
                    pred_node.op == "placeholder"
                    and pred_node.name in inputs_to_buffers
                ):
                    buffer_name = inputs_to_buffers[pred_node.name]

                    if buffer_name in exported_program.constants:
                        tensor = exported_program._constants[buffer_name]
                        if tensor.device.type != "cpu":
                            exported_program._constants[buffer_name] = tensor.to("cpu")

                    # Also update the placeholder metadata
                    if "val" in pred_node.meta:
                        fake_tensor = pred_node.meta["val"]
                        if isinstance(fake_tensor, torch.Tensor):
                            pred_node.meta["val"] = fake_tensor.to("cpu")
        exported_program.validate()
