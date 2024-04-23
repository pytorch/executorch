# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# export_nanogpt.py

# Load partitioner for Xnnpack backend
import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

# Model to be delegated to specific backend should use specific edge compile config
from executorch.backends.xnnpack.utils.configs import get_xnnpack_edge_compile_config
from executorch.exir import to_edge

from model import GPT
from torch._export import capture_pre_autograd_graph
from torch.export import export
from torch.nn.attention import sdpa_kernel, SDPBackend

model = GPT.from_pretrained("gpt2")  # use gpt2 weight as pretrained weight
example_inputs = (
    torch.randint(0, 100, (1, model.config.block_size), dtype=torch.long),
)
dynamic_shape = ({1: torch.export.Dim("token_dim", max=model.config.block_size)},)

# Trace the model, converting it to a portable intermediate representation.
# The torch.no_grad() call tells PyTorch to exclude training-specific logic.
with sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
    m = capture_pre_autograd_graph(model, example_inputs, dynamic_shapes=dynamic_shape)
    traced_model = export(m, example_inputs, dynamic_shapes=dynamic_shape)

# Convert the model into a runnable ExecuTorch program.
# To be further lowered to Xnnpack backend, `traced_model` needs xnnpack-specific edge compile config
edge_config = get_xnnpack_edge_compile_config()
edge_manager = to_edge(traced_model, compile_config=edge_config)

# Delegate exported model to Xnnpack backend by invoking `to_backend` function with Xnnpack partitioner.
edge_manager = edge_manager.to_backend(XnnpackPartitioner())
et_program = edge_manager.to_executorch()

# Save the Xnnpack-delegated ExecuTorch program to a file.
with open("nanogpt.pte", "wb") as file:
    file.write(et_program.buffer)
