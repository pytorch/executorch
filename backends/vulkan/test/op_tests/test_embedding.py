# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import torch
from torch.export import export, Dim
from executorch.exir import to_edge
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner

class EmbeddingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(10, 32)

    def forward(self, x):
        return self.emb(x)

class TestEmbedding(unittest.TestCase):
    def test_embedding_dynamic_shape(self):
        batch = Dim("batch", min=1, max=100)
        
        module = EmbeddingModule()
        x = torch.randint(0, 10, (2, 16))
        
        exported_program = export(module, (x,), dynamic_shapes=({0: batch},))
        edge_program = to_edge(exported_program)
        
        partitioner = VulkanPartitioner({"require_dynamic_shapes": True})
        delegated_program = edge_program.to_backend(partitioner)
        
        delegated_nodes = [
            node for node in delegated_program.exported_program().graph.nodes
            if node.op == "call_function" and "executorch_call_delegate" in str(node.target)
        ]
            
        self.assertGreaterEqual(len(delegated_nodes), 1, "Failed to partition the operations to Vulkan! No delegated nodes found.")
        
        for node in delegated_program.exported_program().graph.nodes:
            if node.op == "call_function":
                target_str = str(node.target)
                self.assertTrue("embedding" not in target_str, "embedding was left in the main graph (not delegated)!")

if __name__ == "__main__":
    unittest.main()
