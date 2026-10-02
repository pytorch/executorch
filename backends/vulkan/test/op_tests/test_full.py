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

class FullModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Using full_like to inherit dynamic batch dimension
        return torch.full_like(x, 3.14)

class TestFull(unittest.TestCase):
    def test_full_dynamic_shape(self):
        # 1. Define a dynamic dimension (Batch size)
        batch = Dim("batch", min=1, max=100)
        
        # 2. Create the module and dummy input
        module = FullModule()
        x = torch.randn(2, 16, 128, 128) # Batch is 2
        
        # 3. Export to EXIR with dynamic shapes
        exported_program = export(module, (x,), dynamic_shapes=({0: batch},))
        edge_program = to_edge(exported_program)
        
        # 4. Partition using Vulkan Partitioner
        partitioner = VulkanPartitioner({"require_dynamic_shapes": True})
        delegated_program = edge_program.to_backend(partitioner)
        
        # 5. Verify that the ops were partitioned successfully (no fallback)
        delegated_nodes = [
            node for node in delegated_program.exported_program().graph.nodes
            if node.op == "call_function" and "executorch_call_delegate" in str(node.target)
        ]
            
        self.assertGreaterEqual(len(delegated_nodes), 1, "Failed to partition the operations to Vulkan! No delegated nodes found.")
        
        # Check that there are no standard aten ops left that were supposed to be delegated
        for node in delegated_program.exported_program().graph.nodes:
            if node.op == "call_function":
                target_str = str(node.target)
                self.assertTrue("full" not in target_str, "full_like/full was left in the main graph (not delegated)!")

if __name__ == "__main__":
    unittest.main()
