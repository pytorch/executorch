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

class AnyDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.any(x, dim=-1)

class TestAnyDim(unittest.TestCase):
    def test_any_dynamic_shape(self):
        batch = Dim("batch", min=1, max=100)
        
        module = AnyDimModule()
        x = torch.randint(0, 2, (2, 16, 128, 128)).to(torch.bool)
        
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
                self.assertTrue("any" not in target_str, "any was left in the main graph (not delegated)!")

if __name__ == "__main__":
    unittest.main()
