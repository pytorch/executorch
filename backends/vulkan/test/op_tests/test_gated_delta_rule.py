import unittest
import torch
from torch.testing._internal.common_utils import TestCase
# Import the custom op so it registers in PyTorch
from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401

# Import Vulkan testing utilities (paths may vary slightly depending on ExecuTorch version)
from executorch.backends.vulkan.test.utils import check_op_on_vulkan

class TestGatedDeltaRule(TestCase):
    def test_gated_delta_rule_vulkan(self):
        # TODO: 1. Define input shapes based on Kev's linear attention dimensions
        batch_size, num_heads, head_k_dim, head_v_dim, seq_len = 1, 4, 32, 32, 16
        
        # TODO: 2. Initialize random dummy tensors for q, k, v, decay, beta, state
        q = torch.randn(batch_size, num_heads, seq_len, head_k_dim, dtype=torch.float32)
        k = torch.randn(batch_size, num_heads, seq_len, head_k_dim, dtype=torch.float32)
        v = torch.randn(batch_size, num_heads, seq_len, head_v_dim, dtype=torch.float32)
        decay = torch.rand(batch_size, num_heads, seq_len, head_k_dim, dtype=torch.float32)
        beta = torch.rand(batch_size, num_heads, seq_len, head_v_dim, dtype=torch.float32)
        state = torch.zeros(batch_size, num_heads, head_k_dim, head_v_dim, dtype=torch.float32)

        # TODO: 3. Create a small nn.Module that wraps the custom operation
        class GatedDeltaRuleModule(torch.nn.Module):
            def forward(self, q, k, v, decay, beta, state):
                # This calls the CPU implementation defined in PyTorch/ExecuTorch
                return torch.ops.llama.gated_delta_rule(q, k, v, decay, beta, state)

        model = GatedDeltaRuleModule().eval()
        inputs = (q, k, v, decay, beta, state)

        # TODO: 4. Check the operation on Vulkan. 
        # This utility traces the graph, lowers it to Vulkan, runs the GPU shader, 
        # runs the CPU reference, and compares the outputs using torch.allclose!
        check_op_on_vulkan(model, inputs)

if __name__ == "__main__":
    unittest.main()
