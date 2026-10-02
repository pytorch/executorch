# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.vulkan.test.utils import lower_module_and_test_output

# Ensure the custom ops are registered in the PyTorch environment
try:
    torch.ops.llama.gated_delta_rule.default
except AttributeError:
    import executorch.extension.llm.custom_ops.custom_ops


class TestGatedDeltaRule(unittest.TestCase):
    def test_gated_delta_rule_vulkan(self):
        # We will use small toy shapes for fast testing
        # Based on logs: query is [Batch, Heads, Sequence, HeadDim]
        B, H, S, D = 1, 2, 4, 16

        query = torch.randn(B, H, S, D, dtype=torch.float32)
        key = torch.randn(B, H, S, D, dtype=torch.float32)
        value = torch.randn(B, H, S, D, dtype=torch.float32)
        decay = torch.randn(B, H, S, dtype=torch.float32)
        beta = torch.randn(B, H, S, dtype=torch.float32)
        initial_state = torch.randn(B, H, D, D, dtype=torch.float32)

        class GatedDeltaRuleModule(torch.nn.Module):
            def forward(self, query, key, value, decay, beta, initial_state):
                return torch.ops.llama.gated_delta_rule.default(
                    query, key, value, decay, beta, initial_state
                )

        model = GatedDeltaRuleModule().eval()
        inputs = (query, key, value, decay, beta, initial_state)

        # This compiles the module for Vulkan, runs it on GPU, and compares with CPU via torch.allclose
        lower_module_and_test_output(model, inputs)


if __name__ == "__main__":
    unittest.main()
