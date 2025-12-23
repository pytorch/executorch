# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import unittest
from pathlib import Path
import torch
from backends.cuda.passes.move_cond_predicate_to_cpu import (
    MoveCondPredicateToCpuPass,
)
from torch.export import export


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
class TestMoveCondPredicateToCpuPass(unittest.TestCase):
    """Test the MoveCondPredicateToCpuPass transformation pass."""

    def test_gpu_buffer_predicate_moved_to_cpu(self):
        """Test that a GPU non-persistent buffer used as predicate is moved to CPU."""

        class CondWithGpuBufferPredicate(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Non-persistent buffer goes to constants
                self.register_buffer(
                    "flag", torch.tensor([False], device="cuda"), persistent=False
                )

            def true_branch(self, x):
                return x * 2

            def false_branch(self, x):
                return x + 1

            def forward(self, x):
                return torch.cond(
                    self.flag,
                    self.true_branch,
                    self.false_branch,
                    (x,),
                )

        module = CondWithGpuBufferPredicate()
        module.eval()
        inputs = (torch.randn(4, 4, device="cuda"),)

        # Export the model
        exported_program = export(module, inputs, strict=True)

        # Verify the buffer is on GPU before the pass
        buffer_name = None
        for name in exported_program.constants:
            if "flag" in name:
                buffer_name = name
                break

        self.assertIsNotNone(buffer_name, "Buffer 'flag' should exist in constants")
        self.assertEqual(
            exported_program._constants[buffer_name].device.type,
            "cuda",
            "Buffer should be on CUDA before the pass",
        )

        # Apply the pass
        pass_instance = MoveCondPredicateToCpuPass()
        pass_instance(exported_program)

        # Verify the buffer is now on CPU
        self.assertEqual(
            exported_program._constants[buffer_name].device.type,
            "cpu",
            "Buffer should be on CPU after the pass",
        )

    def test_cpu_buffer_predicate_unchanged(self):
        """Test that a CPU non-persistent buffer used as predicate remains on CPU."""

        class CondWithCpuBufferPredicate(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Non-persistent buffer on CPU
                self.register_buffer(
                    "flag", torch.tensor([True], device="cpu"), persistent=False
                )

            def true_branch(self, x):
                return x * 2

            def false_branch(self, x):
                return x + 1

            def forward(self, x):
                return torch.cond(
                    self.flag,
                    self.true_branch,
                    self.false_branch,
                    (x,),
                )

        module = CondWithCpuBufferPredicate()
        module.eval()
        # Input still on cuda, but buffer on cpu
        inputs = (torch.randn(4, 4, device="cuda"),)

        exported_program = export(module, inputs, strict=True)

        # Find the buffer
        buffer_name = None
        for name in exported_program.constants:
            if "flag" in name:
                buffer_name = name
                break

        self.assertIsNotNone(buffer_name, "Buffer 'flag' should exist in constants")
        self.assertEqual(
            exported_program._constants[buffer_name].device.type,
            "cpu",
            "Buffer should be on CPU before the pass",
        )

        # Apply the pass
        pass_instance = MoveCondPredicateToCpuPass()
        pass_instance(exported_program)

        # Verify the buffer remains on CPU
        self.assertEqual(
            exported_program._constants[buffer_name].device.type,
            "cpu",
            "Buffer should remain on CPU after the pass",
        )

    def test_computed_predicate_no_change(self):
        """Test that a computed predicate (not a buffer) is not affected."""

        class CondWithComputedPredicate(torch.nn.Module):
            def true_branch(self, x):
                return x * 2

            def false_branch(self, x):
                return x + 1

            def forward(self, x):
                # Predicate is computed from input, not a buffer
                pred = x.sum() > 0
                return torch.cond(
                    pred,
                    self.true_branch,
                    self.false_branch,
                    (x,),
                )

        module = CondWithComputedPredicate()
        module.eval()
        inputs = (torch.randn(4, 4, device="cuda"),)

        exported_program = export(module, inputs, strict=True)

        # Apply the pass - should not raise any errors
        pass_instance = MoveCondPredicateToCpuPass()
        pass_instance(exported_program)

        # Validate the program is still valid
        exported_program.validate()

    def test_multiple_cond_with_buffer_predicates(self):
        """Test that multiple torch.cond calls with non-persistent buffer predicates are handled."""

        class MultipleCondWithBufferPredicates(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Non-persistent buffers go to constants
                self.register_buffer(
                    "flag1", torch.tensor([False], device="cuda"), persistent=False
                )
                self.register_buffer(
                    "flag2", torch.tensor([True], device="cuda"), persistent=False
                )

            def true_branch1(self, x):
                return x * 2

            def false_branch1(self, x):
                return x + 1

            def true_branch2(self, x):
                return x - 1

            def false_branch2(self, x):
                return x / 2

            def forward(self, x):
                y = torch.cond(
                    self.flag1,
                    self.true_branch1,
                    self.false_branch1,
                    (x,),
                )
                z = torch.cond(
                    self.flag2,
                    self.true_branch2,
                    self.false_branch2,
                    (y,),
                )
                return z

        module = MultipleCondWithBufferPredicates()
        module.eval()
        inputs = (torch.randn(4, 4, device="cuda"),)

        exported_program = export(module, inputs, strict=True)

        # Verify both buffers are on GPU before the pass
        flag1_name = None
        flag2_name = None
        for name in exported_program.constants:
            if "flag1" in name:
                flag1_name = name
            elif "flag2" in name:
                flag2_name = name

        self.assertIsNotNone(flag1_name)
        self.assertIsNotNone(flag2_name)

        self.assertEqual(
            exported_program._constants[flag1_name].device.type,
            "cuda",
            "flag1 should be on CUDA before the pass",
        )
        self.assertEqual(
            exported_program._constants[flag2_name].device.type,
            "cuda",
            "flag2 should be on CUDA before the pass",
        )

        # Apply the pass
        pass_instance = MoveCondPredicateToCpuPass()
        pass_instance(exported_program)

        # Verify both buffers are now on CPU
        self.assertEqual(
            exported_program._constants[flag1_name].device.type,
            "cpu",
            "flag1 should be on CPU after the pass",
        )
        self.assertEqual(
            exported_program._constants[flag2_name].device.type,
            "cpu",
            "flag2 should be on CPU after the pass",
        )

    def test_cross_attention_cache_pattern(self):
        """Test the cross-attention cache pattern from the docstring."""

        class CrossAttentionWithCache(torch.nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.k_proj = torch.nn.Linear(hidden_size, hidden_size)
                self.v_proj = torch.nn.Linear(hidden_size, hidden_size)
                self.q_proj = torch.nn.Linear(hidden_size, hidden_size)
                self.out_proj = torch.nn.Linear(hidden_size, hidden_size)
                # Non-persistent buffer used as predicate for torch.cond
                self.register_buffer(
                    "initialized",
                    torch.tensor([False], device="cuda"),
                    persistent=False,
                )
                # Non-persistent k_cache and v_cache (these should not be moved)
                self.register_buffer(
                    "k_cache",
                    torch.zeros(1, 10, hidden_size, device="cuda"),
                    persistent=False,
                )
                self.register_buffer(
                    "v_cache",
                    torch.zeros(1, 10, hidden_size, device="cuda"),
                    persistent=False,
                )

            def compute_kv(self, q, encoder_hidden_states):
                k = self.k_proj(encoder_hidden_states)
                v = self.v_proj(encoder_hidden_states)
                return k, v

            def use_cached_kv(self, q, encoder_hidden_states):
                return self.k_cache.clone(), self.v_cache.clone()

            def forward(self, hidden_states, encoder_hidden_states):
                q = self.q_proj(hidden_states)
                # Use torch.cond with initialized buffer as predicate
                k, v = torch.cond(
                    self.initialized,
                    self.use_cached_kv,
                    self.compute_kv,
                    (q, encoder_hidden_states),
                )
                attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                return self.out_proj(attn_output)

        hidden_size = 64
        module = CrossAttentionWithCache(hidden_size).cuda()
        module.eval()
        inputs = (
            torch.randn(1, 5, hidden_size, device="cuda"),  # hidden_states
            torch.randn(1, 10, hidden_size, device="cuda"),  # encoder_hidden_states
        )

        exported_program = export(module, inputs, strict=True)

        # Find the initialized buffer
        initialized_name = None
        for name in exported_program.constants:
            if "initialized" in name:
                initialized_name = name
                break

        self.assertIsNotNone(
            initialized_name, "Buffer 'initialized' should exist in constants"
        )
        self.assertEqual(
            exported_program._constants[initialized_name].device.type,
            "cuda",
            "initialized buffer should be on CUDA before the pass",
        )

        # Apply the pass
        pass_instance = MoveCondPredicateToCpuPass()
        pass_instance(exported_program)

        # Verify the initialized buffer is now on CPU
        self.assertEqual(
            exported_program._constants[initialized_name].device.type,
            "cpu",
            "initialized buffer should be on CPU after the pass",
        )

        # Other buffers (k_cache, v_cache) should remain on GPU
        for name in exported_program.constants:
            if "k_cache" in name or "v_cache" in name:
                self.assertEqual(
                    exported_program._constants[name].device.type,
                    "cuda",
                    f"{name} should remain on CUDA (not used as cond predicate)",
                )

    def test_placeholder_meta_updated(self):
        """Test that placeholder metadata is updated when buffer is moved."""

        class CondWithGpuBufferPredicate(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Non-persistent buffer goes to constants
                self.register_buffer(
                    "flag", torch.tensor([False], device="cuda"), persistent=False
                )

            def true_branch(self, x):
                return x * 2

            def false_branch(self, x):
                return x + 1

            def forward(self, x):
                return torch.cond(
                    self.flag,
                    self.true_branch,
                    self.false_branch,
                    (x,),
                )

        module = CondWithGpuBufferPredicate()
        module.eval()
        inputs = (torch.randn(4, 4, device="cuda"),)

        exported_program = export(module, inputs, strict=True)

        # Find the predicate placeholder node and verify its metadata
        pred_node = None
        for node in exported_program.graph_module.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.higher_order.cond
            ):
                pred_node = node.args[0]
                break

        self.assertIsNotNone(pred_node, "Should find a cond node with predicate")

        # Check metadata before pass
        if "val" in pred_node.meta:
            fake_tensor = pred_node.meta["val"]
            if isinstance(fake_tensor, torch.Tensor):
                self.assertEqual(
                    fake_tensor.device.type,
                    "cuda",
                    "Placeholder metadata should show CUDA before the pass",
                )

        # Apply the pass
        pass_instance = MoveCondPredicateToCpuPass()
        pass_instance(exported_program)

        # Check metadata after pass
        if "val" in pred_node.meta:
            fake_tensor = pred_node.meta["val"]
            if isinstance(fake_tensor, torch.Tensor):
                self.assertEqual(
                    fake_tensor.device.type,
                    "cpu",
                    "Placeholder metadata should show CPU after the pass",
                )

    def test_requires_exported_program_attribute(self):
        """Test that the pass has requires_exported_program attribute set to True."""
        pass_instance = MoveCondPredicateToCpuPass()
        self.assertTrue(
            pass_instance.requires_exported_program,
            "Pass should require an ExportedProgram",
        )

    def test_program_validates_after_pass(self):
        """Test that exported program is valid after applying the pass."""

        class CondWithGpuBufferPredicate(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Non-persistent buffer goes to constants
                self.register_buffer(
                    "flag", torch.tensor([False], device="cuda"), persistent=False
                )

            def true_branch(self, x):
                return x * 2

            def false_branch(self, x):
                return x + 1

            def forward(self, x):
                return torch.cond(
                    self.flag,
                    self.true_branch,
                    self.false_branch,
                    (x,),
                )

        module = CondWithGpuBufferPredicate()
        module.eval()
        inputs = (torch.randn(4, 4, device="cuda"),)

        exported_program = export(module, inputs, strict=True)

        # Apply the pass - should not raise
        pass_instance = MoveCondPredicateToCpuPass()
        pass_instance(exported_program)

        # validate() is called inside the pass, but we call it again to be sure
        exported_program.validate()

    def test_no_cond_in_graph(self):
        """Test that pass works correctly when there is no torch.cond in the graph."""

        class SimpleModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Non-persistent buffer to test that it stays on GPU
                self.register_buffer(
                    "buffer", torch.tensor([1.0], device="cuda"), persistent=False
                )

            def forward(self, x):
                return x + self.buffer

        module = SimpleModule()
        module.eval()
        inputs = (torch.randn(4, 4, device="cuda"),)

        exported_program = export(module, inputs, strict=True)

        # Apply the pass - should not raise
        pass_instance = MoveCondPredicateToCpuPass()
        pass_instance(exported_program)

        # Buffer should remain on GPU since it's not a cond predicate
        buffer_name = None
        for name in exported_program.constants:
            if "buffer" in name:
                buffer_name = name
                break

        if buffer_name:
            self.assertEqual(
                exported_program._constants[buffer_name].device.type,
                "cuda",
                "Buffer should remain on CUDA since it's not a cond predicate",
            )

    def test_persistent_buffer_predicate_not_moved(self):
        """Test that a persistent buffer (in state_dict) used as predicate is NOT moved.

        The pass only handles non-persistent buffers stored in `constants`.
        Persistent buffers are stored in `state_dict` and should remain unchanged.
        """

        class CondWithPersistentBufferPredicate(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Persistent buffer (default) goes to state_dict, not constants
                self.register_buffer("flag", torch.tensor([False], device="cuda"))

            def true_branch(self, x):
                return x * 2

            def false_branch(self, x):
                return x + 1

            def forward(self, x):
                return torch.cond(
                    self.flag,
                    self.true_branch,
                    self.false_branch,
                    (x,),
                )

        module = CondWithPersistentBufferPredicate()
        module.eval()
        inputs = (torch.randn(4, 4, device="cuda"),)

        exported_program = export(module, inputs, strict=True)

        # Verify the buffer is in state_dict, NOT in constants
        self.assertIn("flag", exported_program.state_dict)
        self.assertNotIn("flag", exported_program.constants)

        # Verify the buffer is on GPU before the pass
        self.assertEqual(
            exported_program.state_dict["flag"].device.type,
            "cuda",
            "Persistent buffer should be on CUDA before the pass",
        )

        # Apply the pass
        pass_instance = MoveCondPredicateToCpuPass()
        pass_instance(exported_program)

        # Verify the buffer remains on GPU (pass should not affect state_dict buffers)
        self.assertEqual(
            exported_program.state_dict["flag"].device.type,
            "cuda",
            "Persistent buffer should remain on CUDA after the pass (not in constants)",
        )


if __name__ == "__main__":
    unittest.main()

