import unittest
from typing import Sequence

import torch

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge, to_edge_transform_and_lower
from executorch.exir.passes import remove_unused_parameters_pass
from executorch.runtime import Runtime
from torch.export import ExportedProgram


class TestRemoveUnusedParametersPass(unittest.TestCase):
    def test_debug(self):
        """Debug test: single linear lowered via to_edge_transform_and_lower for xnnpack."""

        class SingleLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 16)
                self.norm = torch.nn.LayerNorm((16,))

            def forward(self, x):
                return self.norm(self.linear(x))

        model = SingleLinear().eval()
        x = torch.randn(1, 16)
        eager_outputs = model(x)

        ep = torch.export.export(model, (x,), strict=False)
        lowered = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()])

        edge_ep = lowered._edge_programs["forward"]

        # Print debug info
        print("Graph module:")
        print(edge_ep.graph_module)

        for name, mod in edge_ep.graph_module.named_modules():
            if "lowered_module" in name and hasattr(mod, "original_module"):
                if mod.original_module is not None:
                    orig = mod.original_module
                    print(f"\n{name}:")
                    print(
                        f"  input_specs: {[s.arg.name for s in orig.graph_signature.input_specs]}"
                    )
                    print("  original_module graph:")
                    print(f"  {orig.graph_module}")

        # Run post-delegation eager graph
        eager_module = edge_ep.module()
        eager_graph_outputs = eager_module(x)

        self.assertTrue(
            torch.allclose(eager_graph_outputs, eager_outputs, atol=1e-5),
            f"Post-delegation eager mismatch.\n"
            f"  Expected: {eager_outputs}\n"
            f"  Got: {eager_graph_outputs}",
        )

    class SimpleModelWithUnusedParameters(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(16, 16)
            self.unused_linear = torch.nn.Linear(1024, 1024)

        def forward(self, x):
            return self.linear1(x)

    class NestedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mod1 = TestRemoveUnusedParametersPass.SimpleModelWithUnusedParameters()
            self.mod2 = TestRemoveUnusedParametersPass.SimpleModelWithUnusedParameters()

        def forward(self, x):
            y = self.mod1(x) + self.mod2(x)
            y += self.mod1.unused_linear(x.repeat([1, 64]))[:, :16]
            return y

    def test_remove_unused_parameters_simple(self):
        model = self.SimpleModelWithUnusedParameters()
        model.eval()
        example_inputs = (torch.randn(1, 16),)
        eager_outputs = model(*example_inputs)
        ep = torch.export.export(model, example_inputs, strict=False)

        unused_param_names_and_args = {
            "unused_linear.weight": "p_unused_linear_weight",
            "unused_linear.bias": "p_unused_linear_bias",
        }

        self._test_pass(ep, unused_param_names_and_args, example_inputs, eager_outputs)

    def test_remove_unused_parameters_nested(self):
        model = self.NestedModel()
        model.eval()
        example_inputs = (torch.randn(1, 16),)
        eager_outputs = model(*example_inputs)
        ep = torch.export.export(model, example_inputs, strict=False)

        unused_param_names_and_args = {
            "mod2.unused_linear.weight": "p_mod2_unused_linear_weight",
            "mod2.unused_linear.bias": "p_mod2_unused_linear_bias",
        }

        self._test_pass(ep, unused_param_names_and_args, example_inputs, eager_outputs)

    def test_remove_unused_parameters_simple_e2e_to_edge(self):
        model = self.SimpleModelWithUnusedParameters().eval()
        example_inputs = (torch.randn(1, 16),)

        # There are approximately 1M unused fp32 parameters - ~4Mb.
        # Without the unused params, the expected size is ~2.5Kb.
        size_bound = 10000

        for strict in [False, True]:
            for delegate in [False, True]:
                self._test_pass_e2e(
                    model,
                    example_inputs,
                    strict=strict,
                    use_to_edge=True,
                    delegate=delegate,
                    size_bound=size_bound,
                )

    def test_remove_unused_parameters_simple_e2e_to_edge_transform_and_lower(self):
        model = self.SimpleModelWithUnusedParameters().eval()
        example_inputs = (torch.randn(1, 16),)

        # There are approximately 1M unused fp32 parameters - ~4Mb.
        # Without the unused params, the expected size is ~2.5Kb.
        size_bound = 10000

        for strict in [False, True]:
            for delegate in [False, True]:
                self._test_pass_e2e(
                    model,
                    example_inputs,
                    strict=strict,
                    use_to_edge=False,
                    delegate=delegate,
                    size_bound=size_bound,
                )

    def test_remove_unused_parameters_nested_e2e_to_edge(self):
        model = self.NestedModel().eval()
        example_inputs = (torch.randn(1, 16),)

        size_bound = 20000 + 1024 * 1024 * 4

        for strict in [False, True]:
            for delegate in [False, True]:
                self._test_pass_e2e(
                    model,
                    example_inputs,
                    strict=strict,
                    use_to_edge=True,
                    delegate=delegate,
                    size_bound=size_bound,
                )

    def test_remove_unused_parameters_nested_e2e_to_edge_transform_and_lower(self):
        model = self.SimpleModelWithUnusedParameters().eval()
        example_inputs = (torch.randn(1, 16),)

        size_bound = 20000 + 1024 * 1024 * 4

        for strict in [False, True]:
            for delegate in [False, True]:
                self._test_pass_e2e(
                    model,
                    example_inputs,
                    strict=strict,
                    use_to_edge=False,
                    delegate=delegate,
                    size_bound=size_bound,
                )

    def _test_pass(
        self,
        ep: ExportedProgram,
        unused_param_names_and_args: dict[str, str],
        example_inputs: Sequence[torch.Tensor],
        expected_outputs: torch.Tensor,
    ):
        # Verify EP state before running the pass.
        placeholders = {
            n.target for n in ep.graph_module.graph.nodes if n.op == "placeholder"
        }
        for param_name, param_arg in unused_param_names_and_args.items():
            self.assertIn(param_name, ep.state_dict.keys())
            self.assertIn(param_name, ep.graph_signature.parameters)
            self.assertIn(param_arg, placeholders)

        new_ep = remove_unused_parameters_pass(ep)

        # Verify that the unused params are not in the state dict,
        # graph signature, or graph.
        new_placeholders = {
            n.target for n in new_ep.graph_module.graph.nodes if n.op == "placeholder"
        }
        for param_name, param_arg in unused_param_names_and_args.items():
            self.assertNotIn(param_name, new_ep.state_dict.keys())
            self.assertNotIn(param_name, new_ep.graph_signature.parameters)
            self.assertNotIn(param_arg, new_placeholders)

        # Verify that the outputs are unchanged.
        new_outputs = new_ep.module()(*example_inputs)
        self.assertTrue(torch.allclose(new_outputs, expected_outputs))

    def test_unused_user_inputs_not_removed(self):
        """Verify that unused user inputs are not removed by the pass."""

        class ModelWithUnusedInput(torch.nn.Module):
            def forward(self, x, unused_input):
                return x * 2

        model = ModelWithUnusedInput()
        model.eval()
        example_inputs = (torch.randn(1, 16), torch.randn(1, 16))
        eager_outputs = model(*example_inputs)
        ep = torch.export.export(model, example_inputs, strict=False)

        # Verify unused_input is in the graph before the pass.
        placeholders_before = [
            n.target for n in ep.graph_module.graph.nodes if n.op == "placeholder"
        ]
        self.assertIn("unused_input", placeholders_before)

        new_ep = remove_unused_parameters_pass(ep)

        # Verify unused_input is still in the graph after the pass.
        placeholders_after = [
            n.target for n in new_ep.graph_module.graph.nodes if n.op == "placeholder"
        ]
        self.assertIn("unused_input", placeholders_after)

        # Verify outputs are unchanged.
        new_outputs = new_ep.module()(*example_inputs)
        self.assertTrue(torch.allclose(new_outputs, eager_outputs))

    def test_remove_unused_parameters_cond_unused_both_paths(self):
        """Test removal of parameters unused in both branches of cond."""

        class CondModelUnusedBothPaths(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 16)
                self.unused_linear = torch.nn.Linear(32, 32)

            def forward(self, pred: torch.Tensor, x: torch.Tensor):
                def true_fn(x):
                    return x + 1

                def false_fn(x):
                    return x - 1

                return torch.cond(pred, true_fn, false_fn, (self.linear(x),))

        model = CondModelUnusedBothPaths()
        model.eval()
        example_inputs = (torch.tensor(True), torch.randn(1, 16))
        eager_outputs = model(*example_inputs)
        ep = torch.export.export(model, example_inputs, strict=False)

        unused_param_names_and_args = {
            "unused_linear.weight": "p_unused_linear_weight",
            "unused_linear.bias": "p_unused_linear_bias",
        }

        self._test_pass(ep, unused_param_names_and_args, example_inputs, eager_outputs)

    def test_remove_unused_parameters_cond_used_one_path(self):
        """Test that parameters used in only one cond branch are NOT removed."""

        class CondModelUsedOnePath(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear_true = torch.nn.Linear(16, 16)
                self.linear_false = torch.nn.Linear(16, 16)

            def forward(self, pred: torch.Tensor, x: torch.Tensor):
                linear_true = self.linear_true

                def true_fn(x):
                    return linear_true(x)

                def false_fn(x):
                    # Doesn't use linear_true, just returns x (clone to avoid aliasing)
                    return x.clone()

                return torch.cond(pred, true_fn, false_fn, (x,))

        model = CondModelUsedOnePath()
        model.eval()
        example_inputs = (torch.tensor(True), torch.randn(1, 16))
        eager_outputs = model(*example_inputs)
        ep = torch.export.export(model, example_inputs, strict=False)

        # linear_false is completely unused - should be removed
        unused_param_names_and_args = {
            "linear_false.weight": "p_linear_false_weight",
            "linear_false.bias": "p_linear_false_bias",
        }

        self._test_pass(ep, unused_param_names_and_args, example_inputs, eager_outputs)

        # linear_true is used in one path - should NOT be removed
        new_ep = remove_unused_parameters_pass(ep)
        self.assertIn("linear_true.weight", new_ep.state_dict.keys())
        self.assertIn("linear_true.bias", new_ep.state_dict.keys())

    def test_remove_unused_parameters_cond_e2e_all_branches(self):
        """E2E test for cond with delegation, covering both true and false branches.

        Uses same linear in both branches so that both branches have same signature.
        """

        class CondModelWithLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 16)
                self.unused_linear = torch.nn.Linear(32, 32)

            def forward(self, pred: torch.Tensor, x: torch.Tensor):
                linear = self.linear

                def true_fn(x):
                    return torch.relu(linear(x))

                def false_fn(x):
                    return torch.sigmoid(linear(x))

                return torch.cond(pred, true_fn, false_fn, (x,))

        model = CondModelWithLinear().eval()
        x = torch.randn(1, 16)

        # Test both branches
        for pred_val, branch_name in [(True, "true"), (False, "false")]:
            pred = torch.tensor(pred_val)
            example_inputs = (pred, x)
            eager_outputs = model(*example_inputs)

            ep = torch.export.export(model, example_inputs, strict=False)

            # Test via to_edge + to_backend path
            edge_program = to_edge(ep)
            delegated = edge_program.to_backend(XnnpackPartitioner())

            # Verify unused_linear is not in state_dict
            edge_ep = delegated._edge_programs["forward"]
            self.assertNotIn(
                "unused_linear.weight",
                edge_ep.state_dict.keys(),
                f"unused_linear.weight should be removed ({branch_name} branch)",
            )

            # Run post-delegation eager graph
            eager_module = edge_ep.module()
            eager_graph_outputs = eager_module(pred, x)
            self.assertTrue(
                torch.allclose(eager_graph_outputs, eager_outputs, atol=1e-5),
                f"Post-delegation eager mismatch ({branch_name} branch).\n"
                f"  Expected: {eager_outputs}\n"
                f"  Got: {eager_graph_outputs}",
            )

            # Run via PTE runtime
            lowered = delegated.to_executorch()
            runtime = Runtime.get()
            program = runtime.load_program(lowered.buffer)
            method = program.load_method("forward")
            runtime_outputs = method.execute([pred, x])

            self.assertTrue(
                torch.allclose(runtime_outputs[0], eager_outputs, atol=1e-5),
                f"PTE runtime mismatch ({branch_name} branch).\n"
                f"  Expected: {eager_outputs}\n"
                f"  Got: {runtime_outputs[0]}",
            )

    def test_remove_unused_parameters_cond_e2e_to_edge_transform_and_lower(self):
        """E2E test for cond via to_edge_transform_and_lower, covering both branches.

        Uses same linear in both branches so that both branches have same signature.
        """

        class CondModelWithLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 16)
                self.unused_linear = torch.nn.Linear(32, 32)

            def forward(self, pred: torch.Tensor, x: torch.Tensor):
                linear = self.linear

                def true_fn(x):
                    return torch.relu(linear(x))

                def false_fn(x):
                    return torch.sigmoid(linear(x))

                return torch.cond(pred, true_fn, false_fn, (x,))

        model = CondModelWithLinear().eval()
        x = torch.randn(1, 16)

        # Test both branches
        for pred_val, branch_name in [(True, "true"), (False, "false")]:
            pred = torch.tensor(pred_val)
            example_inputs = (pred, x)
            eager_outputs = model(*example_inputs)

            ep = torch.export.export(model, example_inputs, strict=False)

            # Test via to_edge_transform_and_lower path
            lowered = to_edge_transform_and_lower(
                ep,
                partitioner=[XnnpackPartitioner()],
            )

            # Verify unused_linear is not in state_dict
            edge_ep = lowered._edge_programs["forward"]
            self.assertNotIn(
                "unused_linear.weight",
                edge_ep.state_dict.keys(),
                f"unused_linear.weight should be removed ({branch_name} branch)",
            )

            # Run post-delegation eager graph
            eager_module = edge_ep.module()
            eager_graph_outputs = eager_module(pred, x)
            self.assertTrue(
                torch.allclose(eager_graph_outputs, eager_outputs, atol=1e-5),
                f"Post-delegation eager mismatch ({branch_name} branch).\n"
                f"  Expected: {eager_outputs}\n"
                f"  Got: {eager_graph_outputs}",
            )

            # Run via PTE runtime
            et_program = lowered.to_executorch()
            runtime = Runtime.get()
            program = runtime.load_program(et_program.buffer)
            method = program.load_method("forward")
            runtime_outputs = method.execute([pred, x])

            self.assertTrue(
                torch.allclose(runtime_outputs[0], eager_outputs, atol=1e-5),
                f"PTE runtime mismatch ({branch_name} branch).\n"
                f"  Expected: {eager_outputs}\n"
                f"  Got: {runtime_outputs[0]}",
            )

    def test_remove_unused_parameters_map(self):
        """Test removal of unused parameters with map_impl."""
        from torch._higher_order_ops.map import map as torch_map

        class MapModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 16)
                self.unused_linear = torch.nn.Linear(32, 32)

            def forward(self, xs: torch.Tensor):
                linear = self.linear

                def map_fn(x):
                    return linear(x)

                return torch_map(map_fn, xs)

        model = MapModel()
        model.eval()
        example_inputs = (torch.randn(3, 16),)
        eager_outputs = model(*example_inputs)
        ep = torch.export.export(model, example_inputs, strict=False)

        unused_param_names_and_args = {
            "unused_linear.weight": "p_unused_linear_weight",
            "unused_linear.bias": "p_unused_linear_bias",
        }

        self._test_pass(ep, unused_param_names_and_args, example_inputs, eager_outputs)

    def test_remove_unused_parameters_map_e2e(self):
        """E2E test for map with delegation, testing multiple batch sizes."""
        from torch._higher_order_ops.map import map as torch_map

        class MapModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 16)
                self.unused_linear = torch.nn.Linear(32, 32)

            def forward(self, xs: torch.Tensor):
                linear = self.linear

                def map_fn(x):
                    return torch.relu(linear(x))

                return torch_map(map_fn, xs)

        model = MapModel().eval()

        # Test with different batch sizes
        for batch_size in [1, 3, 5]:
            xs = torch.randn(batch_size, 16)
            example_inputs = (xs,)
            eager_outputs = model(*example_inputs)

            ep = torch.export.export(model, example_inputs, strict=False)

            # Test via to_edge_transform_and_lower path
            lowered = to_edge_transform_and_lower(
                ep,
                partitioner=[XnnpackPartitioner()],
            )

            # Verify unused_linear is not in state_dict
            edge_ep = lowered._edge_programs["forward"]
            self.assertNotIn(
                "unused_linear.weight",
                edge_ep.state_dict.keys(),
                f"unused_linear.weight should be removed (batch_size={batch_size})",
            )

            # Run post-delegation eager graph
            eager_module = edge_ep.module()
            eager_graph_outputs = eager_module(xs)
            self.assertTrue(
                torch.allclose(eager_graph_outputs, eager_outputs, atol=1e-5),
                f"Post-delegation eager mismatch (batch_size={batch_size}).\n"
                f"  Expected: {eager_outputs}\n"
                f"  Got: {eager_graph_outputs}",
            )

            # Run via PTE runtime
            et_program = lowered.to_executorch()
            runtime = Runtime.get()
            program = runtime.load_program(et_program.buffer)
            method = program.load_method("forward")
            runtime_outputs = method.execute([xs])

            self.assertTrue(
                torch.allclose(runtime_outputs[0], eager_outputs, atol=1e-5),
                f"PTE runtime mismatch (batch_size={batch_size}).\n"
                f"  Expected: {eager_outputs}\n"
                f"  Got: {runtime_outputs[0]}",
            )

    def test_remove_unused_parameters_scan(self):
        """Test removal of unused parameters with scan."""
        from torch._higher_order_ops.scan import scan

        class ScanModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 16)
                self.unused_linear = torch.nn.Linear(32, 32)

            def forward(self, xs: torch.Tensor):
                linear = self.linear
                init = torch.zeros(1, 16)

                def combine_fn(carry, x):
                    new_carry = carry + linear(x)
                    return new_carry, new_carry.clone()

                final_carry, ys = scan(combine_fn, init, xs)
                return ys

        model = ScanModel()
        model.eval()
        example_inputs = (torch.randn(5, 1, 16),)
        eager_outputs = model(*example_inputs)
        ep = torch.export.export(model, example_inputs, strict=False)

        unused_param_names_and_args = {
            "unused_linear.weight": "p_unused_linear_weight",
            "unused_linear.bias": "p_unused_linear_bias",
        }

        self._test_pass(ep, unused_param_names_and_args, example_inputs, eager_outputs)

    def test_remove_unused_parameters_scan_e2e(self):
        """E2E test for scan with delegation, testing multiple sequence lengths."""
        from torch._higher_order_ops.scan import scan

        class ScanModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 16)
                self.unused_linear = torch.nn.Linear(32, 32)

            def forward(self, xs: torch.Tensor):
                linear = self.linear
                init = torch.zeros(1, 16)

                def combine_fn(carry, x):
                    new_carry = carry + torch.relu(linear(x))
                    return new_carry, new_carry.clone()

                final_carry, ys = scan(combine_fn, init, xs)
                return ys

        model = ScanModel().eval()

        # Test with different sequence lengths
        for seq_len in [1, 3, 5]:
            xs = torch.randn(seq_len, 1, 16)
            example_inputs = (xs,)
            eager_outputs = model(*example_inputs)

            ep = torch.export.export(model, example_inputs, strict=False)

            # Test via to_edge_transform_and_lower path
            lowered = to_edge_transform_and_lower(
                ep,
                partitioner=[XnnpackPartitioner()],
            )

            # Verify unused_linear is not in state_dict
            edge_ep = lowered._edge_programs["forward"]
            self.assertNotIn(
                "unused_linear.weight",
                edge_ep.state_dict.keys(),
                f"unused_linear.weight should be removed (seq_len={seq_len})",
            )

            # Run post-delegation eager graph
            eager_module = edge_ep.module()
            eager_graph_outputs = eager_module(xs)
            self.assertTrue(
                torch.allclose(eager_graph_outputs, eager_outputs, atol=1e-5),
                f"Post-delegation eager mismatch (seq_len={seq_len}).\n"
                f"  Expected: {eager_outputs}\n"
                f"  Got: {eager_graph_outputs}",
            )

            # Run via PTE runtime
            et_program = lowered.to_executorch()
            runtime = Runtime.get()
            program = runtime.load_program(et_program.buffer)
            method = program.load_method("forward")
            runtime_outputs = method.execute([xs])

            self.assertTrue(
                torch.allclose(runtime_outputs[0], eager_outputs, atol=1e-5),
                f"PTE runtime mismatch (seq_len={seq_len}).\n"
                f"  Expected: {eager_outputs}\n"
                f"  Got: {runtime_outputs[0]}",
            )

    @unittest.skip("while_loop is not yet supported by the emitter")
    def test_remove_unused_parameters_while_loop_e2e(self):
        """E2E test for while_loop with delegation."""
        from torch._higher_order_ops.while_loop import while_loop

        class WhileLoopModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 16)
                self.unused_linear = torch.nn.Linear(32, 32)

            def forward(self, x: torch.Tensor, max_iters: torch.Tensor):
                linear = self.linear

                def cond_fn(i, x):
                    return i < max_iters

                def body_fn(i, x):
                    return i + 1, torch.relu(linear(x))

                init_i = torch.tensor(0)
                final_i, result = while_loop(cond_fn, body_fn, (init_i, x))
                return result

        model = WhileLoopModel().eval()

        # Test with different iteration counts
        for max_iters_val in [1, 3, 5]:
            x = torch.randn(1, 16)
            max_iters = torch.tensor(max_iters_val)
            example_inputs = (x, max_iters)
            eager_outputs = model(*example_inputs)

            ep = torch.export.export(model, example_inputs, strict=False)

            # Test via to_edge_transform_and_lower path
            lowered = to_edge_transform_and_lower(
                ep,
                partitioner=[XnnpackPartitioner()],
            )

            # Verify unused_linear is not in state_dict
            edge_ep = lowered._edge_programs["forward"]
            self.assertNotIn(
                "unused_linear.weight",
                edge_ep.state_dict.keys(),
                f"unused_linear.weight should be removed (max_iters={max_iters_val})",
            )

            # Run post-delegation eager graph
            eager_module = edge_ep.module()
            eager_graph_outputs = eager_module(x, max_iters)
            self.assertTrue(
                torch.allclose(eager_graph_outputs, eager_outputs, atol=1e-5),
                f"Post-delegation eager mismatch (max_iters={max_iters_val}).\n"
                f"  Expected: {eager_outputs}\n"
                f"  Got: {eager_graph_outputs}",
            )

            # Run via PTE runtime
            et_program = lowered.to_executorch()
            runtime = Runtime.get()
            program = runtime.load_program(et_program.buffer)
            method = program.load_method("forward")
            runtime_outputs = method.execute([x, max_iters])

            self.assertTrue(
                torch.allclose(runtime_outputs[0], eager_outputs, atol=1e-5),
                f"PTE runtime mismatch (max_iters={max_iters_val}).\n"
                f"  Expected: {eager_outputs}\n"
                f"  Got: {runtime_outputs[0]}",
            )

    def test_delegate_consumes_all_uses_params_removed(self):
        """E2E test: When delegate consumes all uses of a param, it should be removed from top-level."""

        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(16, 16))

            def forward(self, x):
                # Use matmul + relu, which XNNPACK will fully delegate
                return torch.relu(x @ self.weight)

        model = SimpleModel().eval()
        example_inputs = (torch.randn(1, 16),)
        eager_outputs = model(*example_inputs)

        ep = torch.export.export(model, example_inputs, strict=False)
        edge_program = to_edge(ep)
        delegated = edge_program.to_backend(XnnpackPartitioner())

        # Get the edge program after delegation
        edge_ep = delegated._edge_programs["forward"]

        # The weight param should be removed from the top-level graph
        # because the delegate consumed it (tracked via consumed_params metadata)
        self.assertNotIn("weight", edge_ep.state_dict.keys())

        # Verify the program still runs correctly
        lowered = delegated.to_executorch()
        runtime = Runtime.get()
        program = runtime.load_program(lowered.buffer)
        method = program.load_method("forward")
        runtime_outputs = method.execute([*example_inputs])
        self.assertTrue(torch.allclose(runtime_outputs[0], eager_outputs, atol=1e-5))

    def _test_pass_e2e(
        self,
        model: torch.nn.Module,
        example_inputs: Sequence[torch.Tensor],
        strict: bool,
        use_to_edge: bool,
        delegate: bool,
        size_bound: int,
    ):
        eager_outputs = model(*example_inputs)
        ep = torch.export.export(model, example_inputs, strict=strict)

        if use_to_edge:
            lowered = to_edge(ep)
            if delegate:
                lowered = lowered.to_backend(XnnpackPartitioner())
        else:  # use to_edge_transform_and_lower
            lowered = to_edge_transform_and_lower(
                ep,
                partitioner=[XnnpackPartitioner()] if delegate else [],
            )

        lowered = lowered.to_executorch()
        self.assertLess(len(lowered.buffer), size_bound)

        # Make sure we can load and run the serialized .pte.
        runtime = Runtime.get()
        program = runtime.load_program(lowered.buffer)
        method = program.load_method("forward")
        runtime_outputs = method.execute([*example_inputs])

        self.assertEqual(1, len(runtime_outputs))
        self.assertTrue(
            torch.allclose(runtime_outputs[0], eager_outputs, atol=1e-5),
            "Values out of tolerance.\n"
            + f"  Strict: {strict}, ToEdge: {use_to_edge}, Delegate: {delegate}.\n"
            + f"  Eager: {eager_outputs}.\n"
            + f"  Pybind: {runtime_outputs[0]}.\n"
            + f"  Error: {eager_outputs - runtime_outputs[0]}",
        )
