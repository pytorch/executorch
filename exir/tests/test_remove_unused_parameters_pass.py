import unittest
from typing import Sequence

import torch

from executorch.exir import to_edge, to_edge_transform_and_lower
from executorch.exir.backend.test.op_partitioner_demo import AddMulPartitionerDemo
from executorch.exir.passes import remove_unused_parameters_pass
from executorch.runtime import Runtime
from torch.export import ExportedProgram


class TestRemoveUnusedParametersPass(unittest.TestCase):
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
        """Test that parameters used in only one cond branch are not removed."""

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
                    return x.clone()

                return torch.cond(pred, true_fn, false_fn, (x,))

        model = CondModelUsedOnePath()
        model.eval()
        example_inputs = (torch.tensor(True), torch.randn(1, 16))
        eager_outputs = model(*example_inputs)
        ep = torch.export.export(model, example_inputs, strict=False)

        # linear_false should be removed, linear_true should stay.
        unused_param_names_and_args = {
            "linear_false.weight": "p_linear_false_weight",
            "linear_false.bias": "p_linear_false_bias",
        }

        self._test_pass(ep, unused_param_names_and_args, example_inputs, eager_outputs)

        new_ep = remove_unused_parameters_pass(ep)
        self.assertIn("linear_true.weight", new_ep.state_dict.keys())
        self.assertIn("linear_true.bias", new_ep.state_dict.keys())

    def test_remove_unused_parameters_nested_cond(self):
        """Test removal of unused parameters in two-level (nested) conds.

        Uses nn.Parameter + F.linear and explicitly passes all params through
        cond operands so they appear as placeholders in submodule graphs.
        After export, modifies the IR to make w_none/b_none unused (since
        torch.export strips unused operands from HOPs).

        Three categories of parameters:
        - w_all/b_all: used in all paths
        - w_some/b_some: used in some paths (only inner true branch)
        - w_none/b_none: used in no paths (after IR modification)
        """
        from executorch.exir.graph_module import get_control_flow_submodules

        # Note: This test is a bit contrived because export will strip unused
        # params out of HOPs. This simulates partitioning weights by updating
        # IR post-export.
        class NestedCondModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w_all = torch.nn.Parameter(torch.randn(16, 16))
                self.b_all = torch.nn.Parameter(torch.randn(16))
                self.w_some = torch.nn.Parameter(torch.randn(16, 16))
                self.b_some = torch.nn.Parameter(torch.randn(16))
                self.w_none = torch.nn.Parameter(torch.randn(16, 16))
                self.b_none = torch.nn.Parameter(torch.randn(16))

            def forward(
                self, pred1: torch.Tensor, pred2: torch.Tensor, x: torch.Tensor
            ):
                # All params explicitly passed through cond operands so that
                # they appear in every submodule graph after export.
                def true_fn(
                    pred2, x, w_all, b_all, w_some, b_some, w_none, b_none
                ):
                    def inner_true_fn(
                        x, w_all, b_all, w_some, b_some, w_none, b_none
                    ):
                        return (
                            torch.nn.functional.linear(x, w_all, b_all)
                            + torch.nn.functional.linear(x, w_some, b_some)
                            + torch.nn.functional.linear(x, w_none, b_none)
                        )

                    def inner_false_fn(
                        x, w_all, b_all, w_some, b_some, w_none, b_none
                    ):
                        return torch.nn.functional.linear(
                            x, w_all, b_all
                        ) + torch.nn.functional.linear(x, w_none, b_none)

                    return torch.cond(
                        pred2,
                        inner_true_fn,
                        inner_false_fn,
                        (x, w_all, b_all, w_some, b_some, w_none, b_none),
                    )

                def false_fn(
                    pred2, x, w_all, b_all, w_some, b_some, w_none, b_none
                ):
                    return torch.nn.functional.linear(
                        x, w_all, b_all
                    ) * 2 + torch.nn.functional.linear(x, w_none, b_none)

                return torch.cond(
                    pred1,
                    true_fn,
                    false_fn,
                    (
                        pred2,
                        x,
                        self.w_all,
                        self.b_all,
                        self.w_some,
                        self.b_some,
                        self.w_none,
                        self.b_none,
                    ),
                )

        model = NestedCondModel()
        model.eval()
        example_inputs = (
            torch.tensor(True),
            torch.tensor(True),
            torch.randn(1, 16),
        )
        ep = torch.export.export(model, example_inputs, strict=False)

        # Modify IR: in every leaf submodule, replace uses of w_none/b_none
        # placeholders with w_all/b_all (same shape), making them unused.
        # This simulates a transformation that leaves stale params in the graph.
        def _make_params_unused_in_leaves(gm):
            submodules = list(get_control_flow_submodules(gm))
            if not submodules:
                # Leaf submodule: replace w_none/b_none uses with w_all/b_all.
                nodes_by_suffix = {}
                for node in gm.graph.nodes:
                    if node.op == "placeholder":
                        for suffix in ("w_all", "b_all", "w_none", "b_none"):
                            if node.target.endswith(suffix):
                                nodes_by_suffix[suffix] = node
                if "w_none" in nodes_by_suffix and "w_all" in nodes_by_suffix:
                    nodes_by_suffix["w_none"].replace_all_uses_with(
                        nodes_by_suffix["w_all"]
                    )
                if "b_none" in nodes_by_suffix and "b_all" in nodes_by_suffix:
                    nodes_by_suffix["b_none"].replace_all_uses_with(
                        nodes_by_suffix["b_all"]
                    )
                gm.recompile()
            else:
                for _, submod, _ in submodules:
                    _make_params_unused_in_leaves(submod)

        _make_params_unused_in_leaves(ep.graph_module)

        # Verify precondition: w_none is still in submodule placeholders
        # (threaded through cond operands) even though it has no direct users.
        def _has_placeholder_in_submodules(gm, target_substr):
            for _, submod, _ in get_control_flow_submodules(gm):
                for node in submod.graph.nodes:
                    if (
                        node.op == "placeholder"
                        and target_substr in node.target
                    ):
                        return True
                if _has_placeholder_in_submodules(submod, target_substr):
                    return True
            return False

        self.assertTrue(
            _has_placeholder_in_submodules(ep.graph_module, "w_none"),
            "w_none should still be in submodule placeholders after IR modification",
        )

        # Compute expected outputs from the modified (but still valid) graph.
        expected_outputs = ep.module()(*example_inputs)

        unused_param_names_and_args = {
            "w_none": "p_w_none",
            "b_none": "p_b_none",
        }

        new_ep = remove_unused_parameters_pass(ep)

        # Verify unused params removed from top-level state.
        for param_name, param_arg in unused_param_names_and_args.items():
            self.assertNotIn(param_name, new_ep.state_dict.keys())
            self.assertNotIn(param_name, new_ep.graph_signature.parameters)

        # Verify params used in all/some paths are preserved.
        for name in ["w_all", "b_all", "w_some", "b_some"]:
            self.assertIn(name, new_ep.state_dict.keys())

        # Verify unused params removed from submodule graphs too.
        self.assertFalse(
            _has_placeholder_in_submodules(new_ep.graph_module, "w_none"),
            "w_none should be removed from all submodule placeholders",
        )

        # Verify outputs are unchanged.
        new_outputs = new_ep.module()(*example_inputs)
        self.assertTrue(torch.allclose(new_outputs, expected_outputs))

    def test_remove_unused_parameters_cond_e2e_all_branches(self):
        """Test for cond with delegation, covering both true and false branches."""

        class CondModelWithLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 16)
                self.unused_linear = torch.nn.Linear(32, 32)

            def forward(self, pred: torch.Tensor, x: torch.Tensor):
                linear = self.linear

                def true_fn(x):
                    return linear(x) + 1

                def false_fn(x):
                    return linear(x) + 2

                return torch.cond(pred, true_fn, false_fn, (x,))

        model = CondModelWithLinear().eval()
        x = torch.randn(1, 16)

        for pred_val, branch_name in [(True, "true"), (False, "false")]:
            pred = torch.tensor(pred_val)
            example_inputs = (pred, x)
            eager_outputs = model(*example_inputs)

            ep = torch.export.export(model, example_inputs, strict=False)

            edge_program = to_edge(ep)
            delegated = edge_program.to_backend(AddMulPartitionerDemo())

            edge_ep = delegated._edge_programs["forward"]
            self.assertNotIn(
                "unused_linear.weight",
                edge_ep.state_dict.keys(),
                f"unused_linear.weight should be removed ({branch_name} branch)",
            )

            eager_graph_outputs = edge_ep.module()(pred, x)
            self.assertTrue(
                torch.allclose(eager_graph_outputs, eager_outputs, atol=1e-5),
                f"Post-delegation eager mismatch ({branch_name} branch).\n"
                f"  Expected: {eager_outputs}\n"
                f"  Got: {eager_graph_outputs}",
            )

    def test_remove_unused_parameters_cond_e2e_to_edge_transform_and_lower(self):
        """Test for cond via to_edge_transform_and_lower, covering both branches."""

        class CondModelWithLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 16)
                self.unused_linear = torch.nn.Linear(32, 32)

            def forward(self, pred: torch.Tensor, x: torch.Tensor):
                linear = self.linear

                def true_fn(x):
                    return linear(x) + 1

                def false_fn(x):
                    return linear(x) + 2

                return torch.cond(pred, true_fn, false_fn, (x,))

        model = CondModelWithLinear().eval()
        x = torch.randn(1, 16)

        for pred_val, branch_name in [(True, "true"), (False, "false")]:
            pred = torch.tensor(pred_val)
            example_inputs = (pred, x)
            eager_outputs = model(*example_inputs)

            ep = torch.export.export(model, example_inputs, strict=False)

            lowered = to_edge_transform_and_lower(
                ep,
                partitioner=[AddMulPartitionerDemo()],
            )

            edge_ep = lowered._edge_programs["forward"]
            self.assertNotIn(
                "unused_linear.weight",
                edge_ep.state_dict.keys(),
                f"unused_linear.weight should be removed ({branch_name} branch)",
            )

            eager_graph_outputs = edge_ep.module()(pred, x)
            self.assertTrue(
                torch.allclose(eager_graph_outputs, eager_outputs, atol=1e-5),
                f"Post-delegation eager mismatch ({branch_name} branch).\n"
                f"  Expected: {eager_outputs}\n"
                f"  Got: {eager_graph_outputs}",
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
        """Test for map with delegation, testing multiple batch sizes."""
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

        model = MapModel().eval()

        for batch_size in [1, 3, 5]:
            xs = torch.randn(batch_size, 16)
            example_inputs = (xs,)
            eager_outputs = model(*example_inputs)

            ep = torch.export.export(model, example_inputs, strict=False)

            lowered = to_edge_transform_and_lower(
                ep,
                partitioner=[AddMulPartitionerDemo()],
            )

            edge_ep = lowered._edge_programs["forward"]
            self.assertNotIn(
                "unused_linear.weight",
                edge_ep.state_dict.keys(),
                f"unused_linear.weight should be removed (batch_size={batch_size})",
            )

            eager_graph_outputs = edge_ep.module()(xs)
            self.assertTrue(
                torch.allclose(eager_graph_outputs, eager_outputs, atol=1e-5),
                f"Post-delegation eager mismatch (batch_size={batch_size}).\n"
                f"  Expected: {eager_outputs}\n"
                f"  Got: {eager_graph_outputs}",
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

                _, ys = scan(combine_fn, init, xs)
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
        """Test for scan with delegation, testing multiple sequence lengths."""
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

                _, ys = scan(combine_fn, init, xs)
                return ys

        model = ScanModel().eval()

        for seq_len in [1, 3, 5]:
            xs = torch.randn(seq_len, 1, 16)
            example_inputs = (xs,)
            eager_outputs = model(*example_inputs)

            ep = torch.export.export(model, example_inputs, strict=False)

            lowered = to_edge_transform_and_lower(
                ep,
                partitioner=[AddMulPartitionerDemo()],
            )

            edge_ep = lowered._edge_programs["forward"]
            self.assertNotIn(
                "unused_linear.weight",
                edge_ep.state_dict.keys(),
                f"unused_linear.weight should be removed (seq_len={seq_len})",
            )

            eager_graph_outputs = edge_ep.module()(xs)
            self.assertTrue(
                torch.allclose(eager_graph_outputs, eager_outputs, atol=1e-5),
                f"Post-delegation eager mismatch (seq_len={seq_len}).\n"
                f"  Expected: {eager_outputs}\n"
                f"  Got: {eager_graph_outputs}",
            )

    @unittest.skip("while_loop is not yet supported by the emitter")
    def test_remove_unused_parameters_while_loop_e2e(self):
        """Test for while_loop with delegation."""
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
                    return i + 1, linear(x)

                init_i = torch.tensor(0)
                final_i, result = while_loop(cond_fn, body_fn, (init_i, x))
                return result

        model = WhileLoopModel().eval()

        for max_iters_val in [1, 3, 5]:
            x = torch.randn(1, 16)
            max_iters = torch.tensor(max_iters_val)
            example_inputs = (x, max_iters)
            eager_outputs = model(*example_inputs)

            ep = torch.export.export(model, example_inputs, strict=False)

            lowered = to_edge_transform_and_lower(
                ep,
                partitioner=[AddMulPartitionerDemo()],
            )

            edge_ep = lowered._edge_programs["forward"]
            self.assertNotIn(
                "unused_linear.weight",
                edge_ep.state_dict.keys(),
                f"unused_linear.weight should be removed (max_iters={max_iters_val})",
            )

            eager_graph_outputs = edge_ep.module()(x, max_iters)
            self.assertTrue(
                torch.allclose(eager_graph_outputs, eager_outputs, atol=1e-5),
                f"Post-delegation eager mismatch (max_iters={max_iters_val}).\n"
                f"  Expected: {eager_outputs}\n"
                f"  Got: {eager_graph_outputs}",
            )

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
                lowered = lowered.to_backend(AddMulPartitionerDemo())
        else:  # use to_edge_transform_and_lower
            lowered = to_edge_transform_and_lower(
                ep,
                partitioner=[AddMulPartitionerDemo()] if delegate else [],
            )

        lowered = lowered.to_executorch()
        self.assertLess(len(lowered.buffer), size_bound)

        if not delegate:
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
