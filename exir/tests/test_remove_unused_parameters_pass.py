import unittest
from typing import Sequence

import torch

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge, to_edge_transform_and_lower
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
