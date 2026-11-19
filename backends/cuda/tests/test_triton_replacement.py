# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The generic parts of the Triton replacement pass: kernel legality on the
export device, planned replacement and its transactional rewrite."""

import importlib.util
import unittest
from unittest import mock

import torch


class DeviceSupportTest(unittest.TestCase):
    def test_capability_of_the_export_device(self):
        from executorch.backends.cuda.triton.routing import device_supports

        cuda = torch.device("cuda", 0)
        with mock.patch.object(torch.version, "hip", None), mock.patch.object(
            torch.cuda, "get_device_capability", return_value=(8, 6)
        ) as capability:
            self.assertTrue(device_supports(0, cuda))
            self.assertTrue(device_supports(80, cuda))
            self.assertTrue(device_supports(86, cuda))
            self.assertFalse(device_supports(90, cuda))
            self.assertFalse(device_supports(0, torch.device("cpu")))
        capability.assert_called_with(cuda)

    def test_rocm_runs_no_cuda_kernel(self):
        from executorch.backends.cuda.triton.routing import device_supports

        with mock.patch.object(torch.version, "hip", "6.0"):
            self.assertFalse(device_supports(0, torch.device("cuda", 0)))


_PASS_LOGGER = "executorch.backends.cuda.triton.replacement_pass"
_RELU = torch.ops.aten.relu.default
_SIGMOID = torch.ops.aten.sigmoid.default


def _graph():
    """relu(x) with two users."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    y = graph.call_function(_RELU, (x,))
    graph.output((graph.call_function(torch.ops.aten.add.Tensor, (y, 1.0)), y))
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def _nodes(graph_module):
    return [
        (n.op, n.target, n.name, n.args, n.kwargs, sorted(u.name for u in n.users))
        for n in graph_module.graph.nodes
    ]


def _targets(graph_module):
    return [n.target for n in graph_module.graph.nodes if n.op == "call_function"]


class PlannedReplacementTest(unittest.TestCase):
    """A planned op through the pass, with a planner that returns fixed
    candidates."""

    def setUp(self):
        if importlib.util.find_spec("triton") is None:
            self.skipTest("Triton required")
        from executorch.backends.cuda.triton import replacement_pass

        self.legal = {"inductor", "fake"}
        self.static_choice = "fake"
        for patch in (
            mock.patch.dict(replacement_pass.EDGE_TO_TRITON_KERNELS, {_RELU: _SIGMOID}),
            mock.patch.dict(replacement_pass.PLANNERS, {_RELU: self.plan}),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def plan(self, node):
        from executorch.backends.cuda.triton import routing

        return routing.ReplacementPlan(
            "fake",
            (
                routing.Candidate(
                    routing.REGULAR_LOWERING,
                    None if "inductor" in self.legal else "only a kernel",
                ),
                routing.Candidate(
                    "fake",
                    None if "fake" in self.legal else "why",
                    "test kernel",
                    _SIGMOID,
                    node.args,
                ),
            ),
            self.static_choice,
            "not worth it",
            "relu(x)",
        )

    def run_pass(self, graph_module):
        from executorch.backends.cuda.triton.replacement_pass import (
            ReplaceEdgeOpWithTritonOpPass,
        )

        ReplaceEdgeOpWithTritonOpPass()(graph_module)
        graph_module.graph.lint()

    def test_choices(self):
        cases = (
            (
                {"fake"},
                "fake",
                "Replacing relu_default (test kernel, only implementation)",
            ),
            (
                {"inductor"},
                "inductor",
                "Skipping fake replacement of relu_default: why",
            ),
            ({"inductor", "fake"}, "fake", "Replacing relu_default (test kernel)"),
            (
                {"inductor", "fake"},
                "inductor",
                "Skipping fake replacement of relu_default: not worth it",
            ),
        )
        for legal, static_choice, message in cases:
            self.legal, self.static_choice = legal, static_choice
            graph_module = _graph()
            with self.subTest(
                legal=legal, static_choice=static_choice
            ), self.assertLogs(_PASS_LOGGER, "INFO") as logs:
                self.run_pass(graph_module)
            output = "\n".join(logs.output)
            self.assertIn(message, output)
            replaced = message.startswith("Replacing")
            self.assertEqual(_SIGMOID in _targets(graph_module), replaced)
            self.assertEqual(_RELU in _targets(graph_module), not replaced)
            self.assertIn(f"Replaced {int(replaced)} nodes", output)

    def test_invalid_plans_fail_with_the_signature(self):
        from executorch.backends.cuda.triton import routing

        regular = routing.Candidate(routing.REGULAR_LOWERING)
        with self.assertRaisesRegex(
            routing.ReplacementError, r"relu\(x\): the static choice 'fake' cannot"
        ):
            routing.ReplacementPlan(
                "fake",
                (regular, routing.Candidate("fake", "why")),
                "fake",
                "",
                "relu(x)",
            )
        for candidates in (
            (regular, routing.Candidate("a"), routing.Candidate("b")),
            (regular, regular),
            (routing.Candidate("fake", None, "", _SIGMOID, (1,)), regular),
        ):
            with self.subTest(candidates=[c.name for c in candidates]):
                with self.assertRaisesRegex(
                    routing.ReplacementError, "one replacement"
                ):
                    routing.ReplacementPlan("fake", candidates, signature="relu(x)")
        for kernel, args in ((None, (1,)), (_SIGMOID, ())):
            with self.subTest(kernel=kernel, args=args), self.assertRaisesRegex(
                routing.ReplacementError, "no kernel and arguments"
            ):
                routing.ReplacementPlan(
                    "fake",
                    (regular, routing.Candidate("fake", None, "", kernel, args)),
                    signature="relu(x)",
                )

    def test_no_implementation_fails_the_export(self):
        from executorch.backends.cuda.triton import routing

        self.legal = set()
        with self.assertRaisesRegex(
            routing.ReplacementError,
            r"relu\(x\): nothing can run it \(inductor: only a kernel; fake: why\)",
        ):
            self.run_pass(_graph())

    def faults(self):
        def fail(*args, **kwargs):
            raise RuntimeError("injected fault")

        real_erase = torch.fx.Graph.erase_node

        def fail_erasing_the_original(graph, node):
            if node.target == _RELU:
                fail()
            real_erase(graph, node)

        real_replace_input = torch.fx.Node.replace_input_with
        rewired = []

        def fail_after_the_first_user(user, old, new):
            # The first user is rewired, the second fails.
            if rewired:
                fail()
            real_replace_input(user, old, new)
            rewired.append(user)

        return {
            "node creation": mock.patch.object(torch.fx.Graph, "call_function", fail),
            "rewiring the second user": mock.patch.object(
                torch.fx.Node, "replace_input_with", fail_after_the_first_user
            ),
            "erasing": mock.patch.object(
                torch.fx.Graph, "erase_node", fail_erasing_the_original
            ),
        }

    def assertUnchanged(self, graph_module, before):
        graph_module.graph.lint()
        self.assertEqual(_nodes(graph_module), before)
        self.assertNotIn(_SIGMOID, _targets(graph_module))

    def test_failures_keep_the_node_the_regular_lowering_runs(self):
        for step, fault in self.faults().items():
            graph_module = _graph()
            before = _nodes(graph_module)
            with self.subTest(step=step):
                with fault:
                    self.run_pass(graph_module)
                self.assertUnchanged(graph_module, before)

    def test_failures_raise_when_only_the_replacement_runs_it(self):
        from executorch.backends.cuda.triton import routing

        self.legal = {"fake"}
        for step, fault in self.faults().items():
            graph_module = _graph()
            before = _nodes(graph_module)
            with self.subTest(step=step):
                with fault, self.assertRaisesRegex(
                    routing.ReplacementError,
                    r"relu\(x\) \(inductor: only a kernel; fake: legal\)",
                ):
                    self.run_pass(graph_module)
                self.assertUnchanged(graph_module, before)
