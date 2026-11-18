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
    """A planned op through the pass, with a planner that returns a fixed route."""

    def setUp(self):
        if importlib.util.find_spec("triton") is None:
            self.skipTest("Triton required")
        from executorch.backends.cuda.triton import replacement_pass

        self.route = None
        for patch in (
            mock.patch.dict(replacement_pass.EDGE_TO_TRITON_KERNELS, {_RELU: _SIGMOID}),
            mock.patch.dict(replacement_pass.PLANNERS, {_RELU: self.plan}),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def plan(self, node):
        from executorch.backends.cuda.triton import routing

        return routing.ReplacementPlan(
            self.route,
            "only a kernel" if self.route is routing.Route.REQUIRED else "why",
            "fake",
            "test kernel",
            _SIGMOID,
            node.args,
            "relu(x)",
        )

    def run_pass(self, graph_module):
        from executorch.backends.cuda.triton.replacement_pass import (
            ReplaceEdgeOpWithTritonOpPass,
        )

        ReplaceEdgeOpWithTritonOpPass()(graph_module)
        graph_module.graph.lint()

    def test_routes(self):
        from executorch.backends.cuda.triton import routing

        for route in routing.Route:
            self.route = route
            graph_module = _graph()
            with self.subTest(route=route), self.assertLogs(
                _PASS_LOGGER, "INFO"
            ) as logs:
                self.run_pass(graph_module)
            output = "\n".join(logs.output)
            if route is routing.Route.UNSUPPORTED:
                self.assertEqual(_targets(graph_module).count(_RELU), 1)
                self.assertIn("Skipping fake replacement of relu_default: why", output)
                self.assertIn("Replaced 0 nodes", output)
            else:
                self.assertNotIn(_RELU, _targets(graph_module))
                self.assertIn(_SIGMOID, _targets(graph_module))
                self.assertIn(
                    f"Replacing relu_default ({route.value}, test kernel)", output
                )
                self.assertIn("Replaced 1 nodes", output)

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

    def test_optional_failures_keep_the_node(self):
        from executorch.backends.cuda.triton import routing

        self.route = routing.Route.OPTIONAL
        for step, fault in self.faults().items():
            graph_module = _graph()
            before = _nodes(graph_module)
            with self.subTest(step=step):
                with fault:
                    self.run_pass(graph_module)
                self.assertUnchanged(graph_module, before)

    def test_required_failures_raise(self):
        from executorch.backends.cuda.triton import routing

        self.route = routing.Route.REQUIRED
        for step, fault in self.faults().items():
            graph_module = _graph()
            before = _nodes(graph_module)
            with self.subTest(step=step):
                with fault, self.assertRaisesRegex(
                    routing.ReplacementError, r"relu\(x\) \(only a kernel\)"
                ):
                    self.run_pass(graph_module)
                self.assertUnchanged(graph_module, before)
