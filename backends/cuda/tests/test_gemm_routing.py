# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Which dense matrix multiplications ReplaceEdgeOpWithTritonOpPass offers
measured routing as triton::linear, what is measured for them, and the graph
after either choice.

The planner tests use the graphs the CUDA backend receives from both lowering
APIs, exported on fake CUDA tensors, and hand-built graphs for argument forms
export does not produce; they need no GPU. The other tests run on one.
"""

import contextlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn
from executorch.backends.cuda.tests.routing_test_utils import (
    backend_graphs,
    LOWERING_APIS,
)
from torch._subclasses.fake_tensor import FakeTensorMode

_PASS_LOGGER = "executorch.backends.cuda.triton.replacement_pass"
_N, _K = 32, 64


@contextlib.contextmanager
def _matmul_precision(precision):
    previous = torch.backends.cuda.matmul.fp32_precision
    torch.backends.cuda.matmul.fp32_precision = precision
    try:
        yield
    finally:
        torch.backends.cuda.matmul.fp32_precision = previous


class _Linear(nn.Module):
    def __init__(self, bias=True, dtype=torch.float32):
        super().__init__()
        self.linear = nn.Linear(_K, _N, bias=bias, dtype=dtype)

    def forward(self, x):
        return self.linear(x)


class _Matmul(nn.Module):
    """``x @ weight.T`` with the transpose written as ``transpose``."""

    def __init__(self, transpose="t", bias=False, op="matmul"):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(_N, _K))
        self.bias = nn.Parameter(torch.randn(_N)) if bias else None
        self.transpose = transpose
        self.op = op

    def forward(self, x):
        wt = {
            "t": lambda w: w.t(),
            "permute": lambda w: w.permute(1, 0),
            "negative permute": lambda w: w.permute(-1, -2),
            "transpose": lambda w: w.transpose(0, 1),
        }[self.transpose](self.weight)
        if self.op == "addmm":
            return torch.addmm(self.bias, x, wt)
        if self.op == "mm":
            return torch.mm(x, wt)
        return x @ wt


class _SharedTranspose(_Matmul):
    """A transpose the multiplication shares with another user."""

    def forward(self, x):
        wt = self.weight.t()
        return x @ wt, wt.sum(dim=0)


_FORMS = {
    "linear with bias": (_Linear, (2, 5, _K)),
    "linear without bias": (lambda: _Linear(bias=False), (5, _K)),
    "matmul with t": (_Matmul, (2, 5, _K)),
    "matmul with permute": (lambda: _Matmul("permute"), (5, _K)),
    "matmul with a negative permute": (lambda: _Matmul("negative permute"), (5, _K)),
    "matmul with transpose": (lambda: _Matmul("transpose"), (5, _K)),
    "mm": (lambda: _Matmul(op="mm"), (5, _K)),
    "addmm": (lambda: _Matmul(bias=True, op="addmm"), (5, _K)),
    "shared transpose": (_SharedTranspose, (5, _K)),
}
# The op the GEMM node of each form is, by lowering API.
_ORIGINALS = {
    "linear with bias": ("aten::linear.default", "aten::addmm.default"),
    "linear without bias": ("aten::linear.default", "aten::mm.default"),
    "addmm": ("aten::addmm.default", "aten::addmm.default"),
    "mm": ("aten::mm.default", "aten::mm.default"),
}


def _gemm_nodes(graph_module):
    from executorch.backends.cuda.triton.kernels.gemm.routing import GEMM_TARGETS

    return [
        n
        for n in graph_module.graph.nodes
        if n.op == "call_function" and n.target in GEMM_TARGETS
    ]


def _the_gemm(graph_module):
    (node,) = _gemm_nodes(graph_module)
    return node


def _graph(target, *args, output=None, **kwargs):
    """A graph calling the edge op ``target`` once. A tensor argument becomes a
    placeholder holding it; ``("t", tensor)`` the ``t_copy`` of one; ``output``
    replaces the node's fake value. Returns the call's node."""
    from executorch.exir.dialects._ops import ops as exir_ops

    graph = torch.fx.Graph()
    names = iter("abcdefgh")

    def node(value):
        if isinstance(value, tuple) and value and value[0] == "t":
            source = node(value[1])
            with _MODE:
                return _call(graph, exir_ops.edge.aten.t_copy.default, (source,), {})
        if not isinstance(value, torch.Tensor):
            return value
        placeholder = graph.placeholder(next(names))
        placeholder.meta["val"] = value
        return placeholder

    call_args = tuple(node(a) for a in args)
    call_kwargs = {k: node(v) for k, v in kwargs.items()}
    with _MODE:
        call = _call(graph, target, call_args, call_kwargs)
    if output is not None:
        call.meta["val"] = output
    graph.output(call)
    return call


def _call(graph, target, args, kwargs):
    def value(a):
        return a.meta["val"] if isinstance(a, torch.fx.Node) else a

    call = graph.call_function(target, args, kwargs)
    call.meta["val"] = target(
        *[value(a) for a in args], **{k: value(v) for k, v in kwargs.items()}
    )
    return call


_MODE = FakeTensorMode()


def _fake(*shape, strides=None, dtype=torch.float32, offset=0):
    with _MODE:
        t = torch.empty_strided(
            shape,
            strides or torch.empty(shape).stride(),
            dtype=dtype,
            device="cuda",
        )
        return t.as_strided(shape, t.stride(), offset) if offset else t


class _RoutingTest(unittest.TestCase):
    def setUp(self):
        if importlib.util.find_spec("triton") is None:
            self.skipTest("Triton required")

    def plan(self, node):
        from executorch.backends.cuda.triton.kernels.gemm import routing

        return routing.plan(node)

    def assertRefused(self, node, reason):
        plan = self.plan(node)
        self.assertEqual([c.name for c in plan.legal()], ["inductor"])
        self.assertIn(reason, plan.candidates[1].reason)


class PlannerTest(_RoutingTest):
    def test_every_form_offers_both_implementations(self):
        for label, (model, shape) in _FORMS.items():
            graphs = backend_graphs(model, lambda shape=shape: torch.randn(*shape))
            for api, graph_module in graphs.items():
                with self.subTest(label, api=api):
                    node = _the_gemm(graph_module)
                    before = graph_module.graph.python_code("self").src
                    plan = self.plan(node)
                    self.assertEqual(graph_module.graph.python_code("self").src, before)
                    self.assertEqual(
                        [c.name for c in plan.legal()], ["inductor", "linear"]
                    )
                    self.assertEqual(plan.static_choice, "inductor")
                    self.assertFalse(plan.measure)
                    linear = plan.candidate("linear")
                    self.assertIs(linear.kernel, torch.ops.triton.linear.default)
                    x, weight, bias, strides = linear.args
                    self.assertEqual(weight.op, "placeholder")
                    self.assertEqual(list(weight.meta["val"].shape), [_N, _K])
                    self.assertEqual(
                        bias is not None, label in ("linear with bias", "addmm")
                    )
                    self.assertEqual(strides, list(node.meta["val"].stride()))

    def test_arguments_by_position_keyword_and_default(self):
        from executorch.exir.dialects._ops import ops as exir_ops

        addmm, linear = (
            exir_ops.edge.aten.addmm.default,
            exir_ops.edge.aten.linear.default,
        )
        x, w, b = _fake(5, _K), _fake(_N, _K), _fake(_N)
        accepted = {
            "addmm with beta and alpha": (
                _graph(addmm, b, x, ("t", w), beta=1, alpha=1),
                True,
            ),
            "addmm with float beta and alpha": (
                _graph(addmm, b, x, ("t", w), beta=1.0, alpha=1.0),
                True,
            ),
            "linear with a keyword bias": (_graph(linear, x, w, bias=b), True),
            "linear with a keyword weight": (_graph(linear, x, weight=w), False),
            "linear with a positional None bias": (_graph(linear, x, w, None), False),
        }
        for label, (node, has_bias) in accepted.items():
            with self.subTest(label):
                plan = self.plan(node)
                self.assertEqual([c.name for c in plan.legal()], ["inductor", "linear"])
                _, weight, bias, _ = plan.candidate("linear").args
                self.assertEqual(weight.meta["val"].shape, (_N, _K))
                self.assertEqual(bias is not None, has_bias)

    def test_refusals(self):
        from executorch.exir.dialects._ops import ops as exir_ops

        e = exir_ops.edge.aten
        x, w, b = _fake(5, _K), _fake(_N, _K), _fake(_N)
        cases = {
            "beta=2, alpha=1": _graph(e.addmm.default, b, x, ("t", w), beta=2),
            "beta=1, alpha=0.5": _graph(e.addmm.default, b, x, ("t", w), alpha=0.5),
            "is not a [N] bias": _graph(e.addmm.default, _fake(5, _N), x, ("t", w)),
            "needs float32": _graph(
                e.linear.default, x, w, _fake(_N, dtype=torch.float16)
            ),
            "empty input": _graph(e.linear.default, _fake(0, _K), w),
            "not a contiguous [N, K]": _graph(
                e.linear.default, x, _fake(_N, _K, strides=(1, _N))
            ),
            "the leading dims of input": _graph(
                e.linear.default, _fake(3, 2, _K, strides=(_K, 3 * _K, 1)), w
            ),
            "not a dense layout": _graph(
                e.linear.default, x, w, output=_fake(5, _N, strides=(2 * _N, 1))
            ),
            "a matmul with a 3-D right operand": _graph(
                e.matmul.default, _fake(2, 5, _K), _fake(2, _K, _N)
            ),
            "not a transposed [N, K] weight": _graph(e.mm.default, x, _fake(_K, _N)),
            "dtype torch.bfloat16": _graph(
                e.linear.default,
                _fake(5, _K, dtype=torch.bfloat16),
                _fake(_N, _K, dtype=torch.bfloat16),
            ),
        }
        for reason, node in cases.items():
            with self.subTest(reason):
                self.assertRefused(node, reason)

    def test_computed_and_untransposed_weights_are_refused(self):
        class Computed(_Matmul):
            def forward(self, x):
                return x @ (self.weight * 2).t()

        class Untransposed(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(_K, _N))

            def forward(self, x):
                return x @ self.weight

        cases = {
            "computed in the graph": Computed,
            "not a transposed [N, K] weight": Untransposed,
        }
        for reason, model in cases.items():
            for api, graph_module in backend_graphs(
                model, lambda: torch.randn(5, _K)
            ).items():
                with self.subTest(reason, api=api):
                    self.assertRefused(_the_gemm(graph_module), reason)

    def test_symbolic_shapes_are_refused(self):
        batch = torch.export.Dim("batch", min=2, max=64)
        graphs = backend_graphs(
            _Linear, lambda: torch.randn(8, _K), dynamic_shapes=({0: batch},)
        )
        for api, graph_module in graphs.items():
            with self.subTest(api):
                self.assertRefused(_the_gemm(graph_module), "symbolic shapes")

    def test_tf32_and_rocm_keep_the_regular_lowering(self):
        from executorch.exir.dialects._ops import ops as exir_ops

        node = _graph(exir_ops.edge.aten.linear.default, _fake(5, _K), _fake(_N, _K))
        with _matmul_precision("tf32"):
            self.assertRefused(node, "TF32 matmuls are allowed")
        with mock.patch.object(torch.version, "hip", "6.0"):
            plan = self.plan(node)
            self.assertEqual(plan.candidates[1].reason, "ROCm")

    def test_without_measurement_the_regular_lowering_runs_every_call(self):
        from executorch.backends.cuda.triton.replacement_pass import (
            ReplaceEdgeOpWithTritonOpPass,
        )

        for api, graph_module in backend_graphs(
            _Linear, lambda: torch.randn(5, _K)
        ).items():
            with self.subTest(api), self.assertLogs(_PASS_LOGGER, "INFO") as logs:
                ReplaceEdgeOpWithTritonOpPass()(graph_module)
                self.assertEqual(len(_gemm_nodes(graph_module)), 1)
                self.assertIn(
                    "Skipping gemm replacement of aten_",
                    "\n".join(logs.output),
                )
                self.assertIn(
                    "no shape class is routed without measurement",
                    "\n".join(logs.output),
                )


def _cuda(test):
    return unittest.skipUnless(torch.cuda.is_available(), "CUDA required")(test)


def _inputs(graph_module):
    """Random CUDA tensors for the graph's placeholders, with their strides."""
    torch.manual_seed(0)
    inputs = []
    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            val = node.meta["val"]
            t = torch.randn(val.shape, dtype=val.dtype, device="cuda")
            inputs.append(t.as_strided(val.shape, val.stride()))
    return inputs


@_cuda
class ReplacementTest(_RoutingTest):
    def test_either_choice_computes_the_same_outputs(self):
        """Replacing the GEMM node keeps the graph's outputs (within the
        accuracy of strict FP32), removes a transpose left without users and
        keeps one shared with another user."""
        from executorch.backends.cuda.triton import routing
        from executorch.backends.cuda.triton.kernels.gemm.routing import TRANSPOSES

        for label, (model, shape) in _FORMS.items():
            graphs = backend_graphs(model, lambda shape=shape: torch.randn(*shape))
            for api, graph_module in graphs.items():
                with self.subTest(label, api=api):
                    inputs = _inputs(graph_module)
                    expected = graph_module(*inputs)
                    node = _the_gemm(graph_module)
                    plan = self.plan(node)
                    transposes = [
                        n for n in node.all_input_nodes if n.target in TRANSPOSES
                    ]
                    new = routing.replace(
                        graph_module.graph, node, plan, plan.candidate("linear")
                    )
                    self.assertIs(new.target, torch.ops.triton.linear.default)
                    graph_module.graph.lint()
                    graph_module.recompile()
                    calls = [
                        n for n in graph_module.graph.nodes if n.op == "call_function"
                    ]
                    self.assertTrue(all(n.users for n in calls), graph_module.code)
                    for transpose in transposes:
                        self.assertEqual(
                            transpose in calls, label == "shared transpose", label
                        )
                    for got, want in zip(graph_module(*inputs), expected, strict=True):
                        torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


@_cuda
class MeasurementSpecTest(_RoutingTest):
    def test_both_materializations_of_every_form(self):
        """The regular lowering is measured on the GEMM node's own operands
        (the transpose copy a core-ATen graph holds is a contiguous [K, N]
        tensor), the replacement on the [N, K] weight; forms of equal sizes
        are cached apart."""
        from executorch.backends.cuda.triton import measured_routing

        keys = {}
        for label, (model, shape) in _FORMS.items():
            graphs = backend_graphs(model, lambda shape=shape: torch.randn(*shape))
            for api, graph_module in graphs.items():
                with self.subTest(label, api=api):
                    node = _the_gemm(graph_module)
                    spec = measured_routing._spec(node, self.plan(node))
                    if label in _ORIGINALS:
                        original = _ORIGINALS[label][LOWERING_APIS.index(api)]
                        self.assertEqual(spec["original"], original)
                    tensors = [a["tensor"] for a in spec["args"] if isinstance(a, dict)]
                    if spec["original"] != "aten::linear.default":
                        self.assertEqual(tensors[-1], ["float32", [_K, _N], [_N, 1]])
                    replacement = spec["replacement"]
                    self.assertEqual(replacement["op"], "triton::linear.default")
                    self.assertEqual(
                        replacement["args"][1]["tensor"], ["float32", [_N, _K], [_K, 1]]
                    )
                    val = node.meta["val"]
                    self.assertEqual(
                        spec["output"]["tensor"],
                        ["float32", list(val.shape), list(val.stride())],
                    )
                    self.assertEqual(replacement["args"][3], list(val.stride()))
                    key = measured_routing.cache_key(spec, {}, torch.device("cuda"))
                    keys.setdefault(key, set()).add(json.dumps(spec, sort_keys=True))
        self.assertTrue(all(len(specs) == 1 for specs in keys.values()))
        self.assertGreater(len(keys), 4)

    def test_offset_inputs_are_not_measured(self):
        from executorch.backends.cuda.triton import measured_routing
        from executorch.exir.dialects._ops import ops as exir_ops

        node = _graph(
            exir_ops.edge.aten.linear.default, _fake(5, _K, offset=7), _fake(_N, _K)
        )
        plan = self.plan(node)
        self.assertEqual([c.name for c in plan.legal()], ["inductor", "linear"])
        with self.assertRaisesRegex(measured_routing.Unmeasurable, "storage offset"):
            measured_routing._spec(node, plan)


@_cuda
class MeasuredRoutingTest(_RoutingTest):
    def test_default_measurement_keeps_the_static_choice(self):
        """With measured routing on (the default), the GEMM family, which does
        not opt in yet, takes its static choice: no call is measured."""
        from executorch.backends.cuda.cuda_backend import CudaBackend
        from executorch.backends.cuda.triton import measured_routing
        from executorch.exir.backend.compile_spec_schema import CompileSpec

        for api, graph_module in backend_graphs(
            lambda: _Matmul(bias=True, op="addmm"), lambda: torch.randn(5, _K)
        ).items():
            with self.subTest(
                api
            ), tempfile.TemporaryDirectory() as directory, mock.patch.object(
                measured_routing, "_run_worker"
            ) as worker:
                specs = [
                    CudaBackend.generate_method_name_compile_spec("forward"),
                    CompileSpec("measured_routing_cache_dir", directory.encode()),
                ]
                for p in CudaBackend.get_custom_passes(specs):
                    if not getattr(p, "requires_exported_program", False):
                        p(graph_module)
                worker.assert_not_called()
                self.assertFalse((Path(directory) / "logs").exists())
                self.assertEqual(len(_gemm_nodes(graph_module)), 1)
