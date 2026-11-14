# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Which convolutions ReplaceEdgeOpWithTritonOpPass routes to
triton::convolution, and its failure contract.

The graphs are the ones the CUDA backend receives from both lowering APIs,
exported on fake CUDA tensors, so no GPU is needed.
"""

import contextlib
import dataclasses
import importlib.util
import unittest
from unittest import mock

import torch
import torch.nn as nn

_PASS_LOGGER = "executorch.backends.cuda.triton.replacement_pass"
_APIS = ("to_edge_transform_and_lower", "to_edge_to_backend")


def _backend_graphs(model, *inputs, dynamic_shapes=None, device="cuda"):
    """The graph the CUDA backend gets from each lowering API.

    The model is exported on fake CUDA tensors, or on CPU tensors whose fake
    values are then moved to CUDA, as the backend does before the pass runs.
    """
    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
    from executorch.exir import to_edge, to_edge_transform_and_lower
    from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
    from torch._subclasses.fake_tensor import FakeTensorMode

    fake_mode = FakeTensorMode()
    with fake_mode if device == "cuda" else contextlib.nullcontext(), torch.device(
        device
    ):
        model = model()
        args = tuple(make() for make in inputs)
        program = torch.export.export(model, args, dynamic_shapes=dynamic_shapes)
    captured = []

    def capture(cls, edge_program, compile_specs, collector):
        captured.append(edge_program.graph_module)
        return PreprocessResult(processed_bytes=b"")

    # Lowering looks the backend up by class name; patch every class of that
    # name, since the module may have been imported under another name too.
    backends = {CudaBackend}
    backends.update(
        cls for cls in BackendDetails.__subclasses__() if cls.__name__ == "CudaBackend"
    )
    with contextlib.ExitStack() as stack:
        for backend in backends:
            stack.enter_context(
                mock.patch.object(
                    backend, "_preprocess_with_weight_collector", classmethod(capture)
                )
            )
        to_edge_transform_and_lower(program, partitioner=[CudaPartitioner([])])
        to_edge(program).to_backend(CudaPartitioner([]))
    for graph_module in captured if device == "cpu" else ():
        for node in graph_module.graph.nodes:
            val = node.meta.get("val")
            if isinstance(val, torch.Tensor):
                with fake_mode:
                    node.meta["val"] = torch.empty_strided(
                        val.shape, val.stride(), dtype=val.dtype, device="cuda"
                    )
    return dict(zip(_APIS, captured))


def _targets(graph_module):
    return [n.target for n in graph_module.graph.nodes if n.op == "call_function"]


def _nodes(graph_module):
    return [
        (n.op, n.target, n.name, n.args, n.kwargs, sorted(u.name for u in n.users))
        for n in graph_module.graph.nodes
    ]


def _depthwise(channels=8, kernel=3, padding=1, dtype=torch.float32, **kwargs):
    class Depthwise(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(
                channels,
                kwargs.pop("out_channels", channels),
                kernel,
                padding=padding,
                groups=kwargs.pop("groups", channels),
                dtype=dtype,
                **kwargs,
            )

        def forward(self, x):
            return self.conv(x)

    return Depthwise


class _PassTest(unittest.TestCase):
    def setUp(self):
        if importlib.util.find_spec("triton") is None:
            self.skipTest("Triton required")

    def run_pass(self, graph_module):
        from executorch.backends.cuda.triton.replacement_pass import (
            ReplaceEdgeOpWithTritonOpPass,
        )

        with self.assertLogs(_PASS_LOGGER, "INFO") as logs:
            ReplaceEdgeOpWithTritonOpPass()(graph_module)
        graph_module.graph.lint()
        return _targets(graph_module), "\n".join(logs.output)

    def assertRouted(
        self, model, *inputs, template, canonical_args=None, device="cuda"
    ):
        for api, graph_module in _backend_graphs(model, *inputs, device=device).items():
            with self.subTest(api=api):
                targets, logs = self.run_pass(graph_module)
                self.assertEqual(targets.count(torch.ops.triton.convolution.default), 1)
                self.assertIn(f"optional, {template} template", logs)
                self.assertIn("Replaced 1 nodes", logs)
                node = next(
                    n
                    for n in graph_module.graph.nodes
                    if n.target == torch.ops.triton.convolution.default
                )
                if canonical_args is not None and api == _APIS[0]:
                    self.assertEqual(node.args[3:], canonical_args)

    def assertKept(self, model, *inputs, reason, **export_kwargs):
        for api, graph_module in _backend_graphs(
            model, *inputs, **export_kwargs
        ).items():
            with self.subTest(api=api):
                targets, logs = self.run_pass(graph_module)
                self.assertNotIn(torch.ops.triton.convolution.default, targets)
                self.assertIn("Skipping conv replacement", logs)
                self.assertIn(reason, logs)
                self.assertIn("Replaced 0 nodes", logs)


class ConvRoutingTest(_PassTest):
    """The routing gates, with a template that supports every call."""

    def setUp(self):
        super().setUp()
        from executorch.backends.cuda.triton.kernels.conv import routing

        patch = mock.patch.object(routing, "TEMPLATES", (_permissive_template(),))
        patch.start()
        self.addCleanup(patch.stop)

    def test_canonical_arguments(self):
        self.assertRouted(
            _depthwise(),
            lambda: torch.randn(2, 8, 12, 10),
            template="permissive",
            canonical_args=([1, 1], [1, 1], [1, 1], [1, 1], False, [0, 0], 8),
        )
        # An even kernel pads one more element after than before.
        self.assertRouted(
            _depthwise(kernel=4, padding="same"),
            lambda: torch.randn(1, 8, 9, 9),
            template="permissive",
            canonical_args=([1, 1], [1, 1], [2, 2], [1, 1], False, [0, 0], 8),
        )

    def test_channels_last_cpu_export_is_routed(self):
        # torch.export traces with cuDNN off. On CPU it still records the
        # channels-last output ATen's CUDA convolution returns.
        self.assertRouted(
            _depthwise(kernel=7, padding=3, stride=2),
            lambda: torch.randn(1, 8, 15, 9).to(memory_format=torch.channels_last),
            template="permissive",
            device="cpu",
        )

        def channels_last_weight():
            return _depthwise(kernel=5, padding=2)().to(
                memory_format=torch.channels_last
            )

        self.assertRouted(
            channels_last_weight,
            lambda: torch.randn(1, 8, 6, 7),
            template="permissive",
            device="cpu",
        )

    def test_channels_last_cuda_export_is_kept(self):
        # On CUDA, with cuDNN off, the export records a contiguous output.
        self.assertKept(
            _depthwise(),
            lambda: torch.randn(1, 8, 9, 9).to(memory_format=torch.channels_last),
            reason="differs from the node's",
        )

    def test_rocm_is_kept(self):
        for graph_module in _backend_graphs(
            _depthwise(), lambda: torch.randn(1, 8, 9, 9)
        ).values():
            with mock.patch.object(torch.version, "hip", "6.0"):
                targets, logs = self.run_pass(graph_module)
            self.assertNotIn(torch.ops.triton.convolution.default, targets)
            self.assertIn("Skipping conv replacement", logs)
            self.assertIn("ROCm", logs)

    def test_symbolic_shape_is_kept(self):
        batch = torch.export.Dim("batch", min=2, max=8)
        self.assertKept(
            _depthwise(),
            lambda: torch.randn(2, 8, 9, 9),
            reason="symbolic shapes",
            dynamic_shapes=({0: batch},),
        )

    def test_output_layout_mismatch_is_kept(self):
        for graph_module in _backend_graphs(
            _depthwise(), lambda: torch.randn(1, 8, 9, 9)
        ).values():
            conv = next(n for n in graph_module.graph.nodes if "conv" in str(n.target))
            val = conv.meta["val"]
            conv.meta["val"] = val.as_strided(val.shape, (648, 1, 72, 8))
            targets, logs = self.run_pass(graph_module)
            self.assertNotIn(torch.ops.triton.convolution.default, targets)
            self.assertIn("differs from the node's", logs)

    def test_worth_routing_gates_only_optional_calls(self):
        from executorch.backends.cuda.triton.kernels.conv import routing

        unprofitable = dataclasses.replace(
            _permissive_template(), worth_routing=lambda call: "not worth it"
        )
        with mock.patch.object(routing, "TEMPLATES", (unprofitable,)):
            self.assertKept(
                _depthwise(), lambda: torch.randn(1, 8, 9, 9), reason="not worth it"
            )
            # Only a template can compile a dilated convolution.
            graph_module = _backend_graphs(
                _depthwise(dilation=2, padding=2), lambda: torch.randn(1, 8, 9, 9)
            )[_APIS[0]]
            targets, logs = self.run_pass(graph_module)
        self.assertIn(torch.ops.triton.convolution.default, targets)
        self.assertIn("required, permissive template", logs)

    def test_without_a_template_the_node_is_kept(self):
        from executorch.backends.cuda.triton.kernels.conv import routing

        with mock.patch.object(routing, "TEMPLATES", ()):
            self.assertKept(
                _depthwise(), lambda: torch.randn(1, 8, 9, 9), reason="no template"
            )


def _permissive_template():
    """A test template that claims every call, so the routing of calls the
    regular lowering cannot compile (REQUIRED) can be exercised."""
    from executorch.backends.cuda.triton.kernels.conv.conv import ConvTemplate

    return ConvTemplate(
        "permissive",
        lambda call: None,
        ("any",),
        lambda name, call: None,
        lambda *args, **kwargs: None,
        lambda *args, **kwargs: None,
        lambda call: None,
    )


class ConvFailureContractTest(unittest.TestCase):
    """Faults injected at each step: OPTIONAL calls keep their node, REQUIRED
    calls raise; either way the graph is left as it was."""

    def setUp(self):
        if importlib.util.find_spec("triton") is None:
            self.skipTest("Triton required")

    def graph(self, dilation=1):
        """A conv whose output has two users."""
        conv = _depthwise(dilation=dilation, padding=dilation)

        class TwoUsers(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = conv()

            def forward(self, x):
                y = self.conv(x)
                return y + 1, y * 2

        graph_module = _backend_graphs(TwoUsers, lambda: torch.randn(1, 8, 9, 9))[
            _APIS[0]
        ]
        (conv_node,) = (
            n
            for n in graph_module.graph.nodes
            if n.op == "call_function" and "conv" in str(n.target)
        )
        self.assertEqual(len(conv_node.users), 2)
        return graph_module

    def run_pass(self, graph_module):
        from executorch.backends.cuda.triton.replacement_pass import (
            ReplaceEdgeOpWithTritonOpPass,
        )

        ReplaceEdgeOpWithTritonOpPass()(graph_module)

    def faults(self):
        from executorch.backends.cuda.triton.kernels.conv import routing

        def fail(*args, **kwargs):
            raise RuntimeError("injected fault")

        # Faults in the rewrite itself are tested in test_triton_replacement.py.
        return {
            "binding": mock.patch.object(routing, "bind_conv", fail),
            "preflight": mock.patch.object(routing, "out_shape_and_stride", fail),
            "template selection": mock.patch.object(
                routing, "TEMPLATES", (_failing_template(),)
            ),
        }

    def assertUnchanged(self, graph_module, before):
        graph_module.graph.lint()
        self.assertEqual(_nodes(graph_module), before)
        self.assertNotIn(torch.ops.triton.convolution.default, _targets(graph_module))

    def test_optional_failures_keep_the_node(self):
        from executorch.backends.cuda.triton.kernels.conv import routing

        for step, fault in self.faults().items():
            graph_module = self.graph()
            before = _nodes(graph_module)
            with self.subTest(step=step):
                with mock.patch.object(
                    routing, "TEMPLATES", (_permissive_template(),)
                ), fault:
                    self.run_pass(graph_module)
                self.assertUnchanged(graph_module, before)

    def test_required_is_replaced(self):
        from executorch.backends.cuda.triton.kernels.conv import routing

        graph_module = self.graph(dilation=2)
        with self.assertLogs(_PASS_LOGGER, "INFO") as logs:
            with mock.patch.object(routing, "TEMPLATES", (_permissive_template(),)):
                self.run_pass(graph_module)
        self.assertIn("required, permissive template", "\n".join(logs.output))
        self.assertIn(torch.ops.triton.convolution.default, _targets(graph_module))

    def test_required_failures_raise(self):
        from executorch.backends.cuda.triton.kernels.conv import routing
        from executorch.backends.cuda.triton.routing import ReplacementError

        faults = self.faults()
        # Nothing is classified before binding; a call that cannot be bound
        # keeps the regular lowering.
        del faults["binding"]
        for step, fault in faults.items():
            graph_module = self.graph(dilation=2)
            before = _nodes(graph_module)
            with self.subTest(step=step):
                with mock.patch.object(
                    routing, "TEMPLATES", (_permissive_template(),)
                ), fault:
                    with self.assertRaisesRegex(
                        ReplacementError,
                        r"convolution\(input=float32\[1, 8, 9, 9\].*dilation=\[2, 2\]",
                    ):
                        self.run_pass(graph_module)
                self.assertUnchanged(graph_module, before)


def _failing_template():
    from executorch.backends.cuda.triton.kernels.conv.conv import ConvTemplate

    def fail(*args, **kwargs):
        raise RuntimeError("injected fault")

    return ConvTemplate("failing", fail, ("any",), fail, fail, fail, fail)


class DirectTemplateRoutingTest(_PassTest):
    """Which convolutions the direct-stencil template takes."""

    def test_depthwise_is_routed(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self.assertRouted(
                    _depthwise(dtype=dtype),
                    lambda dtype=dtype: torch.randn(2, 8, 12, 10, dtype=dtype),
                    template="direct",
                )
        self.assertRouted(
            _depthwise(kernel=4, padding="same"),
            lambda: torch.randn(1, 8, 9, 9),
            template="direct",
            canonical_args=([1, 1], [1, 1], [2, 2], [1, 1], False, [0, 0], 8),
        )
        self.assertRouted(
            _depthwise(kernel=7, padding=3, stride=2),
            lambda: torch.randn(1, 8, 15, 9).to(memory_format=torch.channels_last),
            template="direct",
            device="cpu",
        )

    def test_grouped_is_routed(self):
        # (channels, out channels, groups, kernel, stride)
        for channels, out_channels, groups, kernel, stride in (
            (8, 16, 8, 3, 1),
            (8, 8, 4, 5, 2),
            (16, 4, 2, 7, 1),
            # 4 input and output channels per group with 1x1 and 3x3 kernels.
            (16, 16, 4, 1, 1),
            (16, 16, 4, 3, 1),
        ):
            with self.subTest(channels=channels, groups=groups):
                self.assertRouted(
                    _depthwise(
                        channels=channels,
                        out_channels=out_channels,
                        groups=groups,
                        kernel=kernel,
                        padding=kernel // 2,
                        stride=stride,
                    ),
                    lambda channels=channels: torch.randn(2, channels, 12, 10),
                    template="direct",
                )

    def test_other_convolutions_are_kept(self):
        x = lambda: torch.randn(1, 8, 9, 9)  # noqa: E731
        self.assertKept(
            _depthwise(channels=18, groups=2),
            lambda: torch.randn(1, 18, 9, 9),
            reason="at most 8 input channels per group",
        )
        for channels, out_channels, groups, kernel, stride, reason in (
            (16, 16, 4, 5, 1, "where Inductor's template is as fast"),
            (8, 16, 4, 9, 1, "grouped kernel [9, 9] is not benchmarked"),
            (8, 16, 4, 3, 3, "grouped stride [3, 3] is not benchmarked"),
        ):
            with self.subTest(groups=groups, out_channels=out_channels, kernel=kernel):
                self.assertKept(
                    _depthwise(
                        channels=channels,
                        out_channels=out_channels,
                        groups=groups,
                        kernel=kernel,
                        padding=kernel // 2,
                        stride=stride,
                    ),
                    lambda channels=channels: torch.randn(1, channels, 12, 12),
                    reason=reason,
                )
        for dtype, reason in (
            (torch.bfloat16, "reduction of 196 per group"),
            (torch.float16, "grouped float16 is not benchmarked"),
        ):
            with self.subTest(dtype=dtype):
                self.assertKept(
                    _depthwise(
                        channels=16,
                        out_channels=8,
                        groups=4,
                        kernel=7,
                        padding=3,
                        dtype=dtype,
                    ),
                    lambda dtype=dtype: torch.randn(1, 16, 12, 12, dtype=dtype),
                    reason=reason,
                )
        self.assertKept(_depthwise(dilation=2, padding=2), x, reason="dilation [2, 2]")
        self.assertKept(_depthwise(kernel=11, padding=5), x, reason="larger than 9")
        self.assertKept(
            _depthwise(dtype=torch.float64),
            lambda: torch.randn(1, 8, 9, 9, dtype=torch.float64),
            reason="dtype torch.float64",
        )

        class Dense(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(8, 4, 3)

            def forward(self, x):
                return self.conv(x)

        self.assertKept(Dense, x, reason="not a depthwise convolution")

        class Depthwise3d(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv3d(8, 8, 3, groups=8)

            def forward(self, x):
                return self.conv(x)

        self.assertKept(
            Depthwise3d, lambda: torch.randn(1, 8, 5, 5, 5), reason="3-D convolution"
        )

        class Transposed(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.ConvTranspose2d(8, 8, 3, groups=8)

            def forward(self, x):
                return self.conv(x)

        self.assertKept(Transposed, x, reason="transposed convolution")

    def test_benchmarked_range(self):
        from executorch.backends.cuda.triton.kernels.conv.core import direct

        # (channels, input size, kernel, stride, padding): the edge of the
        # benchmarked range is routed, one step past it is not.
        edges = (
            (
                (8, (1, 8, 20, 20), 3, 3, 1),
                (8, (1, 8, 20, 20), 3, 4, 1),
                "stride [4, 4] is not benchmarked",
            ),
            (
                (8, (1, 8, 20, 20), 5, 1, 4),
                (8, (1, 8, 20, 20), 5, 1, 5),
                "beyond kernel - 1 is not benchmarked",
            ),
            (
                (
                    direct.MAX_ROUTED_CHANNELS,
                    (1, direct.MAX_ROUTED_CHANNELS, 4, 4),
                    3,
                    1,
                    1,
                ),
                (4100, (1, 4100, 4, 4), 3, 1, 1),
                "4100 channels are not benchmarked",
            ),
            (
                (2, (1, 2, 8, direct.MAX_ROUTED_SPATIAL_SIZE), 3, 1, 1),
                (2, (1, 2, 8, direct.MAX_ROUTED_SPATIAL_SIZE + 1), 3, 1, 1),
                "spatial size [8, 2049] is not benchmarked",
            ),
            (
                (128, (32, 128, 112, 112), 3, 1, 1),
                (128, (33, 128, 112, 112), 3, 1, 1),
                "element input is not benchmarked",
            ),
        )
        for inside, outside, reason in edges:
            for case, routed in ((inside, True), (outside, False)):
                channels, shape, kernel, stride, padding = case
                model = _depthwise(
                    channels=channels, kernel=kernel, padding=padding, stride=stride
                )
                with self.subTest(case=case):
                    if routed:
                        self.assertRouted(
                            model,
                            lambda shape=shape: torch.randn(shape),
                            template="direct",
                        )
                    else:
                        self.assertKept(
                            model, lambda shape=shape: torch.randn(shape), reason=reason
                        )
