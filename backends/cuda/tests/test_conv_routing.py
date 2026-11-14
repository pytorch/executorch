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
import math
import unittest
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensorMode

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
        self,
        model,
        *inputs,
        template,
        canonical_args=None,
        output_strides=None,
        device="cuda",
    ):
        """Routed, with the output strides the conv node recorded (and, if
        given, ``output_strides``)."""
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
                self.assertEqual(node.args[-1], list(node.meta["val"].stride()))
                if output_strides is not None:
                    self.assertEqual(node.args[-1], output_strides)
                if canonical_args is not None and api == _APIS[0]:
                    self.assertEqual(node.args[3:-1], canonical_args)

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

    def test_channels_last_cuda_export_follows_the_graph(self):
        # On CUDA, with cuDNN off, the export records a contiguous output.
        self.assertRouted(
            _depthwise(),
            lambda: torch.randn(1, 8, 9, 9).to(memory_format=torch.channels_last),
            template="permissive",
            output_strides=[648, 81, 9, 1],
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

    def test_recorded_output_layout(self):
        # (size, strides) recorded for the [2, 8, 9, 9] output, and why the
        # node is kept; every dense layout is followed.
        for size, strides, reason in (
            ((2, 8, 9, 9), (648, 81, 9, 1), None),
            ((2, 8, 9, 9), (648, 1, 72, 8), None),
            ((2, 8, 9, 9), (81, 162, 9, 1), None),
            ((2, 8, 9, 9), (700, 81, 9, 1), "are not a dense layout"),
            ((2, 8, 9, 9), (648, 81, 9, 2), "are not a dense layout"),
            ((2, 8, 9, 8), (576, 72, 8, 1), "differs from the node's"),
        ):
            for graph_module in _backend_graphs(
                _depthwise(), lambda: torch.randn(2, 8, 9, 9)
            ).values():
                conv = next(
                    n
                    for n in graph_module.graph.nodes
                    if n.op == "call_function" and "conv" in str(n.target)
                )
                val = conv.meta["val"]
                with val.fake_mode:
                    conv.meta["val"] = torch.empty_strided(
                        size, strides, dtype=val.dtype, device=val.device
                    )
                with self.subTest(size=size, strides=strides):
                    targets, logs = self.run_pass(graph_module)
                    if reason is None:
                        (node,) = [
                            n
                            for n in graph_module.graph.nodes
                            if n.target == torch.ops.triton.convolution.default
                        ]
                        self.assertEqual(node.args[-1], list(strides))
                    else:
                        self.assertNotIn(torch.ops.triton.convolution.default, targets)
                        self.assertIn(reason, logs)

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
    def test_channels_last_cuda_export_is_routed(self):
        # The export on CUDA records a contiguous output for a channels-last
        # input; the converted conv writes exactly that layout.
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self.assertRouted(
                    _depthwise(channels=64, dtype=dtype),
                    lambda dtype=dtype: torch.randn(2, 64, 28, 28, dtype=dtype).to(
                        memory_format=torch.channels_last
                    ),
                    template="direct",
                    output_strides=[50176, 784, 28, 1],
                )

    def test_reordering_size_edge(self):
        # Channels-last input, contiguous output: routed from the smallest
        # measured input that wins on; the same-order layouts are not limited.
        from executorch.backends.cuda.triton.kernels.conv.core import direct
        from executorch.backends.cuda.triton.kernels.conv.core.common import (
            ConvCall,
            ConvParams,
        )

        params = ConvParams((1, 1), (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        for batch, output, routed in (
            (2, "contiguous", True),
            (1, "contiguous", False),
            (1, "channels_last", True),
        ):
            with FakeTensorMode():
                x = torch.empty(batch, 8, 28, 28, device="cuda").contiguous(
                    memory_format=torch.channels_last
                )
                w = torch.empty(8, 1, 3, 3, device="cuda")
                out = torch.empty(batch, 8, 28, 28, device="cuda")
                if output == "channels_last":
                    out = out.contiguous(memory_format=torch.channels_last)
            self.assertEqual(2 * 8 * 28 * 28, direct.MIN_ROUTED_REORDERING_ELEMENTS)
            reason = direct.worth_routing(
                ConvCall(x, w, None, params, tuple(out.stride()))
            )
            with self.subTest(batch=batch, output=output):
                if routed:
                    self.assertIsNone(reason)
                else:
                    self.assertIn("contiguous output is not benchmarked", reason)
        self.assertKept(
            _depthwise(),
            lambda: torch.randn(1, 8, 9, 9).to(memory_format=torch.channels_last),
            reason="contiguous output is not benchmarked faster",
        )

    def test_only_benchmarked_layouts_are_routed(self):
        from executorch.backends.cuda.triton.kernels.conv.core import direct
        from executorch.backends.cuda.triton.kernels.conv.core.common import (
            ConvCall,
            ConvParams,
        )

        params = ConvParams((1, 1), (1, 1), (1, 1), (1, 1), False, (0, 0), 64)
        shape = (2, 64, 28, 28)
        contiguous, channels_last = (50176, 784, 28, 1), (50176, 1, 1792, 64)
        channel_outermost = (784, 1568, 28, 1)
        with FakeTensorMode():
            x = torch.empty(shape, device="cuda")
            w = torch.empty(64, 1, 3, 3, device="cuda")
            x_cl = x.contiguous(memory_format=torch.channels_last)
            w_cl = w.contiguous(memory_format=torch.channels_last)
            sliced = x.as_strided(shape, x.stride(), 1)
            padded = x.as_strided(shape, (64 * 785, 785, 28, 1))
        # (input, weight, output strides, routed); a depthwise weight matches
        # both memory formats.
        for x_, w_, out, routed in (
            (x, w, contiguous, True),
            (x, w, channels_last, True),
            (x, w_cl, channels_last, True),
            (x_cl, w, channels_last, True),
            (x_cl, w_cl, contiguous, True),
            (x_cl, w, contiguous, True),
            (x, w, channel_outermost, False),
            (x_cl, w, channel_outermost, False),
            (sliced, w, contiguous, False),
            (padded, w, contiguous, False),
        ):
            call = ConvCall(x_, w_, None, params, out)
            with self.subTest(input=x_.stride(), weight=w_.stride(), output=out):
                reason = direct.worth_routing(call)
                if routed:
                    self.assertIsNone(reason)
                else:
                    self.assertIn("not a benchmarked layout", reason)

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
                self.conv = nn.Conv2d(8, 4, 1)

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


def _dense(
    conv, in_channels, out_channels, kernel, dtype, view=None, pad=None, **kwargs
):
    class Dense(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = conv(in_channels, out_channels, kernel, dtype=dtype, **kwargs)

        def forward(self, x):
            x = x if view is None else x.view(view)
            return self.conv(x if pad is None else F.pad(x, pad))

    return Dense


# Benchmarked convolutions in the patterns models write them: (row, model,
# input shape). The 3-input-channel ones are routed; the rest
# keep the regular lowering, and the reason.
_ROUTED_ROWS = (
    (
        "2x14x14 kernel == stride",
        lambda dtype: _dense(
            nn.Conv3d,
            3,
            1280,
            (2, 14, 14),
            dtype,
            view=(-1, 3, 2, 14, 14),
            stride=(2, 14, 14),
            bias=False,
        ),
        (256, 3 * 2 * 14 * 14),
    ),
    (
        "2x16x16 kernel == stride",
        lambda dtype: _dense(
            nn.Conv3d,
            3,
            1024,
            (2, 16, 16),
            dtype,
            view=(-1, 3, 2, 16, 16),
            stride=(2, 16, 16),
        ),
        (512, 3 * 2 * 16 * 16),
    ),
    (
        "1x4x4 kernel == stride",
        lambda dtype: _dense(nn.Conv3d, 3, 256, (1, 4, 4), dtype, stride=(1, 4, 4)),
        (1, 3, 2, 544, 736),
    ),
    (
        "14x14 kernel == stride",
        lambda dtype: _dense(nn.Conv2d, 3, 1152, 14, dtype, stride=14, padding="valid"),
        (1, 3, 224, 224),
    ),
    (
        "16x16 kernel == stride",
        lambda dtype: _dense(nn.Conv2d, 3, 768, 16, dtype, stride=16, bias=False),
        (2, 3, 224, 224),
    ),
    (
        "7x7 s4, 3 channels",
        lambda dtype: _dense(nn.Conv2d, 3, 256, 7, dtype, stride=4, padding=3),
        (3, 3, 224, 224),
    ),
    (
        "7x7 s2 p3, 3 channels",
        lambda dtype: _dense(
            nn.Conv2d, 3, 64, 7, dtype, stride=2, padding=3, bias=False
        ),
        (1, 3, 480, 640),
    ),
    (
        "7x7 s2, 3 to 32 channels",
        lambda dtype: _dense(nn.Conv2d, 3, 32, 7, dtype, stride=2),
        (1, 3, 84, 84),
    ),
)
_KEPT_ROWS = (
    (
        "1x2x2 kernel == stride, 48 channels",
        lambda dtype: _dense(nn.Conv3d, 48, 3072, (1, 2, 2), dtype, stride=(1, 2, 2)),
        (1, 48, 1, 14, 28),
        "48 input channels",
    ),
    (
        "3x3",
        lambda dtype: _dense(nn.Conv2d, 64, 64, 3, dtype, padding=1, bias=False),
        (1, 64, 120, 160),
        "64 input channels",
    ),
    (
        "3x3 s2",
        lambda dtype: _dense(
            nn.Conv2d, 256, 512, 3, dtype, stride=2, padding=1, bias=False
        ),
        (1, 256, 30, 40),
        "256 input channels",
    ),
    (
        "1x1",
        lambda dtype: _dense(nn.Conv2d, 512, 512, 1, dtype),
        (1, 512, 15, 20),
        "1x1 convolution",
    ),
    (
        "1x1 s2",
        lambda dtype: _dense(nn.Conv2d, 64, 128, 1, dtype, stride=2, bias=False),
        (1, 64, 120, 160),
        "1x1 convolution",
    ),
    (
        "5x5 s2, 32 channels",
        lambda dtype: _dense(nn.Conv2d, 32, 32, 5, dtype, stride=2),
        (1, 32, 39, 39),
        "32 input channels",
    ),
)


def _layout_variants(routed_layouts):  # noqa: C901
    """(layout, whether ``routed_layouts`` admits it): functions that give a
    dense NCHW call with a contiguous weight and the default output another
    input, weight or output layout. Where size-1 dims make the output strides
    match another format too, whether that one is admitted decides."""
    from executorch.backends.cuda.triton.kernels.conv.core.common import (
        memory_formats,
        out_shape_and_stride,
    )

    def output_in(*formats):
        def routed(call):
            return bool(memory_formats(*out_shape_and_stride(call)) & set(formats))

        return routed

    def channels_last(t):
        fmt = torch.channels_last if t.dim() == 4 else torch.channels_last_3d
        return t.contiguous(memory_format=fmt)

    def output(call, order):
        shape, _ = out_shape_and_stride(call)
        strides, running = [0] * len(shape), 1
        for d in reversed(order(len(shape))):
            strides[d] = running
            running *= shape[d]
        return dataclasses.replace(call, output_strides=tuple(strides))

    def nchw(call):
        return call

    def channels_last_input(call):
        return dataclasses.replace(call, input=channels_last(call.input))

    def channels_last_input_contiguous_output(call):
        return output(channels_last_input(call), lambda n: range(n))

    def contiguous_input_channels_last_output(call):
        return output(call, lambda n: (0, *range(2, n), 1))

    def channels_last_input_channel_outermost_output(call):
        return output(channels_last_input(call), lambda n: (1, 0, *range(2, n)))

    def sliced_input(call):
        x = call.input
        return dataclasses.replace(call, input=x.as_strided(x.shape, x.stride(), 1))

    def padded_input(call):
        x = call.input
        n, c, *spatial = x.shape
        plane = math.prod(spatial) + 1
        strides = (c * plane, plane, *x.stride()[2:])
        return dataclasses.replace(call, input=x.as_strided(x.shape, strides))

    def channels_last_weight(call):
        return dataclasses.replace(call, weight=channels_last(call.weight))

    def sliced_weight(call):
        w = call.weight
        return dataclasses.replace(call, weight=w.as_strided(w.shape, w.stride(), 1))

    contiguous, cl = "contiguous", "channels_last"

    def admitted(input):
        return [o for i, w, o in routed_layouts if (i, w) == (input, contiguous)]

    return (
        (nchw, True),
        (channels_last_input, (cl, contiguous, cl) in routed_layouts),
        (
            channels_last_input_contiguous_output,
            (cl, contiguous, contiguous) in routed_layouts,
        ),
        (contiguous_input_channels_last_output, output_in(*admitted(contiguous))),
        (channels_last_input_channel_outermost_output, output_in(*admitted(cl))),
        (sliced_input, False),
        (padded_input, False),
        (channels_last_weight, False),
        (sliced_weight, False),
    )


# 3-D convolutions: a causal 3x3x3 after an explicit pad, a 1x1x1 and a
# (3,1,1) stride (2,1,1): (row, model, input shape, dtype, why Inductor
# keeps them).
_CONV3D_ROWS = (
    (
        "causal 3x3x3",
        _dense(nn.Conv3d, 160, 160, 3, torch.bfloat16, pad=(1, 1, 1, 1, 2, 0)),
        (1, 160, 1, 112, 224),
        torch.bfloat16,
        "160 input channels",
    ),
    (
        "1x1x1",
        _dense(nn.Conv3d, 160, 320, 1, torch.bfloat16),
        (2, 160, 16, 32, 32),
        torch.bfloat16,
        "1x1 convolution",
    ),
    (
        "3x1x1 s2",
        _dense(nn.Conv3d, 320, 320, (3, 1, 1), torch.bfloat16, stride=(2, 1, 1)),
        (2, 320, 17, 16, 16),
        torch.bfloat16,
        "320 input channels",
    ),
)
# 1-D convolutions over short sequences, likewise.
_CONV1D_ROWS = (
    (
        "k5",
        _dense(nn.Conv1d, 512, 512, 5, torch.float32, padding=2),
        (1, 512, 16),
        torch.float32,
        "512 input channels",
    ),
    (
        "k1",
        _dense(nn.Conv1d, 512, 1024, 1, torch.float32),
        (1, 512, 8),
        torch.float32,
        "1x1 convolution",
    ),
    (
        "k3 s2",
        _dense(nn.Conv1d, 512, 512, 3, torch.float32, stride=2, padding=1),
        (1, 512, 16),
        torch.float32,
        "512 input channels",
    ),
)


def _steps_past(families, shape, c_out, kernel, stride, padding):
    """The calls one step past each limit a benchmarked corner sits on."""
    from executorch.backends.cuda.triton.kernels.conv.core import implicit_gemm

    family = families[(kernel, stride, padding)]
    batch, channels, *extent = shape
    yield (batch, channels - 1, *extent), c_out, kernel, stride, padding
    yield (batch, channels + 1, *extent), c_out, kernel, stride, padding
    per_image = implicit_gemm._outputs(1, extent, kernel, stride, padding)
    if batch > 1 and (batch - 1) * per_image < implicit_gemm.MIN_ROUTED_OUTPUT_PIXELS:
        yield (batch - 1, channels, *extent), c_out, kernel, stride, padding
    pixels = batch * math.prod(extent)
    if (
        batch == family.max_batch
        or pixels + math.prod(extent) > implicit_gemm.MAX_ROUTED_INPUT_PIXELS
    ):
        yield (batch + 1, channels, *extent), c_out, kernel, stride, padding
    for i, e in enumerate(extent):
        grown = [*extent[:i], e + 1, *extent[i + 1 :]]
        if (
            e == family.max_spatial[i]
            or batch * math.prod(grown) > implicit_gemm.MAX_ROUTED_INPUT_PIXELS
        ):
            yield (batch, channels, *grown), c_out, kernel, stride, padding
    outputs = implicit_gemm._outputs(batch, extent, kernel, stride, padding)
    most = min(
        implicit_gemm.MAX_ROUTED_OUT_CHANNELS,
        implicit_gemm.MAX_ROUTED_OUTPUT_ELEMENTS // outputs,
    )
    if c_out == most:
        yield shape, c_out + 1, kernel, stride, padding
    if c_out == implicit_gemm.MIN_ROUTED_OUT_CHANNELS > 1:
        yield shape, c_out - 1, kernel, stride, padding
    for geometry in (
        (kernel, tuple(s + 1 for s in stride), padding),
        (tuple(k + 2 for k in kernel), stride, padding),
        (kernel, stride, tuple(p + 1 for p in padding)),
    ):
        if geometry not in families:
            yield (shape, c_out, *geometry)


class ImplicitGemmRoutingTest(_PassTest):
    """Which dense convolutions the implicit-GEMM template takes."""

    def setUp(self):
        super().setUp()
        from executorch.backends.cuda.triton.kernels.conv.core import implicit_gemm

        # Exported on an A100, FP32 without TF32 unless a test allows it.
        for patch in (
            mock.patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)),
            mock.patch.object(implicit_gemm, "conv_allows_tf32", return_value=False),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def tf32(self, allowed):
        from executorch.backends.cuda.triton.kernels.conv.core import implicit_gemm

        return mock.patch.object(
            implicit_gemm, "conv_allows_tf32", return_value=allowed
        )

    def test_benchmarked_rows(self):
        for dtype in (torch.float32, torch.bfloat16):
            for row, model, shape in _ROUTED_ROWS:
                with self.subTest(row=row, dtype=dtype):
                    self.assertRouted(
                        model(dtype),
                        lambda shape=shape, dtype=dtype: torch.randn(
                            shape, dtype=dtype
                        ),
                        template="implicit_gemm",
                    )
            for row, model, shape, reason in _KEPT_ROWS:
                with self.subTest(row=row, dtype=dtype):
                    self.assertKept(
                        model(dtype),
                        lambda shape=shape, dtype=dtype: torch.randn(
                            shape, dtype=dtype
                        ),
                        reason=reason,
                    )

    def test_channels_last_cuda_export_is_routed(self):
        # The export on CUDA records a contiguous output for a channels-last
        # input; the converted conv writes exactly that layout.
        for (row, model, shape), strides in (
            (_ROUTED_ROWS[6], [4915200, 76800, 320, 1]),
            (_ROUTED_ROWS[7], [48672, 1521, 39, 1]),
        ):
            with self.subTest(row=row):
                self.assertRouted(
                    model(torch.float32),
                    lambda shape=shape: torch.randn(shape).to(
                        memory_format=torch.channels_last
                    ),
                    template="implicit_gemm",
                    output_strides=strides,
                )

    def test_canonical_arguments(self):
        # padding="valid" binds to zero padding on both sides.
        self.assertRouted(
            _ROUTED_ROWS[3][1](torch.float32),
            lambda: torch.randn(1, 3, 224, 224),
            template="implicit_gemm",
            canonical_args=([14, 14], [0, 0], [0, 0], [1, 1], False, [0, 0], 1),
        )

    def test_precision_without_tensor_cores_is_kept(self):
        _, model, shape = _ROUTED_ROWS[6]
        with mock.patch.object(
            torch.cuda, "get_device_capability", return_value=(7, 5)
        ):
            for dtype, tf32 in ((torch.bfloat16, False), (torch.float32, True)):
                with self.subTest(dtype=dtype, tf32=tf32), self.tf32(tf32):
                    self.assertKept(
                        model(dtype),
                        lambda dtype=dtype: torch.randn(shape, dtype=dtype),
                        reason="without tensor cores",
                    )
            self.assertRouted(
                model(torch.float32),
                lambda: torch.randn(shape),
                template="implicit_gemm",
            )

    def worth_routing(self, *call, dtype, tf32, layout=None):
        from executorch.backends.cuda.triton.kernels.conv.core import implicit_gemm

        with self.tf32(tf32):
            return implicit_gemm.worth_routing(self.call(*call, dtype, layout))

    def call(self, shape, out_channels, kernel, stride, before, after, dtype, layout):
        from executorch.backends.cuda.triton.kernels.conv.core.common import (
            ConvCall,
            ConvParams,
        )

        rank = len(kernel)
        params = ConvParams(
            tuple(stride),
            tuple(before),
            tuple(after),
            (1,) * rank,
            False,
            (0,) * rank,
            1,
        )
        with FakeTensorMode():
            x = torch.empty(shape, dtype=dtype, device="cuda")
            w = torch.empty(out_channels, shape[1], *kernel, dtype=dtype, device="cuda")
            call = ConvCall(x, w, None, params)
            return call if layout is None else layout(call)

    def precisions(self):
        """(dtype, TF32 allowed) of the FMA and tensor-core cases."""
        return (
            (torch.float32, False),
            (torch.bfloat16, False),
            (torch.float16, False),
            (torch.float32, True),
        )

    def test_benchmarked_calls_are_routed(self):
        # Every call the benchmark's boundary suite measures.
        from executorch.backends.cuda.triton.kernels.conv.core import implicit_gemm

        for dtype, tf32 in self.precisions():
            for shape, c_out, kernel, stride, padding in implicit_gemm.boundary_calls():
                with self.subTest(
                    shape=shape, c_out=c_out, kernel=kernel, dtype=dtype, tf32=tf32
                ):
                    self.assertIsNone(
                        self.worth_routing(
                            shape,
                            c_out,
                            kernel,
                            stride,
                            padding,
                            padding,
                            dtype=dtype,
                            tf32=tf32,
                        )
                    )

    def test_one_step_past_the_benchmarked_calls_is_kept(self):
        """Each benchmarked corner moved one step past a limit it sits on: one
        input channel more or less, one image, pixel or output channel more,
        one image fewer below the output-pixel minimum, or a stride, kernel or
        padding outside the benchmarked geometries."""
        from executorch.backends.cuda.triton.kernels.conv.core import implicit_gemm

        families = {
            geometry: family
            for family in implicit_gemm.ROUTED_FAMILIES
            for geometry in family.geometries
        }

        for dtype, tf32 in self.precisions():
            for call in implicit_gemm.boundary_calls():
                padding = call[4]
                outside = [(*step, step[-1]) for step in _steps_past(families, *call)]
                # Asymmetric padding: one more element after.
                outside.append((*call[:4], padding, tuple(p + 1 for p in padding)))
                for shape, c_out, kernel, stride, before, after in outside:
                    with self.subTest(
                        call=call, outside=(shape, c_out, kernel, stride, before, after)
                    ):
                        self.assertIsNotNone(
                            self.worth_routing(
                                shape,
                                c_out,
                                kernel,
                                stride,
                                before,
                                after,
                                dtype=dtype,
                                tf32=tf32,
                            )
                        )

    def test_only_benchmarked_layouts_are_routed(self):
        """The (input, weight, output) memory formats the benchmark measures are
        routed; sliced or padded inputs, channels-last or sliced weights and
        other output layouts keep the regular lowering."""
        from executorch.backends.cuda.triton.kernels.conv.core import implicit_gemm

        for dtype, tf32 in self.precisions():
            for call in implicit_gemm.boundary_calls():
                shape, c_out, kernel, stride, padding = call
                for layout, routed in _layout_variants(implicit_gemm.ROUTED_LAYOUTS):
                    with self.subTest(
                        call=call, layout=layout.__name__, dtype=dtype, tf32=tf32
                    ):
                        reason = self.worth_routing(
                            shape,
                            c_out,
                            kernel,
                            stride,
                            padding,
                            padding,
                            dtype=dtype,
                            tf32=tf32,
                            layout=layout,
                        )
                        if callable(routed):
                            routed = routed(
                                self.call(*call[:4], padding, padding, dtype, layout)
                            )
                        if routed:
                            self.assertIsNone(reason)
                        else:
                            self.assertIn("not a benchmarked layout", reason)

    def test_conv3d_rows(self):
        # Inductor keeps every row: none is benchmarked faster.
        for row, model, shape, dtype, reason in _CONV3D_ROWS:
            with self.subTest(row=row):
                self.assertKept(
                    model,
                    lambda shape=shape, dtype=dtype: torch.randn(shape, dtype=dtype),
                    reason=reason,
                )

    def test_conv1d_rows(self):
        # to_edge leaves a 1-D convolution as aten.convolution, which Inductor
        # cannot lower, so the template takes it. to_edge_transform_and_lower
        # leaves aten.conv1d, which the backend lowers as a 2-D convolution;
        # no row is benchmarked faster, so Inductor keeps it.
        for row, model, shape, dtype, reason in _CONV1D_ROWS:
            graphs = _backend_graphs(
                model, lambda shape=shape, dtype=dtype: torch.randn(shape, dtype=dtype)
            )
            for api, graph_module in graphs.items():
                with self.subTest(row=row, api=api):
                    targets, logs = self.run_pass(graph_module)
                    if api == "to_edge_to_backend":
                        self.assertEqual(
                            targets.count(torch.ops.triton.convolution.default), 1
                        )
                        self.assertIn("required, implicit_gemm template", logs)
                    else:
                        self.assertNotIn(torch.ops.triton.convolution.default, targets)
                        self.assertIn(reason, logs)

    def test_same_padded_conv1d_is_required(self):
        # The backend decomposes only conv1d.default into a 2-D convolution.
        graphs = _backend_graphs(
            _dense(nn.Conv1d, 16, 32, 4, torch.float32, padding="same"),
            lambda: torch.randn(1, 16, 16),
        )
        for api, graph_module in graphs.items():
            with self.subTest(api=api):
                targets, logs = self.run_pass(graph_module)
                self.assertEqual(targets.count(torch.ops.triton.convolution.default), 1)
                self.assertIn("required, implicit_gemm template", logs)

    def test_explicit_pad_stays_its_own_node(self):
        from executorch.backends.cuda.triton.kernels.conv import routing

        take_all = tuple(
            dataclasses.replace(t, worth_routing=lambda call: None)
            for t in routing.TEMPLATES
        )
        for row, model, shape, dtype, _ in _CONV3D_ROWS[:1]:
            graphs = _backend_graphs(
                model, lambda shape=shape, dtype=dtype: torch.randn(shape, dtype=dtype)
            )
            for api, graph_module in graphs.items():
                with self.subTest(row=row, api=api), mock.patch.object(
                    routing, "TEMPLATES", take_all
                ):
                    self.run_pass(graph_module)
                    (conv,) = [
                        n
                        for n in graph_module.graph.nodes
                        if n.target == torch.ops.triton.convolution.default
                    ]
                    self.assertRegex(
                        str(conv.args[0].target), r"aten\.(constant_pad_nd|pad)\."
                    )
                    rank = len(shape) - 2
                    self.assertEqual(conv.args[4:6], ([0] * rank, [0] * rank))

    def test_transposed(self):
        # (model, input shape, dtype, whether each API's graph needs the
        # template): to_edge keeps aten.convolution, which Inductor cannot lower
        # transposed on CUDA; to_edge_transform_and_lower keeps
        # conv_transpose1d/2d, which the backend rewrites as convolutions, and
        # conv_transpose3d, which it cannot lower.
        cases = (
            (
                "k4 s2 upsample",
                _dense(
                    nn.ConvTranspose1d,
                    1024,
                    1024,
                    4,
                    torch.float32,
                    stride=2,
                    padding=1,
                ),
                (1, 1024, 4),
                torch.float32,
                {"to_edge_transform_and_lower": False, "to_edge_to_backend": True},
            ),
            (
                "3-channel 2-D",
                _dense(
                    nn.ConvTranspose2d, 3, 64, 3, torch.bfloat16, stride=2, padding=1
                ),
                (1, 3, 56, 56),
                torch.bfloat16,
                {"to_edge_transform_and_lower": False, "to_edge_to_backend": True},
            ),
            (
                "3-D",
                _dense(
                    nn.ConvTranspose3d,
                    8,
                    4,
                    3,
                    torch.float32,
                    stride=2,
                    output_padding=1,
                ),
                (1, 8, 3, 4, 5),
                torch.float32,
                {"to_edge_transform_and_lower": True, "to_edge_to_backend": True},
            ),
        )
        for row, model, shape, dtype, required in cases:
            graphs = _backend_graphs(
                model, lambda shape=shape, dtype=dtype: torch.randn(shape, dtype=dtype)
            )
            for api, graph_module in graphs.items():
                with self.subTest(row=row, api=api):
                    targets, logs = self.run_pass(graph_module)
                    if required[api]:
                        (node,) = [
                            n
                            for n in graph_module.graph.nodes
                            if n.target == torch.ops.triton.convolution.default
                        ]
                        self.assertIn("required, implicit_gemm template", logs)
                        self.assertTrue(node.args[7])
                        self.assertEqual(node.args[-1], list(node.meta["val"].stride()))
                    else:
                        self.assertNotIn(torch.ops.triton.convolution.default, targets)
                        self.assertIn("transposed convolution is not benchmarked", logs)

    def test_unsupported_transposed_is_kept(self):
        for model, reason in (
            (
                _dense(nn.ConvTranspose1d, 8, 8, 3, torch.float32, dilation=2),
                "implicit GEMM: dilation [2]",
            ),
            (
                _dense(nn.ConvTranspose1d, 8, 8, 3, torch.float32, groups=2),
                "implicit GEMM: grouped convolution",
            ),
        ):
            with self.subTest(reason=reason):
                self.assertKept(model, lambda: torch.randn(1, 8, 9), reason=reason)
