# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.passes import LegalizePortableDimOrderPass
from executorch.extension.pybindings.portable_lib import (  # @manual
    _load_for_executorch_from_buffer,
)
from torch.export import Dim, export
from torch.testing import FileCheck


class MeanIssueReproModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.mean(x, dim=[2, 3], keepdim=True)
        return x.reshape([-1, 2])


class MeanNoReshapeModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=[2, 3], keepdim=True)


class MeanKeepdimFalseModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=[2, 3], keepdim=False)


class MeanSingleDimModule(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=[self.dim], keepdim=True)


class LinearIssueReproModule(torch.nn.Module):
    def __init__(self, out_features: int = 10):
        super().__init__()
        self.linear = torch.nn.Linear(10, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ClonePoolModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.avg_pool2d(torch.relu(x).clone(), 2)


class CloneCopyModule(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.relu(x).copy_(torch.relu(y).clone())


class CopyIntoBufferModule(torch.nn.Module):
    def __init__(self, memory_format: torch.memory_format, shape=(2, 3, 4, 6)):
        super().__init__()
        self.memory_format = memory_format
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        buffer = torch.empty(self.shape, memory_format=self.memory_format)
        return buffer.copy_(x)


class CopyIntoReluModule(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.relu(x).copy_(y)


class CopyIntoMutableBufferModule(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.register_buffer(
            "buffer", torch.zeros(shape).to(memory_format=torch.channels_last)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        previous = self.buffer.clone()
        self.buffer.copy_(x)
        return previous


class CopyIntoDataSizedBufferModule(torch.nn.Module):
    def forward(self, channels: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        c = channels.item()
        torch._check(c >= 1)
        torch._check(c <= x.shape[1])
        buffer = torch.empty((2, c, 4, 6), memory_format=torch.channels_last)
        return buffer.copy_(x[:, :c])


class AvgPoolCopyModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.avg_pool2d(x, 2)
        return torch.empty_like(x, memory_format=torch.channels_last).copy_(x)


class TestLegalizePortableDimOrderPass(unittest.TestCase):
    @staticmethod
    def _build_edge(module: torch.nn.Module, sample_input, dynamic_shapes=None):
        return to_edge_transform_and_lower(
            export(module, sample_input, dynamic_shapes=dynamic_shapes, strict=True),
            partitioner=[],
        )

    @staticmethod
    def _build_executorch_program(
        module: torch.nn.Module, sample_input, dynamic_shapes=None
    ):
        edge = TestLegalizePortableDimOrderPass._build_edge(
            module, sample_input, dynamic_shapes
        )
        executorch_prog = edge.to_executorch(
            config=ExecutorchBackendConfig(extract_delegate_segments=False)
        )
        return edge, executorch_prog

    @staticmethod
    def _find_call_function_nodes(graph_module, target_substring: str):
        return [
            node
            for node in graph_module.graph.nodes
            if node.op == "call_function" and target_substring in str(node.target)
        ]

    def _assert_runtime_matches_eager(
        self,
        module: torch.nn.Module,
        sample_input,
        *,
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ):
        edge, executorch_prog = self._build_executorch_program(module, sample_input)
        executorch_module = _load_for_executorch_from_buffer(executorch_prog.buffer)

        with torch.no_grad():
            expected = module(*sample_input)
        runtime_output = executorch_module.run_method("forward", sample_input)[0]

        torch.testing.assert_close(
            runtime_output,
            expected,
            atol=atol,
            rtol=rtol,
        )
        return edge.exported_program().graph_module

    def test_channels_last_mean_inserts_copy_before_mean(self) -> None:
        module = MeanIssueReproModule().eval().to(memory_format=torch.channels_last)
        sample_input = (torch.randn(1, 2, 3, 4).to(memory_format=torch.channels_last),)

        edge = self._build_edge(module, sample_input)
        updated = edge.transform([LegalizePortableDimOrderPass()])
        graph_module = updated.exported_program().graph_module

        copy_nodes = self._find_call_function_nodes(graph_module, "_to_dim_order_copy")
        self.assertEqual(len(copy_nodes), 1)
        mean_nodes = self._find_call_function_nodes(graph_module, "aten.mean.dim")
        self.assertEqual(len(mean_nodes), 1)

        copy_arg = mean_nodes[0].args[0]
        self.assertIsInstance(copy_arg, torch.fx.Node)
        self.assertIn("_to_dim_order_copy", str(copy_arg.target))

    def test_channels_last_linear_inserts_copy_before_expand_copy(self) -> None:
        module = LinearIssueReproModule(out_features=8).eval()
        sample_input = (torch.randn(1, 2, 3, 10).to(memory_format=torch.channels_last),)

        edge = self._build_edge(module, sample_input)
        updated = edge.transform([LegalizePortableDimOrderPass()])
        graph_module = updated.exported_program().graph_module

        copy_nodes = self._find_call_function_nodes(graph_module, "_to_dim_order_copy")
        self.assertEqual(len(copy_nodes), 1)
        expand_nodes = self._find_call_function_nodes(graph_module, "aten.expand_copy")
        self.assertTrue(expand_nodes)
        self.assertTrue(
            any(
                isinstance(node.args[0], torch.fx.Node)
                and "_to_dim_order_copy" in str(node.args[0].target)
                for node in expand_nodes
            ),
            "Expected one expand_copy path to consume a legalized contiguous copy.",
        )

    def test_contiguous_inputs_skip_copy(self) -> None:
        mean_module = (
            MeanIssueReproModule().eval().to(memory_format=torch.channels_last)
        )
        linear_module = LinearIssueReproModule().eval()
        copy_module = CopyIntoBufferModule(torch.contiguous_format)
        channels_last_3d = torch.randn(2, 3, 4, 5, 6).to(
            memory_format=torch.channels_last_3d
        )
        cases = (
            ("mean", mean_module, (torch.randn(1, 2, 3, 4).contiguous(),)),
            ("linear", linear_module, (torch.randn(1, 2, 3, 10).contiguous(),)),
            ("copy", copy_module, (torch.randn(2, 3, 4, 6),)),
            ("broadcast_copy", copy_module, (torch.randn(3, 4, 6),)),
            (
                "channels_last_broadcast_copy",
                copy_module,
                (torch.randn(1, 3, 4, 6).to(memory_format=torch.channels_last),),
            ),
            (
                "single_value_copy",
                CopyIntoBufferModule(torch.channels_last),
                (torch.tensor(7.0),),
            ),
            # Channels last memory without the leading 1.
            (
                "leading_one_copy",
                CopyIntoBufferModule(torch.channels_last, (1, 3, 4, 6)),
                (torch.randn(1, 3, 4, 6).to(memory_format=torch.channels_last)[0],),
            ),
            # With one channel, both layouts are the same memory.
            (
                "single_channel_copy",
                CopyIntoBufferModule(torch.channels_last, (2, 1, 4, 6)),
                (torch.randn(2, 1, 4, 6),),
            ),
            # A layout that exir cannot convert to is left as it is.
            (
                "channels_last_3d_copy",
                CopyIntoReluModule(),
                (channels_last_3d, channels_last_3d.clone()),
            ),
            # No size is known when lowering, so none is known to broadcast.
            (
                "dynamic_single_value_copy",
                CopyIntoReluModule(),
                (
                    torch.randn(2, 3, 4, 6).to(memory_format=torch.channels_last),
                    torch.tensor(7.0),
                ),
                ({i: Dim.DYNAMIC for i in range(4)}, None),
            ),
        )

        for name, module, sample_input, *dynamic_shapes in cases:
            with self.subTest(name=name):
                edge = self._build_edge(module, sample_input, *dynamic_shapes)
                updated = edge.transform([LegalizePortableDimOrderPass()])
                FileCheck().check_not("_to_dim_order_copy").run(
                    updated.exported_program().graph_module.code
                )

    def test_issue_16507_runtime_regressions(self) -> None:
        cases = (
            (
                "baseline",
                MeanIssueReproModule().eval().to(memory_format=torch.channels_last),
                (torch.randn(1, 2, 3, 4).to(memory_format=torch.channels_last),),
            ),
            (
                "no_reshape",
                MeanNoReshapeModule().eval().to(memory_format=torch.channels_last),
                (torch.randn(1, 2, 3, 4).to(memory_format=torch.channels_last),),
            ),
            (
                "keepdim_false",
                MeanKeepdimFalseModule().eval().to(memory_format=torch.channels_last),
                (torch.randn(1, 2, 3, 4).to(memory_format=torch.channels_last),),
            ),
            (
                "single_dim_h",
                MeanSingleDimModule(2).eval().to(memory_format=torch.channels_last),
                (torch.randn(1, 2, 3, 4).to(memory_format=torch.channels_last),),
            ),
            (
                "single_dim_w",
                MeanSingleDimModule(3).eval().to(memory_format=torch.channels_last),
                (torch.randn(1, 2, 3, 4).to(memory_format=torch.channels_last),),
            ),
        )

        for name, module, sample_input in cases:
            with self.subTest(name=name):
                graph_module = self._assert_runtime_matches_eager(module, sample_input)
                FileCheck().check("_to_dim_order_copy").check("aten.mean.out").run(
                    graph_module.code
                )

    def test_issue_11086_runtime_regressions(self) -> None:
        cases = (
            (
                "baseline",
                LinearIssueReproModule().eval(),
                (torch.randn(1, 2, 3, 10).to(memory_format=torch.channels_last),),
            ),
            (
                "larger_spatial",
                LinearIssueReproModule(out_features=6).eval(),
                (torch.randn(2, 5, 7, 10).to(memory_format=torch.channels_last),),
            ),
        )

        for name, module, sample_input in cases:
            with self.subTest(name=name):
                graph_module = self._assert_runtime_matches_eager(module, sample_input)
                FileCheck().check("_to_dim_order_copy").check(
                    "aten.expand_copy.out"
                ).run(graph_module.code)

    def test_issue_11086_c1_control_uses_alternate_path(self) -> None:
        module = LinearIssueReproModule(out_features=8).eval()
        sample_input = (torch.randn(1, 1, 3, 10).to(memory_format=torch.channels_last),)

        edge = self._build_edge(module, sample_input)
        graph_module = edge.exported_program().graph_module
        FileCheck().check_not("aten.expand_copy").run(graph_module.code)

        updated = edge.transform([LegalizePortableDimOrderPass()])
        FileCheck().check_not("_to_dim_order_copy").run(
            updated.exported_program().graph_module.code
        )

        runtime_graph = self._assert_runtime_matches_eager(module, sample_input)
        FileCheck().check_not("aten.expand_copy.out").run(runtime_graph.code)

    def test_copy_source_gets_dim_order_of_destination(self) -> None:
        channels_last_input = torch.randn(2, 3, 4, 6).to(
            memory_format=torch.channels_last
        )
        cases = (
            (
                "contiguous_into_channels_last",
                CopyIntoBufferModule(torch.channels_last),
                (torch.randn(2, 3, 4, 6),),
            ),
            (
                "channels_last_into_contiguous",
                CopyIntoBufferModule(torch.contiguous_format),
                (channels_last_input,),
            ),
            (
                "int_into_channels_last",
                CopyIntoBufferModule(torch.channels_last),
                (torch.arange(144, dtype=torch.int32).reshape(2, 3, 4, 6),),
            ),
            # The pass itself makes the pool output contiguous.
            (
                "avg_pool2d_into_channels_last",
                AvgPoolCopyModule(),
                (channels_last_input,),
            ),
        )

        for name, module, sample_input in cases:
            with self.subTest(name=name):
                graph_module = self._assert_runtime_matches_eager(module, sample_input)
                (copy_node,) = self._find_call_function_nodes(graph_module, "aten.copy")
                source = copy_node.args[1]
                self.assertIn("_to_dim_order_copy", str(source.target))
                self.assertNotIn("expand_copy", str(source.args[0].target))

    def test_copy_source_gets_shape_of_channels_last_destination(self) -> None:
        cases = (
            (
                "broadcast",
                CopyIntoBufferModule(torch.channels_last),
                (torch.randn(3, 4, 6),),
            ),
            # Sizes that match apart from leading ones take another kernel path.
            (
                "leading_one",
                CopyIntoBufferModule(torch.channels_last, (1, 3, 4, 6)),
                (torch.randn(3, 4, 6),),
            ),
            (
                "channels_last_broadcast",
                CopyIntoBufferModule(torch.channels_last),
                (torch.randn(1, 3, 4, 6).to(memory_format=torch.channels_last),),
            ),
        )

        for name, module, sample_input in cases:
            with self.subTest(name=name):
                graph_module = self._assert_runtime_matches_eager(module, sample_input)
                (copy_node,) = self._find_call_function_nodes(graph_module, "aten.copy")
                source = copy_node.args[1]
                self.assertIn("_to_dim_order_copy", str(source.target))
                self.assertIn("expand_copy", str(source.args[0].target))

    def test_copy_with_symbolic_size(self) -> None:
        source = torch.randn(1, 3, 4, 6).to(memory_format=torch.channels_last)
        x = torch.randn(2, 8, 4, 6)
        cases = (
            # Tracing assumes the batch is not 1, but at runtime it can be.
            (
                "dynamic",
                CopyIntoReluModule(),
                ({0: Dim("batch", min=1, max=4)}, None),
                [(torch.randn(batch, 3, 4, 6), source) for batch in (2, 1)],
            ),
            (
                "channels_last_destination",
                CopyIntoReluModule(),
                ({0: Dim("batch", min=1, max=4)}, None),
                [
                    (
                        torch.randn(batch, 3, 4, 6).to(
                            memory_format=torch.channels_last
                        ),
                        source,
                    )
                    for batch in (2, 1)
                ],
            ),
            # Broadcasts when the batch is 1.
            (
                "batch_up_to_destination",
                CopyIntoBufferModule(torch.channels_last),
                ({0: Dim("batch", min=1, max=2)},),
                [
                    (torch.randn(batch, 3, 4, 6).to(memory_format=torch.channels_last),)
                    for batch in (2, 1)
                ],
            ),
            (
                "data_dependent",
                CopyIntoDataSizedBufferModule(),
                None,
                [(torch.tensor(channels), x) for channels in (3, 1, 8)],
            ),
        )

        for name, module, dynamic_shapes, inputs in cases:
            with self.subTest(name=name):
                _, executorch_prog = self._build_executorch_program(
                    module, inputs[0], dynamic_shapes
                )
                executorch_module = _load_for_executorch_from_buffer(
                    executorch_prog.buffer
                )
                for sample_input in inputs:
                    torch.testing.assert_close(
                        executorch_module.run_method("forward", sample_input)[0],
                        module(*sample_input),
                    )

    def test_buffer_write_back_gets_dim_order_of_buffer(self) -> None:
        inputs = (
            torch.randn(2, 3, 4, 6),
            # With one channel, both layouts are the same memory but not the
            # same dim order.
            torch.randn(2, 1, 4, 6),
            # The same dim order, but a broadcast writes the buffer as contiguous.
            torch.randn(1, 3, 4, 6).to(memory_format=torch.channels_last),
        )
        for x in inputs:
            with self.subTest(shape=x.shape, stride=x.stride()):
                shape = (2, *x.shape[1:])
                module = CopyIntoMutableBufferModule(shape)
                _, executorch_prog = self._build_executorch_program(module, (x,))
                executorch_module = _load_for_executorch_from_buffer(
                    executorch_prog.buffer
                )

                executorch_module.run_method("forward", (x,))
                previous = executorch_module.run_method(
                    "forward", (torch.zeros_like(x),)
                )
                torch.testing.assert_close(previous[0], x.expand(shape))

    def test_reinplaced_program_matches_eager(self) -> None:
        channels_last = torch.randn(2, 3, 4, 6).to(memory_format=torch.channels_last)
        cases = (
            ("avg_pool2d", ClonePoolModule(), (channels_last,)),
            ("copy", CloneCopyModule(), (channels_last, torch.randn(2, 3, 4, 6))),
            (
                "broadcast_copy",
                CloneCopyModule(),
                (channels_last, torch.randn(3, 4, 6)),
            ),
            (
                "single_value_copy",
                CloneCopyModule(),
                (channels_last, torch.tensor(7.0)),
            ),
        )
        config = ExecutorchBackendConfig(
            extract_delegate_segments=False,
            # Lets the clone reuse the memory of its input, and makes the copy
            # an in-place copy_.
            reinplace_extra_ops=frozenset(
                {
                    exir_ops.edge.dim_order_ops._clone_dim_order.default,
                    exir_ops.edge.aten.copy.default,
                }
            ),
        )

        for name, module, sample_input in cases:
            with self.subTest(name=name):
                edge = self._build_edge(module, sample_input)
                executorch_module = _load_for_executorch_from_buffer(
                    edge.to_executorch(config).buffer
                )
                torch.testing.assert_close(
                    executorch_module.run_method("forward", sample_input)[0],
                    module(*sample_input),
                )
