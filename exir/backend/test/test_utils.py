# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import unittest
from unittest.mock import patch

import torch
from executorch import exir
from executorch.exir import to_edge
from executorch.exir.backend.backend_api import to_backend
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.test.backend_with_compiler_demo import (
    BackendWithCompilerDemo,
)
from executorch.exir.backend.test.demos.rpc.executor_backend_preprocess import (
    ExecutorBackend,
)
from executorch.exir.backend.test.op_partitioner_demo import AddMulPartitionerDemo
from executorch.exir.backend.utils import (
    _maybe_duplicate_constant_nodes,
    format_delegated_graph,
    get_delegated_payload,
    get_delegates,
    get_non_lowered_nodes,
    is_identical_graph,
    tag_constant_data,
    tag_mutated_buffer,
)

from executorch.exir.dialects._ops import bind_pattern_to_op, ops as exir_ops
from executorch.exir.lowered_backend_module import get_lowered_backend_modules
from executorch.extension.pybindings.portable_lib import (  # @manual
    _get_registered_backend_names,
    _load_for_executorch_from_buffer,
)
from torch.export import export, ExportedProgram
from torch.fx import symbolic_trace
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher
from torch.library import Library

T_QuantPerTensor = exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
T_DQuantPerTensor = exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default


class TestUtils(unittest.TestCase):
    def test_identical_graph_with_unused_args(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                # y is not used arg
                return x

        m = MyModule()
        graph_module: torch.fx.GraphModule = symbolic_trace(m)
        is_matched = is_identical_graph(graph_module, graph_module)
        self.assertTrue(is_matched)

    def test_identical_graph_with_used_args(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x, y

        m = MyModule()
        graph_module: torch.fx.GraphModule = symbolic_trace(m)
        is_matched = is_identical_graph(graph_module, graph_module)
        self.assertTrue(is_matched)

    def test_identical_graph_for_linear(self):
        graph_module: torch.fx.GraphModule = symbolic_trace(torch.nn.Linear(10, 10))
        is_matched = is_identical_graph(graph_module, graph_module)
        self.assertTrue(is_matched)

    def test_identical_graph_for_composite_module(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 4))
                self.linear = torch.nn.Linear(4, 5)

            def forward(self, x):
                return self.linear(x + self.param).clamp(min=0.0, max=1.0)

        graph_module: torch.fx.GraphModule = symbolic_trace(MyModule())
        is_matched = is_identical_graph(graph_module, graph_module)
        self.assertTrue(is_matched)

    def test_not_identical_graph_for_args(self):
        class MyModule1(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                # y is not used arg
                return x + 1

        class MyModule2(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x + 1, y + 2

        graph_module_1: torch.fx.GraphModule = (
            to_edge(
                export(MyModule1(), (torch.rand(3, 4), torch.rand(3, 4)), strict=True)
            )
            .exported_program()
            .graph_module
        )
        graph_module_2: torch.fx.GraphModule = (
            to_edge(
                export(MyModule2(), (torch.rand(3, 4), torch.rand(3, 4)), strict=True)
            )
            .exported_program()
            .graph_module
        )
        is_matched = is_identical_graph(graph_module_1, graph_module_2)
        self.assertFalse(is_matched)

    def test_match_attrs(self):
        class LargeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weght = torch.nn.Parameter(torch.ones(3, 3))
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                a = x + self.weght
                b = self.linear(x)
                return a, b

        inputs = (torch.ones(3, 3),)

        large_model = (
            to_edge(
                export(LargeModel(), inputs, strict=True),
                compile_config=exir.EdgeCompileConfig(_check_ir_validity=False),
            )
            .exported_program()
            .graph_module
        )

        pattern = (
            to_edge(
                export(torch.nn.Linear(3, 3), inputs, strict=True),
                compile_config=exir.EdgeCompileConfig(_check_ir_validity=False),
            )
            .exported_program()
            .graph_module.graph
        )

        subgraph_matcher = SubgraphMatcher(pattern)
        match_result = subgraph_matcher.match(large_model.graph)

        # Should find exact one match
        self.assertEqual(len(match_result), 1)

    def test_invalid_partitioner_without_partitioner(self):
        """
        Tests replacing literals with placeholders in the case there are
        `getitem` calls which do not have a schema.
        """

        class InvalidPartitioner(Partitioner):
            """
            Partitions all add/mul nodes regardless of order
            """

            def __init__(self) -> None:
                # A valid partitioner should have partition_tags
                self.test = "a"

            def partition(
                self, edge_exported_program: ExportedProgram
            ) -> PartitionResult:
                return PartitionResult(
                    tagged_exported_program=edge_exported_program, partition_tags=None
                )

        exported_program = to_edge(
            export(torch.nn.Linear(3, 3), (torch.randn(3, 3),), strict=True)
        )

        error_msg = r"needs a `partition_tags` field containing a mapping of tags to delegate spec"
        with self.assertRaisesRegex(
            AssertionError,
            error_msg,
        ):
            _ = to_backend(exported_program.exported_program(), InvalidPartitioner())

    test_lib = Library("test_lib", "DEF")

    @staticmethod
    @bind_pattern_to_op(
        test_lib, "test_q_linear(Tensor x, Tensor weight, Tensor bias) -> Tensor"
    )
    def q_linear(x, weight, bias):
        return x

    def test_get_non_lowered_nodes(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, x, b):
                y = torch.mm(a, x)
                z = y + b
                a = z - a
                y = torch.mm(a, x)
                z = y + b
                return z

        m = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2))
        edge = to_edge(export(m, inputs, strict=True))
        edge = edge.to_backend(AddMulPartitionerDemo())
        number_of_cpu_nodes = get_non_lowered_nodes(edge.exported_program().graph)
        # Only sub is not not lowerable
        self.assertEqual(len(number_of_cpu_nodes), 1)

    def test_get_delegates(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, x, b):
                y = torch.mm(a, x)
                z = y + b
                a = z - a
                y = torch.mm(a, x)
                z = y + b
                return z

        m = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2))
        edge = to_edge(export(m, inputs, strict=True))
        edge = edge.to_backend(AddMulPartitionerDemo())
        number_of_delegates = get_delegates(edge.exported_program().graph)
        # there will be 2 delegates: (mm + add) -> sub -> (mm + add)
        self.assertEqual(len(number_of_delegates), 2)

    def test_print_delegted_graph(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, x, b):
                y = torch.mm(a, x)
                z = y + b
                a = z - a
                y = torch.mm(a, x)
                z = y + b
                return z

        m = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2))

        edge = to_edge(export(m, inputs, strict=True)).to_backend(
            AddMulPartitionerDemo()
        )

        graph_str = format_delegated_graph(edge.exported_program().graph_module)
        self.assertIn(
            "BackendWithCompilerDemo",
            graph_str,
            "Expect to find the backend id in the graph format string",
        )
        self.assertIn(
            "executorch.exir.dialects.edge._ops.aten.mm.default",
            graph_str,
            "Expect to see the aten.mm in the delegated graph",
        )

    def test_get_delegated_payload_with_delegates(self):
        """Test get_delegated_payload returns correct payload for delegated modules."""

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, x, b):
                y = torch.mm(a, x)
                z = y + b
                a = z - a
                y = torch.mm(a, x)
                z = y + b
                return z

        m = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2))

        edge = to_edge(export(m, inputs, strict=True)).to_backend(
            AddMulPartitionerDemo()
        )

        payloads = get_delegated_payload(edge.exported_program().graph_module)

        # Should have 2 delegates: (mm + add) -> sub -> (mm + add)
        self.assertEqual(len(payloads), 2)

        # Verify payload structure for each delegate
        for name, (backend_id, compile_specs, processed_bytes) in payloads.items():
            # Check delegate name format
            self.assertTrue(
                name.startswith("lowered_module_"),
                f"Delegate name should start with 'lowered_module_', got {name}",
            )

            # Check backend_id
            self.assertEqual(
                backend_id,
                "BackendWithCompilerDemo",
                f"Expected backend_id 'BackendWithCompilerDemo', got {backend_id}",
            )

            # Check compile_specs is a list
            self.assertIsInstance(
                compile_specs,
                list,
                f"compile_specs should be a list, got {type(compile_specs)}",
            )

            # Check processed_bytes is bytes
            self.assertIsInstance(
                processed_bytes,
                bytes,
                f"processed_bytes should be bytes, got {type(processed_bytes)}",
            )

            # Verify processed_bytes is not empty (backend should produce some output)
            self.assertGreater(
                len(processed_bytes),
                0,
                "processed_bytes should not be empty",
            )

    def test_get_delegated_payload_without_delegates(self):
        """Test get_delegated_payload returns empty dict when no delegates present."""

        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x + 1

        m = SimpleModel()
        inputs = (torch.randn(2, 2),)

        # Create edge program without delegation
        edge = to_edge(export(m, inputs, strict=True))

        payloads = get_delegated_payload(edge.exported_program().graph_module)

        # Should have no delegates
        self.assertEqual(
            len(payloads),
            0,
            "Expected empty payload dict when no delegates present",
        )

    def test_get_delegated_payload_keys_match_delegates(self):
        """Test that get_delegated_payload keys match get_delegates node names."""

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, x, b):
                y = torch.mm(a, x)
                z = y + b
                a = z - a
                y = torch.mm(a, x)
                z = y + b
                return z

        m = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2))

        edge = to_edge(export(m, inputs, strict=True)).to_backend(
            AddMulPartitionerDemo()
        )

        graph_module = edge.exported_program().graph_module

        # Get delegates using existing utility
        delegate_nodes = get_delegates(graph_module.graph)
        delegate_names = {node.name for node in delegate_nodes}

        # Get payloads using new utility
        payloads = get_delegated_payload(graph_module)
        payload_names = set(payloads.keys())

        # Names should match
        self.assertEqual(
            delegate_names,
            payload_names,
            f"Delegate names mismatch: {delegate_names} vs {payload_names}",
        )

    def test_to_backend_keeps_a_mutated_buffer_out_of_a_partial_delegate(  # noqa: C901
        self,
    ):
        class UpdateState(torch.nn.Module):
            def __init__(self, read: bool, returned: bool, copy_input: bool):
                super().__init__()
                self.read = read
                self.returned = returned
                self.copy_input = copy_input
                self.register_buffer("state", torch.zeros(2))

            def forward(self, x):
                if self.read:
                    out = torch.sin(self.state) + (x + self.state)
                else:
                    out = torch.sin(x)
                self.state.copy_(x if self.copy_input else x + x)
                if self.returned:
                    return out, self.state
                # In read mode, the last add reads the new value.
                return out + self.state if self.read else out

        class TagNodes(Partitioner):
            def __init__(self, tags):
                super().__init__()
                self.tags = tags

            def partition(self, program: ExportedProgram) -> PartitionResult:
                for node in program.graph.nodes:
                    if node.name in self.tags:
                        node.meta["delegation_tag"] = self.tags[node.name]
                tag_mutated_buffer(program)
                spec = DelegationSpec(
                    BackendWithCompilerDemo.__name__,
                    [CompileSpec("max_value", bytes([4]))],
                )
                return PartitionResult(
                    tagged_exported_program=program,
                    partition_tags={tag: spec for tag in self.tags.values()},
                )

        # The delegate owns the buffer only when it reads and writes all of
        # it. Otherwise the owning program keeps the one buffer: a copy in the
        # delegate would never see the updates.
        sin, add, writer = "aten_sin_default", "aten_add_tensor", "aten_add_tensor_2"
        # Also the add that makes "out", and the add that reads the new value.
        whole = {sin, add, "aten_add_tensor_1", writer, "aten_add_tensor_3"}
        has_demo_backend = (
            BackendWithCompilerDemo.__name__ in _get_registered_backend_names()
        )
        for read, returned, copy_input, tags, owned_by_program in (
            (True, False, False, dict.fromkeys(whole, "tag0"), False),
            (True, False, False, {sin: "tag0", add: "tag0", writer: "tag0"}, True),
            (True, False, False, {sin: "tag0", add: "tag0"}, True),
            (True, False, False, {sin: "tag0"}, True),
            (True, False, False, {sin: "tag0", add: "tag1"}, True),
            # Lowering cannot give a delegate a buffer whose new value is returned.
            (True, True, False, {sin: "tag0", add: "tag0", writer: "tag0"}, True),
            # Without readers, "add" is the writer. Some partitioners tag such a
            # buffer themselves.
            (False, False, False, {add: "tag0"}, True),
            (False, False, False, {add: "tag0", "b_state": "tag0"}, False),
            (False, True, False, {add: "tag0", "b_state": "tag0"}, True),
            # The new value is the graph input, which no partition holds.
            (True, False, True, dict.fromkeys(whole, "tag0"), True),
        ):
            with self.subTest(
                read=read, returned=returned, copy_input=copy_input, tags=tags
            ):
                x = torch.full((2,), 0.25)
                lowered = to_edge(
                    export(UpdateState(read, returned, copy_input), (x,), strict=True)
                ).to_backend(TagNodes(tags))
                self.assertEqual(
                    [
                        spec.arg.name
                        for spec in lowered.exported_program().graph_signature.input_specs
                        if spec.target == "state"
                    ],
                    ["b_state"] if owned_by_program else [],
                )
                if copy_input:
                    # Every op is delegated, so only the new value keeps the
                    # buffer out.
                    self.assertEqual(
                        get_non_lowered_nodes(lowered.exported_program().graph), []
                    )
                if tags == {sin: "tag0"} and has_demo_backend:
                    eager = UpdateState(read, returned, copy_input)
                    module = _load_for_executorch_from_buffer(
                        lowered.to_executorch().buffer
                    )
                    # The demo backend approximates sin with two Taylor terms.
                    for _ in range(2):
                        torch.testing.assert_close(
                            module.forward((x,))[0], eager(x), atol=1e-3, rtol=0
                        )
        if not has_demo_backend:
            self.skipTest("the runtime has no demo backend to run the lowered program")

    def test_to_backend_copies_a_constant_read_by_ops_in_two_places(self):
        class CatConst(torch.nn.Module):
            def __init__(self, returned: bool):
                super().__init__()
                self.returned = returned
                self.register_buffer("const", torch.ones(2))

            def forward(self, x):
                out = torch.cat([x, self.const]), torch.cat([self.const, x])
                return (*out, self.const) if self.returned else out

        class TagEachCat(Partitioner):
            def partition(self, program: ExportedProgram) -> PartitionResult:
                tags = {}
                for node in program.graph.nodes:
                    if node.target == exir_ops.edge.aten.cat.default:
                        tag = f"tag{len(tags)}"
                        node.meta["delegation_tag"] = tag
                        tags[tag] = DelegationSpec(ExecutorBackend.__name__, [])
                tag_constant_data(program)
                return PartitionResult(
                    tagged_exported_program=program, partition_tags=tags
                )

        # cat reads the constant from a list. Each delegate gets its own copy,
        # and the graph output keeps the original, which the output specs name.
        for returned, constants in (
            (False, [["b_const"], ["b_const_copy_0"]]),
            (True, [["b_const_copy_0"], ["b_const_copy_1"]]),
        ):
            with self.subTest(returned=returned):
                lowered = to_edge(
                    export(CatConst(returned), (torch.ones(2),), strict=True)
                ).to_backend(TagEachCat())
                program = lowered.exported_program()
                self.assertEqual(
                    [
                        [
                            spec.arg.name
                            for spec in module.original_module.graph_signature.input_specs
                            if spec.target == "const"
                        ]
                        for _, module in program.graph_module.named_children()
                    ],
                    constants,
                )
                if returned:
                    self.assertEqual(
                        program.graph.output_node().args[0][-1].name, "b_const"
                    )
                lowered.to_executorch()

    def test_to_backend_gives_each_program_one_copy_of_a_shared_constant(self):
        class UseConst(torch.nn.Module):
            def __init__(self, cpu_first, kind="constant"):
                super().__init__()
                const = torch.ones(2)
                if kind == "buffer":
                    self.register_buffer("const", const)
                else:
                    # A plain tensor attribute is lifted as a constant tensor.
                    self.const = (
                        torch.nn.Parameter(const) if kind == "parameter" else const
                    )
                self.cpu_first = cpu_first

            def forward(self, x):
                if self.cpu_first:
                    s, c = torch.sin(self.const), torch.cos(self.const)
                y = (x + self.const) * self.const
                z = (x - self.const) / self.const
                if not self.cpu_first:
                    s, c = torch.sin(self.const), torch.cos(self.const)
                return y, z, s, c

        class TagNodes(Partitioner):
            def __init__(self, tags):
                super().__init__()
                self.tags = tags

            def partition(self, program: ExportedProgram) -> PartitionResult:
                for node in program.graph.nodes:
                    if node.name in self.tags:
                        node.meta["delegation_tag"] = self.tags[node.name]
                tag_constant_data(program)
                return PartitionResult(
                    tagged_exported_program=program,
                    partition_tags=dict.fromkeys(
                        self.tags.values(),
                        DelegationSpec(ExecutorBackend.__name__, []),
                    ),
                )

        def counts(program):
            programs = [program] + [
                module.original_module
                for module in get_lowered_backend_modules(program.graph_module)
            ]
            return [
                [spec.target for spec in p.graph_signature.input_specs].count("const")
                for p in programs
            ]

        # Each delegate reads the constant twice, and sin and cos read it
        # outside of them. Each program takes it once, whichever reads first.
        for kind, cpu_first in itertools.product(
            ("constant", "buffer", "parameter"), (False, True)
        ):
            with self.subTest(kind=kind, cpu_first=cpu_first):
                lowered = to_edge(
                    export(UseConst(cpu_first, kind), (torch.ones(2),), strict=True)
                )
                lowered = lowered.to_backend(
                    TagNodes({"aten_add_tensor": "tag0", "aten_mul_tensor": "tag0"})
                )
                self.assertEqual(counts(lowered.exported_program()), [1, 1])
                lowered.exported_program().module()
                lowered = lowered.to_backend(
                    TagNodes({"aten_sub_tensor": "tag1", "aten_div_tensor": "tag1"})
                )
                self.assertEqual(counts(lowered.exported_program()), [1, 1, 1])
                lowered.to_executorch()

        # One partitioner takes both delegates, so one call copies it twice.
        with self.subTest(one_partitioner=True):
            lowered = to_edge(export(UseConst(False), (torch.ones(2),), strict=True))
            lowered = lowered.to_backend(
                TagNodes(
                    {
                        "aten_add_tensor": "tag0",
                        "aten_mul_tensor": "tag0",
                        "aten_sub_tensor": "tag1",
                        "aten_div_tensor": "tag1",
                    }
                )
            )
            self.assertEqual(counts(lowered.exported_program()), [1, 1, 1])
            lowered.to_executorch()

    def test_maybe_duplicate_constant_nodes_refuses_one_tag_string(self):
        program = to_edge(
            export(torch.nn.Linear(2, 2), (torch.ones(2),), strict=True)
        ).exported_program()
        # A str is an iterable of tags too, one per character.
        with self.assertRaises(TypeError):
            _maybe_duplicate_constant_nodes(program, "tag0")

    def test_constants_shared_by_many_delegates_are_copied_in_one_pass(self):
        class ShareConsts(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("a", torch.ones(2))
                self.register_buffer("b", torch.ones(2))

            def forward(self, x):
                return x + self.a, x * self.a, self.a - self.b, x / self.b

        program = to_edge(
            export(ShareConsts(), (torch.ones(2),), strict=True)
        ).exported_program()
        ops = [node for node in program.graph.nodes if node.op == "call_function"]
        for i, node in enumerate(ops):
            node.meta["delegation_tag"] = f"tag{i}"
        tag_constant_data(program)

        # Recompiling and validating once per shared constant is quadratic in
        # the graph size, which a large model with many delegates cannot afford.
        with patch.object(
            ExportedProgram,
            "_validate",
            autospec=True,
            side_effect=ExportedProgram._validate,
        ) as validate, patch.object(
            torch.fx.GraphModule,
            "recompile",
            autospec=True,
            side_effect=torch.fx.GraphModule.recompile,
        ) as recompile:
            _maybe_duplicate_constant_nodes(
                program, [node.meta["delegation_tag"] for node in ops]
            )
        self.assertEqual(validate.call_count, 1)
        self.assertEqual(recompile.call_count, 1)

        # Copying b_a rewires sub, which must not change who keeps b_b: the
        # first reader in graph order does.
        self.assertEqual(
            {
                node.meta["delegation_tag"]: [arg.name for arg in node.args]
                for node in ops
            },
            {
                "tag0": ["x", "b_a"],
                "tag1": ["x", "b_a_copy_0"],
                "tag2": ["b_a_copy_1", "b_b"],
                "tag3": ["x", "b_b_copy_0"],
            },
        )
        for node in program.graph.nodes:
            if node.op == "placeholder" and node.name != "x":
                self.assertEqual(
                    {user.meta["delegation_tag"] for user in node.users},
                    {node.meta["delegation_tag"]},
                )
        self.assertEqual(
            {
                spec.arg.name: spec.target
                for spec in program.graph_signature.input_specs
            },
            {
                "b_a_copy_0": "a",
                "b_a_copy_1": "a",
                "b_a": "a",
                "b_b_copy_0": "b",
                "b_b": "b",
                "x": None,
            },
        )
