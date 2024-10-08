# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pye-strict

import copy
import unittest
from typing import Any, Dict

import torch
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig
from executorch.exir.backend.test.op_partitioner_demo import (
    AddMulPartitionerDemo,
    NonDecompTestPartitioner,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.error import ExportError
from executorch.exir.lowered_backend_module import get_lowered_submodules
from executorch.exir.pass_base import ExportPass
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.program._program import (
    EdgeProgramManager,
    ExecutorchProgramManager,
    to_edge,
    to_edge_transform_and_lower,
    to_edge_with_preserved_ops,
)
from executorch.exir.tracer import _default_decomposition_table
from executorch.exir.verification.verifier import EXIREdgeDialectVerifier

from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
)
from torch.export import Dim, export, ExportedProgram
from torch.export._trace import _export

from torch.library import impl, Library
from torch.nn import functional as F


class TestLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 16, bias=True)

    def forward(self, x):
        return self.linear(x)

    @classmethod
    def _get_random_inputs(cls):
        x = torch.rand(8, 32)
        return (x,)


class TestSDPA(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value):
        return torch.ops.aten.scaled_dot_product_attention.default(query, key, value)

    @classmethod
    def _get_random_inputs(cls):
        d_k = 64
        batch = 16
        seq_len = 10
        query = torch.rand(batch, seq_len, d_k)
        key = torch.rand(batch, seq_len, d_k)
        value = torch.rand(batch, seq_len, d_k)
        return (query, key, value)


class TestLinearSDPACombined(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 16, bias=True)

    def forward(self, x, query, key, value):
        x = self.linear(x)
        return (
            x,
            torch.ops.aten.scaled_dot_product_attention.default(query, key, value),
        )

    @classmethod
    def _get_random_inputs(cls):
        return TestLinear._get_random_inputs() + TestSDPA._get_random_inputs()


class TestUpsample(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return x

    @classmethod
    def _get_random_inputs(cls):
        x = torch.randn(1, 1, 8, 8)
        return (x,)


class TestLSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=8, hidden_size=16, batch_first=True)

    def forward(self, x):
        return self.lstm(x)

    @classmethod
    def _get_random_inputs(cls):
        return (torch.rand(1, 10, 8),)


class WrapperModule(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


lib = Library("exir_program_test_op", "DEF")

# Fake a operator for testing.
# This operator takes two tensors as input and returns the first one.
lib.define("foo(Tensor self, Tensor other) -> Tensor")


@impl(lib, "foo", "CPU")
def foo(a, b):
    # do nothing and return a.
    return a + b


@impl(lib, "foo", "Meta")
def foo_meta(a, b):
    # do nothing and return a.
    return torch.empty_like(a)


def get_exported_programs() -> Dict[str, ExportedProgram]:
    class Forward(torch.nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            z = torch.mul(x, y)
            return torch.add(z, x)

    forward = Forward()

    class Foo(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.add(x, torch.ones(1))

    foo = Foo()

    programs = {}
    programs["forward"] = export(
        forward,
        args=(
            torch.ones(1),
            torch.zeros(1),
        ),
    ).run_decompositions()
    programs["foo"] = export(
        foo,
        (torch.ones(1),),
    ).run_decompositions()
    return programs


def get_config_methods() -> Dict[str, Any]:
    def bam():
        return 3

    def bar():
        return "bar"

    return {"bam": bam(), "bar": bar()}


class AddToMulPassEdge(ExportPass):
    def call_operator(self, op, args, kwargs, meta):
        if op == exir_ops.edge.aten.add.Tensor:
            return super().call_operator(
                exir_ops.edge.aten.mul.Tensor, args, kwargs, meta
            )
        else:
            return super().call_operator(op, args, kwargs, meta)


class TestProgramManagers(unittest.TestCase):
    def test_edge_manager_basic_api(self):
        edge_manager: EdgeProgramManager = to_edge(
            get_exported_programs(), get_config_methods()
        )

        # test basic apis
        self.assertEqual(edge_manager.methods, {"forward", "foo"})
        self.assertEqual(edge_manager.config_methods, {"bam", "bar"})

        # test dialect is correct
        try:
            EXIREdgeDialectVerifier()(
                edge_manager.exported_program("forward").graph_module
            )
            EXIREdgeDialectVerifier()(edge_manager.exported_program("foo").graph_module)
        except ExportError as e:
            self.assertTrue(False, msg="Graph not in edge dialect : " + e.msg)

    def test_executorch_manager_basic_api(self):
        executorch_manager: ExecutorchProgramManager = to_edge(
            get_exported_programs(), get_config_methods()
        ).to_executorch()

        # test basic apis
        self.assertEqual(executorch_manager.methods, {"forward", "foo"})
        self.assertEqual(executorch_manager.config_methods, {"bam", "bar"})

        # test that the emitted output is correct
        self.assertEqual(
            len(executorch_manager._emitter_output.program.execution_plan), 4
        )

        # test that the buffer is correct
        executorch_module = _load_for_executorch_from_buffer(executorch_manager.buffer)
        self.assertEqual(
            executorch_module.run_method("forward", (torch.ones(1), torch.zeros(1)))[0],
            torch.ones(1),
        )
        self.assertEqual(
            executorch_module.run_method("foo", (torch.ones(1),))[0],
            torch.ones(1) + torch.ones(1),
        )
        self.assertEqual(
            executorch_module.run_method("bar", ())[0],
            "bar",
        )
        self.assertEqual(
            executorch_module.run_method("bam", ())[0],
            3,
        )

    def test_executorch_manager_multi_config(self):
        def get_executorch_memory_planning_passes() -> Dict[str, MemoryPlanningPass]:
            return {
                "forward": MemoryPlanningPass(
                    alloc_graph_input=True,
                    alloc_graph_output=False,
                ),
                "foo": MemoryPlanningPass(
                    alloc_graph_input=False,
                    alloc_graph_output=True,
                ),
            }

        executorch_manager: ExecutorchProgramManager = to_edge(
            get_exported_programs(), get_config_methods()
        ).to_executorch(
            ExecutorchBackendConfig(
                memory_planning_pass=get_executorch_memory_planning_passes()
            )
        )

        method = executorch_manager._emitter_output.program.execution_plan[0]
        if method.name == "forward":
            for input_val in method.inputs:
                evalue = method.values[input_val]
                self.assertEqual(evalue.val.allocation_info, None)
            for output_val in method.outputs:
                evalue = method.values[output_val]
                self.assertNotEqual(evalue.val.allocation_info, None)
        else:
            for input_val in method.inputs:
                evalue = method.values[input_val]
                self.assertEqual(evalue.val.allocation_info, None)
            for output_val in method.outputs:
                evalue = method.values[output_val]
                self.assertNotEqual(evalue.val.allocation_info, None)

    def test_no_getattr(self):
        class Mul(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * 3.14

        mul = Mul()
        ep = to_edge(torch.export.export(mul, (torch.ones(1),))).exported_program()
        for node in ep.graph.nodes:
            self.assertNotEqual(node.op, "get_attr")
        self.assertEqual(
            len([node for node in ep.graph.nodes if node.op == "placeholder"]), 2
        )

    def test_constraint_present_after_dce(self):
        import executorch.exir as exir

        class M(torch.nn.Module):
            def forward(self, x, y):
                z = y.item()
                torch._check(z > 0)
                torch._check(z < 4)
                return x[z : z + y.shape[0]]

        ep = torch.export.export(M(), (torch.randn(10), torch.tensor([3])))

        edge_manager = to_edge(
            ep, compile_config=exir.EdgeCompileConfig(_check_ir_validity=False)
        )
        edge_manager.to_executorch()

    def test_edge_manager_transform(self):
        edge_manager: EdgeProgramManager = to_edge(
            get_exported_programs(), get_config_methods()
        )

        original_res = edge_manager.exported_program("forward").module()(
            torch.ones(1), torch.ones(1)
        )

        # perform transformation
        transformed_edge = edge_manager.transform(
            [
                AddToMulPassEdge(),
            ]
        )

        # still have all our methods
        self.assertEqual(len(transformed_edge.methods), 2)
        self.assertEqual(len(transformed_edge.config_methods), 2)

        # transformation was applied
        self.assertEqual(
            transformed_edge.exported_program("forward").module()(
                torch.ones(1), torch.ones(1)
            ),
            torch.ones(1),  # x * y * x
        )

        # original unchanged
        self.assertEqual(
            edge_manager.exported_program("forward").module()(
                torch.ones(1), torch.ones(1)
            ),
            original_res,  # x * y + x
        )

    def test_issue_3659(self):

        class Mul(torch.nn.Module):
            def __init__(self):
                super(Mul, self).__init__()

            def forward(self, x: torch.Tensor, y: torch.Tensor):
                return torch.matmul(x, y)

            def get_eager_model(self) -> torch.nn.Module:
                return self

            def get_example_inputs(self):
                return (torch.randn(1, 3, 10), torch.randn(1, 10, 3))

            def get_dynamic_shapes(self):
                dim1_x = Dim("Dot_dim1_x", min=2, max=100)
                dim2_x = Dim("Dot_dim2_x", min=2, max=100)
                return {"x": {1: dim1_x, 2: dim2_x}, "y": {1: dim2_x, 2: dim1_x}}

        model = Mul()
        ep = torch.export.export(
            model, model.get_example_inputs(), dynamic_shapes=model.get_dynamic_shapes()
        )

        to_edge(
            ep,
            compile_config=EdgeCompileConfig(
                _check_ir_validity=True,
            ),
        )

    def test_transform_dict_api(self):
        edge_manager = to_edge(get_exported_programs(), get_config_methods())

        transformed_edge = edge_manager.transform(
            {
                "forward": [
                    AddToMulPassEdge(),
                ]
            }
        )

        self.assertEqual(
            transformed_edge.exported_program("forward").module()(
                torch.ones(1), torch.ones(1)
            ),
            torch.ones(1),  # x * y * x
        )

        self.assertEqual(
            transformed_edge.exported_program("foo").module()(
                torch.ones(1),
            ),
            torch.ones(1) + 1,  # x + 1
        )

    def test_edge_to_backend_replaces_subgraph(self):
        edge_manager: EdgeProgramManager = to_edge(
            get_exported_programs(), get_config_methods()
        )
        delegate_manager: EdgeProgramManager = edge_manager.to_backend(
            AddMulPartitionerDemo()
        )

        forward_program = delegate_manager.exported_program("forward")
        self.assertEqual(
            forward_program.module()(torch.ones(1), torch.ones(1)),
            torch.ones(1) + 1,  # x * y + x
        )

        add_nodes = [
            node
            for node in forward_program.graph_module.graph.nodes
            if node.op == "call_function"
            and node.target == exir_ops.edge.aten.add.Tensor
        ]
        self.assertEqual(len(add_nodes), 0)

        foo_program = delegate_manager.exported_program("foo")
        add_nodes = [
            node
            for node in foo_program.graph_module.graph.nodes
            if node.op == "call_function"
            and node.target == exir_ops.edge.aten.add.Tensor
        ]
        self.assertEqual(len(add_nodes), 0)

        lowered_submods = get_lowered_submodules(foo_program.graph_module)
        self.assertEqual(len(lowered_submods), 1)

        # original unchanged
        lowered_submods = get_lowered_submodules(
            edge_manager.exported_program("forward").graph_module
        )
        self.assertEqual(len(lowered_submods), 0)

        # two delegate blobs for forward and foo
        self.assertEqual(
            len(
                delegate_manager.to_executorch(ExecutorchBackendConfig())
                ._emitter_output.program.execution_plan[0]
                .delegates
            ),
            1,
        )
        self.assertEqual(
            len(
                delegate_manager.to_executorch(ExecutorchBackendConfig())
                ._emitter_output.program.execution_plan[1]
                .delegates
            ),
            1,
        )

    def test_edge_to_backend_selective(self):
        edge_manager: EdgeProgramManager = to_edge(
            get_exported_programs(), get_config_methods()
        )
        delegate_manager: EdgeProgramManager = edge_manager.to_backend(
            {"forward": AddMulPartitionerDemo()}
        )

        forward_program = delegate_manager.exported_program("forward")
        self.assertEqual(
            forward_program.module()(torch.ones(1), torch.ones(1)),
            torch.ones(1) + 1,  # x * y + x
        )

        add_nodes = [
            node
            for node in forward_program.graph_module.graph.nodes
            if node.op == "call_function"
            and node.target == exir_ops.edge.aten.add.Tensor
        ]
        self.assertEqual(len(add_nodes), 0)

        # foo unchanged
        lowered_submods = get_lowered_submodules(
            delegate_manager.exported_program("foo").graph_module
        )
        self.assertEqual(len(lowered_submods), 0)

        # original unchanged
        lowered_submods = get_lowered_submodules(
            edge_manager.exported_program("forward").graph_module
        )
        self.assertEqual(len(lowered_submods), 0)

        # one delegate blob for forward
        self.assertEqual(
            len(
                delegate_manager.to_executorch(
                    ExecutorchBackendConfig(
                        extract_delegate_segments=False,
                    )
                )
                ._emitter_output.program.execution_plan[0]  # foo
                .delegates
            ),
            0,
        )
        self.assertEqual(
            len(
                delegate_manager.to_executorch(
                    ExecutorchBackendConfig(
                        extract_delegate_segments=False,
                    )
                )
                ._emitter_output.program.execution_plan[1]  # forward
                .delegates
            ),
            1,
        )

    def test_edge_manager_dialect(self):
        edge_manager: EdgeProgramManager = to_edge(
            get_exported_programs(), get_config_methods()
        )
        self.assertTrue(edge_manager.exported_program().dialect == "EDGE")

    def _test_edge_dialect_verifier(
        self, callable, validate_ir=True, exception_list=None
    ):
        from executorch.exir import EdgeCompileConfig

        edge_compile_config = EdgeCompileConfig(
            _check_ir_validity=validate_ir,
            _core_aten_ops_exception_list=exception_list,
        )
        # pre-autograd export. eventually this will become torch.export
        one = torch.ones(1, dtype=torch.float)
        two = torch.ones(1, dtype=torch.int32)
        inputs = (
            one,
            two,
        )
        if not isinstance(callable, torch.nn.Module):
            callable = WrapperModule(callable)

        exported_foo = export(callable, inputs)
        _ = to_edge(exported_foo, compile_config=edge_compile_config)

    def test_edge_dialect_custom_op(self):
        # We shouldn't error out if there's a custom op in the graph.
        def _use_foo_add(a: torch.Tensor, b: torch.Tensor):
            return torch.ops.exir_program_test_op.foo(a, b)

        from torch._export.verifier import SpecViolationError

        try:
            # This should not raise error
            self._test_edge_dialect_verifier(_use_foo_add)
            self._test_edge_dialect_verifier(_use_foo_add, False)
        except SpecViolationError:
            self.fail("Should not error out on custom op")

    def get_num_nondecomposed_ops(self, ep, partitioner):
        # count the number of aten ops that the partitioner can delegate
        # we do this by running run_decompositions() with the preserved ops given
        # to us by the partitioner. Then we count the number of preserved aten ops
        # which pass the filter_ops fn given by the partitioner
        reference_ep = copy.deepcopy(ep)
        aten_ops_not_decomposed, filter_ops = partitioner.ops_to_not_decompose(ep)
        reference_decomp_ep = reference_ep.run_decompositions(
            decomp_table=_default_decomposition_table(),
            _preserve_ops=tuple(aten_ops_not_decomposed),
        )
        num_non_decomposed_aten_ops = 0
        for node in reference_decomp_ep.graph.nodes:
            if (
                node.op == "call_function"
                and node.target in aten_ops_not_decomposed
                and (filter_ops(node) if filter_ops else True)
            ):
                num_non_decomposed_aten_ops += 1
        return num_non_decomposed_aten_ops

    def _test_model_with_non_decomp_partitioner(self, model: torch.nn.Module):
        # This is the pre-dispatch export that we will be switching to primarily
        # in the near future. The input to to_edge_transform_and_lower needs to
        # be a graph generated by this pre dispatch export.
        ep = _export(model, model._get_random_inputs(), pre_dispatch=True)
        non_decomp_partitioner = NonDecompTestPartitioner()

        num_non_decomposed_aten_ops = self.get_num_nondecomposed_ops(
            ep, non_decomp_partitioner
        )

        # run to_edge_trasnform_and_lower
        edge = to_edge_transform_and_lower(
            ep,
            compile_config=EdgeCompileConfig(),
            partitioner=[NonDecompTestPartitioner()],
        )
        # Check that non_decomposed_edge_ops are all consumed by the delegate
        non_decomposed_edge_ops = (
            non_decomp_partitioner.supported_non_decomposed_edge_ops
        )
        for node in edge.exported_program().graph.nodes:
            if node.op == "call_function":
                self.assertTrue(node.target not in non_decomposed_edge_ops)

        # check that the number of call_delegate_nodes is equal to the number of
        # non_decomposed_aten_ops we found above
        num_call_delegates = 0
        for node in edge.exported_program().graph_module.graph.nodes:
            # There should only be a single call_function node in the graph
            # and that should be a call_delegate node.
            if (
                node.op == "call_function"
                and node.target == torch.ops.higher_order.executorch_call_delegate
            ):
                num_call_delegates += 1

        self.assertEqual(num_call_delegates, num_non_decomposed_aten_ops)

    def test_to_edge_transform_and_lower(self):
        self._test_model_with_non_decomp_partitioner(TestLinear())

        self._test_model_with_non_decomp_partitioner(TestSDPA())

        self._test_model_with_non_decomp_partitioner(TestLinearSDPACombined())

        self._test_model_with_non_decomp_partitioner(TestUpsample())

        self._test_model_with_non_decomp_partitioner(TestLSTM())

    def test_to_edge_transform_and_lower_with_exception(self):
        class TestLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(32, 16, bias=True)
                self.linear_no_bias = torch.nn.Linear(32, 16, bias=False)

            def forward(self, x):
                return (self.linear(x), self.linear_no_bias(x))

            @classmethod
            def _get_random_inputs(cls):
                x = torch.rand(8, 32)
                return (x,)

        model = TestLinear()
        ep = _export(model, model._get_random_inputs(), pre_dispatch=True)
        edge = to_edge_transform_and_lower(
            ep,
            compile_config=EdgeCompileConfig(),
            partitioner=[NonDecompTestPartitioner()],
        )

        def count_nodes(graph_module, target):
            count = 0
            for node in graph_module.graph.nodes:
                if node.op == "call_function" and node.target == target:
                    count += 1
            return count

        # There should be 1 call_delegate node and 1 node for aten.mm.default for the
        # linear that doesn't have a bias which was decomposed as the partitioner
        # said this node wasn't supported.
        self.assertEqual(
            count_nodes(
                edge.exported_program().graph_module,
                torch.ops.higher_order.executorch_call_delegate,
            ),
            1,
        )
        self.assertEqual(
            count_nodes(
                edge.exported_program().graph_module, exir_ops.edge.aten.mm.default
            ),
            1,
        )

    def test_edge_dialect_non_core_aten_ops(self):
        class LinalgNorm(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.linalg.norm(x)

        from torch._export.verifier import SpecViolationError

        input = torch.arange(9, dtype=torch.float) - 4
        ep = torch.export.export(LinalgNorm(), (input,))

        # aten::linalg_norm is not a core op, so it should error out
        with self.assertRaises(SpecViolationError):
            _ = to_edge(ep, compile_config=EdgeCompileConfig(_check_ir_validity=True))

        # with exception list, it should not error out
        try:
            # This should not raise error
            _ = to_edge(
                ep,
                compile_config=EdgeCompileConfig(
                    _check_ir_validity=True,
                    _core_aten_ops_exception_list=[
                        torch.ops.aten.linalg_vector_norm.default
                    ],
                ),
            )
        except SpecViolationError:
            self.fail("Should not error out on linalg_vector_norm op")

    def _test_to_edge_with_preserved_ops(
        self, program, preserved_ops, expected_preserved_ops
    ):
        edge = to_edge_with_preserved_ops(program, preserve_ops=preserved_ops)

        def count_nodes(graph_module, target):
            count = 0
            for node in graph_module.graph.nodes:
                if node.op == "call_function" and node.target in target:
                    count += 1
            return count

        aten_ops_non_decomposed = count_nodes(
            program.graph_module,
            preserved_ops,
        )

        edge_ops_non_decomposed = count_nodes(
            edge.exported_program().graph_module,
            expected_preserved_ops,
        )

        self.assertEqual(aten_ops_non_decomposed, edge_ops_non_decomposed)

    def test_to_edge_with_single_preserved_op(self):
        model = TestLinear()
        program = torch.export.export(model, model._get_random_inputs())

        ops_not_to_decompose = [
            torch.ops.aten.linear.default,
        ]
        expected_non_decomposed_edge_ops = [
            exir_ops.edge.aten.linear.default,
        ]

        self._test_to_edge_with_preserved_ops(
            program, ops_not_to_decompose, expected_non_decomposed_edge_ops
        )

    def test_to_edge_with_partial_ops_preserved(self):
        model = TestLinearSDPACombined()
        program = torch.export.export(model, model._get_random_inputs())

        ops_not_to_decompose = [
            torch.ops.aten.linear.default,
        ]
        expected_non_decomposed_edge_ops = [
            exir_ops.edge.aten.linear.default,
        ]

        self._test_to_edge_with_preserved_ops(
            program, ops_not_to_decompose, expected_non_decomposed_edge_ops
        )

    def test_to_edge_with_multiple_ops_preserved(self):
        model = TestLinearSDPACombined()
        program = torch.export.export(model, model._get_random_inputs())

        ops_not_to_decompose = [
            torch.ops.aten.linear.default,
            torch.ops.aten.scaled_dot_product_attention.default,
        ]
        expected_non_decomposed_edge_ops = [
            exir_ops.edge.aten.linear.default,
            exir_ops.edge.aten.scaled_dot_product_attention.default,
        ]

        self._test_to_edge_with_preserved_ops(
            program, ops_not_to_decompose, expected_non_decomposed_edge_ops
        )

    def test_to_edge_with_preserved_ops_not_in_model(self):
        model = TestSDPA()
        program = torch.export.export(model, model._get_random_inputs())

        ops_not_to_decompose = [
            torch.ops.aten.linear.default,
        ]
        expected_non_decomposed_edge_ops = [
            exir_ops.edge.aten.linear.default,
        ]

        self._test_to_edge_with_preserved_ops(
            program, ops_not_to_decompose, expected_non_decomposed_edge_ops
        )
