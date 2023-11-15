# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pye-strict

import unittest
from typing import Any, Dict

import torch
from executorch.exir import ExecutorchBackendConfig
from executorch.exir.backend.test.op_partitioner_demo import AddMulPartitionerDemo
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.error import ExportError
from executorch.exir.lowered_backend_module import get_lowered_submodules
from executorch.exir.pass_base import ExportPass
from executorch.exir.program._program import (
    EdgeProgramManager,
    ExecutorchProgramManager,
    to_edge,
)
from executorch.exir.verification.verifier import EXIREdgeDialectVerifier

from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
)
from torch.export import export, ExportedProgram

from torch.library import impl, Library

lib = Library("test_op", "DEF")

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
    def forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = torch.mul(x, y)
        return torch.add(z, x)

    def foo(x: torch.Tensor) -> torch.Tensor:
        return torch.add(x, torch.ones(1))

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

    def test_edge_manager_transform(self):
        edge_manager: EdgeProgramManager = to_edge(
            get_exported_programs(), get_config_methods()
        )

        original_res = edge_manager.exported_program("forward")(
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
        print(transformed_edge.exported_program("forward").graph_module.graph)

        # transformation was applied
        self.assertEqual(
            transformed_edge.exported_program("forward")(torch.ones(1), torch.ones(1)),
            torch.ones(1),  # x * y * x
        )

        # original unchanged
        self.assertEqual(
            edge_manager.exported_program("forward")(torch.ones(1), torch.ones(1)),
            original_res,  # x * y + x
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
            transformed_edge.exported_program("forward")(torch.ones(1), torch.ones(1)),
            torch.ones(1),  # x * y * x
        )

        self.assertEqual(
            transformed_edge.exported_program("foo")(
                torch.ones(1),
            ),
            torch.ones(1) + 1,  # x + 1
        )

    def test_edge_to_backend_replaces_subgraph(self):
        edge_manager: EdgeProgramManager = to_edge(
            get_exported_programs(), get_config_methods()
        )
        delegate_manager: EdgeProgramManager = edge_manager.to_backend(
            AddMulPartitionerDemo
        )

        forward_program = delegate_manager.exported_program("forward")
        self.assertEqual(
            forward_program(torch.ones(1), torch.ones(1)),
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
                delegate_manager.to_executorch(
                    ExecutorchBackendConfig(extract_segments=True)
                )
                ._emitter_output.program.execution_plan[0]
                .delegates
            ),
            1,
        )
        self.assertEqual(
            len(
                delegate_manager.to_executorch(
                    ExecutorchBackendConfig(extract_segments=True)
                )
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
            {"forward": AddMulPartitionerDemo}
        )

        forward_program = delegate_manager.exported_program("forward")
        self.assertEqual(
            forward_program(torch.ones(1), torch.ones(1)),
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
                delegate_manager.to_executorch(ExecutorchBackendConfig())
                ._emitter_output.program.execution_plan[0]  # foo
                .delegates
            ),
            0,
        )
        self.assertEqual(
            len(
                delegate_manager.to_executorch(ExecutorchBackendConfig())
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

    def _test_edge_dialect_verifier(self, callable, validate_ir=True):
        from executorch.exir import EdgeCompileConfig

        edge_compile_config = EdgeCompileConfig(
            _check_ir_validity=validate_ir,
        )
        # pre-autograd export. eventually this will become torch.export
        one = torch.ones(1, dtype=torch.float)
        two = torch.ones(1, dtype=torch.int32)
        inputs = (
            one,
            two,
        )
        exported_foo = export(callable, inputs)
        _ = to_edge(exported_foo, compile_config=edge_compile_config)

    def test_edge_dialect_custom_op(self):
        def _use_foo_add(a: torch.Tensor, b: torch.Tensor):
            return torch.ops.test_op.foo(a, b)

        from torch._export.verifier import SpecViolationError

        with self.assertRaises(SpecViolationError):
            self._test_edge_dialect_verifier(_use_foo_add)

        # This should not raise error
        self._test_edge_dialect_verifier(_use_foo_add, False)
