# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import executorch.exir.tests.models as models

import torch
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.lowered_backend_module import (
    create_submodule_from_nodes,
    LoweredBackendModule,
)
from executorch.exir.schema import (
    BackendDelegate,
    BackendDelegateDataReference,
    DataLocation,
    DelegateCall,
)
from executorch.exir.tests.common import register_additional_test_aten_ops
from torch.export import export
from torch.testing import FileCheck


class WrapperModule(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class TestDelegate(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        register_additional_test_aten_ops()

    def test_call_delegate(self) -> None:
        def g(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

        inputs = (torch.ones(1, 3), torch.ones(1, 3))
        edge_ir_m = to_edge(export(WrapperModule(g), inputs))
        lowered_module: LoweredBackendModule = LoweredBackendModule(
            edge_ir_m.exported_program(), "BackendWithCompilerDemo", b"moo", []
        )

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.ops.higher_order.executorch_call_delegate(lowered_module, x, y)

        orig_res = f(*inputs)
        gm = export(
            WrapperModule(f),
            inputs,
        )
        FileCheck().check("lowered_module_0").check(
            "torch.ops.higher_order.executorch_call_delegate"
        ).run(gm.graph_module.code)
        self.assertTrue(torch.allclose(orig_res, gm.module()(*inputs)))

    def test_to_backend(self) -> None:
        """Check if we have patched a lowered module correctly (for delegation)"""

        m = models.CompositeDelegateModule()

        exec_prog = to_edge(
            export(m, m.get_random_inputs()),
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        ).to_executorch()  # TODO(larryliu): fix split_copy.Tensor
        graph_module = exec_prog.exported_program().graph_module
        program = exec_prog._emitter_output.program

        # Check that there exists a call_delegate, representing the call to the
        # delegated function
        FileCheck().check("lowered_module_0").check(
            "torch.ops.higher_order.executorch_call_delegate"
        ).run(graph_module.code)

        # Check that there does not exist an add node (from the non-delegated
        # BasicModuleAdd.forward function)
        self.assertTrue(
            exir_ops.edge.aten.add.default
            not in {node.target for node in graph_module.graph.nodes}
        )

        for node in graph_module.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.higher_order.executorch_call_delegate
            ):
                # Check that the first argument is the lowered backend module
                # (which we got from a getattr)
                self.assertEqual(node.args[0].op, "get_attr")
                get_attr_backend = getattr(graph_module, node.args[0].target)
                self.assertEqual(
                    get_attr_backend._backend_id, m.lowered_module._backend_id
                )
                self.assertEqual(
                    get_attr_backend._processed_bytes, m.lowered_module._processed_bytes
                )
                self.assertEqual(
                    get_attr_backend._compile_specs, m.lowered_module._compile_specs
                )

        # Check the BackendDelegate object itself
        delegate: BackendDelegate = program.execution_plan[0].delegates[0]
        self.assertEqual(delegate.id, "backend_demo")
        processed: BackendDelegateDataReference = delegate.processed
        self.assertEqual(processed.location, DataLocation.INLINE)
        self.assertLess(processed.index, len(program.backend_delegate_data))
        self.assertEqual(
            program.backend_delegate_data[processed.index].data, b"basic_module_add"
        )

        # Check the delegate instruction
        self.assertTrue(
            isinstance(
                program.execution_plan[0].chains[0].instructions[0].instr_args,
                DelegateCall,
            )
        )

    def test_cannot_assign_attr(self) -> None:
        deleg = LoweredBackendModule(None, "", b"", [])  # pyre-ignore
        with self.assertRaises(AttributeError):
            deleg.backend_id = "123"  # pyre-ignore

    def test_create_submodule_single_return(self) -> None:
        """
        Original graph:
            add_tensor = add(x, y)
            mul_tensor = mul(add_tensor, y)
            sub_tensor = sub(mul_tensor, y)
            div_tensor = div(sub_tensor, y)
            return [div_tensor]

        Partitioned graph:
            add_tensor = add(x, y)
            mul_tensor = mul(add_tensor, y)
            return [mul_tensor]  # Output is pytree.flatten-ed

        Final graph:
            partitioned_res = partitioned_graph(x, y)
            getitem_0 = partitioned_res[0]
            sub_tensor = sub(getitem_0, y)
            div_tensor = div(sub_tensor, y)
            return [div_tensor]
        """
        inputs = (torch.randn(1, 3), torch.randn(1, 3))

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                x = x + y
                x = x * y
                x = x - y
                x = x / y
                return x

        orig_res = Model()(*inputs)
        prog = to_edge(export(Model(), inputs))
        gm = prog.exported_program().graph_module

        node_list = []
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target in {
                exir_ops.edge.aten.add.Tensor,
                exir_ops.edge.aten.mul.Tensor,
            }:
                node_list.append(node)

        sub_gm, node = create_submodule_from_nodes(gm, node_list, "tag")
        sub_gm.recompile()
        gm.recompile()

        for node in sub_gm.graph.nodes:
            if node.op == "output":
                self.assertEqual(len(node.args), 1)
                self.assertTrue(isinstance(node.args[0], list))
                self.assertEqual(len(node.args[0]), 1)

        new_res = prog.exported_program().module()(*inputs)
        self.assertTrue(torch.allclose(new_res, orig_res))

    def test_create_submodule_multiple_return(self) -> None:
        """
        Original graph:
            add_tensor = add(x, y)
            mul_tensor = mul(add_tensor, y)
            sub_tensor = sub(add_tensor, mul_tensor)
            div_tensor = div(sub_tensor, mul_tensor)
            return [div_tensor]

        Partitioned graph:
            add_tensor = add(x, y)
            mul_tensor = mul(add_tensor, y)
            return [add_tensor, mul_tensor]

        Final graph:
            partitioned_res = partitioned_graph(x, y)
            getitem_0 = partitioned_res[0]
            getitem_1 = partitioned_res[1]
            sub_tensor = sub(getitem_0, getitem_1)
            div_tensor = div(sub_tensor, getitem_1)
            return [div_tensor]
        """
        inputs = (torch.randn(1, 3), torch.randn(1, 3))

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                x = x + y
                y = x * y
                x = x - y
                x = x / y
                return x

        orig_res = Model()(*inputs)
        prog = to_edge(export(Model(), inputs))
        gm = prog.exported_program().graph_module

        node_list = []
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target in {
                exir_ops.edge.aten.add.Tensor,
                exir_ops.edge.aten.mul.Tensor,
            }:
                node_list.append(node)

        sub_gm, node = create_submodule_from_nodes(gm, node_list, "tag")
        sub_gm.recompile()
        gm.recompile()

        for node in sub_gm.graph.nodes:
            if node.op == "output":
                self.assertEqual(len(node.args), 1)
                self.assertTrue(isinstance(node.args[0], list))
                self.assertEqual(len(node.args[0]), 2)

        new_res = prog.exported_program().module()(*inputs)
        self.assertTrue(torch.allclose(new_res, orig_res))

    def test_create_submodule_list_return(self) -> None:
        """
        Original graph:
            split_tensor = split(x, 5)
            getitem_0 = split_tensor[0]
            sub_tensor = sub(getitem_0, y)
            div_tensor = div(sub_tensor, y)
            return [div_tensor]

        Partitioned graph:
            split_tensor = split(x, 5)
            getitem_0 = split_tensor[0]
            getitem_1 = split_tensor[1]
            return [getitem_0, getitem_1]  # List output is "opened"

        Final graph:
            partitioned_res = partitioned_graph(x, y)
            getitem_0 = partitioned_res[0]
            sub_tensor = sub(getitem_0, y)
            div_tensor = div(sub_tensor, y)
            return [div_tensor]
        """
        inputs = (torch.randn(10), torch.randn(5))

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                x = torch.split(x, 5)
                x = x[0] - y
                x = x / y
                return x

        orig_res = Model()(*inputs)
        prog = to_edge(export(Model(), inputs))
        gm = prog.exported_program().graph_module

        node_list = []
        for node in gm.graph.nodes:
            # TODO(ssjia): split.Tensor now gets decomposed to split_with_sizes. Due to how executorch uses a pinned Pytorch
            # nightly, the CI may not catch the changes to Pytorch's core decomposition table. As a temporary workaround,
            # make the test backwards compatible with the old decomposition table. Remove the or statement once Pytorch nightly
            # has been updated.
            if node.op == "call_function" and (
                node.target == exir_ops.edge.aten.split_with_sizes_copy.default
                or node.target == exir_ops.edge.aten.split_copy.Tensor
            ):
                node_list.append(node)

        sub_gm, node = create_submodule_from_nodes(gm, node_list, "tag")

        for node in sub_gm.graph.nodes:
            if node.op == "output":
                self.assertEqual(len(node.args), 1)
                self.assertTrue(isinstance(node.args[0], list))
                self.assertEqual(len(node.args[0]), 2)

        new_res = prog.exported_program().module()(*inputs)
        self.assertTrue(torch.allclose(new_res, orig_res))
