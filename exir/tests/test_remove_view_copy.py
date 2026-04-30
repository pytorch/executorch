# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

import torch
import torch.nn as nn
from executorch.exir import memory, to_edge
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.replace_view_copy_with_view_pass import _ViewSpec


class TestModel1(nn.Module):
    __test__ = False

    def __init__(self):
        super().__init__()
        self.parameter = nn.Parameter(torch.rand(5, 6))
        self.parameter.requires_grad = False
        self.parameter2 = nn.Parameter(torch.rand(30))
        self.parameter2.requires_grad = False

    def forward(self, x):
        v1 = self.parameter.view(
            6, 5
        )  # removed, lifetime of parameter will be extended
        v2 = x.view(6, 5)  # not removed
        v3 = torch.ops.aten.mul.Tensor(v1, v2).view(
            30
        )  # removed, lifetime of mul.Tensor will be extended
        v4 = torch.ops.aten.mul.Tensor(v3, self.parameter2)
        v5 = v4.view(6, 5)  # not removed, output of the graph
        v6 = v4.view(2, 15)  # not removed, output of the graph
        return v5, v6

    def get_example_inputs(self):
        return (torch.rand(5, 6),)


class TestRemoveViewCopy(unittest.TestCase):
    def test_disable(self) -> None:
        model = TestModel1()
        model.eval()
        example_inputs = model.get_example_inputs()
        ep = torch.export.export(model, example_inputs, strict=True)
        etpm = to_edge(ep).to_executorch(
            config=ExecutorchBackendConfig(
                remove_view_copy=False,
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            ),
        )

        for node in etpm.exported_program().graph_module.graph.nodes:
            assert node.target != memory.view

    def test_output_matches(self) -> None:
        model = TestModel1()
        model.eval()
        example_inputs = model.get_example_inputs()
        ep = torch.export.export(model, example_inputs, strict=True)

        epm_remove = to_edge(ep)
        epm_no_remove = copy.deepcopy(
            epm_remove
        )  # to_executorch modifies the edge_program, so we make a copy

        # Run pass with no removal
        etpm_remove = epm_remove.to_executorch(
            config=ExecutorchBackendConfig(
                remove_view_copy=True,
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            ),
        )

        # Run pass with removal
        etpm_no_remove = epm_no_remove.to_executorch(
            config=ExecutorchBackendConfig(
                remove_view_copy=True,
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            ),
        )

        out_remove_v5, out_remove_v6 = etpm_remove.exported_program().module()(
            *example_inputs
        )
        out_no_remove_v5, out_no_remove_v6 = etpm_no_remove.exported_program().module()(
            *example_inputs
        )

        self.assertTrue(torch.allclose(out_remove_v5, out_no_remove_v5))
        self.assertTrue(torch.allclose(out_remove_v6, out_no_remove_v6))

    def test_spec(self) -> None:
        model = TestModel1()
        model.eval()
        example_inputs = model.get_example_inputs()
        ep = torch.export.export(model, example_inputs, strict=True)

        etpm = to_edge(ep).to_executorch(
            config=ExecutorchBackendConfig(
                remove_view_copy=True,
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            ),
        )

        # etpm.exported_program().graph.print_tabular()

        # idx  opcode         name                      target                              args                                                kwargs
        # ---  -------------  ------------------------  ----------------------------------  --------------------------------------------------  ----------------
        # 0    placeholder    p_parameter               p_parameter                         ()                                                  {}
        # 1    placeholder    p_parameter2              p_parameter2                        ()                                                  {}
        # 2    placeholder    x                         x                                   ()                                                  {}
        # 3    call_function  aten_view_copy_default    <function view at 0x7fe57bea6d40>   (p_parameter, [6, 5])                               {}
        # 4    call_function  aten_view_copy_default_1  <function view at 0x7fe57bea6d40>   (x, [6, 5])                                         {}
        # 5    call_function  alloc                     <function alloc at 0x7fe57bea6c20>  (((6, 5), torch.float32),)                          {}
        # 6    call_function  aten_mul_tensor           aten.mul.out                        (aten_view_copy_default, aten_view_copy_default_1)  {'out': alloc}
        # 7    call_function  aten_view_copy_default_2  <function view at 0x7fe57bea6d40>   (aten_mul_tensor, [30])                             {}
        # 8    call_function  alloc_1                   <function alloc at 0x7fe57bea6c20>  (((30,), torch.float32),)                           {}
        # 9    call_function  aten_mul_tensor_1         aten.mul.out                        (aten_view_copy_default_2, p_parameter2)            {'out': alloc_1}
        # 10   call_function  alloc_2                   <function alloc at 0x7fe57bea6c20>  (((6, 5), torch.float32),)                          {}
        # 11   call_function  aten_view_copy_default_3  aten.view_copy.out                  (aten_mul_tensor_1, [6, 5])                         {'out': alloc_2}
        # 12   output         output_1                  output                              ((aten_view_copy_default_3,),)                      {}

        for node in etpm.exported_program().graph.nodes:
            if node.name == "p_parameter":
                # p_parameter's lifetime is extended through aten_view_copy_default (memory.view) to idx 6
                self.assertEqual(node.meta["spec"].lifetime, [0, 6])
            elif node.name == "aten_view_copy_default":
                # aten_view_copy_default is a memory.view of p_parameter.
                # p_parameter is a constant with storage, so we check that the view's storage matches the base

                # assert base is p_parameter
                self.assertEqual(node.args[0].name, "p_parameter")

                # assert base is const with storage
                self.assertTrue(node.args[0].meta["spec"].const)
                self.assertTrue(node.args[0].meta["spec"].storage is not None)
                self.assertTrue(node.args[0].meta["spec"].mem_id is None)
                self.assertTrue(node.args[0].meta["spec"].mem_offset is None)

                # assert self is const with storage
                self.assertTrue(node.meta["spec"].const)
                self.assertTrue(node.meta["spec"].storage is not None)
                self.assertTrue(node.meta["spec"].mem_id is None)
                self.assertTrue(node.meta["spec"].mem_offset is None)

                # assert storage matches
                self.assertEqual(
                    node.meta["spec"].storage, node.args[0].meta["spec"].storage
                )

                # assert lifetime matches
                self.assertEqual(
                    node.meta["spec"].lifetime, node.args[0].meta["spec"].lifetime
                )
            elif node.name == "aten_mul_tensor":
                # aten_mul_tensor's lifetime is extended through aten_view_copy_default_2 (memory.view) to idx 9
                self.assertEqual(node.meta["spec"].lifetime, [5, 9])
            elif node.name == "aten_view_copy_default_2":
                # aten_view_copy_default_2 is a memory.view of aten_mul_tensor

                # assert base is aten_mul_tensor
                self.assertEqual(node.args[0].name, "aten_mul_tensor")

                # assert base and self are not const, do not have storage,
                # but do have mem_id and mem_offset
                self.assertFalse(node.args[0].meta["spec"].const)
                self.assertTrue(node.args[0].meta["spec"].storage is None)
                self.assertTrue(node.args[0].meta["spec"].mem_id is not None)
                self.assertTrue(node.args[0].meta["spec"].mem_offset is not None)

                self.assertFalse(node.meta["spec"].const)
                self.assertTrue(node.meta["spec"].storage is None)
                self.assertTrue(node.meta["spec"].mem_id is not None)
                self.assertTrue(node.meta["spec"].mem_offset is not None)

                # assert self and base mem_id, mem_offset, and lifetime matches
                self.assertEqual(
                    node.meta["spec"].mem_id, node.args[0].meta["spec"].mem_id
                )
                self.assertEqual(
                    node.meta["spec"].mem_offset, node.args[0].meta["spec"].mem_offset
                )
                self.assertEqual(
                    node.meta["spec"].lifetime, node.args[0].meta["spec"].lifetime
                )

        # Test evalues in execution plan
        plan = etpm.executorch_program.execution_plan[0]
        self.assertEqual(plan.operators[0].name, "executorch_prim::et_view")
        self.assertEqual(plan.operators[1].name, "aten::mul")
        self.assertEqual(plan.operators[2].name, "aten::view_copy")

        instructions = plan.chains[0].instructions
        self.assertEqual(len(instructions), 5)

        self.assertEqual(instructions[0].instr_args.op_index, 0)  # view @ idx2
        self.assertEqual(instructions[1].instr_args.op_index, 1)  # aten:mul @ idx6
        self.assertEqual(instructions[2].instr_args.op_index, 1)  # aten:mul @ idx9
        self.assertEqual(
            instructions[3].instr_args.op_index, 2
        )  # aten:view_copy @ idx11
        self.assertEqual(
            instructions[4].instr_args.op_index, 2
        )  # aten:view_copy @ idx11

    def test_elide_static_views_does_not_remove_dynamic_views(self) -> None:
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x = x + x
                x = x.view(-1, 1)
                return 2 * x

        model = TestModel()
        model.eval()
        example_inputs = (torch.rand(5, 6),)
        dynamic_shapes = {"x": {0: torch.export.Dim("dim0", min=1, max=10)}}
        ep = torch.export.export(
            model, example_inputs, strict=True, dynamic_shapes=dynamic_shapes
        )
        etpm = to_edge(ep).to_executorch(
            config=ExecutorchBackendConfig(
                remove_view_copy=True,
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=True),
            ),
        )
        plan = etpm.executorch_program.execution_plan[0]
        op_names = [op.name for op in plan.operators]
        self.assertTrue("executorch_prim::et_view" in op_names)

    def test_contiguous_select_replaced(self) -> None:
        class SelectDim0Model(nn.Module):
            __test__ = False

            def forward(self, x):
                y = x + 1
                z = y.select(0, 2)
                return z * 2

        model = SelectDim0Model()
        model.eval()
        example_inputs = (torch.rand(4, 3),)
        ep = torch.export.export(model, example_inputs, strict=True)
        etpm = to_edge(ep).to_executorch(
            config=ExecutorchBackendConfig(
                remove_view_copy=True,
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            ),
        )

        found_select_view = False
        for node in etpm.exported_program().graph.nodes:
            if node.target == memory.select:
                found_select_view = True
                self.assertIsInstance(node.meta["spec"], _ViewSpec)
                base = node.args[0]
                self.assertEqual(
                    node.meta["spec"].mem_id, base.meta["spec"].mem_id
                )
                self.assertEqual(
                    node.meta["spec"].lifetime, base.meta["spec"].lifetime
                )
        self.assertTrue(found_select_view)

    def test_non_contiguous_select_not_replaced(self) -> None:
        class SelectDim1Model(nn.Module):
            __test__ = False

            def forward(self, x):
                y = x + 1
                z = y.select(1, 1)
                return z * 2

        model = SelectDim1Model()
        model.eval()
        example_inputs = (torch.rand(4, 3),)
        ep = torch.export.export(model, example_inputs, strict=True)
        etpm = to_edge(ep).to_executorch(
            config=ExecutorchBackendConfig(
                remove_view_copy=True,
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            ),
        )

        for node in etpm.exported_program().graph.nodes:
            self.assertNotEqual(node.target, memory.select)

    def test_select_output_matches(self) -> None:
        class SelectModel(nn.Module):
            __test__ = False

            def forward(self, x):
                y = x + 1
                z = y.select(0, 2)
                return z * 2

        model = SelectModel()
        model.eval()
        example_inputs = (torch.rand(4, 3),)
        ep = torch.export.export(model, example_inputs, strict=True)

        epm_remove = to_edge(ep)
        epm_no_remove = copy.deepcopy(epm_remove)

        etpm_remove = epm_remove.to_executorch(
            config=ExecutorchBackendConfig(
                remove_view_copy=True,
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            ),
        )
        etpm_no_remove = epm_no_remove.to_executorch(
            config=ExecutorchBackendConfig(
                remove_view_copy=False,
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            ),
        )

        out_remove = etpm_remove.exported_program().module()(*example_inputs)
        out_no_remove = etpm_no_remove.exported_program().module()(*example_inputs)
        self.assertTrue(torch.allclose(out_remove, out_no_remove))

    def test_select_spec_mem_offset(self) -> None:
        class SelectModel(nn.Module):
            __test__ = False

            def forward(self, x):
                y = x + 1
                z = y.select(0, 2)
                return z * 2

        model = SelectModel()
        model.eval()
        example_inputs = (torch.rand(4, 3),)
        ep = torch.export.export(model, example_inputs, strict=True)
        etpm = to_edge(ep).to_executorch(
            config=ExecutorchBackendConfig(
                remove_view_copy=True,
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            ),
        )

        for node in etpm.exported_program().graph.nodes:
            if node.target == memory.select:
                base = node.args[0]
                base_offset = base.meta["spec"].mem_offset
                select_offset = node.meta["spec"].mem_offset
                # select(dim=0, index=2) on [4,3] float32: offset = 2 * 3 * 4 = 24
                self.assertEqual(select_offset, base_offset + 24)

    def test_dynamic_shape_select_replaced(self) -> None:
        class SelectDim0Model(nn.Module):
            __test__ = False

            def forward(self, x):
                y = x + 1
                z = y.select(0, 2)
                return z * 2

        model = SelectDim0Model()
        model.eval()
        example_inputs = (torch.rand(4, 3),)
        dynamic_shapes = {"x": {1: torch.export.Dim("dim1", min=1, max=10)}}
        ep = torch.export.export(
            model, example_inputs, strict=True, dynamic_shapes=dynamic_shapes
        )

        epm = to_edge(ep)
        epm_copy = copy.deepcopy(epm)

        etpm_on = epm.to_executorch(
            config=ExecutorchBackendConfig(
                remove_view_copy=True,
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=True),
            ),
        )
        etpm_off = epm_copy.to_executorch(
            config=ExecutorchBackendConfig(
                remove_view_copy=False,
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=True),
            ),
        )

        found_select_view = False
        for node in etpm_on.exported_program().graph.nodes:
            if node.target == memory.select:
                found_select_view = True
        self.assertTrue(found_select_view)

    def test_dynamic_shape_select_no_allocation(self) -> None:
        class SelectDim0Model(nn.Module):
            __test__ = False

            def forward(self, x):
                y = x + 1
                z = y.select(0, 2)
                return z * 2

        model = SelectDim0Model()
        model.eval()
        example_inputs = (torch.rand(4, 3),)
        dynamic_shapes = {"x": {1: torch.export.Dim("dim1", min=1, max=10)}}
        ep = torch.export.export(
            model, example_inputs, strict=True, dynamic_shapes=dynamic_shapes
        )

        etpm = to_edge(ep).to_executorch(
            config=ExecutorchBackendConfig(
                remove_view_copy=True,
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=True),
            ),
        )

        for node in etpm.exported_program().graph.nodes:
            if node.target == memory.select:
                spec = node.meta["spec"]
                self.assertIsNone(spec.mem_id)
                self.assertIsNone(spec.mem_offset)

    def test_input_select_shares_allocation(self) -> None:
        class InputSelectModel(nn.Module):
            __test__ = False

            def forward(self, x):
                z = x.select(0, 1)
                return z * 2

        model = InputSelectModel()
        model.eval()
        example_inputs = (torch.rand(4, 3),)
        ep = torch.export.export(model, example_inputs, strict=True)

        etpm = to_edge(ep).to_executorch(
            config=ExecutorchBackendConfig(
                remove_view_copy=True,
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=True),
            ),
        )

        found_select = False
        for node in etpm.exported_program().graph.nodes:
            if node.target == memory.select:
                found_select = True
                spec = node.meta["spec"]
                base = node.args[0]
                self.assertEqual(spec.mem_id, base.meta["spec"].mem_id)
                base_offset = base.meta["spec"].mem_offset
                # select(0, 1) on [4, 3] float32: byte_offset = 1 * 3 * 4 = 12
                self.assertEqual(spec.mem_offset, base_offset + 12)
        self.assertTrue(found_select)

    def test_dynamic_dim_selected_away_replaced(self) -> None:
        class SelectDynDimModel(nn.Module):
            __test__ = False

            def forward(self, x):
                y = x + 1
                z = y.select(1, 0)
                return z * 2

        model = SelectDynDimModel()
        model.eval()
        example_inputs = (torch.rand(1, 4, 3),)
        dynamic_shapes = {"x": {1: torch.export.Dim("seq", min=1, max=128)}}
        ep = torch.export.export(
            model, example_inputs, strict=True, dynamic_shapes=dynamic_shapes
        )
        etpm = to_edge(ep).to_executorch(
            config=ExecutorchBackendConfig(
                remove_view_copy=True,
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=True),
            ),
        )

        found_select_view = False
        for node in etpm.exported_program().graph.nodes:
            if node.target == memory.select:
                found_select_view = True
        self.assertTrue(found_select_view)

    def test_dynamic_leading_dim_select_not_replaced(self) -> None:
        class SelectDim1Model(nn.Module):
            __test__ = False

            def forward(self, x):
                y = x + 1
                z = y.select(1, 0)
                return z * 2

        model = SelectDim1Model()
        model.eval()
        example_inputs = (torch.rand(2, 3),)
        dynamic_shapes = {"x": {0: torch.export.Dim("batch", min=1, max=10)}}
        ep = torch.export.export(
            model, example_inputs, strict=True, dynamic_shapes=dynamic_shapes
        )
        etpm = to_edge(ep).to_executorch(
            config=ExecutorchBackendConfig(
                remove_view_copy=True,
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=True),
            ),
        )

        for node in etpm.exported_program().graph.nodes:
            self.assertNotEqual(node.target, memory.select)

    def test_view_then_select_chained(self) -> None:
        class ViewThenSelectModel(nn.Module):
            __test__ = False

            def forward(self, x):
                y = x + 1
                z = y.view(3, 1, 4)
                a = z.select(0, 0)
                b = z.select(0, 1)
                c = z.select(0, 2)
                return a + b + c

        model = ViewThenSelectModel()
        model.eval()
        example_inputs = (torch.rand(3, 4),)
        ep = torch.export.export(model, example_inputs, strict=True)
        etpm = to_edge(ep).to_executorch(
            config=ExecutorchBackendConfig(
                remove_view_copy=True,
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            ),
        )

        found_select_view = False
        for node in etpm.exported_program().graph.nodes:
            if node.target == memory.select:
                found_select_view = True
                spec = node.meta["spec"]
                self.assertIsInstance(spec, _ViewSpec)
                self.assertIsNotNone(spec.mem_id)
                self.assertIsNotNone(spec.mem_offset)
        self.assertTrue(found_select_view)

        import math
        from executorch.exir.schema import ScalarType

        plan = etpm.executorch_program.execution_plan[0]
        buffer_sizes = plan.non_const_buffer_sizes
        for tensor in plan.values:
            t = tensor.val
            if hasattr(t, "allocation_info") and t.allocation_info is not None:
                mem_id = t.allocation_info.memory_id
                if mem_id >= len(buffer_sizes):
                    continue
                buf_size = buffer_sizes[mem_id]
                offset = (
                    t.allocation_info.memory_offset_high << 32
                ) | t.allocation_info.memory_offset_low
                elem_size = {
                    ScalarType.FLOAT: 4,
                    ScalarType.INT: 4,
                    ScalarType.LONG: 8,
                    ScalarType.DOUBLE: 8,
                    ScalarType.HALF: 2,
                    ScalarType.BYTE: 1,
                }.get(t.scalar_type, 4)
                nbytes = math.prod(t.sizes) * elem_size
                self.assertLessEqual(
                    offset + nbytes,
                    buf_size,
                    f"Tensor with sizes={t.sizes} at offset={offset} "
                    f"exceeds buffer[{mem_id}] size {buf_size}",
                )

    def test_constant_base_not_replaced(self) -> None:
        class ConstSelectModel(nn.Module):
            __test__ = False

            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.rand(4, 3))
                self.param.requires_grad = False

            def forward(self, x):
                z = self.param.select(0, 1)
                return x + z

        model = ConstSelectModel()
        model.eval()
        example_inputs = (torch.rand(3),)
        ep = torch.export.export(model, example_inputs, strict=True)
        etpm = to_edge(ep).to_executorch(
            config=ExecutorchBackendConfig(
                remove_view_copy=True,
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            ),
        )

        for node in etpm.exported_program().graph.nodes:
            self.assertNotEqual(node.target, memory.select)
