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


class TestModel1(nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = nn.Parameter(torch.rand(5, 6))
        self.parameter.requires_grad = False

    def forward(self, x):
        v1 = self.parameter.view(
            6, 5
        )  # removed, lifetime of parameter will be extended
        v2 = x.view(6, 5)  # not removed
        v3 = torch.ops.aten.mul.Tensor(v1, v2).view(
            30
        )  # removed, lifetime of mul.Tensor will be extended
        return v3

    def get_example_inputs(self):
        return (torch.rand(5, 6),)


class TestTryRemoveViewCopy(unittest.TestCase):
    def test_disable(self) -> None:
        model = TestModel1()
        model.eval()
        example_inputs = model.get_example_inputs()
        ep = torch.export.export(model, example_inputs)
        etpm = to_edge(ep).to_executorch(
            config=ExecutorchBackendConfig(
                try_remove_view_copy=False,
                memory_planning_pass=MemoryPlanningPass(
                    "greedy", alloc_graph_input=False
                ),
            ),
        )

        for node in etpm.exported_program().graph_module.graph.nodes:
            assert node.target != memory.view

    def test_output_matches(self) -> None:
        model = TestModel1()
        model.eval()
        example_inputs = model.get_example_inputs()
        ep = torch.export.export(model, example_inputs)

        epm_remove = to_edge(ep)
        epm_no_remove = copy.deepcopy(
            epm_remove
        )  # to_executorch modifies the edge_program, so we make a copy

        # Run pass with no removal
        etpm_remove = epm_remove.to_executorch(
            config=ExecutorchBackendConfig(
                try_remove_view_copy=True,
                memory_planning_pass=MemoryPlanningPass(
                    "greedy", alloc_graph_input=False
                ),
            ),
        )

        # Run pass with removal
        etpm_no_remove = epm_no_remove.to_executorch(
            config=ExecutorchBackendConfig(
                try_remove_view_copy=True,
                memory_planning_pass=MemoryPlanningPass(
                    "greedy", alloc_graph_input=False
                ),
            ),
        )

        out_remove = etpm_remove.exported_program().module()(*example_inputs)
        out_no_remove = etpm_no_remove.exported_program().module()(*example_inputs)

        self.assertTrue(torch.allclose(out_remove, out_no_remove))

    def test_spec(self) -> None:
        model = TestModel1()
        model.eval()
        example_inputs = model.get_example_inputs()
        ep = torch.export.export(model, example_inputs)

        etpm = to_edge(ep).to_executorch(
            config=ExecutorchBackendConfig(
                try_remove_view_copy=True,
                memory_planning_pass=MemoryPlanningPass(
                    "greedy", alloc_graph_input=False
                ),
            ),
        )

        # etpm.exported_program().graph.print_tabular()

        # idx  opcode         name                      target                              args                                                kwargs
        # ---  -------------  ------------------------  ----------------------------------  --------------------------------------------------  ----------------
        # 0    placeholder    arg0_1                    arg0_1                              ()                                                  {}
        # 1    placeholder    arg1_1                    arg1_1                              ()                                                  {}
        # 2    call_function  aten_view_copy_default    <function view at 0x7f10a6dfeb00>   (arg0_1, [6, 5])                                    {}
        # 3    call_function  alloc                     <function alloc at 0x7f10a6dfe9e0>  (((6, 5), torch.float32),)                          {}
        # 4    call_function  aten_view_copy_default_1  aten.view_copy.out                  (arg1_1, [6, 5])                                    {'out': alloc}
        # 5    call_function  alloc_1                   <function alloc at 0x7f10a6dfe9e0>  (((6, 5), torch.float32),)                          {}
        # 6    call_function  aten_mul_tensor           aten.mul.out                        (aten_view_copy_default, aten_view_copy_default_1)  {'out': alloc_1}
        # 7    call_function  aten_view_copy_default_2  <function view at 0x7f10a6dfeb00>   (aten_mul_tensor, [30])                             {}
        # 8    output         output_1                  output                              ((aten_view_copy_default_2,),)                      {}

        # arg0_1 is the parameter
        # arg1_1 is the user input

        for node in etpm.exported_program().graph.nodes:
            if node.name == "arg0_1":
                # arg0_1's lifetime is extended through aten_view_copy_default (memory.view) to idx 6
                self.assertEqual(node.meta["spec"].lifetime, [0, 6])
            elif node.name == "aten_view_copy_default":
                # aten_view_copy_default is a memory.view of arg0_1.
                # arg0_1 is a constant with storage, so we check that the view's storage matches the base

                # assert base is arg0_1
                self.assertEqual(node.args[0].name, "arg0_1")

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
                # aten_mul_tensor's lifetime is extended through aten_view_copy_default_2 (memory.view) to idx 8
                self.assertEqual(node.meta["spec"].lifetime, [5, 8])
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
        evalues = etpm.executorch_program.execution_plan[0].values

        # evalue 0 is the parameter arg0_1 and evalue 2 is view aten_view_copy_default
        # assert their sizes are as expected and constant_buffer_idx != 0
        self.assertEqual(evalues[0].val.sizes, [5, 6])  # pyre-ignore
        self.assertNotEqual(evalues[0].val.constant_buffer_idx, 0)  # pyre-ignore
        self.assertEqual(evalues[2].val.sizes, [6, 5])  # pyre-ignore
        self.assertNotEqual(evalues[2].val.constant_buffer_idx, 0)  # pyre-ignore

        # assert they have the same constant_buffer_idx
        self.assertEqual(evalues[0].val.constant_buffer_idx, evalues[2].val.constant_buffer_idx)  # pyre-ignore

        # evalue 7 is alloc_1 (aten_mul_tensor) and evalue 8 is aten_view_copy_default_2
        # assert their sizes are as expected and constant_buffer_idx == 0
        self.assertEqual(evalues[7].val.sizes, [6, 5])  # pyre-ignore
        self.assertEqual(evalues[7].val.constant_buffer_idx, 0)  # pyre-ignore
        self.assertEqual(evalues[8].val.sizes, [30])  # pyre-ignore
        self.assertEqual(evalues[8].val.constant_buffer_idx, 0)  # pyre-ignore

        # assert they have the same mem_id and mem_offset low and high
        self.assertEqual(evalues[7].val.allocation_info.memory_id, evalues[8].val.allocation_info.memory_id)  # pyre-ignore
        self.assertEqual(evalues[7].val.allocation_info.memory_offset_low, evalues[8].val.allocation_info.memory_offset_low)  # pyre-ignore
        self.assertEqual(evalues[7].val.allocation_info.memory_offset_high, evalues[8].val.allocation_info.memory_offset_high)  # pyre-ignore
