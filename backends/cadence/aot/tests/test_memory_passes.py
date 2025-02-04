# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import logging
import math
import unittest
from typing import cast

import executorch.backends.cadence.aot.ops_registrations  # noqa
import torch
from executorch.backends.cadence.aot import compiler
from executorch.backends.cadence.aot.memory_planning import find_peak_memory_usage
from executorch.backends.cadence.aot.pass_utils import count_node
from executorch.exir import memory
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.tests.models import MultiLayerPerceptron


class TestMemPlanningPasses(unittest.TestCase):
    def test_calculate_peak_memory_pass(self):
        class PeakMemoryTestModel(torch.nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
                super().__init__()
                self.linear = torch.nn.Linear(input_dim, hidden_dim)
                self.relu = torch.nn.ReLU()
                self.linear2 = torch.nn.Linear(hidden_dim, output_dim)

            def forward(self, x: torch.Tensor):
                x = self.linear(x)
                x = self.relu(x)
                x = self.linear2(x)
                return x

        def calculate_aligned_num_bytes(num: int, alignment: int = 16):
            return math.ceil(num / alignment) * alignment

        # model 1
        batch_size, input_dim, hidden_dim, output_dim = 3, 16, 10, 20

        inputs = (torch.ones(batch_size, input_dim),)
        model = PeakMemoryTestModel(input_dim, hidden_dim, output_dim)

        graph_module = (
            compiler.export_to_executorch_gen_etrecord(model, inputs)
            .exported_program()
            .graph_module
        )

        peak_usage, _ = find_peak_memory_usage(
            graph_module,
            mem_constraints=None,
            alloc_graph_input=True,
            alloc_graph_output=True,
        )
        expected_peak_usage = calculate_aligned_num_bytes(
            hidden_dim * batch_size * 4
        ) + calculate_aligned_num_bytes(
            output_dim * batch_size * 4
        )  # Align data on a 16 byte boundary
        self.assertEqual(peak_usage, expected_peak_usage)

        # model 2
        batch_size, input_dim, hidden_dim, output_dim = 11, 10, 16, 8

        inputs = (torch.ones(batch_size, input_dim),)
        model = MultiLayerPerceptron(
            input_dim, hidden_dim, hidden_dim, hidden_dim, output_dim
        )

        graph_module = (
            compiler.export_to_executorch_gen_etrecord(model, inputs)
            .exported_program()
            .graph_module
        )

        peak_usage, _ = find_peak_memory_usage(
            graph_module,
            mem_constraints=None,
            alloc_graph_input=True,
            alloc_graph_output=True,
        )

        expected_peak_usage = 2 * calculate_aligned_num_bytes(
            hidden_dim * batch_size * 4
        )  # Align data on a 16 byte boundary
        self.assertEqual(peak_usage, expected_peak_usage)

    def test_zero_memory_pass(self):
        class ZeroMem(torch.nn.Module):
            def forward(self, x):
                return x[:, 2::3, ...]

        x = torch.randn(2, 7, 3, 2)

        # Compiler with alloc_graph_input=False and alloc_graph_output=False.
        # Cadence won't allocate memory for input and output, and the total memory
        # usage will be 0
        executorch_prog = compiler.export_to_executorch_gen_etrecord(
            ZeroMem(),
            (x,),
            alloc_graph_input=False,
            alloc_graph_output=False,
        )
        graph_module = executorch_prog.exported_program().graph_module
        graph_module.graph.eliminate_dead_code()
        peak_usage, _ = find_peak_memory_usage(
            graph_module,
            alloc_graph_input=False,
            alloc_graph_output=False,
            mem_constraints=None,
        )
        self.assertEqual(peak_usage, 0)


class TestMemTransform(unittest.TestCase):
    def _verify_cat_nop_memory_alloc(self, node: torch.fx.Node) -> None:
        spec = node.meta.get("spec", None)
        self.assertIsNotNone(spec)
        dim: int = cast(int, node.args[1]) if len(node.args) > 1 else 0
        outer_size = math.prod(spec.shape[:dim])
        self.assertEqual(
            outer_size,
            1,
            f"{node=} has wrong outer size: {outer_size=}, expected 1.",
        )
        inner_dim_elements = math.prod(spec.shape[dim + 1 :]) * spec.dtype.itemsize
        dim_offset = 0
        for arg in cast(list[torch.fx.Node], node.args[0]):
            arg_spec = arg.meta.get("spec", None)
            self.assertEqual(arg_spec.mem_id, spec.mem_id)
            self.assertEqual(
                arg_spec.mem_offset,
                spec.mem_offset + dim_offset * inner_dim_elements,
                f"{arg=} for node {node=} has wrong memory offset: {arg_spec.mem_offset=} {dim_offset=} for cat on {dim=}, but output has {spec.mem_offset=}",
            )
            dim_offset += arg_spec.shape[dim]

    def _verify_slice_nop_memory_alloc(self, node: torch.fx.Node) -> None:
        spec = node.meta.get("spec", None)
        self.assertIsNotNone(spec)
        dim: int = cast(int, node.args[1]) if len(node.args) > 1 else 0
        outer_size = math.prod(spec.shape[:dim])
        self.assertEqual(
            outer_size,
            1,
            f"{node=} has wrong outer size: {outer_size=}, expected 1.",
        )
        inner_dim_elements = math.prod(spec.shape[dim + 1 :]) * spec.dtype.itemsize
        start: int = (
            cast(int, node.args[2])
            if (len(node.args) > 2 and node.args[2] is not None)
            else 0
        )
        arg = cast(torch.fx.Node, node.args[0])
        arg_spec = arg.meta.get("spec", None)
        self.assertEqual(arg_spec.mem_id, spec.mem_id)
        self.assertEqual(
            spec.mem_offset,
            arg_spec.mem_offset + start * inner_dim_elements,
            f"{arg=} for node {node=} has wrong memory offset: {arg_spec.mem_offset=} {start=} for slice on {dim=}, but output has {spec.mem_offset=}",
        )

    def _verify_select_nop_memory_alloc(self, node: torch.fx.Node) -> None:
        spec = node.meta.get("spec", None)
        self.assertIsNotNone(spec)
        dim: int = cast(int, node.args[1]) if len(node.args) > 1 else 0
        outer_size = math.prod(spec.shape[:dim])
        self.assertEqual(
            outer_size,
            1,
            f"{node=} has wrong outer size: {outer_size=}, expected 1.",
        )
        inner_dim_elements = math.prod(spec.shape[dim:]) * spec.dtype.itemsize
        index: int = (
            cast(int, node.args[2])
            if (len(node.args) > 2 and node.args[2] is not None)
            else 0
        )
        arg = cast(torch.fx.Node, node.args[0])
        arg_spec = arg.meta.get("spec", None)
        self.assertEqual(arg_spec.mem_id, spec.mem_id)
        self.assertEqual(
            spec.mem_offset,
            arg_spec.mem_offset + index * inner_dim_elements,
            f"{arg=} for node {node=} has wrong memory offset: {arg_spec.mem_offset=} for select on {dim=} {index=}, "
            f"but output has {spec.mem_offset=}"
            f"{spec=} {arg_spec=}",
        )

    def verify_nop_memory_alloc(self, graph_module):
        for node in graph_module.graph.find_nodes(
            op="call_function", target=torch.ops.aten._cat_nop.out
        ):
            self._verify_cat_nop_memory_alloc(node)

        for node in graph_module.graph.find_nodes(
            op="call_function", target=torch.ops.aten._slice_copy_nop.Tensor_out
        ):
            self._verify_slice_nop_memory_alloc(node)

        for node in graph_module.graph.find_nodes(
            op="call_function", target=torch.ops.aten._select_copy_nop.int_out
        ):
            self._verify_select_nop_memory_alloc(node)

    def test_optimize_cat_on_placeholders(self):
        class Cat(torch.nn.Module):
            def forward(self, x, y):
                return torch.ops.aten.cat((x, y))

        x = torch.ones(3, 6)
        y = torch.ones(2, 6)
        # Optimizing cat ops is only at opt_level 2+, and requires the memory planning
        # pass to run:
        graph_module = (
            compiler.export_to_executorch_gen_etrecord(
                Cat(), (x, y), opt_level=2, mem_algo=1
            )
            .exported_program()
            .graph_module
        )
        logging.info(f"graph_module: {graph_module.print_readable(print_output=False)}")
        graph_module.graph.eliminate_dead_code()
        # Assert that cat op is optimized away
        self.assertEqual(count_node(graph_module, torch.ops.aten.cat.out), 0)
        # Assert that cat op is replaced by its nop version post optimization
        self.assertEqual(count_node(graph_module, torch.ops.aten._cat_nop.out), 1)
        self.verify_nop_memory_alloc(graph_module)

    def test_optimize_cat_outermost(self):
        class OptimizeCatFeasible1(torch.nn.Module):
            def forward(self, x, y):
                x1 = torch.add(x, 2.4, 3.1)
                y1 = torch.add(y, 1, 2)
                # Cat along the outermost dimension can be optimized away after
                # adding constraints on the locations of x1 and y1.
                return torch.ops.aten.cat((x1, y1))

        x = torch.ones(3, 6)
        y = torch.ones(2, 6)
        # Optimizing cat ops is only at opt_level 2+, and requires the memory planning
        # pass to run:
        graph_module = (
            compiler.export_to_executorch_gen_etrecord(
                OptimizeCatFeasible1(), (x, y), opt_level=2, mem_algo=1
            )
            .exported_program()
            .graph_module
        )
        graph_module.graph.eliminate_dead_code()
        # Assert that cat op is optimized away
        self.assertEqual(count_node(graph_module, torch.ops.aten.cat.out), 0)
        # Assert that cat op is replaced by its nop version post optimization
        self.assertEqual(count_node(graph_module, torch.ops.aten._cat_nop.out), 1)
        self.verify_nop_memory_alloc(graph_module)

    def test_optimize_cat_non_outermost(self):
        class OptimizeCatFeasible2(torch.nn.Module):
            def forward(self, x, y):
                x1 = torch.add(x, 2.4, 3.1)
                y1 = torch.add(y, 1, 2)
                # Cat along the outermost dimension can be optimized away after
                # adding constraints on the locations of x1 and y1.
                return torch.ops.aten.cat((x1, y1), 1)

        x = torch.ones(1, 3, 6)
        y = torch.ones(1, 2, 6)
        # Optimizing cat ops is only at opt_level 2+, and requires the memory planning
        # pass to run:
        graph_module = (
            compiler.export_to_executorch_gen_etrecord(
                OptimizeCatFeasible2(), (x, y), opt_level=2, mem_algo=1
            )
            .exported_program()
            .graph_module
        )
        graph_module.graph.eliminate_dead_code()
        # Assert that cat op is optimized away
        self.assertEqual(count_node(graph_module, torch.ops.aten.cat.out), 0)
        # Assert that cat op is replaced by its nop version post optimization
        self.assertEqual(count_node(graph_module, torch.ops.aten._cat_nop.out), 1)
        self.verify_nop_memory_alloc(graph_module)

    def test_no_optimize_cat_non_outermost(self):
        class OptimizeCatInfeasible1(torch.nn.Module):
            def forward(self, x, y):
                x1 = torch.add(x, 2.4, 3.1)
                y1 = torch.add(y, 1, 2)
                # Cat along the outermost dimension can be optimized away after
                # adding constraints on the locations of x1 and y1.
                return torch.ops.aten.cat((x1, y1), 1)

        x = torch.ones(2, 4, 5)
        y = torch.ones(2, 2, 5)
        # Optimizing cat ops is only at opt_level 2+, and requires the memory planning
        # pass to run
        graph_module = (
            compiler.export_to_executorch_gen_etrecord(
                OptimizeCatInfeasible1(), (x, y), opt_level=2, mem_algo=1
            )
            .exported_program()
            .graph_module
        )
        graph_module.graph.eliminate_dead_code()
        # Assert that cat op is not optimized away, since the concat is not
        # along the outermost dim
        self.assertEqual(count_node(graph_module, torch.ops.aten.cat.out), 1)
        self.verify_nop_memory_alloc(graph_module)

    def test_no_optimize_cat_non_outermost1(self):
        class OptimizeCatInfeasible2(torch.nn.Module):
            def forward(self, x, y):
                x1 = torch.add(x, 2.4, 3.1)
                y1 = torch.add(y, 1, 2)
                # Cat along the outermost dimension can be optimized away after
                # adding constraints on the locations of x1 and y1.
                return torch.ops.aten.cat((x1, y1), 0) + 2

        x = torch.ones(5, 5)
        y = torch.ones(3, 5)
        # Optimizing cat ops is only at opt_level 2+, and requires the memory planning
        # pass to run:
        graph_module = (
            compiler.export_to_executorch_gen_etrecord(
                OptimizeCatInfeasible2(), (x, y), opt_level=2, mem_algo=1
            )
            .exported_program()
            .graph_module
        )
        graph_module.graph.eliminate_dead_code()
        # Assert that cat op is not optimized away, since the concat relative
        # offsets are not multiple of 8 bytes, and the cat is not the output
        # of the graph.
        self.assertEqual(count_node(graph_module, torch.ops.aten.cat.out), 1)
        self.verify_nop_memory_alloc(graph_module)

    def test_optimize_cat_with_slice(self):
        class OptimizeCatSliceFeasible(torch.nn.Module):
            def forward(self, x):
                x1 = torch.add(x, 2.4, 3.1)
                x2 = torch.ops.aten.slice(x, 0, 0, 1)
                x3 = torch.ops.aten.cat((x1, x2))
                return torch.add(x3, x3)

        x = torch.randn(5, 6)
        # Compile, and set alloc_graph_input to False so that slice op is not
        # optimized away.
        # Optimizing cat ops is only at opt_level 2+, and requires the memory planning
        # pass to run:
        graph_module = (
            compiler.export_to_executorch_gen_etrecord(
                OptimizeCatSliceFeasible(),
                (x,),
                opt_level=2,
                mem_algo=1,
                alloc_graph_input=False,
            )
            .exported_program()
            .graph_module
        )
        graph_module.graph.eliminate_dead_code()
        # Assert that cat op is optimized away
        self.assertEqual(count_node(graph_module, torch.ops.aten._cat_nop.out), 1)
        self.verify_nop_memory_alloc(graph_module)

    def test_optimize_cat_with_slice_infeasible(self):
        class OptimizeCatSliceInfeasible(torch.nn.Module):
            def forward(self, x, y):
                x1 = torch.add(x, 2.4, 3.1)
                y1 = torch.add(y, 1, 2)
                y2 = torch.ops.aten.slice(y1, 0, 0, 1)
                # Cat can't be optimized away if any of the tensor (e.g., y1)
                # is slice_nop
                return torch.ops.aten.cat((y2, x1))

        x = torch.ones(3, 5)
        y = torch.ones(2, 5)
        # Optimizing cat ops is only at opt_level 2+, and requires the memory planning
        # pass to run:
        graph_module = (
            compiler.export_to_executorch_gen_etrecord(
                OptimizeCatSliceInfeasible(), (x, y), opt_level=2, mem_algo=1
            )
            .exported_program()
            .graph_module
        )
        graph_module.graph.eliminate_dead_code()
        # Assert that cat op is not optimized away
        self.assertEqual(count_node(graph_module, torch.ops.aten.cat.out), 1)
        self.verify_nop_memory_alloc(graph_module)

    def test_optimize_slice_Tensor(self):
        class SliceTensor(torch.nn.Module):
            def forward(self, x, y, z):
                x1 = torch.add(x, 2.4, 3.1)
                # This slice should always be optimized, since x1 is not placeholder
                # and the slice is along the outermost dim
                t1 = torch.ops.aten.slice(x1, 0, 1, 2)
                # This slice should not be optimized when alloc_graph_input=False,
                # since y is a placeholder node
                t2 = torch.ops.aten.slice(y, 0, 0, 1)
                # This slice should be always optimized, since the dims before
                # sliced dims are 1
                z1 = torch.add(z, 2.4, 3.1)
                t3 = torch.ops.aten.slice(z1, 1, 4, 5)
                return (t1 + t2) * t3

        x = torch.ones(3, 6)
        y = torch.ones(2, 6)
        z = torch.ones(1, 6)
        # Run the memory planning pass and get the graph module
        graph_module = (
            compiler.export_to_executorch_gen_etrecord(
                SliceTensor(),
                (x, y, z),
                opt_level=2,
                mem_algo=1,
                alloc_graph_input=False,
            )
            .exported_program()
            .graph_module
        )
        graph_module.graph.eliminate_dead_code()
        # Assert that t2 is not optimized away
        self.assertEqual(
            count_node(graph_module, torch.ops.aten.slice_copy.Tensor_out), 1
        )
        # Assert that t1 and t3 are optimized to slice_copy_nop veresion
        self.assertEqual(
            count_node(graph_module, torch.ops.aten._slice_copy_nop.Tensor_out), 2
        )
        # When we compile with alloc_graph_input=True, all the slice ops must
        # be optimized.
        # Optimizing cat ops is only at opt_level 2+, and requires the memory planning
        # pass to run:
        graph_module = (
            compiler.export_to_executorch_gen_etrecord(
                SliceTensor(),
                (x, y, z),
                opt_level=3,
                mem_algo=1,
                alloc_graph_input=True,
            )
            .exported_program()
            .graph_module
        )
        graph_module.graph.eliminate_dead_code()
        self.assertFalse(count_node(graph_module, torch.ops.aten.slice_copy.Tensor_out))
        self.assertEqual(
            count_node(graph_module, torch.ops.aten._slice_copy_nop.Tensor_out), 3
        )
        self.verify_nop_memory_alloc(graph_module)

    def test_optimize_select_Tensor(self):
        class SelectTensor(torch.nn.Module):
            def forward(self, x, y, z):
                x1 = torch.add(x, 2.4, 3.1)
                # This select should always be optimized, since x1 is not
                # placeholder, and the select is along the outermost dim
                t1 = torch.select_copy(x1, 0, 1)
                # This select should not be optimized if alloc_graph_input=False,
                # since y is a placeholder node.
                t2 = torch.select_copy(y, 0, 0)
                # This select should always be optimized, since the dims before
                # select dims are 1
                z1 = torch.add(z, 2.4, 3.1)
                t3 = torch.select(z1, 1, 4)
                return (t1 + t2) * t3

        x = torch.ones(3, 6)
        y = torch.ones(2, 6)
        z = torch.ones(1, 6)
        # Optimizing select ops is only at opt_level 2+, and requires the memory planning
        # pass to run:
        graph_module = (
            compiler.export_to_executorch_gen_etrecord(
                SelectTensor(),
                (x, y, z),
                opt_level=2,
                mem_algo=1,
                alloc_graph_input=False,
            )
            .exported_program()
            .graph_module
        )
        graph_module.graph.eliminate_dead_code()
        # Assert that t2 is not optimized away
        self.assertEqual(
            count_node(graph_module, torch.ops.aten.select_copy.int_out), 1
        )
        # Assert that t1 and t3 are optimized to select_copy_nop veresion
        self.assertEqual(
            count_node(graph_module, torch.ops.aten._select_copy_nop.int_out), 2
        )
        # When we compile with alloc_graph_input=True, all the select ops must
        # be optimized.
        # Optimizing select ops is only at opt_level 2+, and requires the memory planning
        # pass to run:
        graph_module = (
            compiler.export_to_executorch_gen_etrecord(
                SelectTensor(),
                (x, y, z),
                opt_level=3,
                mem_algo=1,
                alloc_graph_input=True,
            )
            .exported_program()
            .graph_module
        )
        graph_module.graph.eliminate_dead_code()
        self.assertEqual(
            count_node(graph_module, torch.ops.aten.select_copy.int_out), 0
        )
        self.assertEqual(
            count_node(graph_module, torch.ops.aten._select_copy_nop.int_out), 3
        )
        self.verify_nop_memory_alloc(graph_module)

    # TODO: Test fails due to memory planning
    @unittest.expectedFailure
    def test_optimize_cat_with_param(self):
        class CatWithPadding(torch.nn.Module):
            def __init__(self, padding_shape):
                super().__init__()
                zeros = torch.zeros(padding_shape)
                self.register_buffer("padding", zeros)

            def forward(self, x, y):
                x1 = torch.add(x, 2.4, 3.1)
                y1 = torch.add(y, 1, 2)
                # Cat along the outermost dimension cannot be optimized away
                # because padding is a param
                return torch.ops.aten.cat((x1, y1, self.padding))

        x = torch.ones(3, 5)
        y = torch.ones(2, 5)
        graph_module = (
            compiler.export_to_executorch_gen_etrecord(
                CatWithPadding((1, 5)), (x, y), opt_level=2
            )
            .exported_program()
            .graph_module
        )
        graph_module.graph.eliminate_dead_code()
        # Assert that cat op is not optimized away
        self.assertEqual(count_node(graph_module, exir_ops.edge.aten.cat.default), 1)
        self.verify_nop_memory_alloc(graph_module)

    def test_optimize_cat_then_slice_on_mutable_buffer(self):
        class CatWithPadding(torch.nn.Module):
            def __init__(self, padding_shape):
                super().__init__()
                zeros = torch.zeros(padding_shape)
                self.register_buffer("padding", zeros)

            def forward(self, x, y):
                x = x.view(3, 5)
                cat = torch.ops.aten.cat((x, self.padding.clone()))
                slice_copy = torch.ops.aten.slice(cat, dim=0, start=x.shape[0])
                self.padding.copy_(slice_copy)
                return cat.view(-1) + y

        x = torch.ones(15)
        y = torch.ones(1)
        et_prog_manager = compiler.export_to_executorch_gen_etrecord(
            CatWithPadding((1, 5)), (x, y), opt_level=3
        )
        graph_module = et_prog_manager.exported_program().graph_module
        logging.info(f"graph_module: {graph_module.print_readable(print_output=False)}")
        self.assertEqual(count_node(graph_module, torch.ops.aten.cat.out), 0)
        self.assertEqual(count_node(graph_module, torch.ops.aten._cat_nop.out), 1)
        self.verify_nop_memory_alloc(graph_module)

    def test_optimize_cat_with_view(self):
        class CatViewFeasible(torch.nn.Module):
            def forward(self, x, y):
                x1 = torch.add(x, 2.4, 3.1)
                x2 = x1.view((5, 3))
                y1 = torch.add(y, 2.4, 3.1)
                y2 = y1.view((2, 3))
                # Cat can be optimized away since x2 and y2 are not mem-equivalent
                return torch.ops.aten.cat((y2, x2))

        x = torch.ones(3, 5)
        y = torch.ones(3, 2)
        # Optimizing cat ops is only at opt_level 2+, and requires the memory planning
        # pass to run:
        graph_module = (
            compiler.export_to_executorch_gen_etrecord(
                CatViewFeasible(), (x, y), opt_level=2, mem_algo=1
            )
            .exported_program()
            .graph_module
        )
        graph_module.graph.eliminate_dead_code()
        # Assert that cat op is optimized away
        self.assertEqual(count_node(graph_module, torch.ops.aten._cat_nop.out), 1)
        self.assertEqual(count_node(graph_module, torch.ops.aten.cat.out), 0)
        self.verify_nop_memory_alloc(graph_module)

    def test_no_optimize_cat_with_repeated_args(self):
        class CatViewInfeasible(torch.nn.Module):
            def forward(self, x):
                x1 = torch.add(x, 2.4, 3.1)
                # Repeat will be decomposed into a cat. The cat cannot be optimized
                # away since all its args are mem-equivalent
                return torch.ops.aten.repeat(x1, [1, 2])

        x = torch.ones(3, 5)
        # Optimizing cat ops is only at opt_level 2+, and requires the memory planning
        # pass to run:
        graph_module = (
            compiler.export_to_executorch_gen_etrecord(
                CatViewInfeasible(), (x,), opt_level=2, mem_algo=1
            )
            .exported_program()
            .graph_module
        )
        graph_module.graph.eliminate_dead_code()
        # Assert that cat op is not optimized away
        self.assertEqual(count_node(graph_module, torch.ops.aten.cat.out), 1)
        self.assertEqual(count_node(graph_module, torch.ops.aten._cat_nop.out), 0)
        self.verify_nop_memory_alloc(graph_module)

    def test_no_optimize_cat_with_placeholder(self):
        class CatViewInfeasible(torch.nn.Module):
            def forward(self, x, y):
                # Repeat will be decomposed into a cat. The cat cannot be optimized
                # away since all its args are mem-equivalent
                return torch.cat((x, y), dim=0)

        x = torch.ones(3, 5)
        y = torch.ones(2, 5)
        # Optimizing cat ops is only at opt_level 2+, and requires the memory planning
        # pass to run:
        graph_module = (
            compiler.export_to_executorch_gen_etrecord(
                CatViewInfeasible(),
                (x, y),
                opt_level=2,
                mem_algo=1,
                alloc_graph_input=False,
            )
            .exported_program()
            .graph_module
        )
        graph_module.graph.eliminate_dead_code()
        # Assert that cat op is not optimized away
        self.assertEqual(count_node(graph_module, torch.ops.aten.cat.out), 1)
        self.assertEqual(count_node(graph_module, torch.ops.aten._cat_nop.out), 0)
        self.verify_nop_memory_alloc(graph_module)

    def test_no_optimize_cat(self) -> None:
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x) -> torch.Tensor:
                x0 = torch.slice_copy(x, dim=0, start=0, end=4)
                x0 = x0.view(-1)
                x1 = torch.slice_copy(x, dim=0, start=4, end=8)
                x1 = x1.view(-1)
                return torch.cat((x0, x1), dim=0)

        model = Model()
        inputs = (torch.randn(16, 16),)

        # Check that both view ops and slice copy are optimized.
        # We can't optimize cat op in this case.
        graph_module = (
            compiler.export_to_executorch_gen_etrecord(
                model, inputs, opt_level=3, alloc_graph_input=True
            )
            .exported_program()
            .graph_module
        )
        self.assertEqual(count_node(graph_module, torch.ops.aten._cat_nop.out), 0)
        self.assertEqual(
            count_node(graph_module, torch.ops.aten._slice_copy_nop.Tensor_out), 2
        )
        self.assertEqual(count_node(graph_module, memory.view), 2)
        self.verify_nop_memory_alloc(graph_module)

    def test_optimize_slice_copy(self) -> None:
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x) -> torch.Tensor:
                x0 = torch.slice_copy(x, dim=0, start=0, end=4)
                x0 = x0.view(-1)
                x1 = torch.slice_copy(x, dim=0, start=4, end=8)
                x1 = x1.view(-1)
                return torch.cat((x0, x1), dim=0)

        model = Model()
        inputs = (torch.randn(16, 16),)

        # Check that view ops and cat are optimized.
        # We can't optimize slice_copy op in this case.
        graph_module = (
            compiler.export_to_executorch_gen_etrecord(
                model, inputs, opt_level=3, alloc_graph_input=False
            )
            .exported_program()
            .graph_module
        )
        graph_module.print_readable()
        self.assertEqual(count_node(graph_module, torch.ops.aten._cat_nop.out), 1)
        self.assertEqual(
            count_node(graph_module, torch.ops.aten._slice_copy_nop.Tensor_out), 0
        )
        self.assertEqual(count_node(graph_module, memory.view), 2)
        self.verify_nop_memory_alloc(graph_module)

    def test_cat_then_cat(self) -> None:
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x) -> torch.Tensor:
                x1 = x + 1
                x2 = x1 + 1
                x3 = x2 + 1
                return torch.cat((torch.cat((x1, x2), dim=0), x3), dim=0)

        model = Model()
        inputs = (torch.randn(16, 16),)

        # Check that both the cat ops can be optimized.
        graph_module = (
            compiler.export_to_executorch_gen_etrecord(
                model, inputs, opt_level=3, alloc_graph_input=False
            )
            .exported_program()
            .graph_module
        )
        graph_module.print_readable()
        self.assertEqual(count_node(graph_module, torch.ops.aten._cat_nop.out), 2)
        self.assertEqual(count_node(graph_module, torch.ops.aten.cat.out), 0)
        self.verify_nop_memory_alloc(graph_module)

    def test_view_for_unallocated_output(self):
        class Model(torch.nn.Module):
            def __init__(self, padding_shape):
                super().__init__()

            def forward(self, x, y):
                x = x + 1
                # x_view will be a memory.view.
                x_view = torch.ops.aten.view_copy(x, [15])
                return x, x_view + y

        x = torch.ones(3, 5)
        y = torch.ones(15)
        # Check that memory planning passes for unallocated output `x`.
        graph_module = (
            compiler.export_to_executorch_gen_etrecord(
                Model((1, 5)), (x, y), opt_level=2, alloc_graph_output=False
            )
            .exported_program()
            .graph_module
        )
        self.assertEqual(count_node(graph_module, memory.view), 1)
        self.verify_nop_memory_alloc(graph_module)
