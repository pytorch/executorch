# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import logging
import math
import unittest
from typing import cast

import executorch.backends.cadence.aot.ops_registrations  # noqa
import torch
from executorch.backends.cadence.aot import compiler
from executorch.backends.cadence.aot.graph_builder import GraphBuilder
from executorch.backends.cadence.aot.memory_planning import (
    CadenceMemoryPlanning,
    find_peak_memory_usage,
)
from executorch.backends.cadence.aot.pass_utils import count_node
from executorch.backends.cadence.aot.utils import (
    get_default_memory_config,
    MemoryConfig,
)
from executorch.exir import memory
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.memory_planning import collect_specs_from_nodes
from executorch.exir.passes.spec_prop_pass import SpecPropPass
from executorch.exir.tests.models import MultiLayerPerceptron
from parameterized.parameterized import parameterized
from torch.fx import GraphModule


class TestMemPlanningPasses(unittest.TestCase):
    def test_calculate_peak_memory_pass(self) -> None:
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

        def calculate_aligned_num_bytes(num: int, alignment: int = 16) -> int:
            return math.ceil(num / alignment) * alignment

        # model 1
        batch_size, input_dim, hidden_dim, output_dim = 3, 16, 10, 20

        inputs = (torch.ones(batch_size, input_dim),)
        model = PeakMemoryTestModel(input_dim, hidden_dim, output_dim)

        exported_program = compiler.export_to_executorch_gen_etrecord(
            model, inputs
        ).exported_program()

        peak_usage, _ = find_peak_memory_usage(
            exported_program.graph_module,
            exported_program.graph_signature,
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

        exported_program = compiler.export_to_executorch_gen_etrecord(
            model, inputs
        ).exported_program()

        peak_usage, _ = find_peak_memory_usage(
            exported_program.graph_module,
            exported_program.graph_signature,
            mem_constraints=None,
            alloc_graph_input=True,
            alloc_graph_output=True,
        )

        expected_peak_usage = 2 * calculate_aligned_num_bytes(
            hidden_dim * batch_size * 4
        )  # Align data on a 16 byte boundary
        self.assertEqual(peak_usage, expected_peak_usage)

    def test_zero_memory_pass(self) -> None:
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
            executorch_prog.exported_program().graph_signature,
            alloc_graph_input=False,
            alloc_graph_output=False,
            mem_constraints=None,
        )
        self.assertEqual(peak_usage, 0)


class TestMemTransform(unittest.TestCase):
    def _verify_cat_nop_memory_alloc(self, node: torch.fx.Node) -> None:
        node_spec = node.meta.get("spec", None)
        self.assertIsNotNone(node_spec)
        dim: int = cast(int, node.kwargs["dim"]) if "dim" in node.kwargs else 0
        outer_size = math.prod(node_spec.shape[:dim])
        self.assertEqual(
            outer_size,
            1,
            f"{node=} has wrong outer size: {outer_size=}, expected 1.",
        )
        inner_dim_elements = (
            math.prod(node_spec.shape[dim + 1 :]) * node_spec.dtype.itemsize
        )
        dim_offset = 0
        for arg in cast(list[torch.fx.Node], node.args[0]):
            arg_spec = arg.meta.get("spec", None)
            self.assertEqual(arg_spec.mem_id, node_spec.mem_id)
            actual_offset = node_spec.mem_offset + dim_offset * inner_dim_elements
            self.assertEqual(
                arg_spec.mem_offset,
                actual_offset,
                f"{arg=} of node {node=} has wrong memory offset: expected {arg_spec.mem_offset=}, but got {actual_offset=} = {node_spec.mem_offset=} + {dim_offset=} * {inner_dim_elements=}",
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

    def verify_nop_memory_alloc(self, graph_module: torch.fx.GraphModule) -> None:
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

    # Initializes the nodes metadata and runs the GenerateMemoryViewConstraints,
    # GenerateSliceAndSelectNopConstraints, and GenerateCatNopConstraints passes.
    def run_memory_planning(
        self, original, opt_level=2, alloc_graph_input=True
    ) -> GraphModule:
        graph_module = SpecPropPass().call(original).graph_module
        return CadenceMemoryPlanning(
            get_default_memory_config(),
            opt_level=opt_level,
            mem_algo=1,  # greedy_by_size_for_offset_calculation_with_hierarchy
            alloc_graph_input=alloc_graph_input,
        )(graph_module).graph_module

    @parameterized.expand(
        [
            [
                [3, 6],  # x_shape
                [2, 6],  # y_shape
                0,  # concat dim
            ],
        ]
    )
    def test_optimize_cat_on_placeholders(self, x_shape, y_shape, concat_dim) -> None:
        concat_shape = [x_shape[concat_dim] + y_shape[concat_dim], x_shape[1]]
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.ones(*x_shape))
        y = builder.placeholder("y", torch.ones(*y_shape))
        pre_created_output = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=(concat_shape, 0.0),
            kwargs={"dtype": torch.float32},
        )
        graph_output = builder.call_operator(
            op=torch.ops.aten.cat.out,
            args=([x, y],),
            kwargs={"dim": concat_dim, "out": pre_created_output},
        )
        builder.output([graph_output])
        original = builder.get_graph_module()

        graph_module = self.run_memory_planning(original)
        graph_module.graph.eliminate_dead_code()
        # Assert that cat op is optimized away
        self.assertEqual(count_node(graph_module, torch.ops.aten.cat.out), 0)
        # Assert that cat op is replaced by its nop version post optimization
        self.assertEqual(count_node(graph_module, torch.ops.aten._cat_nop.out), 1)
        self.verify_nop_memory_alloc(graph_module)

    # Returns a GraphModule with the following structure:
    # "add_add_cat_model" : cat(x + 123, y + 456)
    # "add_add_cat_add_model": cat(x + 123, y + 456) + 789
    def get_graph_module(
        self, model_name, x_shape, y_shape, concated_shape, concat_dim
    ) -> GraphModule:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.ones(*x_shape, dtype=torch.float32))
        y = builder.placeholder("y", torch.ones(*y_shape, dtype=torch.float32))
        to_add_to_x = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=(x_shape, 123.0),
            kwargs={"dtype": torch.float32},
        )
        add_x = builder.call_operator(
            op=exir_ops.edge.aten.add.Tensor,
            args=(x, to_add_to_x),
        )
        to_add_to_y = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=(y_shape, 456.0),
            kwargs={"dtype": torch.float32},
        )
        add_y = builder.call_operator(
            op=exir_ops.edge.aten.add.Tensor,
            args=(y, to_add_to_y),
        )
        pre_created_output = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=(concated_shape, 0.0),
            kwargs={"dtype": torch.float32},
        )
        cat = builder.call_operator(
            op=torch.ops.aten.cat.out,
            args=([add_x, add_y],),
            kwargs={"dim": concat_dim, "out": pre_created_output},
        )
        if model_name == "add_add_cat_model":
            builder.output([cat])
            return builder.get_graph_module()

        if model_name == "add_add_cat_add_model":
            to_add_to_cat = builder.call_operator(
                op=exir_ops.edge.aten.full.default,
                args=(concated_shape, 789.0),
                kwargs={"dtype": torch.float32},
            )
            graph_output = builder.call_operator(
                op=exir_ops.edge.aten.add.Tensor,
                args=(cat, to_add_to_cat),
            )
            builder.output([graph_output])
            return builder.get_graph_module()

        raise ValueError(f"Unknown model name {model_name}")

    @parameterized.expand(
        [
            (
                "outermost",
                [3, 6],  # x_shape
                [2, 6],  # y_shape
                [5, 6],  # concated_shape
                0,  # concat dim
            ),
            (
                "non_outermost",
                [1, 3, 6],  # x_shape
                [1, 2, 6],  # y_shape
                [1, 5, 6],  # concated_shape
                1,  # concat dim
            ),
        ],
        name_func=lambda f, _, param: f"{f.__name__}_{param.args[0]}",
    )
    def test_cat_optimized(
        self, _, x_shape, y_shape, concated_shape, concat_dim
    ) -> None:
        original = self.get_graph_module(
            "add_add_cat_model", x_shape, y_shape, concated_shape, concat_dim
        )
        graph_module = self.run_memory_planning(original)
        graph_module.graph.eliminate_dead_code()
        # Assert that cat op is optimized away
        self.assertEqual(count_node(graph_module, torch.ops.aten.cat.out), 0)
        # Assert that cat op is replaced by its nop version post optimization
        self.assertEqual(count_node(graph_module, torch.ops.aten._cat_nop.out), 1)
        self.verify_nop_memory_alloc(graph_module)

    @parameterized.expand(
        [
            (
                "non_outermost",
                [2, 4, 5],  # x_shape
                [2, 2, 5],  # y_shape
                [2, 6, 5],  # concated_shape
                1,  # concat dim
            ),
        ],
        name_func=lambda f, _, param: f"{f.__name__}_{param.args[0]}",
    )
    def test_cat_not_optimized(
        self, _, x_shape, y_shape, concated_shape, concat_dim
    ) -> None:
        original = self.get_graph_module(
            "add_add_cat_model", x_shape, y_shape, concated_shape, concat_dim
        )
        graph_module = self.run_memory_planning(original)
        graph_module.graph.eliminate_dead_code()
        # Assert that cat op is not optimized away, since the concat is not along the outermost dim.
        # The first dimension is 2, but all dims before cat_dim should be == 1.
        self.assertEqual(count_node(graph_module, torch.ops.aten.cat.out), 1)
        self.verify_nop_memory_alloc(graph_module)

    @parameterized.expand(
        [
            (
                "aligned",
                [5, 8],  # x_shape
                [3, 8],  # y_shape
                [8, 8],  # concated_shape
                0,  # concat dim
                0,  # expected cat nodes
            ),
            (
                "unaligned",  # 5 * 5 * 4 % 8 != 0
                [5, 5],  # x_shape
                [3, 5],  # y_shape
                [8, 5],  # concated_shape
                0,  # concat dim
                1,  # expected cat nodes
            ),
        ],
        name_func=lambda f, _, param: f"{f.__name__}_{param.args[0]}",
    )
    def test_cat_not_graph_output(
        self, _, x_shape, y_shape, concated_shape, concat_dim, expected_cat_nodes
    ) -> None:
        original = self.get_graph_module(
            "add_add_cat_add_model", x_shape, y_shape, concated_shape, concat_dim
        )
        graph_module = self.run_memory_planning(original)
        graph_module.graph.eliminate_dead_code()

        # Assert that cat op is optimized away only if its arguments offsets are multiple of 8 bytes.
        self.assertEqual(
            count_node(graph_module, torch.ops.aten.cat.out), expected_cat_nodes
        )
        self.verify_nop_memory_alloc(graph_module)

    def test_optimize_cat_with_slice(self) -> None:
        x_shape = [5, 6]
        concated_shape = [6, 6]
        concat_dim = 0
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.ones(*x_shape, dtype=torch.float32))
        to_add_to_x = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=(x_shape, 123.0),
            kwargs={"dtype": torch.float32},
        )
        add_x = builder.call_operator(
            op=exir_ops.edge.aten.add.Tensor,
            args=(x, to_add_to_x),
        )
        slice_x = builder.call_operator(
            op=exir_ops.edge.aten.slice.Tensor,
            args=(
                x,
                0,  # dim
                0,  # start
                1,  # end
                1,  # step
            ),
        )
        pre_created_output = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=(concated_shape, 0.0),
            kwargs={"dtype": torch.float32},
        )
        cat = builder.call_operator(
            op=torch.ops.aten.cat.out,
            args=([add_x, slice_x],),
            kwargs={"dim": concat_dim, "out": pre_created_output},
        )
        graph_output = builder.call_operator(
            op=exir_ops.edge.aten.add.Tensor,
            args=(cat, cat),
        )
        builder.output([graph_output])
        original = builder.get_graph_module()

        graph_module = self.run_memory_planning(original, alloc_graph_input=False)
        graph_module.graph.eliminate_dead_code()

        # Assert that cat op is optimized away.
        self.assertEqual(count_node(graph_module, torch.ops.aten.cat.out), 0)
        # Assert that cat op is replaced by its nop version post optimization.
        self.assertEqual(count_node(graph_module, torch.ops.aten._cat_nop.out), 1)
        # Assert that slice op was not optimized away.
        self.assertEqual(count_node(graph_module, exir_ops.edge.aten.slice.Tensor), 1)
        self.verify_nop_memory_alloc(graph_module)

    def test_optimize_cat_with_slice_infeasible(self) -> None:
        x_shape = [5, 6]
        y_shape = [3, 6]
        concated_shape = [8, 6]
        concat_dim = 0
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.ones(*x_shape, dtype=torch.float32))
        y = builder.placeholder("y", torch.ones(*y_shape, dtype=torch.float32))
        to_add_to_x = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=(x_shape, 123.0),
            kwargs={"dtype": torch.float32},
        )
        add_x = builder.call_operator(
            op=exir_ops.edge.aten.add.Tensor,
            args=(x, to_add_to_x),
        )
        to_add_to_y = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=(y_shape, 123.0),
            kwargs={"dtype": torch.float32},
        )
        add_y = builder.call_operator(
            op=exir_ops.edge.aten.add.Tensor,
            args=(y, to_add_to_y),
        )
        slice_out = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=(y_shape, 0.0),
            kwargs={"dtype": torch.float32},
        )
        slice_y = builder.call_operator(
            op=torch.ops.aten.slice_copy.Tensor_out,
            args=(
                add_y,
                0,  # dim
                0,  # start
                1,  # end
                1,  # step
            ),
            kwargs={"out": slice_out},
        )
        pre_created_output = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=(concated_shape, 0.0),
            kwargs={"dtype": torch.float32},
        )
        cat = builder.call_operator(
            op=torch.ops.aten.cat.out,
            args=([slice_y, add_x],),
            kwargs={"dim": concat_dim, "out": pre_created_output},
        )
        builder.output([cat])
        original = builder.get_graph_module()
        graph_module = self.run_memory_planning(original, alloc_graph_input=False)
        graph_module.graph.eliminate_dead_code()
        # # Assert that slice op is optimized away.
        self.assertEqual(
            count_node(graph_module, torch.ops.aten._slice_copy_nop.Tensor_out), 1
        )
        # # Assert that cat op is not optimized away
        self.assertEqual(count_node(graph_module, torch.ops.aten.cat.out), 1)
        self.verify_nop_memory_alloc(graph_module)

    def test_optimize_slice_outermost(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.ones(3, 6, dtype=torch.float32))
        to_add_to_x = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=([3, 6], 123.0),
            kwargs={"dtype": torch.float32},
        )
        add_x = builder.call_operator(
            op=exir_ops.edge.aten.add.Tensor,
            args=(x, to_add_to_x),
        )
        slice_out = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=([1, 6], 0.0),
            kwargs={"dtype": torch.float32},
        )
        # This slice should always be optimized, since add_x is not placeholder
        # and the slice is along the outermost dim
        slice_result = builder.call_operator(
            op=torch.ops.aten.slice_copy.Tensor_out,
            args=(
                add_x,
                0,  # dim
                1,  # start
                2,  # end
                1,  # step
            ),
            kwargs={"out": slice_out},
        )
        builder.output([slice_result])
        original = builder.get_graph_module()
        graph_module = self.run_memory_planning(original, alloc_graph_input=False)
        graph_module.graph.eliminate_dead_code()
        self.assertEqual(
            count_node(graph_module, torch.ops.aten._slice_copy_nop.Tensor_out), 1
        )
        self.verify_nop_memory_alloc(graph_module)

    def test_optimize_slice_non_outermost(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.ones(1, 6, dtype=torch.float32))
        to_add_to_x = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=([1, 6], 123.0),
            kwargs={"dtype": torch.float32},
        )
        add_x = builder.call_operator(
            op=exir_ops.edge.aten.add.Tensor,
            args=(x, to_add_to_x),
        )
        slice_out = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=([1, 2], 0.0),
            kwargs={"dtype": torch.float32},
        )
        # This slice should be always optimized, since the dims before
        # sliced dims are 1.
        slice_result = builder.call_operator(
            op=torch.ops.aten.slice_copy.Tensor_out,
            args=(
                add_x,
                1,  # dim
                4,  # start
                6,  # end
                1,  # step
            ),
            kwargs={"out": slice_out},
        )
        builder.output([slice_result])
        original = builder.get_graph_module()
        graph_module = self.run_memory_planning(original, alloc_graph_input=False)
        graph_module.graph.eliminate_dead_code()
        self.assertEqual(
            count_node(graph_module, torch.ops.aten._slice_copy_nop.Tensor_out), 1
        )
        self.verify_nop_memory_alloc(graph_module)

    def test_optimize_slice_depending_on_opt_level(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.ones(2, 6, dtype=torch.float32))
        slice_out = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=([1, 6], 0.0),
            kwargs={"dtype": torch.float32},
        )
        # This slice should not be optimized when alloc_graph_input=False,
        # since y is a placeholder node
        slice_result = builder.call_operator(
            op=torch.ops.aten.slice_copy.Tensor_out,
            args=(
                x,
                0,  # dim
                0,  # start
                1,  # end
                1,  # step
            ),
            kwargs={"out": slice_out},
        )
        builder.output([slice_result])
        original = builder.get_graph_module()
        graph_module = self.run_memory_planning(
            original, opt_level=2, alloc_graph_input=False
        )
        graph_module.graph.eliminate_dead_code()
        self.assertEqual(
            count_node(graph_module, torch.ops.aten.slice_copy.Tensor_out), 1
        )
        self.verify_nop_memory_alloc(graph_module)

        # When we compile with alloc_graph_input=True, all the slice ops must
        # be optimized, which is available only at opt_level 2+.
        graph_module = self.run_memory_planning(
            original, opt_level=3, alloc_graph_input=True
        )
        graph_module.graph.eliminate_dead_code()
        self.assertEqual(
            count_node(graph_module, torch.ops.aten._slice_copy_nop.Tensor_out), 1
        )
        self.verify_nop_memory_alloc(graph_module)

    def test_optimize_select_outermost(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.ones(3, 6, dtype=torch.float32))
        to_add_to_x = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=([3, 6], 123.0),
            kwargs={"dtype": torch.float32},
        )
        add_x = builder.call_operator(
            op=exir_ops.edge.aten.add.Tensor,
            args=(x, to_add_to_x),
        )
        slice_out = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=([1, 6], 0.0),
            kwargs={"dtype": torch.float32},
        )
        # This select should always be optimized, since add_x is not placeholder
        # and the select is along the outermost dim
        slice_result = builder.call_operator(
            op=torch.ops.aten.select_copy.int_out,
            args=(
                add_x,
                0,  # dim
                1,  # index
            ),
            kwargs={"out": slice_out},
        )
        builder.output([slice_result])
        original = builder.get_graph_module()
        graph_module = self.run_memory_planning(original, alloc_graph_input=False)
        graph_module.graph.eliminate_dead_code()
        self.assertEqual(
            count_node(graph_module, torch.ops.aten._select_copy_nop.int_out), 1
        )
        self.verify_nop_memory_alloc(graph_module)

    def test_optimize_select_non_outermost(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.ones(1, 6, dtype=torch.float32))
        to_add_to_x = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=([1, 6], 123.0),
            kwargs={"dtype": torch.float32},
        )
        add_x = builder.call_operator(
            op=exir_ops.edge.aten.add.Tensor,
            args=(x, to_add_to_x),
        )
        slice_out = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=([1, 2], 0.0),
            kwargs={"dtype": torch.float32},
        )
        # This select should always be optimized, since the dims before
        # select dims are 1
        slice_result = builder.call_operator(
            op=torch.ops.aten.select_copy.int_out,
            args=(
                add_x,
                1,  # dim
                4,  # index
            ),
            kwargs={"out": slice_out},
        )
        builder.output([slice_result])
        original = builder.get_graph_module()
        graph_module = self.run_memory_planning(original, alloc_graph_input=False)
        graph_module.graph.eliminate_dead_code()
        self.assertEqual(
            count_node(graph_module, torch.ops.aten._select_copy_nop.int_out), 1
        )
        self.verify_nop_memory_alloc(graph_module)

    def test_optimize_select_depending_on_opt_level(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.ones(2, 6, dtype=torch.float32))
        slice_out = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=([1, 6], 0.0),
            kwargs={"dtype": torch.float32},
        )
        # This select should not be optimized if alloc_graph_input=False,
        # since y is a placeholder node.
        slice_result = builder.call_operator(
            op=torch.ops.aten.select_copy.int_out,
            args=(
                x,
                0,  # dim
                0,  # index
            ),
            kwargs={"out": slice_out},
        )
        builder.output([slice_result])
        original = builder.get_graph_module()
        graph_module = self.run_memory_planning(
            original, opt_level=2, alloc_graph_input=False
        )
        graph_module.graph.eliminate_dead_code()
        self.assertEqual(
            count_node(graph_module, torch.ops.aten.select_copy.int_out), 1
        )
        self.verify_nop_memory_alloc(graph_module)

        # When we compile with alloc_graph_input=True, all the slice ops must
        # be optimized, which is available only at opt_level 2+.
        graph_module = self.run_memory_planning(
            original, opt_level=3, alloc_graph_input=True
        )
        graph_module.graph.eliminate_dead_code()
        self.assertEqual(
            count_node(graph_module, torch.ops.aten._select_copy_nop.int_out), 1
        )
        self.verify_nop_memory_alloc(graph_module)

    # TODO: Test fails due to memory planning
    @unittest.expectedFailure
    def test_optimize_cat_with_param(self) -> None:
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

    def test_optimize_cat_then_slice_on_mutable_buffer(self) -> None:
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

    def test_optimize_cat_with_view(self) -> None:
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

    def test_no_optimize_cat_with_repeated_args(self) -> None:
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

    def test_no_optimize_cat_with_placeholder(self) -> None:
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
        self.assertEqual(count_node(graph_module, torch.ops.aten._cat_nop.out), 2)
        self.assertEqual(count_node(graph_module, torch.ops.aten.cat.out), 0)
        self.verify_nop_memory_alloc(graph_module)

    def test_view_for_unallocated_output(self) -> None:
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

    def test_start_alignment_constraints(self) -> None:
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x: torch.Tensor, y: torch.Tensor):
                add_0 = torch.add(x, y)
                add_1 = torch.add(x, add_0)
                add_2 = torch.add(add_0, add_1)
                add_3 = torch.add(add_1, add_2)
                return add_3

        model = Model()
        inputs = (torch.randn(4, 17), torch.randn(4, 17))
        for mem_algo in range(0, 2):
            graph_module = (
                compiler.export_to_executorch_gen_etrecord(
                    model,
                    inputs,
                    opt_level=1,
                    mem_algo=mem_algo,
                    alloc_graph_input=False,
                    alloc_graph_output=False,
                    memory_config=MemoryConfig(
                        memory_sizes=[0x1000000000], memory_alignments=[37]
                    ),
                )
                .exported_program()
                .graph_module
            )
            # Assert that all memory allocations are aligned to 32B start address
            for spec in collect_specs_from_nodes(
                graph_module.graph.nodes,
                ignore_graph_input=True,
                ignore_graph_output=True,
            ):
                if spec and spec.mem_offset:
                    self.assertEqual(spec.mem_offset % 37, 0)
