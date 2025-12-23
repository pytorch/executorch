# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import math
import unittest
from typing import cast, List, Optional, Sequence

import executorch.backends.cadence.aot.ops_registrations  # noqa
import torch
from executorch.backends.cadence.aot import compiler
from executorch.backends.cadence.aot.graph_builder import GraphBuilder
from executorch.backends.cadence.aot.memory_constraints import (
    ConstraintsGenPass,
    MemConstraints,
)
from executorch.backends.cadence.aot.memory_planning import (
    CadenceMemoryPlanning,
    find_peak_memory_usage,
    PositionBasedGreedyWithHierarchy,
)
from executorch.backends.cadence.aot.memory_planning_algo import (
    MemoryPlanningAlgo,
    MemoryPlanningState,
)
from executorch.backends.cadence.aot.pass_utils import (
    CadencePassAttribute,
    count_node,
    register_cadence_pass,
)
from executorch.backends.cadence.aot.program_builder import ProgramBuilder
from executorch.backends.cadence.aot.typing_stubs import expand
from executorch.backends.cadence.aot.utils import (
    get_default_memory_config,
    MemoryConfig,
)
from executorch.exir import EdgeProgramManager, ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.memory_planning import (
    collect_specs_from_nodes,
    update_all_tensors_lifetime,
)
from executorch.exir.pass_base import PassBase, PassResult
from executorch.exir.passes.spec_prop_pass import SpecPropPass
from executorch.exir.tests.models import MultiLayerPerceptron
from parameterized import parameterized
from torch.fx import GraphModule


class TestMemPlanningPasses(unittest.TestCase):
    def setUp(self) -> None:
        logging.basicConfig(
            format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d:%H:%M:%S",
            level=logging.getLevelName(logging.INFO),
            force=True,
        )
        return super().setUp()

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
        self,
        original: GraphModule,
        opt_level: int = 2,
        mem_algo: int = 1,  # greedy_by_size_for_offset_calculation_with_hierarchy
        alloc_graph_input: bool = True,
        alloc_graph_output: bool = True,
        memory_config: Optional[MemoryConfig] = None,
        additional_constraint_gen_passes: Optional[Sequence[ConstraintsGenPass]] = None,
    ) -> GraphModule:
        if memory_config is None:
            memory_config = get_default_memory_config()
        graph_module = SpecPropPass().call(original).graph_module
        return CadenceMemoryPlanning(
            memory_config,
            opt_level=opt_level,
            mem_algo=mem_algo,
            alloc_graph_input=alloc_graph_input,
            alloc_graph_output=alloc_graph_output,
            additional_constraint_gen_passes=additional_constraint_gen_passes,
        )(graph_module).graph_module

    @expand(
        [
            [
                [3, 6],  # x_shape
                [2, 6],  # y_shape
                0,  # concat dim
                False,  # alloc_graph_input
            ],
            [
                [3, 6],  # x_shape
                [2, 6],  # y_shape
                0,  # concat dim
                True,  # alloc_graph_input
            ],
        ]
    )
    def test_optimize_cat_on_placeholders(
        self,
        x_shape: List[int],
        y_shape: List[int],
        concat_dim: int,
        alloc_graph_input: bool,
    ) -> None:
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

        graph_module = self.run_memory_planning(
            original, alloc_graph_input=alloc_graph_input
        )
        graph_module.graph.eliminate_dead_code()
        if alloc_graph_input:
            self.assertEqual(count_node(graph_module, torch.ops.aten.cat.out), 0)
            self.assertEqual(count_node(graph_module, torch.ops.aten._cat_nop.out), 1)
        else:
            self.assertEqual(count_node(graph_module, torch.ops.aten.cat.out), 1)
            self.assertEqual(count_node(graph_module, torch.ops.aten._cat_nop.out), 0)
        self.verify_nop_memory_alloc(graph_module)

    # Returns a GraphModule with the following structure:
    # "add_add_cat_model" : cat(x + 123, y + 456)
    # "add_add_cat_add_model": cat(x + 123, y + 456) + 789
    def get_graph_module(
        self,
        model_name: str,
        x_shape: List[int],
        y_shape: List[int],
        concated_shape: List[int],
        concat_dim: int,
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

    @expand(
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
    )
    def test_cat_optimized(
        self,
        _,
        x_shape: List[int],
        y_shape: List[int],
        concated_shape: List[int],
        concat_dim: int,
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

    @expand(
        [
            (
                "non_outermost",
                [2, 4, 5],  # x_shape
                [2, 2, 5],  # y_shape
                [2, 6, 5],  # concated_shape
                1,  # concat dim
            ),
        ],
    )
    def test_cat_not_optimized(
        self,
        _,
        x_shape: List[int],
        y_shape: List[int],
        concated_shape: List[int],
        concat_dim: int,
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

    @expand(
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
    )
    def test_cat_not_graph_output(
        self,
        _,
        x_shape: List[int],
        y_shape: List[int],
        concated_shape: List[int],
        concat_dim: int,
        expected_cat_nodes: int,
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

    @expand(
        [
            (True,),  # alloc_graph_input
            (False,),  # alloc_graph_input
        ],
    )
    def test_optimize_cat_with_slice_infeasible(self, alloc_graph_input: bool) -> None:
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
        graph_module = self.run_memory_planning(
            original, opt_level=3, alloc_graph_input=alloc_graph_input
        )
        graph_module.graph.eliminate_dead_code()
        if alloc_graph_input:
            self.assertEqual(count_node(graph_module, torch.ops.aten._cat_nop.out), 0)
            self.assertEqual(
                count_node(graph_module, torch.ops.aten._slice_copy_nop.Tensor_out), 1
            )
        else:
            self.assertEqual(
                count_node(graph_module, torch.ops.aten._slice_copy_nop.Tensor_out), 1
            )
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

    def test_optimize_cat_then_slice_on_mutable_buffer(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.ones(3, 6, dtype=torch.float32))
        y = builder.placeholder("y", torch.ones(1, 6, dtype=torch.float32))
        pre_created_output = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=([4, 6], 0.0),
            kwargs={"dtype": torch.float32},
        )
        cat = builder.call_operator(
            op=torch.ops.aten.cat.out,
            args=([x, y],),
            kwargs={"dim": 0, "out": pre_created_output},
        )
        slice_out = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=([1, 6], 0.0),
            kwargs={"dtype": torch.float32},
        )
        slice_result = builder.call_operator(
            op=torch.ops.aten.slice_copy.Tensor_out,
            args=(
                cat,
                0,  # dim
                3,  # start
                4,  # end
                1,  # step
            ),
            kwargs={"out": slice_out},
        )
        builder.output([slice_result])
        original = builder.get_graph_module()
        graph_module = self.run_memory_planning(original, opt_level=3)
        graph_module.graph.eliminate_dead_code()
        self.assertEqual(count_node(graph_module, torch.ops.aten.cat.out), 0)
        self.assertEqual(count_node(graph_module, torch.ops.aten._cat_nop.out), 1)
        self.verify_nop_memory_alloc(graph_module)

    def test_cat_then_cat(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.ones(16, 16, dtype=torch.float32))
        to_add_to_x = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=([16, 16], 1.0),
            kwargs={"dtype": torch.float32},
        )
        x1 = builder.call_operator(
            op=exir_ops.edge.aten.add.Tensor,
            args=(x, to_add_to_x),
        )
        x2 = builder.call_operator(
            op=exir_ops.edge.aten.add.Tensor,
            args=(x1, to_add_to_x),
        )
        x3 = builder.call_operator(
            op=exir_ops.edge.aten.add.Tensor,
            args=(x2, to_add_to_x),
        )
        pre_created_output1 = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=([32, 16], 0.0),
            kwargs={"dtype": torch.float32},
        )
        cat1 = builder.call_operator(
            op=torch.ops.aten.cat.out,
            args=([x1, x2],),
            kwargs={"dim": 0, "out": pre_created_output1},
        )
        pre_created_output2 = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=([32, 16], 0.0),
            kwargs={"dtype": torch.float32},
        )
        cat2 = builder.call_operator(
            op=torch.ops.aten.cat.out,
            args=([cat1, x3],),
            kwargs={"dim": 0, "out": pre_created_output2},
        )
        builder.output([cat2])
        original = builder.get_graph_module()
        graph_module = self.run_memory_planning(
            original, opt_level=3, alloc_graph_input=False
        )

        self.assertEqual(count_node(graph_module, torch.ops.aten._cat_nop.out), 2)
        self.assertEqual(count_node(graph_module, torch.ops.aten.cat.out), 0)
        self.verify_nop_memory_alloc(graph_module)

    def test_cat_with_duplicate_input_tensor(self) -> None:
        """
        Test that cat is NOT optimized when the same tensor appears multiple
        times in the cat input list. This is because we cannot place the same
        tensor at multiple different offsets relative to the output.
        """
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
        pre_created_output = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=([6, 6], 0.0),
            kwargs={"dtype": torch.float32},
        )
        # Same tensor (add_x) appears twice in the cat inputs
        cat = builder.call_operator(
            op=torch.ops.aten.cat.out,
            args=([add_x, add_x],),
            kwargs={"dim": 0, "out": pre_created_output},
        )
        builder.output([cat])
        original = builder.get_graph_module()
        graph_module = self.run_memory_planning(original)
        graph_module.graph.eliminate_dead_code()

        # Assert that cat op is NOT optimized away since the same tensor
        # appears multiple times in the input list
        self.assertEqual(count_node(graph_module, torch.ops.aten.cat.out), 1)
        self.assertEqual(count_node(graph_module, torch.ops.aten._cat_nop.out), 0)
        self.verify_nop_memory_alloc(graph_module)

    def test_cat_with_tensor_having_existing_constraint(self) -> None:
        """
        Test that the second cat is NOT optimized when a tensor already has a
        relative placement constraint from a previous cat operation.
        """
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.ones(8, 8, dtype=torch.float32))
        to_add = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=([8, 8], 1.0),
            kwargs={"dtype": torch.float32},
        )
        x1 = builder.call_operator(
            op=exir_ops.edge.aten.add.Tensor,
            args=(x, to_add),
        )
        x2 = builder.call_operator(
            op=exir_ops.edge.aten.add.Tensor,
            args=(x1, to_add),
        )
        x3 = builder.call_operator(
            op=exir_ops.edge.aten.add.Tensor,
            args=(x2, to_add),
        )
        # First cat: cat(x1, x2) - this will give x1 and x2 relative placement constraints
        pre_created_output1 = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=([16, 8], 0.0),
            kwargs={"dtype": torch.float32},
        )
        cat1 = builder.call_operator(
            op=torch.ops.aten.cat.out,
            args=([x1, x2],),
            kwargs={"dim": 0, "out": pre_created_output1},
        )
        # Second cat: cat(x2, x3) - x2 already has a constraint from cat1,
        # so this cat cannot be optimized
        pre_created_output2 = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=([16, 8], 0.0),
            kwargs={"dtype": torch.float32},
        )
        cat2 = builder.call_operator(
            op=torch.ops.aten.cat.out,
            args=([x2, x3],),
            kwargs={"dim": 0, "out": pre_created_output2},
        )
        # Use both cat results to keep them alive
        graph_output = builder.call_operator(
            op=exir_ops.edge.aten.add.Tensor,
            args=(cat1, cat2),
        )
        builder.output([graph_output])
        original = builder.get_graph_module()
        graph_module = self.run_memory_planning(
            original, opt_level=3, alloc_graph_input=False
        )
        graph_module.graph.eliminate_dead_code()

        # The first cat should be optimized to _cat_nop, but the second cat
        # cannot be optimized because x2 already has a relative placement constraint
        self.assertEqual(count_node(graph_module, torch.ops.aten._cat_nop.out), 1)
        self.assertEqual(count_node(graph_module, torch.ops.aten.cat.out), 1)
        self.verify_nop_memory_alloc(graph_module)

    def test_view_for_unallocated_output(self) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.ones(3, 5, dtype=torch.float32))
        y = builder.placeholder("y", torch.ones(15, dtype=torch.float32))
        to_add_to_x = builder.call_operator(
            op=exir_ops.edge.aten.full.default,
            args=([3, 5], 1.0),
            kwargs={"dtype": torch.float32},
        )
        add_x = builder.call_operator(
            op=exir_ops.edge.aten.add.Tensor,
            args=(x, to_add_to_x),
        )
        add_x_view = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default,
            args=(add_x, [15]),
        )
        add_x_y = builder.call_operator(
            op=exir_ops.edge.aten.add.Tensor,
            args=(add_x_view, y),
        )
        builder.output([add_x, add_x_y])
        original = builder.get_graph_module()
        graph_module = self.run_memory_planning(
            original, opt_level=2, alloc_graph_output=False
        )
        self.assertEqual(
            count_node(graph_module, exir_ops.edge.aten.view_copy.default), 1
        )
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

    @parameterized.expand([0, 1])
    def test_block_mem_id(self, mem_algo: int) -> None:
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(16))
        add = builder.call_operator(
            op=torch.ops.aten.add.Scalar,
            args=(x, 2.0),
        )
        mul = builder.call_operator(
            op=torch.ops.aten.mul.Scalar,
            args=(add, 2.0),
        )
        builder.output([mul])
        original = builder.get_graph_module()

        dummy_memory_config = MemoryConfig([1024, 1024, 1024, 1024])

        add_scalar_block_mem_ids = [2, 3]
        mul_scalar_block_mem_ids = [1, 3]

        @register_cadence_pass(CadencePassAttribute(opt_level=0))
        class DummyMemIdBlockConstraintGen(PassBase):
            """Blocks placement based on op type.
            add: blocks 2, 3
            mul: blocks 1, 3
            """

            def __init__(self, memory_constraints: MemConstraints):
                self.memory_constraints = memory_constraints

            def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
                for node in graph_module.graph.find_nodes(
                    op="call_function", target=torch.ops.aten.add.Scalar
                ):
                    spec = node.meta["spec"]
                    logging.error(f"add node: {node} {id(spec)=}")
                    for mem_id in add_scalar_block_mem_ids:
                        self.memory_constraints.add_mem_id_to_blocklist(spec, mem_id)
                for node in graph_module.graph.find_nodes(
                    op="call_function", target=torch.ops.aten.mul.Scalar
                ):
                    spec = node.meta["spec"]
                    logging.error(f"mul node: {node} {id(spec)=}")
                    for mem_id in mul_scalar_block_mem_ids:
                        self.memory_constraints.add_mem_id_to_blocklist(spec, mem_id)

        graph_module = self.run_memory_planning(
            original,
            mem_algo=mem_algo,
            memory_config=dummy_memory_config,
            additional_constraint_gen_passes=[DummyMemIdBlockConstraintGen],
        )
        for node in graph_module.graph.find_nodes(
            op="call_function", target=torch.ops.aten.add.Scalar
        ):
            spec = node.meta["spec"]
            self.assertIsNotNone(spec.mem_id)
            self.assertNotIn(spec.mem_id, add_scalar_block_mem_ids)
        for node in graph_module.graph.find_nodes(
            op="call_function", target=torch.ops.aten.mul.Scalar
        ):
            spec = node.meta["spec"]
            self.assertIsNotNone(spec.mem_id)
            self.assertNotIn(spec.mem_id, mul_scalar_block_mem_ids)


class TestConstraintsBase(unittest.TestCase):
    def get_view_then_add_graph(self) -> EdgeProgramManager:
        builder = ProgramBuilder()
        x = builder.placeholder("x", torch.ones(3, 5, dtype=torch.float32))
        y = builder.placeholder("y", torch.ones(2, 15, dtype=torch.float32))
        x_reshape = builder.call_operator(
            op=exir_ops.edge.aten.view_copy.default,
            args=(x, [15]),
        )
        add_x_y = builder.call_operator(
            op=exir_ops.edge.aten.add.Tensor,
            args=(x_reshape, y),
        )
        builder.output([add_x_y])
        edge_program = builder.get_edge_program()
        edge_program = edge_program.transform([SpecPropPass()])
        return edge_program

    @staticmethod
    def get_aligned(num: int) -> int:
        return ((num + 16 - 1) // 16) * 16

    def _run_mem_planning(
        self,
        program: ExportedProgram,
        memory_planning: MemoryPlanningAlgo,
        state: MemoryPlanningState,
        placement_constraints: MemConstraints,
    ) -> None:
        gm = program.graph_module
        graph_signature = program.graph_signature
        # Difficult to just filter the list of specs returned by this due to
        # how we flag trainable weights.
        _ = update_all_tensors_lifetime(gm, graph_signature)

        # Filter specs based on alloc_graph_input and alloc_graph_output
        specs = set(
            collect_specs_from_nodes(
                gm.graph.nodes,
                graph_signature,
                do_assertion=False,
                ignore_graph_input=False,
                ignore_graph_output=False,
                ignore_mutable_buffers=False,
            )
        )
        memory_planning.plan_with_constraints(
            specs,
            gm,
            # pyre-ignore[6]
            None,
            state,
            placement_constraints,
        )


class TestAbsolutePlacementConstraint(TestConstraintsBase):

    def test_manually_planned_specs(self) -> None:
        edge_program = self.get_view_then_add_graph()
        x, y, x_view, add, _ = edge_program.exported_program().graph_module.graph.nodes

        # Create constraints for all nodes.
        memory_config = MemoryConfig([1000, 10000])
        mem_planning = PositionBasedGreedyWithHierarchy(memory_config)
        state = MemoryPlanningState(memory_config=memory_config)
        placement_constraints = MemConstraints()
        x_offset = 8000
        y_offset = 7000
        x_view_offset = 20
        add_offset = 400
        placement_constraints.add_absolute_placement_constraint(x, 2, x_offset)
        placement_constraints.add_absolute_placement_constraint(y, 2, y_offset)
        placement_constraints.add_absolute_placement_constraint(
            x_view, 1, x_view_offset
        )
        placement_constraints.add_absolute_placement_constraint(add, 1, add_offset)

        self._run_mem_planning(
            edge_program.exported_program(), mem_planning, state, placement_constraints
        )
        self.assertListEqual(
            state.bufsizes,
            [
                0,
                self.get_aligned(add_offset + 2 * 3 * 5 * 4),
                self.get_aligned(x_offset + 3 * 5 * 4),
            ],
            msg=f"{state}",
        )

    def test_pinned_memory_id(self) -> None:
        edge_program = self.get_view_then_add_graph()
        x, y, x_view, add, _ = edge_program.exported_program().graph_module.graph.nodes
        # Create both mem_id+mem_offset and mem_offset constraints for all nodes.
        memory_config = MemoryConfig([1000, 10000])
        mem_planning = PositionBasedGreedyWithHierarchy(memory_config)
        state = MemoryPlanningState(memory_config=memory_config)
        placement_constraints = MemConstraints()
        x_offset = None
        y_offset = 8000
        x_view_offset = 800
        add_offset = None
        placement_constraints.add_absolute_placement_constraint(x, 2, x_offset)
        placement_constraints.add_absolute_placement_constraint(y, 2, y_offset)
        placement_constraints.add_absolute_placement_constraint(
            x_view, 1, x_view_offset
        )
        placement_constraints.add_absolute_placement_constraint(add, 1, add_offset)

        self._run_mem_planning(
            edge_program.exported_program(), mem_planning, state, placement_constraints
        )
        self.assertListEqual(
            state.bufsizes,
            [
                0,
                self.get_aligned(x_view_offset + 3 * 5 * 4),
                self.get_aligned(y_offset + 2 * 3 * 5 * 4),
            ],
            msg=f"{state}",
        )


class TestMixedPlacementConstraints(TestConstraintsBase):
    def get_slice_graph(self) -> EdgeProgramManager:
        builder = ProgramBuilder()
        x = builder.placeholder("x", torch.ones(3, 5, dtype=torch.float32))
        x_slice = builder.call_operator(
            op=exir_ops.edge.aten.slice_copy.Tensor,
            args=(x, 0, 2),
        )
        builder.output([x_slice])
        edge_program = builder.get_edge_program()
        edge_program = edge_program.transform([SpecPropPass()])
        return edge_program

    def test_slice_pinned_output(self) -> None:
        edge_program = self.get_slice_graph()
        x, x_slice, _ = edge_program.exported_program().graph_module.graph.nodes
        # Create both mem_id+mem_offset and mem_offset constraints for all nodes.
        memory_config = MemoryConfig([1000])
        mem_planning = PositionBasedGreedyWithHierarchy(memory_config)
        state = MemoryPlanningState(memory_config=memory_config)
        placement_constraints = MemConstraints()
        x_offset = 20
        placement_constraints.add_absolute_placement_constraint(x, 1, x_offset)
        placement_constraints.add_relative_placement_constraint(
            x, x_slice, 40, update_lifetime=False
        )
        self._run_mem_planning(
            edge_program.exported_program(), mem_planning, state, placement_constraints
        )

        # Check that x is placed correctly at `x_offset` and x_slice is placed at `x_offset + 40`.
        self.assertEqual(x.meta["spec"].mem_id, 1)
        self.assertEqual(x.meta["spec"].mem_offset, x_offset)
        self.assertEqual(x_slice.meta["spec"].mem_id, 1)
        self.assertEqual(x_slice.meta["spec"].mem_offset, x_offset + 2 * 5 * 4)

    def test_slice_pinned_input_fail(self) -> None:
        edge_program = self.get_slice_graph()
        x, x_slice, _ = edge_program.exported_program().graph_module.graph.nodes
        # Create both mem_id+mem_offset and mem_offset constraints for all nodes.
        placement_constraints = MemConstraints()
        x_slice_offset = 20
        x_offset = 40
        pin_memory_id = 1
        placement_constraints.add_absolute_placement_constraint(
            x_slice, pin_memory_id, x_slice_offset
        )
        with self.assertRaisesRegex(
            RuntimeError,
            f"Cannot add relative placement constraint for aten_slice_copy_tensor with non-zero offset {x_offset} when it has an absolute placement constraint AbsolutePlacementConstraint\\(pinned_memory_id={pin_memory_id}, offset={x_slice_offset}\\)",
        ):
            placement_constraints.add_relative_placement_constraint(
                x, x_slice, x_offset, update_lifetime=False
            )
