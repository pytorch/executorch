# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import itertools
import unittest
import warnings
from typing import Any, Callable, cast, List, Optional, Tuple, Type

import executorch.exir as exir

try:
    import executorch.kernels.portable  # noqa: F401
except ModuleNotFoundError:
    import logging

    logging.warning(
        "Failed to load portable_custom_ops_aot_lib. This is expected only if running in BUCK "
        "where the library is loaded via preload_deps in the TARGETS file."
    )
    del logging

import torch
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge
from executorch.exir.capture._capture import patch_forward
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.memory import view
from executorch.exir.memory_planning import (
    _do_user_inputs_exist,
    _extend_storage_base_lifetimes,
    _is_inplace_node,
    apply_algo,
    collect_specs_from_nodes,
    filter_nodes,
    get_node_tensor_specs,
    greedy,
    MemoryAlgoResult,
    MemoryPlanningAlgorithmSuite,
    naive,
    Verifier,
)
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.pass_manager import PassManager
from executorch.exir.passes import (  # noqa
    MemoryPlanningPass,
    SpecPropPass,
    ToOutVarPass,
)
from executorch.exir.passes.init_mutable_pass import InitializedMutableBufferPass
from executorch.exir.passes.memory_planning_pass import _iter_unique_specs
from executorch.exir.passes.reinplace import DEFAULT_INPLACEABLE_OPS, reinplace_pass
from executorch.exir.passes.replace_view_copy_with_view_pass import _ViewSpec
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass
from executorch.exir.schema import DeviceType
from executorch.exir.tensor import TensorSpec
from functorch.experimental.control_flow import map as torch_map
from parameterized import parameterized
from torch import nn
from torch.ao.quantization import (  # @manual=//caffe2:torch
    float_qparams_weight_only_qconfig,
)
from torch.ao.quantization.backend_config.executorch import (
    get_executorch_backend_config,
)
from torch.ao.quantization.observer import (
    default_dynamic_quant_observer,
    default_per_channel_weight_observer,
)
from torch.ao.quantization.qconfig_mapping import QConfig, QConfigMapping
from torch.ao.quantization.quantize_fx import (
    _convert_to_reference_decomposed_fx,
    prepare_fx,
)
from torch.export import export
from torch.export.experimental import _export_forward_backward
from torch.export.exported_program import ExportGraphSignature
from torch.fx import Graph, GraphModule, Node
from torch.nn import functional as F
from torch.utils import _pytree as pytree

try:
    from executorch.extension.pybindings.portable_lib import (
        _load_for_executorch_from_buffer,
        _load_program_from_buffer,
    )

    _HAS_RUNTIME = True
except ImportError:
    _HAS_RUNTIME = False


def swap_modules(
    module: torch.nn.Module,
    condition: Callable[[torch.nn.Module], bool],
    convert_func: Callable[[torch.nn.Module], torch.nn.Module],
) -> None:
    reassign = {}
    for name, mod in module.named_children():
        swap_modules(mod, condition, convert_func)
        if condition(mod):
            out = convert_func(mod)
            reassign[name] = out
    for key, value in reassign.items():
        module._modules[key] = value


class ToyModelForMemPlanning(torch.nn.Module):
    def __init__(self) -> None:
        super(ToyModelForMemPlanning, self).__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        o = a
        for _ in range(10):
            o = o * a
            o = o + b
        return o

    def get_random_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.randn(10), torch.randn(10))


class MultiEntryPointStatefulModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("state", torch.zeros(2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.state.add_(x).view(-1) * 2

    def set_state(self, state: torch.Tensor) -> None:
        self.state.copy_(state)

    def get_state(self) -> torch.Tensor:
        return self.state

    def get_example_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.ones(1),)


class ModelWithDifferentTensorSizes(torch.nn.Module):
    def __init__(self) -> None:
        super(ModelWithDifferentTensorSizes, self).__init__()
        self.linears = torch.nn.ModuleList()
        for x in [2, 4, 8, 16, 32, 64, 128]:
            self.linears.append(torch.nn.Linear(x, x * 2))

    def forward(self, i: torch.Tensor) -> torch.Tensor:
        o1 = i
        for linear in self.linears:
            o1 = linear(o1)
        o2 = i
        for linear in self.linears:
            o2 = linear(o2)
        return o1 + o2

    def get_random_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.randn(2),)


class LinearsWithDifferentSizeAndViewOps(torch.nn.Module):
    def __init__(self) -> None:
        super(LinearsWithDifferentSizeAndViewOps, self).__init__()
        self.linears = torch.nn.ModuleList()
        for x in [8, 16, 32, 64]:
            self.linears.append(torch.nn.Linear(x, x * 2))

    def forward(self, i: torch.Tensor) -> torch.Tensor:
        o1 = i
        for linear in self.linears:
            o1 = linear(o1)
        o1 = o1.view(-1, 64, 2)
        o1 = o1 + 1
        o2 = i
        for linear in self.linears:
            o2 = linear(o2)
        return o1.view(-1, 128) + o2

    def get_random_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.randn(3, 8),)


class ModuleReturnTwo(nn.Module):
    def __init__(self) -> None:
        super(ModuleReturnTwo, self).__init__()
        self.linear1 = nn.Linear(8, 8)
        self.linear2 = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        o1 = self.linear1(x)
        o2 = self.linear2(x)
        return o1, o2

    def get_random_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.randn(8),)


class ModuleListArg(nn.Module):
    r"""
    The module split a tensor and concat the parts again. The cat op will receive
    a list of tensors as argument. We want to make sure we can handle lifetime
    of tensors embedded inside a list arg correctly.
    """

    def __init__(self) -> None:
        super(ModuleListArg, self).__init__()

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        s0, s1 = torch.tensor_split(a, 2)
        s = torch.cat([s0, s1], 0)
        return s

    def get_random_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.randn(8),)

    @staticmethod
    def extra_check(
        testcase: unittest.TestCase, graph_module: torch.fx.GraphModule
    ) -> None:
        """
        Make sure the getitem nodes live as long as when the cat node starts alive
        since the cat node should have a list argument containing all the getitem nodes.
        """
        getitem_specs = []
        cat_specs = []
        for node in graph_module.graph.nodes:
            if node.target == torch.ops.aten.cat.out:
                cat_specs.append(node.meta["spec"])
            elif node.target == torch.ops.aten.slice_copy.Tensor_out:
                getitem_specs.append(node.meta["spec"])

        testcase.assertEqual(1, len(cat_specs))
        testcase.assertEqual(2, len(getitem_specs))
        for getitem_spec in getitem_specs:
            testcase.assertTrue(getitem_spec.lifetime[1] >= cat_specs[0].lifetime[0])


class CustomPoolMemoryPlanningPass(MemoryPlanningPass):
    def call(self, graph_module: GraphModule) -> PassResult:
        for subgm in graph_module.modules():
            if not isinstance(subgm, GraphModule):
                continue
            for node in subgm.graph.nodes:
                # mem_id = 1 placeholder and outputs of mul
                # mem_id = 3 for outputs of add
                # parent class will copy spec will to alloc nodes
                if node.op == "placeholder":
                    node.meta["spec"].mem_id = 1
                    continue

                if node.op != "call_function":
                    continue

                if node.target == torch.ops.aten.add.out:
                    node.meta["spec"].mem_id = 3
                elif node.target == torch.ops.aten.mul.out:
                    node.meta["spec"].mem_id = 1

        return super().run(graph_module)

    def run(
        self,
        graph_module: torch.fx.GraphModule,
        graph_signature: Optional[ExportGraphSignature] = None,
    ) -> PassResult:
        return self.call(graph_module)


class MultiplePoolsToyModel(torch.nn.Module):
    def forward(self, a: torch.Tensor) -> torch.Tensor:
        # a: mem_id = 1, offset = 0
        # b: mem_id = 3, offset = 0
        # c: mem_id = 1, offset = 4
        # d: mem_id = 3, offset = 4
        # greedy:
        # e: mem_id = 1, offset = 0
        # naive:
        # e: mem_id = 1, offset = 8
        b = a + a
        c = a * b
        d = c + b
        e = c * d
        return e


def maketest(
    module_cls: Type[torch.nn.Module],
    criteria: Optional[List[Tuple[Callable[..., MemoryAlgoResult], bool]]] = None,
    extra_check: Optional[Callable[..., None]] = None,
    use_functionalization: bool = True,
    alloc_graph_input: bool = True,
    alloc_graph_output: bool = True,
    alloc_mutable_buffer: bool = True,
    has_unused_graph_input: bool = False,
) -> Callable[..., None]:
    # parameterized.expand is not compatible with maketest. I'll just loop thru
    # the test setups in the wrapper.
    def wrapper(self: "TestMemoryPlanning") -> None:
        nonlocal criteria
        if not criteria:
            criteria = [
                # naive algorithm does not reuse tensor storages
                (naive, False),
                # greedy algorithm should reuse tensor storages in the testing model
                (greedy, True),
            ]

        for algo, expect_reuse in criteria:
            print(
                f"algo {getattr(algo, '__name__', repr(algo))}, expect_reuse {expect_reuse}"
            )
            eager_module = module_cls().eval()
            # pyre-fixme[29]: `Union[nn.modules.module.Module,
            #  torch._tensor.Tensor]` is not a function.
            inputs = eager_module.get_random_inputs()
            graph_module = (
                to_edge(export(eager_module, inputs, strict=True))
                .exported_program()
                .graph_module
            )
            mem_algo = MemoryPlanningAlgorithmSuite(algo_list=[algo])
            graph_module = PassManager(
                passes=[
                    SpecPropPass(),
                    ToOutVarPass(),
                    MemoryPlanningPass(
                        mem_algo,
                        alloc_graph_input=alloc_graph_input,
                        alloc_graph_output=alloc_graph_output,
                    ),
                ],
            )(graph_module).graph_module

            self.verify_reuse(
                graph_module,
                expect_reuse,
                alloc_graph_input,
                alloc_graph_output,
                alloc_mutable_buffer,
            )
            self.verify_graph_input_output(
                graph_module,
                alloc_graph_input,
                alloc_graph_output,
                alloc_mutable_buffer,
            )

            self.verify_overlap_placeholders(has_unused_graph_input, graph_module)

            # print(f"Final code: {graph_module.code}")
            # print(f"Final graph: {graph_module.graph}")

            if extra_check:
                extra_check(self, graph_module)

    return wrapper


class TestMemoryPlanningUserInputs(unittest.TestCase):
    """
    Ensure that MemoryPlanning Verifer only assumes a model
    has a user input if it has at least one tensor input.
    """

    def test_tensor_only_inputs(self) -> None:
        class TensorModel(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

        model = TensorModel()
        inputs = (torch.randn(2), torch.randn(2))
        ep = export(model, inputs, strict=True)
        result = _do_user_inputs_exist(graph_signature=ep.graph_signature)
        self.assertTrue(result)

    def test_mixed_inputs(self) -> None:
        class MixedModel(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: int) -> torch.Tensor:
                return x * y

        model = MixedModel()
        inputs = (torch.randn(2), 3)
        ep = export(model, inputs, strict=True)
        result = _do_user_inputs_exist(graph_signature=ep.graph_signature)
        self.assertTrue(result)

    def test_primitive_only_inputs(self) -> None:
        class PrimModel(torch.nn.Module):
            def forward(self, x: int, y: float) -> float:
                return x * y

        model = PrimModel()
        inputs = (2, 3.0)
        ep = export(model, inputs, strict=True)
        result = _do_user_inputs_exist(graph_signature=ep.graph_signature)
        self.assertFalse(result)

    def test_no_inputs(self) -> None:
        class NoInputModel(torch.nn.Module):
            def forward(self) -> torch.Tensor:
                return torch.tensor(1.0)

        model = NoInputModel()
        ep = export(model, (), strict=True)
        result = _do_user_inputs_exist(graph_signature=ep.graph_signature)
        self.assertFalse(result)


class TestMemoryPlanning(unittest.TestCase):
    def verify_reuse(
        self,
        graph_module: torch.fx.GraphModule,
        expect_reuse: bool,
        alloc_graph_input: bool,
        alloc_graph_output: bool,
        alloc_mutable_buffer: bool,
    ) -> None:
        r"""
        Do sanity check and verify tensor storage reuse.

        There should NOT be any tensor storage overlapping between tensors that have
        overlapping lifetime.

        expect_reuse is True if we expect the algorithm reuse tensor storages
        for at least a pair of tensors in the current testing setup.
        """
        # this method throws if 2 tensors overlap both lifetime and storage.
        num_reuse_pairs = Verifier(
            graph_module,
            alloc_graph_input=alloc_graph_input,
            alloc_graph_output=alloc_graph_output,
            alloc_mutable_buffers=alloc_mutable_buffer,
        ).verify_storage_reuse()

        print(f"num_reuse_pairs is {num_reuse_pairs}")
        if expect_reuse:
            self.assertTrue(num_reuse_pairs > 0)
        else:
            self.assertTrue(num_reuse_pairs == 0)

    def verify_graph_input_output(
        self,
        graph_module: torch.fx.GraphModule,
        alloc_graph_input: bool,
        alloc_graph_output: bool,
        alloc_mutable_buffers: bool,
    ) -> None:
        Verifier(
            graph_module, alloc_graph_input, alloc_graph_output, alloc_mutable_buffers
        ).verify_graph_input_output()

    def verify_overlap_placeholders(
        self, has_unused_graph_input: bool, graph_module: GraphModule
    ) -> None:
        """
        If every placholder node is used somewhere, then each pair should have
        overlapped lifetime.
        """
        if has_unused_graph_input:
            return

        ph_list = []
        for nd in graph_module.graph.nodes:
            if nd.op == "placeholder":
                ph_list.append(nd)

        # since all placeholders are used somewhere. Their lifetime should
        # overlap.
        for i in range(len(ph_list)):
            for j in range(i + 1, len(ph_list)):
                ph_lhs = ph_list[i]
                ph_rhs = ph_list[j]
                self.assertTrue(
                    Verifier.lifetime_overlap(ph_lhs.meta["spec"], ph_rhs.meta["spec"])
                )

    test_basic: Callable[..., None] = maketest(ToyModelForMemPlanning)
    # TODO(zhxchen17) re-enable this.
    # test_while: Callable[..., None] = maketest(
    #     ModuleWhile,
    #     criteria=[
    #         ("naive", False),
    #         ("greedy", False),
    #     ],
    # )
    test_different_tensor_sizes: Callable[..., None] = maketest(
        ModelWithDifferentTensorSizes
    )

    test_return_two: Callable[..., None] = maketest(
        ModuleReturnTwo,
        criteria=[
            (naive, False),
            (greedy, True),
        ],
    )

    test_linear_with_view: Callable[..., None] = maketest(
        LinearsWithDifferentSizeAndViewOps,
        criteria=[
            (greedy, True),
        ],
    )

    # greedy algorithm will reuse memory if we let the algorithm allocate
    # memory for both graph input and output.
    test_list_arg: Callable[..., None] = maketest(
        ModuleListArg,
        criteria=[
            (naive, False),
            (greedy, True),
        ],
        extra_check=ModuleListArg.extra_check,
    )

    def test_graph_input_output(self) -> None:
        for (
            alloc_graph_input,
            alloc_graph_output,
            alloc_mutable_buffers,
        ) in itertools.product([True, False], [True, False], [True, False]):
            test = maketest(
                ModelWithDifferentTensorSizes,
                alloc_graph_input=alloc_graph_input,
                alloc_graph_output=alloc_graph_output,
                alloc_mutable_buffer=alloc_mutable_buffers,
            )
            test(self)


class TestVerifier(unittest.TestCase):
    def test_overlap(self) -> None:
        # first enclose second
        self.assertTrue(Verifier.has_overlap([1, 10], [2, 3]))
        # second enclose first
        self.assertTrue(Verifier.has_overlap([2, 3], [1, 10]))
        # first on the left side
        self.assertTrue(Verifier.has_overlap([1, 4], [2, 5]))
        # first on the right side
        self.assertTrue(Verifier.has_overlap([2, 5], [1, 4]))

        # non overlap. first on the left side
        self.assertFalse(Verifier.has_overlap([1, 2], [5, 6]))
        # non overlap. first on the right side
        self.assertFalse(Verifier.has_overlap([5, 6], [1, 2]))


class TestMisc(unittest.TestCase):
    def test_filter_nodes(self) -> None:
        g = Graph()
        nd_pool = [
            Node(g, f"n{idx}", "placeholder", f"n{idx}", (), {}) for idx in range(10)
        ]
        actual_list = list(
            filter_nodes(
                [
                    nd_pool[0],
                    (nd_pool[1], nd_pool[2]),
                    None,
                    [nd_pool[3]],
                    {"first": nd_pool[4]},
                ]
            )
        )
        expected_list = nd_pool[:5]
        self.assertEqual(len(actual_list), len(expected_list))
        for act, exp in zip(actual_list, expected_list):
            self.assertEqual(id(act), id(exp))

    def quantize(self, eager_model: nn.Module) -> nn.Module:
        quantized_model = eager_model
        linear_qconfig_mapping = QConfigMapping().set_object_type(
            F.linear,
            QConfig(
                activation=default_dynamic_quant_observer,
                weight=default_per_channel_weight_observer,
            ),
        )
        embedding_qconfig_mapping = QConfigMapping().set_object_type(
            F.embedding,
            float_qparams_weight_only_qconfig,
        )
        # quantize module
        swap_modules(
            quantized_model,
            lambda mod: isinstance(mod, torch.nn.Linear),
            lambda mod: _convert_to_reference_decomposed_fx(
                prepare_fx(
                    mod,
                    linear_qconfig_mapping,
                    (torch.rand(1, mod.in_features),),
                    backend_config=get_executorch_backend_config(),
                ),
                backend_config=get_executorch_backend_config(),
            ),
        )
        swap_modules(
            quantized_model,
            lambda mod: isinstance(mod, torch.nn.Embedding),
            lambda mod: _convert_to_reference_decomposed_fx(
                prepare_fx(
                    mod,
                    embedding_qconfig_mapping,
                    (torch.ones(1, 1),),
                    backend_config=get_executorch_backend_config(),
                ),
                backend_config=get_executorch_backend_config(),
            ),
        )
        return quantized_model

    @parameterized.expand(
        [
            (
                naive,
                [(1, 0), (3, 0), (1, 4), (3, 4), (1, 8)],
                [0, 12, 0, 8],
            ),
            (
                greedy,
                [(1, 0), (3, 0), (1, 4), (3, 4), (1, 0)],
                [0, 8, 0, 8],
            ),
        ]
    )
    def test_multiple_pools(
        self,
        algo: Callable[..., MemoryAlgoResult],
        expected_allocs: List[Tuple[int, int]],
        expected_bufsizes: List[int],
    ) -> None:
        edge_program = to_edge(
            export(MultiplePoolsToyModel(), (torch.ones(1),), strict=True)
        )

        mem_algo = MemoryPlanningAlgorithmSuite(algo_list=[algo])
        edge_program.to_executorch(
            exir.ExecutorchBackendConfig(
                memory_planning_pass=CustomPoolMemoryPlanningPass(
                    memory_planning_algo=mem_algo,
                    alignment=1,
                ),
            )
        )
        graph_module = edge_program.exported_program().graph_module

        verifier = Verifier(
            graph_module,
            alloc_graph_input=True,
            alloc_graph_output=True,
            alloc_mutable_buffers=True,
        )
        verifier.verify_storage_reuse()
        verifier.verify_graph_input_output()

        idx = 0
        reference_output = {}
        actual_output = {}
        for node in graph_module.graph.nodes:
            if node.op == "placeholder" or (
                node.op == "call_function"
                and node.target in (torch.ops.aten.add.out, torch.ops.aten.mul.out)
            ):
                mem_id, mem_offset = expected_allocs[idx]
                actual_mem_id, actual_mem_offset = (
                    node.meta["spec"].mem_id,
                    node.meta["spec"].mem_offset,
                )
                if (mem_id, mem_offset) not in reference_output:
                    reference_output[(mem_id, mem_offset)] = 1
                    actual_output[(actual_mem_id, actual_mem_offset)] = 1
                else:
                    reference_output[(mem_id, mem_offset)] += 1
                    actual_output[(actual_mem_id, actual_mem_offset)] += 1
                idx += 1
        self.assertEqual(reference_output, actual_output)
        self.assertEqual(graph_module.meta["non_const_buffer_sizes"], expected_bufsizes)

    def test_mutation_not_double_allocated(self) -> None:
        class Simple(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("constant", torch.ones(5, 5))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.constant.add_(1)
                return x - self.constant

        model = Simple()
        inputs = (torch.ones(5, 5),)

        et = to_edge(export(model, inputs, strict=True)).to_executorch()

        # The mutable buffer (5x5 float32 = 100 bytes) should not be
        # double allocated. After the upstream emit dedup
        # (`_emit_spec` reusing value_id when two FX nodes share a
        # TensorSpec via the planner's `_alias_inplace_result_specs`),
        # the `copy_` writeback's "out" arg uses the SAME value_id as
        # its "self" arg (the buffer), rather than creating a separate
        # Value at the same (mem_id, offset).
        execution_plan = et.executorch_program.execution_plan[0]
        values = execution_plan.values

        # Find the `copy_` writeback instruction.
        copy_instructions = []
        for chain in execution_plan.chains:
            for ins in chain.instructions:
                inner = ins.instr_args
                if hasattr(inner, "op_index"):
                    op = execution_plan.operators[inner.op_index]
                    if op.name == "aten::copy_":
                        copy_instructions.append(inner)
        self.assertEqual(
            len(copy_instructions),
            1,
            "Expected exactly one copy_ writeback for the buffer mutation",
        )

        # For an in-place copy_(self, src, ..., out), self (arg 0) and
        # out (the emitted synthetic last arg) must share a value_id
        # per the `(a!)` schema annotation. Emit's spec2id_dict
        # dedup enforces this.
        copy_args = list(copy_instructions[0].args)
        self.assertEqual(
            copy_args[0],
            copy_args[-1],
            f"copy_'s out arg should reference the same value_id as its "
            f"self arg (buffer) via emit dedup. args={copy_args}",
        )

        # Additionally verify no distinct second Value at the buffer's
        # (mem_id, offset): after dedup, the buffer occupies its slot alone.
        buffer_value_id = copy_args[0]
        buffer_val = values[buffer_value_id].val
        self.assertTrue(
            hasattr(buffer_val, "allocation_info") and buffer_val.allocation_info,
            "Buffer value should have allocation_info",
        )
        buffer_alloc = buffer_val.allocation_info
        duplicates_at_buffer_slot = [
            i
            for i, val in enumerate(values)
            if i != buffer_value_id
            and hasattr(val.val, "allocation_info")
            and val.val.allocation_info
            and val.val.allocation_info.memory_id == buffer_alloc.memory_id
            and val.val.allocation_info.memory_offset == buffer_alloc.memory_offset
        ]
        self.assertEqual(
            duplicates_at_buffer_slot,
            [],
            f"Expected no other Values at the buffer's allocation "
            f"(mem_id={buffer_alloc.memory_id}, "
            f"offset={buffer_alloc.memory_offset}); emit dedup should "
            f"collapse placeholder + writeback into one value_id. "
            f"Found duplicates at indices: {duplicates_at_buffer_slot}",
        )

    def test_mutable_buffers_infinite_lifespan(self) -> None:
        class Simple(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("state", torch.zeros(1))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.state.index_put_(
                    [
                        torch.tensor([0]),
                    ],
                    x,
                )
                y = x + self.state
                z = x * y
                return z

        model = Simple()
        inputs = (torch.ones(1),)

        et = to_edge(export(model, inputs, strict=True)).to_executorch(
            ExecutorchBackendConfig(
                emit_mutable_buffer_names=True, run_reinplace_pass=True
            )
        )

        serialized_state = et.executorch_program.execution_plan[0].values[0].val
        self.assertEqual(
            serialized_state.extra_tensor_info.fully_qualified_name, "state"
        )
        memory_base = serialized_state.allocation_info.memory_offset_low
        memory_size = memory_base + 4  # 4 bytes for a single float
        for value in et.executorch_program.execution_plan[0].values[1:]:
            val = value.val
            if hasattr(val, "allocation_info") and val.allocation_info is not None:
                not_overlapping = (
                    val.allocation_info.memory_offset_low < memory_base
                    or val.allocation_info.memory_offset_low >= memory_size
                )
                self.assertTrue(not_overlapping)

    def test_custom_inplace_op_memory_aliasing(self) -> None:
        """Memory planning correctly handles in-place ops registered via
        the ``ops_to_inplace`` extension API (i.e. outside
        ``DEFAULT_INPLACEABLE_OPS``).

        Uses the HF-static-cache pattern: ``index_copy_`` updates two
        mutable buffers (``keys``, ``values``). We:
          1. Preserve ``index_copy`` through edge lowering.
          2. Manually call ``reinplace_pass`` with a custom set that
             includes ``index_copy`` (the in-place form is
             auto-derived).
          3. Lower with ``run_reinplace_pass=False`` (the pass already
             ran).

        Then assert that no other planned tensor's allocation overlaps
        either buffer's storage region. This pins the schema-driven
        ``_alias_inplace_result_specs`` path for non-default ops: the
        ``index_copy_`` result spec must be aliased to the buffer's
        spec, otherwise the planner would carve out a separate
        allocation that could land inside the buffer's slot.
        """
        max_batch_size, num_heads, max_cache_len, head_dim = 1, 2, 4, 8

        class HFStyleStaticCache(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "keys",
                    torch.zeros((max_batch_size, num_heads, max_cache_len, head_dim)),
                )
                self.register_buffer(
                    "values",
                    torch.zeros((max_batch_size, num_heads, max_cache_len, head_dim)),
                )

            def forward(
                self,
                key_states: torch.Tensor,
                value_states: torch.Tensor,
                cache_position: torch.Tensor,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                self.keys.index_copy_(2, cache_position, key_states)
                self.values.index_copy_(2, cache_position, value_states)
                return self.keys, self.values

        model = HFStyleStaticCache()
        key_states = torch.full((max_batch_size, num_heads, 1, head_dim), 1.0)
        value_states = torch.full((max_batch_size, num_heads, 1, head_dim), 2.0)
        cache_position = torch.tensor([1])

        exported_program = export(
            model, (key_states, value_states, cache_position), strict=True
        )

        edge = to_edge(
            exported_program,
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
                preserve_ops=[torch.ops.aten.index_copy.default],
            ),
        )

        # Manually run reinplace_pass with a custom set that
        # includes the (non-default) index_copy edge op. The in-place
        # form is auto-derived by name + schema match.
        custom_set = DEFAULT_INPLACEABLE_OPS | {
            exir_ops.edge.aten.index_copy.default,
        }
        edge_program = reinplace_pass(
            edge.exported_program(), ops_to_inplace=custom_set
        )
        # Sanity: both updates are now in-place.
        inplace_nodes = [
            n
            for n in edge_program.graph.nodes
            if n.op == "call_function" and "index_copy_" in str(n.target)
        ]
        self.assertEqual(
            len(inplace_nodes),
            2,
            "Both buffer updates should be reinplaced before lowering",
        )

        # Lower with run_reinplace_pass=False — the pass already ran
        # with our custom set above. Memory planning should now
        # correctly alias the index_copy_ result spec onto the buffer
        # placeholder spec via _alias_inplace_result_specs.
        et = edge.to_executorch(
            ExecutorchBackendConfig(
                emit_mutable_buffer_names=True,
                run_reinplace_pass=False,
            )
        )

        execution_plan = et.executorch_program.execution_plan[0]
        values = execution_plan.values

        # Collect the keys / values buffer Values by FQN.
        buffer_value_ids: dict[str, int] = {}
        for i, value in enumerate(values):
            val = value.val
            extra = getattr(val, "extra_tensor_info", None)
            fqn = getattr(extra, "fully_qualified_name", None) if extra else None
            if fqn in ("keys", "values"):
                buffer_value_ids[fqn] = i

        self.assertEqual(
            set(buffer_value_ids.keys()),
            {"keys", "values"},
            "Both keys and values buffers should appear in the program "
            "with their FQN",
        )

        # For each buffer, verify no other planned Value's allocation
        # overlaps the buffer's memory region.
        for fqn, vid in buffer_value_ids.items():
            buf_alloc = values[vid].val.allocation_info
            self.assertIsNotNone(buf_alloc, f"Buffer {fqn} should have allocation_info")
            buf_base = buf_alloc.memory_offset_low
            # 4 bytes per float32 element.
            num_elements = max_batch_size * num_heads * max_cache_len * head_dim
            buf_end = buf_base + num_elements * 4

            for j, other in enumerate(values):
                if j == vid:
                    continue
                other_alloc = getattr(other.val, "allocation_info", None)
                if other_alloc is None:
                    continue
                if other_alloc.memory_id != buf_alloc.memory_id:
                    continue
                offset = other_alloc.memory_offset_low
                overlaps = buf_base <= offset < buf_end
                self.assertFalse(
                    overlaps,
                    f"Value {j} (alloc offset={offset}) overlaps the "
                    f"{fqn} buffer's region [{buf_base}, {buf_end}) — "
                    "the in-place index_copy_ result spec was not "
                    "correctly aliased to the buffer spec",
                )

    def test_constants_not_memory_planned(self) -> None:
        class Simple(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)
                self.register_buffer("constant", torch.ones(5, 5))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.nn.functional.sigmoid(self.linear(x) + self.constant + 1)

        def count_planned_inputs(
            nodes: List[Node],
            graph_signature: Any,  # pyre-ignore
        ) -> Tuple[int, int]:
            num_mem_planned_placeholders = 0
            num_placeholders = 0
            for node in nodes:
                if node.op == "placeholder":
                    num_placeholders += 1
                    specs = get_node_tensor_specs(node)
                    self.assertGreaterEqual(len(specs), 1)
                    for spec in specs:
                        if spec.mem_id is not None:
                            num_mem_planned_placeholders += 1
            return num_placeholders, num_mem_planned_placeholders

        model = Simple()
        inputs = (torch.randn(5, 5),)

        ep_no_input_planning = to_edge(
            export(model, inputs, strict=True)
        ).to_executorch(
            config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
                sym_shape_eval_pass=ConstraintBasedSymShapeEvalPass(),
            )
        )

        num_placeholders, num_planned_placeholders = count_planned_inputs(
            ep_no_input_planning.exported_program().graph_module.graph.nodes,
            ep_no_input_planning.exported_program().graph_signature,
        )
        self.assertEqual(
            num_planned_placeholders,
            0,
        )  # one unplanned user input and 4 constants that shouldnt be planned
        self.assertEqual(
            num_placeholders,
            5,  # x, self.constant, linear weight, linear bias, '1' scalar promoted to tensor
        )

        ep_input_planning = to_edge(export(model, inputs, strict=True)).to_executorch(
            config=ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=True),
                sym_shape_eval_pass=ConstraintBasedSymShapeEvalPass(),
            )
        )

        num_placeholders, num_planned_placeholders = count_planned_inputs(
            ep_input_planning.exported_program().graph_module.graph.nodes,
            ep_input_planning.exported_program().graph_signature,
        )
        self.assertEqual(
            num_planned_placeholders,
            1,
        )  # one planned user input and 4 constants that shouldnt be planned
        self.assertEqual(
            num_placeholders,
            5,
        )

    def test_placeholder_lifetime(self) -> None:
        class TestModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, a, b, x):
                a = a + b
                b = a + b
                y = self.linear(x)
                return a, b, y

        model = TestModel()
        example_inputs = (torch.rand(1, 6, 2), torch.rand(1, 6, 2), torch.randn(5, 5))
        exported_model = torch.export.export(model, example_inputs, strict=True)
        edge = to_edge(exported_model)

        class TestPass(ExportPass):
            def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
                permute_dims = [1, 0, 2]
                for node in graph_module.graph.nodes:
                    if node.op == "placeholder" and str(node) == "a":
                        inverse_dims = [
                            permute_dims.index(x) for x in range(len(permute_dims))
                        ]

                        with graph_module.graph.inserting_after(node):
                            permute = graph_module.graph.call_function(
                                exir_ops.edge.aten.permute_copy.default,
                                args=(node, inverse_dims),
                            )
                            permute.meta = node.meta.copy()
                            node.meta["val"] = node.meta["val"].permute(permute_dims)
                            node.replace_all_uses_with(
                                permute, lambda x, permute=permute: x is not permute
                            )
                            break
                return PassResult(graph_module, True)

        edge = edge.transform([TestPass()])
        et = edge.to_executorch()
        et_program = et.executorch_program
        inputs = et_program.execution_plan[0].inputs
        self.assertNotEqual(
            et_program.execution_plan[0]
            .values[inputs[0]]
            .val.allocation_info.memory_offset_low,
            et_program.execution_plan[0]
            .values[inputs[1]]
            .val.allocation_info.memory_offset_low,
        )

        constants = 0
        for node in et.exported_program().graph_module.graph.nodes:
            if node.op == "placeholder" and node.meta.get("spec"):
                meta_spec = node.meta["spec"]
                if meta_spec.const is True:
                    constants += 1
                    self.assertIsNone(node.meta["spec"].mem_offset)
                    self.assertIsNone(node.meta["spec"].mem_id)
        self.assertEqual(constants, 2)

    def test_none_output(self) -> None:
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(6, 6, 5)
                self.linear = nn.Linear(6, 2)

            def forward(self, x):
                return self.linear(self.conv1(x).flatten(1))

        class TrainingNet(nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net
                self.loss = nn.CrossEntropyLoss()

            def forward(self, input, label):
                pred = self.net(input)
                return self.loss(pred, label)

        net = TrainingNet(Net())
        inputs = (torch.randn(1, 6, 5, 5), torch.ones(1, dtype=torch.int64))

        ep = export(net, inputs, strict=True)
        ep = _export_forward_backward(ep)
        ep = to_edge(ep)
        ep = ep.to_executorch()

        ep.dump_executorch_program(True)

        # 149 just so happens to be the index of the user_grad output arg of
        # convolution_backward.out. This is fairly fragile.
        # Check that the None output is not memory planned.
        # TODO(masnesral): restore after https://github.com/pytorch/pytorch/pull/144765
        # self.assertEqual(len(ep.executorch_program.execution_plan[0].values), 151)
        # self.assertEqual(
        #     ep.executorch_program.execution_plan[0]
        #     .values[149]
        #     .val.data_buffer_idx,  # pyright: ignore
        #     0,
        # )
        # self.assertEqual(
        #     ep.executorch_program.execution_plan[0]
        #     .values[149]
        #     .val.allocation_info,  # pyright: ignore
        #     None,
        # )


def _get_specs(gm: torch.fx.GraphModule) -> set[TensorSpec]:
    return set(
        filter(
            None,
            pytree.tree_flatten(
                pytree.tree_map_only(
                    torch.fx.Node,
                    lambda n: n.meta.get("spec", None),
                    list(gm.graph.nodes),
                )
            )[0],
        )
    )


class MapModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Use actual torch.map function for memory planning testing
        def add_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a + b

        # Use torch.map to apply function over first dimension
        # pyre-ignore[6]: For 3rd argument expected `TypeVarTuple` but got `Tensor`.
        map_output = torch_map(add_fn, x, y)

        return map_output + y

    def get_random_inputs(self) -> Tuple[torch.Tensor, ...]:
        return (torch.randn(5, 3), torch.randn(3))


class MultiMapModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.map_model = MapModel()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Use actual torch.map function for memory planning testing
        def add_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a + b

        # pyre-ignore[6]: For 3rd argument expected `TypeVarTuple` but got `Tensor`.
        x = torch_map(add_fn, x, y)
        # pyre-ignore[6]: For 3rd argument expected `TypeVarTuple` but got `Tensor`.
        x = torch_map(add_fn, x, y)
        # pyre-ignore[6]: For 3rd argument expected `TypeVarTuple` but got `Tensor`.
        x = torch_map(add_fn, x, y)
        return x

    def get_random_inputs(self) -> tuple[torch.Tensor, ...]:
        return self.map_model.get_random_inputs()


class TestMap(unittest.TestCase):

    def test_map(self) -> None:
        """Test memory planning for torch.map operations."""

        eager_module = MapModel().eval()
        inputs = eager_module.get_random_inputs()

        # Export and convert to edge
        graph_module = (
            to_edge(export(eager_module, inputs, strict=True))
            .exported_program()
            .graph_module
        )

        # Apply memory planning.
        mem_algo = MemoryPlanningAlgorithmSuite(algo_list=[naive])
        graph_module = PassManager(
            passes=[
                SpecPropPass(),
                ToOutVarPass(),
            ],
        )(graph_module).graph_module
        mem_planning_pass = MemoryPlanningPass(
            mem_algo,
            alloc_graph_input=True,
            alloc_graph_output=True,
            alloc_mutable_buffers=True,
        )
        graph_module = mem_planning_pass.run(graph_module).graph_module

        # Verify memory planning results
        verifier = Verifier(
            graph_module,
            alloc_graph_input=True,
            alloc_graph_output=True,
            alloc_mutable_buffers=True,
        )
        verifier.verify_graph_input_output()
        verifier.verify_storage_reuse(allow_lifetime_and_storage_overlap=False)

        map_nodes = graph_module.graph.find_nodes(
            op="call_function", target=torch.ops.higher_order.map_impl
        )
        assert len(map_nodes) == 1
        map_fn_node = map_nodes[0].args[0]
        self.assertEqual(map_fn_node.op, "get_attr")
        map_fn = getattr(graph_module, map_fn_node.target)

        map_lifetime = map_nodes[0].meta.get("spec", None)[0].lifetime[0]

        # Check that there is no storage overlap between nodes of the outer program and submodule of map.
        for outer_spec in _get_specs(graph_module):
            for inner_spec in _get_specs(map_fn):
                self.assertFalse(
                    verifier.has_overlap(
                        outer_spec.lifetime, [map_lifetime, map_lifetime]
                    )
                    and (verifier.storage_overlap(outer_spec, inner_spec)),
                    f"Outer spec {outer_spec.shape=} {outer_spec.dtype=} {outer_spec.lifetime=} and inner spec {inner_spec} have storage overlap",
                )

    def test_multi_map(self) -> None:
        """Test memory planning for torch.map operations."""

        eager_module = MultiMapModel().eval()
        inputs = eager_module.get_random_inputs()

        # Export and convert to edge
        graph_module = (
            to_edge(export(eager_module, inputs, strict=True))
            .exported_program()
            .graph_module
        )

        # Apply memory planning.
        mem_algo = MemoryPlanningAlgorithmSuite(algo_list=[naive])
        graph_module = PassManager(
            passes=[
                SpecPropPass(),
                ToOutVarPass(),
            ],
        )(graph_module).graph_module
        mem_planning_pass = MemoryPlanningPass(
            mem_algo,
            alloc_graph_input=True,
            alloc_graph_output=True,
            alloc_mutable_buffers=True,
        )
        graph_module = mem_planning_pass.run(graph_module).graph_module

        # Verify memory planning results
        verifier = Verifier(
            graph_module,
            alloc_graph_input=True,
            alloc_graph_output=True,
            alloc_mutable_buffers=True,
        )
        verifier.verify_graph_input_output()
        verifier.verify_storage_reuse(allow_lifetime_and_storage_overlap=False)

        # Check that bufsizes are [0, 320]:
        # 1. 48 (3 * 16 bytes) for map body,
        # 2. 64 * 4 (4 * 16 bytes) input0/map outputs, and
        # 3. 16 bytes for input1.
        self.assertEqual(graph_module.meta["non_const_buffer_sizes"], [0, 320])
        for map_node in graph_module.graph.find_nodes(
            op="call_function", target=torch.ops.higher_order.map_impl
        ):
            map_fn_node = map_node.args[0]
            self.assertEqual(map_fn_node.op, "get_attr")
            map_fn = getattr(graph_module, map_fn_node.target)
            self.assertEqual(map_fn.meta["non_const_buffer_sizes"], [0, 48])

        # Check there is no lifetime and storage overlap between nodes of the outer program and submodule of map.
        for map_node in graph_module.graph.find_nodes(
            op="call_function", target=torch.ops.higher_order.map_impl
        ):
            map_fn_node = map_node.args[0]
            self.assertEqual(map_fn_node.op, "get_attr")
            map_fn = getattr(graph_module, map_fn_node.target)
            map_lifetime = map_node.meta.get("spec", None)[0].lifetime[0]
            outer_specs_with_overlap = set(
                filter(
                    lambda spec: verifier.has_overlap(
                        spec.lifetime, [map_lifetime, map_lifetime]
                    ),
                    _get_specs(graph_module),
                )
            )

            # Check that there is no storage overlap between nodes of the outer program and submodule of map.
            for inner_spec in _get_specs(map_fn):
                for outer_spec in outer_specs_with_overlap:
                    self.assertFalse(
                        verifier.storage_overlap(outer_spec, inner_spec),
                        f"Outer spec {outer_spec.shape=} {outer_spec.dtype=} {outer_spec.lifetime=} and inner spec {inner_spec} have storage overlap",
                    )

    def test_multi_state_plan(self) -> None:
        eager_module = MultiEntryPointStatefulModel().eval()
        forward = export(eager_module, eager_module.get_example_inputs())
        with patch_forward(eager_module, eager_module.get_state):
            get_state = export(eager_module, ())
        with patch_forward(eager_module, eager_module.set_state):
            set_state = export(eager_module, (torch.zeros(1),))
        edge = to_edge(
            {"forward": forward, "set_state": set_state, "get_state": get_state}
        )
        et = edge.to_executorch(
            ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(share_mutable_buffers=True),
                emit_mutable_buffer_names=True,
            )
        )
        et_prog = et.executorch_program
        count = 0
        for plan in et_prog.execution_plan:
            for value in plan.values:
                if (
                    hasattr(value.val, "allocation_info")
                    and value.val.allocation_info is not None
                    and value.val.allocation_info.memory_id == 2
                ):
                    count += 1
                    self.assertEqual(value.val.allocation_info.memory_offset_low, 0)
                    self.assertTrue(value.val.extra_tensor_info is not None)
                    self.assertEqual(
                        value.val.extra_tensor_info.fully_qualified_name, "state"
                    )
        self.assertEqual(count, 3)

    def test_custom_kv_cache_shared_buffers(self) -> None:
        from executorch.examples.models.llama.source_transformation.custom_kv_cache import (
            CustomKVCache,
        )
        from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401

        class KVCacheModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.kv_cache = CustomKVCache(
                    max_batch_size=1,
                    max_context_length=8,
                    n_heads=2,
                    head_dim=4,
                )

            def forward(
                self,
                input_pos: torch.Tensor,
                k_val: torch.Tensor,
                v_val: torch.Tensor,
            ) -> torch.Tensor:
                k_out, v_out = self.kv_cache.update(input_pos, k_val, v_val)
                return (k_out + v_out).sum(dim=-1)

            def reset(self, k_zeros: torch.Tensor, v_zeros: torch.Tensor) -> None:
                self.kv_cache.k_cache.copy_(k_zeros)
                self.kv_cache.v_cache.copy_(v_zeros)

        model = KVCacheModel().eval()
        cache_shape = (1, 8, 2, 4)  # [B, S, H, D]

        forward_ep = export(
            model,
            (torch.tensor([0]), torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4)),
        )
        with patch_forward(model, model.reset):
            reset_ep = export(
                model, (torch.zeros(cache_shape), torch.zeros(cache_shape))
            )

        edge = to_edge({"forward": forward_ep, "reset": reset_ep})
        et = edge.to_executorch(
            ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(
                    share_mutable_buffers=True,
                ),
                emit_mutable_buffer_names=True,
            )
        )
        et_prog = et.executorch_program

        self.assertEqual(len(et_prog.execution_plan[0].non_const_buffer_sizes), 3)
        self.assertEqual(len(et_prog.execution_plan[1].non_const_buffer_sizes), 3)

        # Verify that mem_id=2 has the same buffer size in both execution plans.
        self.assertEqual(
            et_prog.execution_plan[0].non_const_buffer_sizes[2],
            512,  # 2 * (1*8*2*4) = 128 * 4 bytes = 512 bytes
        )
        self.assertEqual(
            et_prog.execution_plan[1].non_const_buffer_sizes[2],
            512,  # 2 * (1*8*2*4) = 128 * 4 bytes = 512 bytes
        )

        for plan in et_prog.execution_plan:
            k_cache = [
                v
                for v in plan.values
                if hasattr(v.val, "extra_tensor_info")
                and v.val.extra_tensor_info is not None
                and v.val.extra_tensor_info.fully_qualified_name == "kv_cache.k_cache"
            ]
            self.assertEqual(len(k_cache), 1)
            self.assertEqual(k_cache[0].val.allocation_info.memory_id, 2)
            self.assertEqual(k_cache[0].val.allocation_info.memory_offset_low, 0)
            self.assertEqual(k_cache[0].val.allocation_info.memory_offset_high, 0)
            v_cache = [
                v
                for v in plan.values
                if hasattr(v.val, "extra_tensor_info")
                and v.val.extra_tensor_info is not None
                and v.val.extra_tensor_info.fully_qualified_name == "kv_cache.v_cache"
            ]
            self.assertEqual(len(v_cache), 1)
            self.assertEqual(v_cache[0].val.allocation_info.memory_id, 2)
            self.assertEqual(v_cache[0].val.allocation_info.memory_offset_low, 256)
            self.assertEqual(v_cache[0].val.allocation_info.memory_offset_high, 0)

    def test_a_reused_pass_instance_replans_shared_mutable_buffers(self) -> None:
        """One `share_mutable_buffers` pass instance, two whole programs.

        `ExecutorchBackendConfig` holds the pass, so a caller who exports twice
        through one config plans the second program on an instance that already
        carries the first. Everything this path records describes one program
        and nothing in `run_multimethod` clears it, so the second export has to
        start from an empty state.
        """
        mem_pass = MemoryPlanningPass(share_mutable_buffers=True)
        config = ExecutorchBackendConfig(
            memory_planning_pass=mem_pass, emit_mutable_buffer_names=True
        )

        def export_two_methods(n: int) -> Any:  # pyre-ignore[3]
            class StatefulModel(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.register_buffer("state", torch.zeros(n))

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self.state.add_(x) * 2

                def get_state(self) -> torch.Tensor:
                    return self.state

            model = StatefulModel().eval()
            forward_ep = export(model, (torch.ones(n),))
            with patch_forward(model, model.get_state):
                get_state_ep = export(model, ())
            return to_edge(
                {"forward": forward_ep, "get_state": get_state_ep}
            ).to_executorch(config)

        first = export_two_methods(4)
        second = export_two_methods(1024)

        # Each program's shared arena holds its own buffer, in both methods.
        for et, arena_size in ((first, 16), (second, 4096)):
            for plan in et.executorch_program.execution_plan:
                self.assertEqual(plan.non_const_buffer_sizes[2], arena_size, plan.name)

        # And the second program is the only one left on the pass. A leftover
        # spec is what sizes the arena from the wrong tensor, and which of them
        # is read depends on set iteration order, so the count is asserted
        # rather than the size alone.
        self.assertEqual(len(mem_pass.state.graph_modules), 2)
        self.assertEqual(len(mem_pass.state.mutable_buffers["state"]), 1)
        self.assertEqual(len(mem_pass.state.maybe_mutable_buffers["state"]), 1)


class TestDeviceAwareMemoryPlanning(unittest.TestCase):
    """Tests for per-device memory planning (separate buffers per device type)."""

    def _prepare_model(
        self,
    ) -> Tuple[GraphModule, ExportGraphSignature]:
        """Prepare ToyModelForMemPlanning through SpecPropPass + ToOutVarPass."""
        model = ToyModelForMemPlanning()
        inputs = model.get_random_inputs()
        edge = to_edge(export(model, inputs, strict=True))
        gm = edge.exported_program().graph_module
        gs = edge.exported_program().graph_signature
        gm = PassManager(passes=[SpecPropPass(), ToOutVarPass()])(gm).graph_module
        return gm, gs

    def _get_planned_specs(
        self,
        gm: GraphModule,
        gs: ExportGraphSignature,
    ) -> list[TensorSpec]:
        """Get the unique set of specs that apply_algo would plan."""
        return list(
            collect_specs_from_nodes(
                gm.graph.nodes,
                gs,
                do_assertion=False,
                ignore_graph_input=False,
                ignore_graph_output=False,
                ignore_mutable_buffers=False,
            )
        )

    def test_cpu_only_unchanged(self) -> None:
        """CPU-only specs produce bufsizes = [0, X] and an empty device list."""
        gm, gs = self._prepare_model()

        algo = MemoryPlanningAlgorithmSuite(algo_list=[greedy])
        bufsizes = apply_algo(algo, gm, 16, gs, enable_non_cpu_memory_planning=True)

        # The CUDA spec is the only tensor in its buffer
        self.assertEqual(bufsizes[0], 0)  # constants
        self.assertGreater(bufsizes[1], 0)  # CPU activations
        self.assertEqual(gm.meta["non_const_buffer_device"], [])

    def test_custom_pool_with_device_planning_raises(self) -> None:
        """Pre-assigned mem_ids + enable_non_cpu_memory_planning raises."""
        gm, gs = self._prepare_model()
        specs = self._get_planned_specs(gm, gs)

        # Pre-assign a custom mem_id AND set a non-CPU device
        specs[0].mem_id = 3
        specs[-1].device = DeviceType.CUDA

        algo = MemoryPlanningAlgorithmSuite(algo_list=[greedy])
        with self.assertRaises(NotImplementedError):
            apply_algo(algo, gm, 16, gs, enable_non_cpu_memory_planning=True)

    def test_pinning_a_non_cpu_tensor_is_allowed_without_device_planning(self) -> None:
        """Without per-device planning there is one block, so the pin holds.

        Every spec goes into the CPU bucket whatever its device attribute, and
        the pinned id is the arena index, so the refusal above does not reach a
        caller who has turned per-device planning off. The arena the pin names
        is an ordinary CPU:0 arena, and `non_const_buffer_device` says nothing
        about the CUDA tensor sitting in it.
        """
        gm, gs = self._prepare_model()
        specs = self._get_planned_specs(gm, gs)

        specs[-1].device = DeviceType.CUDA
        specs[-1].mem_id = 3

        algo = MemoryPlanningAlgorithmSuite(algo_list=[greedy])
        bufsizes = apply_algo(algo, gm, 16, gs, enable_non_cpu_memory_planning=False)

        self.assertEqual(len(bufsizes), 4)
        self.assertEqual(specs[-1].mem_id, 3)
        self.assertGreater(bufsizes[3], 0)
        self.assertEqual(gm.meta["non_const_buffer_device"], [])

    def test_all_cuda_no_wasted_slots(self) -> None:
        """CUDA-only specs produce [0, X] with CUDA at buffer index 1."""
        gm, gs = self._prepare_model()
        specs = self._get_planned_specs(gm, gs)
        for spec in specs:
            spec.device = DeviceType.CUDA

        algo = MemoryPlanningAlgorithmSuite(algo_list=[greedy])
        bufsizes = apply_algo(algo, gm, 16, gs, enable_non_cpu_memory_planning=True)

        # [0, cuda_size] — no wasted CPU buffer slot
        self.assertEqual(len(bufsizes), 2)
        self.assertEqual(bufsizes[0], 0)
        self.assertGreater(bufsizes[1], 0)
        # Device mapping should only contain non-CPU entries
        self.assertIn("non_const_buffer_device", gm.meta)
        device_map = gm.meta["non_const_buffer_device"]
        self.assertEqual(len(device_map), 1)
        self.assertEqual(device_map[0].buffer_idx, 1)
        self.assertEqual(device_map[0].device_type, DeviceType.CUDA)
        self.assertEqual(device_map[0].device_index, 0)

    def test_mixed_cpu_cuda_separate_buffers(self) -> None:
        """CPU specs at mem_id=1, CUDA specs at mem_id=2, separate sizes."""
        gm, gs = self._prepare_model()
        specs = self._get_planned_specs(gm, gs)

        # Set second half of specs to CUDA
        mid = len(specs) // 2
        self.assertGreater(mid, 0)
        cpu_specs = specs[:mid]
        cuda_specs = specs[mid:]
        for spec in cuda_specs:
            spec.device = DeviceType.CUDA

        algo = MemoryPlanningAlgorithmSuite(algo_list=[greedy])
        bufsizes = apply_algo(algo, gm, 16, gs, enable_non_cpu_memory_planning=True)

        # [constants, cpu_activations, cuda_activations]
        self.assertEqual(len(bufsizes), 3)
        self.assertEqual(bufsizes[0], 0)
        self.assertGreater(bufsizes[1], 0)
        self.assertGreater(bufsizes[2], 0)

        # CPU specs should have mem_id=1, CUDA specs should have mem_id=2
        for spec in cpu_specs:
            self.assertEqual(
                spec.mem_id, 1, f"CPU spec has wrong mem_id: {spec.mem_id}"
            )
        for spec in cuda_specs:
            self.assertEqual(
                spec.mem_id, 2, f"CUDA spec has wrong mem_id: {spec.mem_id}"
            )

    def test_mem_offset_correct_after_remap(self) -> None:
        """After remapping, mem_offset is relative to its own buffer."""
        gm, gs = self._prepare_model()
        specs = self._get_planned_specs(gm, gs)

        # Set the last spec to CUDA (sole CUDA tensor)
        cuda_spec = specs[-1]
        cuda_spec.device = DeviceType.CUDA

        algo = MemoryPlanningAlgorithmSuite(algo_list=[greedy])
        bufsizes = apply_algo(algo, gm, 16, gs, enable_non_cpu_memory_planning=True)

        # The CUDA spec is the only tensor in its buffer, so offset should be 0
        self.assertEqual(cuda_spec.mem_offset, 0)
        # The CUDA buffer should fit exactly this tensor
        cuda_mem_id = cuda_spec.mem_id
        self.assertIsNotNone(cuda_mem_id)
        assert cuda_mem_id is not None
        self.assertGreaterEqual(bufsizes[cuda_mem_id], cuda_spec.allocated_memory)

    def test_no_cross_device_memory_sharing(self) -> None:
        """Specs on different devices never share buffers, regardless of lifetime."""
        gm, gs = self._prepare_model()
        specs = self._get_planned_specs(gm, gs)
        self.assertGreaterEqual(len(specs), 2)

        # Assign alternating specs to CUDA to ensure some pairs have
        # non-overlapping lifetimes (which greedy would normally share).
        for i, spec in enumerate(specs):
            if i % 2 == 0:
                spec.device = DeviceType.CUDA

        algo = MemoryPlanningAlgorithmSuite(algo_list=[greedy])
        apply_algo(algo, gm, 16, gs, enable_non_cpu_memory_planning=True)

        # Verify CPU and CUDA specs have disjoint mem_ids
        cpu_mem_ids: set[int] = set()
        cuda_mem_ids: set[int] = set()
        for i, spec in enumerate(specs):
            if spec.mem_id is not None:
                if i % 2 == 0:
                    cuda_mem_ids.add(spec.mem_id)
                else:
                    cpu_mem_ids.add(spec.mem_id)

        self.assertTrue(
            cpu_mem_ids.isdisjoint(cuda_mem_ids),
            f"CPU {cpu_mem_ids} and CUDA {cuda_mem_ids} should not share buffers",
        )

    def test_different_device_indices_separate_buffers(self) -> None:
        """CUDA:0 and CUDA:1 specs get separate buffers."""
        gm, gs = self._prepare_model()
        specs = self._get_planned_specs(gm, gs)
        self.assertGreaterEqual(len(specs), 3)

        # specs[0] → CUDA:0, specs[1] → CUDA:1, rest → CPU
        specs[0].device = DeviceType.CUDA
        specs[0].device_index = 0
        specs[1].device = DeviceType.CUDA
        specs[1].device_index = 1

        algo = MemoryPlanningAlgorithmSuite(algo_list=[greedy])
        bufsizes = apply_algo(algo, gm, 16, gs, enable_non_cpu_memory_planning=True)

        # [constants, cpu, cuda:0, cuda:1]
        self.assertEqual(len(bufsizes), 4)

        # CUDA:0 and CUDA:1 should have different mem_ids
        self.assertNotEqual(specs[0].mem_id, specs[1].mem_id)
        # Both should differ from the CPU spec
        self.assertNotEqual(specs[0].mem_id, specs[2].mem_id)
        self.assertNotEqual(specs[1].mem_id, specs[2].mem_id)

        # Device mapping should only contain non-CPU entries with correct indices
        device_map = gm.meta["non_const_buffer_device"]
        for entry in device_map:
            self.assertEqual(entry.device_type, DeviceType.CUDA)
        cuda_indices = sorted(e.device_index for e in device_map)
        self.assertEqual(cuda_indices, [0, 1])

    def test_device_index_propagated(self) -> None:
        """NonConstBufferDevice entries carry the actual device_index, not 0."""
        gm, gs = self._prepare_model()
        specs = self._get_planned_specs(gm, gs)

        # Set the first spec to CUDA device index 3
        specs[0].device = DeviceType.CUDA
        specs[0].device_index = 3

        algo = MemoryPlanningAlgorithmSuite(algo_list=[greedy])
        apply_algo(algo, gm, 16, gs, enable_non_cpu_memory_planning=True)

        device_map = gm.meta["non_const_buffer_device"]
        self.assertEqual(len(device_map), 1)
        self.assertEqual(device_map[0].device_type, DeviceType.CUDA)
        self.assertEqual(device_map[0].device_index, 3)

    def test_disabled_falls_back_to_cpu(self) -> None:
        """With enable_non_cpu_memory_planning=False (default), CUDA specs are
        planned into CPU memory — one CPU pool, and nothing in the device
        list."""
        gm, gs = self._prepare_model()
        specs = self._get_planned_specs(gm, gs)
        for spec in specs:
            spec.device = DeviceType.CUDA

        algo = MemoryPlanningAlgorithmSuite(algo_list=[greedy])
        # Default: enable_non_cpu_memory_planning=False
        bufsizes = apply_algo(algo, gm, 16, gs)

        # All specs planned into a single CPU pool — same as CPU-only
        self.assertEqual(len(bufsizes), 2)
        self.assertEqual(bufsizes[0], 0)
        self.assertGreater(bufsizes[1], 0)
        self.assertEqual(gm.meta["non_const_buffer_device"], [])


class TestStorageBaseMemoryPlanning(unittest.TestCase):
    def _empty_graph_module(self) -> GraphModule:
        graph = Graph()
        graph.output(())
        return GraphModule({}, graph)

    def _make_storage_backed_specs(self) -> Tuple[TensorSpec, TensorSpec]:
        base = TensorSpec.from_tensor(torch.empty(10))
        child = TensorSpec.from_tensor(torch.empty(2))

        base.lifetime = [0, 1]
        child.lifetime = [0, 1]
        base.mem_id = 1
        child.mem_id = 1
        child.storage_base = base
        child.storage_base_offset = 16
        return base, child

    def test_greedy_places_storage_backed_spec_inside_base_object(self) -> None:
        base, child = self._make_storage_backed_specs()

        algo = MemoryPlanningAlgorithmSuite(algo_list=[greedy])
        algo(
            16,
            {base, child},
            self._empty_graph_module(),
            cast(ExportGraphSignature, None),
            0,
        )

        self.assertEqual(child.mem_id, base.mem_id)
        self.assertEqual(child.mem_obj_id, base.mem_obj_id)
        base_mem_offset = base.mem_offset
        self.assertIsNotNone(base_mem_offset)
        assert base_mem_offset is not None
        self.assertEqual(child.mem_offset, base_mem_offset + 16)

    def test_greedy_result_contains_storage_backed_full_plan(self) -> None:
        base, child = self._make_storage_backed_specs()
        base.realign(1)
        child.realign(1)

        result = greedy(
            1,
            {base, child},
            self._empty_graph_module(),
            cast(ExportGraphSignature, None),
            0,
        )

        base_result = result.spec_dict[base]
        child_result = result.spec_dict[child]
        self.assertEqual(child_result.mem_id, base_result.mem_id)
        self.assertEqual(child_result.mem_obj_id, base_result.mem_obj_id)
        self.assertEqual(child_result.mem_offset, base_result.mem_offset + 16)

    def test_greedy_resolves_chained_storage_base(self) -> None:
        # Build a storage chain where `base` owns the allocation, `child`
        # aliases `base`, and `grandchild` aliases `child`.
        base = TensorSpec.from_tensor(torch.empty(16, dtype=torch.uint8))
        child = TensorSpec.from_tensor(torch.empty(6, dtype=torch.uint8))
        grandchild = TensorSpec.from_tensor(torch.empty(2, dtype=torch.uint8))
        for spec in (base, child, grandchild):
            spec.lifetime = [0, 1]
            spec.mem_id = 1
        child.storage_base = base
        child.storage_base_offset = 8
        grandchild.storage_base = child
        grandchild.storage_base_offset = 4

        # Greedy should resolve the chain in dependency order and assign all
        # three specs to the same memory object.
        algo = MemoryPlanningAlgorithmSuite(algo_list=[greedy])
        algo(
            1,
            {base, child, grandchild},
            self._empty_graph_module(),
            cast(ExportGraphSignature, None),
            0,
        )

        self.assertEqual(child.mem_id, base.mem_id)
        self.assertEqual(grandchild.mem_id, base.mem_id)
        self.assertEqual(child.mem_obj_id, base.mem_obj_id)
        self.assertEqual(grandchild.mem_obj_id, base.mem_obj_id)
        base_mem_offset = base.mem_offset
        self.assertIsNotNone(base_mem_offset)
        assert base_mem_offset is not None
        # Offsets are accumulated through the chain: child is +8 from base,
        # grandchild is +4 from child, so grandchild is +12 from base.
        self.assertEqual(child.mem_offset, base_mem_offset + 8)
        self.assertEqual(grandchild.mem_offset, base_mem_offset + 12)

    def test_greedy_reserves_storage_base_lifetime_before_reuse(self) -> None:
        base = TensorSpec.from_tensor(torch.empty(16, dtype=torch.uint8))
        child = TensorSpec.from_tensor(torch.empty(8, dtype=torch.uint8))
        other = TensorSpec.from_tensor(torch.empty(12, dtype=torch.uint8))
        for spec in (base, child, other):
            spec.mem_id = 1
        base.lifetime = [0, 1]
        child.lifetime = [4, 5]
        other.lifetime = [4, 5]
        child.storage_base = base
        child.storage_base_offset = 8

        _extend_storage_base_lifetimes({base, child, other})

        algo = MemoryPlanningAlgorithmSuite(algo_list=[greedy])
        algo(
            1,
            {base, child, other},
            self._empty_graph_module(),
            cast(ExportGraphSignature, None),
            0,
        )

        self.assertEqual(base.lifetime, [0, 5])
        self.assertEqual(child.mem_id, base.mem_id)
        self.assertEqual(child.mem_obj_id, base.mem_obj_id)
        self.assertNotEqual(other.mem_obj_id, base.mem_obj_id)

    def test_set_alloc_node_spec_uses_shared_alloc_offset(self) -> None:
        base = TensorSpec.from_tensor(torch.empty(10))
        child = TensorSpec.from_tensor(torch.empty(2))

        graph = Graph()
        input_node = graph.placeholder("input")
        input_node.meta["spec"] = base
        other_node = graph.placeholder("other")
        other_node.meta["spec"] = base
        out_node = graph.placeholder("out")
        add_node = graph.call_function(
            torch.ops.aten.add.out,
            args=(input_node, other_node),
            kwargs={"out": out_node},
        )
        add_node.meta["spec"] = child
        add_node.meta["_share_alloc_with_arg_idx"] = 0
        add_node.meta["_shared_alloc_offset"] = 16
        graph.output(add_node)
        graph_module = GraphModule({}, graph)

        MemoryPlanningPass()._set_alloc_node_spec(graph_module)

        self.assertIs(child.storage_base, base)
        self.assertEqual(child.storage_base_offset, 16)

    def test_verifier_allows_storage_base_overlap(self) -> None:
        base, child = self._make_storage_backed_specs()

        algo = MemoryPlanningAlgorithmSuite(algo_list=[greedy])
        algo(
            1,
            {base, child},
            self._empty_graph_module(),
            cast(ExportGraphSignature, None),
            0,
        )

        graph = Graph()
        base_node = graph.placeholder("base")
        base_node.meta["spec"] = base
        child_node = graph.placeholder("child")
        child_node.meta["spec"] = child
        graph.output((base_node, child_node))
        graph_module = GraphModule({}, graph)

        verifier = Verifier(
            graph_module,
            alloc_graph_input=True,
            alloc_graph_output=True,
            alloc_mutable_buffers=True,
        )
        verifier.verify_storage_reuse()

    def test_verifier_allows_chained_storage_base_overlap(self) -> None:
        outer = TensorSpec.from_tensor(torch.empty(10))
        base = TensorSpec.from_tensor(torch.empty(6))
        child = TensorSpec.from_tensor(torch.empty(2))
        for spec in (outer, base, child):
            spec.lifetime = [0, 1]
            spec.mem_id = 1
        base.storage_base = outer
        base.storage_base_offset = 8
        child.storage_base = base
        child.storage_base_offset = 4

        algo = MemoryPlanningAlgorithmSuite(algo_list=[greedy])
        algo(
            1,
            {outer, base, child},
            self._empty_graph_module(),
            cast(ExportGraphSignature, None),
            0,
        )

        graph = Graph()
        outer_node = graph.placeholder("outer")
        outer_node.meta["spec"] = outer
        base_node = graph.placeholder("base")
        base_node.meta["spec"] = base
        child_node = graph.placeholder("child")
        child_node.meta["spec"] = child
        graph.output((outer_node, base_node, child_node))
        graph_module = GraphModule({}, graph)

        verifier = Verifier(
            graph_module,
            alloc_graph_input=True,
            alloc_graph_output=True,
            alloc_mutable_buffers=True,
        )
        verifier.verify_storage_reuse()


class TestInPlaceElemWise(unittest.TestCase):
    def _run_inplace_pipeline(
        self,
        model: torch.nn.Module,
        inputs: Tuple[torch.Tensor, ...],
        eligible_ops: set,  # pyre-ignore[2]
        algo: Callable[..., MemoryAlgoResult] = greedy,
    ) -> torch.fx.GraphModule:
        edge = to_edge(export(model.eval(), inputs, strict=True))
        ep = edge.exported_program()
        reinplace_pass(ep, ops_to_inplace=eligible_ops)
        graph_module = ep.graph_module
        mem_algo = MemoryPlanningAlgorithmSuite(algo_list=[algo])
        return PassManager(
            passes=[
                SpecPropPass(),
                ToOutVarPass(),
                MemoryPlanningPass(
                    memory_planning_algo=mem_algo,
                    alignment=1,
                ),
            ],
        )(graph_module).graph_module

    def test_basic_inplace_sharing(self) -> None:
        class Model(torch.nn.Module):
            def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                c = a + b
                d = c * b
                return d

        gm = self._run_inplace_pipeline(
            Model(),
            (torch.randn(10), torch.randn(10)),
            {exir_ops.edge.aten.mul.Tensor},
        )

        add_spec = None
        inplace_node_found = False
        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target == torch.ops.aten.add.out:
                add_spec = node.meta["spec"]
            if _is_inplace_node(node):
                inplace_node_found = True
                self.assertIs(node.meta["spec"], add_spec)

        self.assertIsNotNone(add_spec)
        self.assertTrue(inplace_node_found)

    def test_verifier_allows_inplace_overlap(self) -> None:
        class Model(torch.nn.Module):
            def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                c = a + b
                d = c * b
                return d

        gm = self._run_inplace_pipeline(
            Model(),
            (torch.randn(10), torch.randn(10)),
            {exir_ops.edge.aten.mul.Tensor},
        )

        verifier = Verifier(
            gm,
            alloc_graph_input=True,
            alloc_graph_output=True,
            alloc_mutable_buffers=True,
        )
        verifier.verify_storage_reuse()

    def test_verifier_allows_chained_inplace_overlap(self) -> None:
        class Model(torch.nn.Module):
            def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                c = a + b
                d = c * b
                e = d * b
                return e

        gm = self._run_inplace_pipeline(
            Model(),
            (torch.randn(10), torch.randn(10)),
            {exir_ops.edge.aten.mul.Tensor},
        )

        inplace_nodes = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function" and _is_inplace_node(node)
        ]
        self.assertEqual(len(inplace_nodes), 2)

        verifier = Verifier(
            gm,
            alloc_graph_input=True,
            alloc_graph_output=True,
            alloc_mutable_buffers=True,
        )
        verifier.verify_storage_reuse()

    def test_multi_user_blocks_inplace(self) -> None:
        class Model(torch.nn.Module):
            def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                c = a + b
                d = c * b
                e = c + d
                return e

        gm = self._run_inplace_pipeline(
            Model(),
            (torch.randn(10), torch.randn(10)),
            {exir_ops.edge.aten.mul.Tensor},
        )

        has_mul_out = any(
            node.target == torch.ops.aten.mul.out
            for node in gm.graph.nodes
            if node.op == "call_function"
        )
        self.assertTrue(has_mul_out)

    def test_no_inplace_when_ops_not_eligible(self) -> None:
        class Model(torch.nn.Module):
            def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                c = a + b
                d = c * b
                return d

        gm = self._run_inplace_pipeline(
            Model(),
            (torch.randn(10), torch.randn(10)),
            set(),
        )

        has_inplace = any(
            _is_inplace_node(node)
            for node in gm.graph.nodes
            if node.op == "call_function"
        )
        self.assertFalse(has_inplace)


class _DeviceTaggingPass(MemoryPlanningPass):
    """Tags the outputs of one op onto CUDA, in the method that contains it.

    A stand-in for a delegate that only some methods lower onto an accelerator,
    so that two methods of the same program disagree about which devices exist.
    """

    # pyre-ignore[2]: torch.fx targets are untyped.
    def __init__(self, target, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.target = target

    def run(
        self,
        graph_module: torch.fx.GraphModule,
        graph_signature: Optional[ExportGraphSignature] = None,
    ) -> PassResult:
        for node in graph_module.graph.nodes:
            if node.op == "call_function" and node.target == self.target:
                for spec in get_node_tensor_specs(node):
                    spec.device = DeviceType.CUDA
        return super().run(graph_module, graph_signature)


class _CustomPoolPass(MemoryPlanningPass):
    """Pins every mul.out output onto arena 3, the pattern documented in
    docs/source/compiler-memory-planning.md.
    """

    def run(
        self,
        graph_module: torch.fx.GraphModule,
        graph_signature: Optional[ExportGraphSignature] = None,
    ) -> PassResult:
        for node in graph_module.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.mul.out:
                for spec in get_node_tensor_specs(node):
                    spec.mem_id = 3
        return super().run(graph_module, graph_signature)


class _BufferPinningPass(MemoryPlanningPass):
    """Pins the placeholder of one named buffer onto arena 3."""

    def __init__(self, fqn: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.fqn = fqn

    def run(
        self,
        graph_module: torch.fx.GraphModule,
        graph_signature: Optional[ExportGraphSignature] = None,
    ) -> PassResult:
        assert graph_signature is not None
        for node in graph_module.graph.nodes:
            if node.op == "placeholder" and isinstance(node.target, str):
                if graph_signature.inputs_to_buffers.get(node.target) == self.fqn:
                    get_node_tensor_specs(node)[0].mem_id = 3
        return super().run(graph_module, graph_signature)


class _PoolOrPinPass(MemoryPlanningPass):
    """Pins `mul.out` outputs, and one read-only buffer, onto arena 3.

    The named buffer's placeholder is pinned only in the methods that do not
    mutate it, so a program whose first method mutates the buffer and whose
    second only reads it is refused by ``run`` after the first has been
    recorded on the pass -- with a CPU block the pool has widened. A
    single-method program that mutates the buffer and has no `mul.out` carries
    no pin at all, so the same instance plans it.
    """

    def __init__(self, fqn: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.fqn = fqn

    def run(
        self,
        graph_module: torch.fx.GraphModule,
        graph_signature: Optional[ExportGraphSignature] = None,
    ) -> PassResult:
        assert graph_signature is not None
        mutated = self.fqn in graph_signature.buffers_to_mutate.values()
        for node in graph_module.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.mul.out:
                for spec in get_node_tensor_specs(node):
                    spec.mem_id = 3
            if (
                not mutated
                and node.op == "placeholder"
                and isinstance(node.target, str)
                and graph_signature.inputs_to_buffers.get(node.target) == self.fqn
            ):
                get_node_tensor_specs(node)[0].mem_id = 3
        return super().run(graph_module, graph_signature)


class _AddOnlyStateModel(nn.Module):
    """Mutates `state` with no `mul.out` in the graph.

    `_PoolOrPinPass` therefore pins nothing here, leaving a single activation
    arena.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("state", torch.zeros(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.state.add_(x)
        return self.state + x


class _BufferDeviceTaggingPass(MemoryPlanningPass):
    """Plans the placeholder of one named buffer onto CUDA.

    A declared shared buffer takes its arena from its own device, so tagging
    the buffer rather than an op is what puts a dedicated arena on a device.
    """

    def __init__(self, fqn: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.fqn = fqn

    def run(
        self,
        graph_module: torch.fx.GraphModule,
        graph_signature: Optional[ExportGraphSignature] = None,
    ) -> PassResult:
        assert graph_signature is not None
        for node in graph_module.graph.nodes:
            if node.op == "placeholder" and isinstance(node.target, str):
                if graph_signature.inputs_to_buffers.get(node.target) == self.fqn:
                    get_node_tensor_specs(node)[0].device = DeviceType.CUDA
        return super().run(graph_module, graph_signature)


class _BufferDeviceIndexPass(MemoryPlanningPass):
    """Plans the placeholder of `state` onto CPU at a chosen device index.

    Nothing in the tree produces a host tensor at a non-zero index, so the
    index is set by hand here, the way the pass above sets a device type by
    hand.
    """

    def __init__(self, index: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.index = index

    def run(
        self,
        graph_module: torch.fx.GraphModule,
        graph_signature: Optional[ExportGraphSignature] = None,
    ) -> PassResult:
        assert graph_signature is not None
        for node in graph_module.graph.nodes:
            if node.op == "placeholder" and isinstance(node.target, str):
                if graph_signature.inputs_to_buffers.get(node.target) == "state":
                    get_node_tensor_specs(node)[0].device_index = self.index
        return super().run(graph_module, graph_signature)


class _EverythingButBuffersOnDevicePass(MemoryPlanningPass):
    """Plans every top-level tensor except the buffers onto CUDA.

    A stand-in for a program whose whole body runs on an accelerator. The
    buffers it skips are the declared shared ones, which are withheld from the
    algorithm too, so the top-level graph has no CPU partition at all, while
    the control-flow submodules -- which apply_algo plans by recursing with
    per-device planning off -- still carry CPU arena indices.
    """

    def run(
        self,
        graph_module: torch.fx.GraphModule,
        graph_signature: Optional[ExportGraphSignature] = None,
    ) -> PassResult:
        assert graph_signature is not None
        for node in graph_module.graph.nodes:
            if (
                node.op == "placeholder"
                and isinstance(node.target, str)
                and node.target in graph_signature.inputs_to_buffers
            ):
                continue
            for spec in get_node_tensor_specs(node):
                spec.device = DeviceType.CUDA
        return super().run(graph_module, graph_signature)


class _PlannedMetaRecordingPass(_EverythingButBuffersOnDevicePass):
    """Keeps each planned graph module and the arena sizes it planned itself."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.planned: List[Tuple[torch.fx.GraphModule, List[int]]] = []

    def run(
        self,
        graph_module: torch.fx.GraphModule,
        graph_signature: Optional[ExportGraphSignature] = None,
    ) -> PassResult:
        result = super().run(graph_module, graph_signature)
        self.planned.append(
            (graph_module, list(graph_module.meta["non_const_buffer_sizes"]))
        )
        return result


class CondStateModel(nn.Module):
    """Reads the buffer AFTER a control-flow op and updates it afterwards.

    The ordinary persistent-state shape. Functionalization hoists the read of a
    mutated buffer to the front of the graph, so a buffer that is only written
    would hide a collision with the submodule region; reading after the `cond`
    exposes it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("state", torch.zeros(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def true_branch(t: torch.Tensor) -> torch.Tensor:
            return (t * 3.0 + 1.0) * 2.0

        def false_branch(t: torch.Tensor) -> torch.Tensor:
            return (t + 5.0) * 0.5 - 1.0

        y = torch.cond(x.sum() > 4.0, true_branch, false_branch, [x])
        z = self.state * 2.0
        self.state.add_(x)
        return z + y


class TwoBufferModel(nn.Module):
    """Each method's body touches a different one of the two buffers."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("cache_a", torch.zeros(4))
        self.register_buffer("cache_b", torch.zeros(8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.cache_a.add_(x)
        return self.cache_a * 2.0

    def touch_b(self, x: torch.Tensor) -> torch.Tensor:
        self.cache_b.add_(torch.cat([x, x]))
        return self.cache_b * 3.0


class ThreeBufferModel(nn.Module):
    """Three declared buffers of differing sizes, all mutated by one method.

    12, 20 and 32 bytes: none of the three is the same size, and the first is
    not a whole number of 16-byte alignment units.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("cache_x", torch.zeros(3))
        self.register_buffer("cache_y", torch.zeros(5))
        self.register_buffer("cache_z", torch.zeros(8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.cache_x.add_(1.0)
        self.cache_y.add_(2.0)
        self.cache_z.add_(x)
        return self.cache_z * 2.0


class PeekStateModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("state", torch.zeros(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.state.add_(x)
        return self.state * 2.0

    def peek(self, x: torch.Tensor) -> torch.Tensor:
        return self.state + x


class InitializedStateModel(nn.Module):
    """`PeekStateModel` with a state worth serializing.

    The buffer starts at 1.0 rather than 0.0, so a pass setting et_init_buffer
    on it emits bytes that a re-initialization visibly puts back.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("state", torch.ones(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.state.add_(x)
        return self.state * 2.0

    def peek(self, x: torch.Tensor) -> torch.Tensor:
        return self.state + x


class ViewedDeviceTensorModel(nn.Module):
    """A view over a tensor a pass will tag onto a device, plus a shared buffer.

    `remove_view_copy` turns the `view_copy` into a `memory.view` node carrying
    a `_ViewSpec`: an object of its own that forwards mem_id to the spec of its
    base. Reaching both while renumbering arenas would move one mem_id twice.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("state", torch.zeros(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.state.add_(x)
        y = torch.mul(x, x)
        return y.view(-1) + self.state


class TwoMutableBufferModel(nn.Module):
    """Two mutable buffers, both written by both methods.

    Declaring one of them leaves the other to the planning algorithm, which is
    what a caller adding `shared_buffer_fqns` to an existing
    `share_mutable_buffers` pass gives up.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("kept", torch.zeros(4))
        self.register_buffer("dropped", torch.zeros(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.kept.add_(x)
        self.dropped.add_(x)
        return self.kept + self.dropped

    def second(self, x: torch.Tensor) -> torch.Tensor:
        self.kept.add_(x * 2.0)
        self.dropped.add_(x * 3.0)
        return self.kept * self.dropped


class EmptyStateModel(nn.Module):
    """A mutated buffer with no elements, so its dedicated arena is zero bytes.

    It is the only declared buffer, so nothing else gives that arena a size, and
    it is mutated, so no other check on the declared names fires first.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("state", torch.zeros(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.state.add_(1.0)
        return x + 1.0


class TestDedicatedSharedArena(unittest.TestCase):
    """Sharing declared buffers through a dedicated arena per device.

    ``shared_buffer_fqns`` withholds the named buffers from the planning
    algorithm and gives them an arena of their own, at an index and offset that
    mean the same thing in every method of the program.
    """

    def _state_placements(
        self, et: Any, fqn: str
    ) -> List[Tuple[int, int]]:  # pyre-ignore[2]
        placements = []
        for plan in et.executorch_program.execution_plan:
            values = [
                v
                for v in plan.values
                if getattr(v.val, "extra_tensor_info", None) is not None
                and v.val.extra_tensor_info.fully_qualified_name == fqn
            ]
            self.assertEqual(len(values), 1, f"{fqn} in {plan.name}")
            info = values[0].val.allocation_info
            placements.append((info.memory_id, info.memory_offset_low))
        return placements

    # pyre-ignore[2]: schema types are untyped here.
    def _cuda_arenas(self, plan) -> List[int]:
        return sorted(
            e.buffer_idx
            for e in (plan.non_const_buffer_device or [])
            if e.device_type == DeviceType.CUDA
        )

    def _to_executorch(
        self,
        eps: dict[str, Any],  # pyre-ignore[2]
        mem_pass: MemoryPlanningPass,
        **config: Any,
    ) -> Any:  # pyre-ignore[3]
        return to_edge(eps).to_executorch(
            ExecutorchBackendConfig(
                memory_planning_pass=mem_pass,
                emit_mutable_buffer_names=True,
                **config,
            )
        )

    # pyre-ignore[3]: the pybindings module is untyped.
    def _load(self, et: Any) -> Any:  # pyre-ignore[2]
        return _load_for_executorch_from_buffer(et.buffer)

    def test_methods_with_different_device_sets_agree_on_arena_indices(self) -> None:
        """One method has a CUDA arena and the other has none.

        Arena indices are positional in the .pte, so the method without the
        device must still be padded to the same length rather than having its
        own arenas renumbered. This checks the emitted plans; that the padded
        method still loads on a build with no CUDA allocator is
        test_padded_device_arena_method_runs_without_a_device_allocator.
        """
        model = PeekStateModel().eval()
        forward_ep = export(model, (torch.ones(4),))
        with patch_forward(model, model.peek):
            peek_ep = export(model, (torch.ones(4),))

        et = self._to_executorch(
            {"forward": forward_ep, "peek": peek_ep},
            _DeviceTaggingPass(
                torch.ops.aten.mul.out,  # only `forward` has a mul.out
                share_mutable_buffers=True,
                shared_buffer_fqns=frozenset({"state"}),
            ),
            enable_non_cpu_memory_planning=True,
        )

        plans = et.executorch_program.execution_plan
        lengths = {len(plan.non_const_buffer_sizes) for plan in plans}
        self.assertEqual(len(lengths), 1, "arena counts differ across methods")

        cuda_indices = [self._cuda_arenas(plan) for plan in plans]
        tagged = [i for i, indices in enumerate(cuda_indices) if indices]
        self.assertEqual(len(tagged), 1, "only one method plans onto CUDA")
        on_cuda, padded = tagged[0], 1 - tagged[0]
        self.assertEqual(len(cuda_indices[on_cuda]), 1)
        cuda_arena = cuda_indices[on_cuda][0]

        # The method that has no CUDA tensors still carries the slot, empty --
        # that is what keeps the two numberings aligned -- but leaves it CPU, so
        # loading it does not require a CUDA allocator.
        self.assertGreater(plans[on_cuda].non_const_buffer_sizes[cuda_arena], 0)
        self.assertEqual(plans[padded].non_const_buffer_sizes[cuda_arena], 0)
        self.assertIsNone(plans[padded].non_const_buffer_device)

        placements = self._state_placements(et, "state")
        self.assertEqual(placements[0], placements[1])
        self.assertNotIn(placements[0][0], cuda_indices[on_cuda])

    def test_shared_arena_of_a_device_a_method_never_declared_is_empty(self) -> None:
        """A method carries the slot of another method's device arena, at zero.

        The numbering is program-wide, so every method has an index for every
        shared arena. A method that declared nothing on that device names none
        of it, so reserving its bytes would only cost host memory.

        The two methods come from different modules because non-strict
        `torch.export` lifts every registered buffer of one module into every
        one of its methods, which would leave both methods declaring both
        buffers. The first method is `touch_b` rather than `forward` because
        `cache_b` has to be mutated somewhere.
        """

        class OneBufferModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("cache_a", torch.zeros(4))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.cache_a.add_(x)
                return self.cache_a * 2.0

        mem_pass = _BufferDeviceTaggingPass(
            "cache_b",
            share_mutable_buffers=True,
            shared_buffer_fqns=frozenset({"cache_a", "cache_b"}),
        )
        # to_executorch pushes this onto the pass instance from the top-level
        # config; these methods are planned directly, so set it here.
        mem_pass.enable_non_cpu_memory_planning = True

        two_buffer = TwoBufferModel().eval()
        with patch_forward(two_buffer, two_buffer.touch_b):
            exported = [export(two_buffer, (torch.ones(4),))]
        exported.append(export(OneBufferModel().eval(), (torch.ones(4),)))

        graph_modules = []
        for ep in exported:
            edge = to_edge(ep)
            gm = edge.exported_program().graph_module
            gs = edge.exported_program().graph_signature
            gm = PassManager(passes=[SpecPropPass(), ToOutVarPass()])(gm).graph_module
            mem_pass.run(gm, gs)
            graph_modules.append(gm)
        mem_pass.run_multimethod()

        both, only_a = graph_modules
        sizes = [gm.meta["non_const_buffer_sizes"] for gm in graph_modules]
        self.assertEqual(len(sizes[0]), len(sizes[1]))

        cuda_arenas = {
            entry.buffer_idx
            for entry in both.meta["non_const_buffer_device"]
            if entry.device_type == DeviceType.CUDA
        }
        self.assertEqual(len(cuda_arenas), 1, cuda_arenas)
        cuda_arena = cuda_arenas.pop()

        self.assertGreater(sizes[0][cuda_arena], 0)
        self.assertEqual(sizes[1][cuda_arena], 0)
        # And the method that leaves it empty also leaves it CPU, so it does not
        # ask for a CUDA allocator to load.
        self.assertEqual(only_a.meta["non_const_buffer_device"], [])
        # Nothing in that method names the arena it zeroed.
        self.assertNotIn(
            cuda_arena,
            {
                spec.mem_id
                for node in only_a.graph.nodes
                for spec in get_node_tensor_specs(node)
                if spec.mem_id is not None
            },
        )

    @unittest.skipUnless(_HAS_RUNTIME, "portable_lib not built")
    def test_padded_device_arena_method_runs_without_a_device_allocator(self) -> None:
        """The padded method loads where the device it pads for does not exist.

        DeviceMemoryBuffer::create fails on a missing allocator before it looks
        at the size, so a padding slot tagged CUDA fails the load outright where
        no CUDA allocator is registered -- which is what makes this discriminate
        here. Whether one is registered is a property of the build, so against
        a build that links the CUDA backend this still passes, but on the
        padding slot reaching a live allocator rather than on it staying
        untagged.
        """
        model = PeekStateModel().eval()
        forward_ep = export(model, (torch.ones(4),))
        with patch_forward(model, model.peek):
            peek_ep = export(model, (torch.ones(4),))

        et = self._to_executorch(
            {"forward": forward_ep, "peek": peek_ep},
            _DeviceTaggingPass(
                torch.ops.aten.mul.out,  # only `forward` has a mul.out
                share_mutable_buffers=True,
                shared_buffer_fqns=frozenset({"state"}),
            ),
            enable_non_cpu_memory_planning=True,
        )

        peek_plan = next(
            plan for plan in et.executorch_program.execution_plan if plan.name == "peek"
        )
        # Asserted here as well as through the load, because the load only
        # discriminates on a build with no CUDA allocator registered: `peek`
        # carries the slot for `forward`'s CUDA arena and it is not tagged CUDA.
        self.assertEqual(self._cuda_arenas(peek_plan), [])

        actual = self._load(et).run_method("peek", [torch.ones(4)])[0]
        self.assertTrue(torch.allclose(actual, torch.ones(4)), actual)

    def test_every_method_places_every_declared_buffer(self) -> None:
        """Two declared buffers, one arena, the same addresses in both methods.

        Each method's body touches only one of the two buffers, but non-strict
        `torch.export` -- the default, and what this test uses -- lifts every
        registered buffer into every method's signature, so both are placed in
        both. That lifting is why a test wanting methods with differing declared
        sets has to export them from separate modules, as
        `test_shared_arena_of_a_device_a_method_never_declared_is_empty` does.
        """
        model = TwoBufferModel().eval()
        forward_ep = export(model, (torch.ones(4),))
        with patch_forward(model, model.touch_b):
            other_ep = export(model, (torch.ones(4),))

        et = self._to_executorch(
            {"forward": forward_ep, "touch_b": other_ep},
            MemoryPlanningPass(
                share_mutable_buffers=True,
                shared_buffer_fqns=frozenset({"cache_a", "cache_b"}),
            ),
        )

        plans = et.executorch_program.execution_plan
        placements = [
            [self._find_placement(plan, fqn) for fqn in ("cache_a", "cache_b")]
            for plan in plans
        ]
        self.assertNotIn(None, placements[0] + placements[1])
        # Both buffers are placed in both methods, at the same two addresses...
        self.assertEqual(placements[0], placements[1])
        # ...in one arena, at non-overlapping offsets sorted by fqn:
        # cache_a (16B once aligned) then cache_b.
        arena = placements[0][0][0]
        self.assertEqual(placements[0], [(arena, 0), (arena, 16)])
        # ...and that arena is the same size in both, since it holds both.
        self.assertEqual(
            plans[0].non_const_buffer_sizes[arena],
            plans[1].non_const_buffer_sizes[arena],
        )
        self.assertEqual(plans[0].non_const_buffer_sizes[arena], 48)

    # pyre-ignore[2,3]: schema types are untyped here.
    def _find_placement(self, plan, fqn: str):
        for value in plan.values:
            info = getattr(value.val, "extra_tensor_info", None)
            if info is not None and info.fully_qualified_name == fqn:
                return (
                    value.val.allocation_info.memory_id,
                    value.val.allocation_info.memory_offset_low,
                )
        return None

    def test_custom_pool_keeps_its_arena(self) -> None:
        """A pass pinning mem_id=3 must not be displaced by the shared arena.

        Indices are handed out in order, so the custom pool widens the CPU block
        and the shared arena lands after it. Only `forward` has the pinned op,
        so the block is also wider than `peek` needs: `peek` carries the extra
        slots empty rather than having its own arena renumbered into one.
        """
        model = PeekStateModel().eval()
        forward_ep = export(model, (torch.ones(4),))
        with patch_forward(model, model.peek):
            peek_ep = export(model, (torch.ones(4),))

        et = self._to_executorch(
            {"forward": forward_ep, "peek": peek_ep},
            _CustomPoolPass(
                share_mutable_buffers=True,
                shared_buffer_fqns=frozenset({"state"}),
            ),
        )

        plans = et.executorch_program.execution_plan
        wide, narrow = plans[0], plans[1]
        # The constants placeholder, the three arenas the pinned pool stretches
        # the CPU block to, and the shared arena closing it.
        self.assertEqual(len(wide.non_const_buffer_sizes), 5)
        self.assertEqual(len(narrow.non_const_buffer_sizes), 5)
        self.assertGreater(wide.non_const_buffer_sizes[3], 0)
        self.assertEqual(narrow.non_const_buffer_sizes[3], 0)
        self.assertGreater(wide.non_const_buffer_sizes[1], 0)
        self.assertGreater(narrow.non_const_buffer_sizes[1], 0)
        placements = self._state_placements(et, "state")
        self.assertEqual(placements[0], placements[1])
        self.assertEqual(placements[0], (4, 0))

    def test_pinning_a_declared_buffer_to_a_custom_pool_raises(self) -> None:
        """A declared buffer goes to the dedicated arena, so a pin cannot hold.

        The two placements are mutually exclusive and the pass cannot honor
        both, so it says so rather than overwriting the caller's mem_id and
        landing the tensor in a pool the caller did not choose.
        """
        model = PeekStateModel().eval()
        mem_pass = _BufferPinningPass(
            "state",
            share_mutable_buffers=True,
            shared_buffer_fqns=frozenset({"state"}),
        )
        with self.assertRaises(ValueError) as cm:
            self._to_executorch({"forward": export(model, (torch.ones(4),))}, mem_pass)
        message = str(cm.exception)
        self.assertIn("'state'", message)
        # Wording only this check uses. The refusal for a declared buffer no
        # method mutates names the buffer and explains mem_id in prose, so
        # matching the two separately would leave this test green on it.
        self.assertIn("already has a mem_id or mem_offset assigned", message)

    @unittest.skipUnless(_HAS_RUNTIME, "portable_lib not built")
    def test_one_method_reads_what_another_wrote_to_the_shared_buffer(self) -> None:
        """The cross-method sharing the feature exists for, observed end to end.

        `_load_program_from_buffer` hands every host-only method the same
        allocation for each host arena, so a CPU shared arena is backed by one
        buffer for the whole program. `forward` writes `state`, `peek` reads it,
        and the same program loaded through `_load_for_executorch_from_buffer`,
        where arena sharing is off, does not see the write.

        Run through the real pipeline rather than a bare PassManager because
        that is what reaches SpecPropPass.update_placeholder_tensor_specs, which
        marks `peek`'s read-only placeholder const and so drops it from the
        planning algorithm.

        The custom pool is what makes this a test of the dedicated arena rather
        than of sharing in general: without it the buffer lands at arena two,
        offset zero, which is where the legacy `share_mutable_buffers` path
        hardcodes it, so every assertion below would hold on a build that never
        ran this feature.
        """
        model = PeekStateModel().eval()
        forward_ep = export(model, (torch.ones(4),))
        with patch_forward(model, model.peek):
            peek_ep = export(model, (torch.ones(4),))

        et = self._to_executorch(
            {"forward": forward_ep, "peek": peek_ep},
            _CustomPoolPass(
                share_mutable_buffers=True,
                shared_buffer_fqns=frozenset({"state"}),
            ),
        )
        # Both methods on the one CPU arena the program path shares, past the
        # arenas the pinned pool stretched the CPU block to.
        self.assertEqual(self._state_placements(et, "state"), [(4, 0), (4, 0)])

        program = _load_program_from_buffer(et.buffer)
        forward = program.load_method("forward")
        peek = program.load_method("peek")

        forward([torch.full((4,), 3.0)])
        seen = peek([torch.zeros(4)])[0]
        self.assertTrue(torch.allclose(seen, torch.full((4,), 3.0)), seen)

        # The same program through the loader that does not share arenas: the
        # agreement is compile-time only, so the write is invisible there.
        module = self._load(et)
        module.run_method("forward", [torch.full((4,), 3.0)])
        unshared = module.run_method("peek", [torch.zeros(4)])[0]
        self.assertTrue(torch.allclose(unshared, torch.zeros(4)), unshared)

    def test_an_initialized_buffer_shared_by_two_methods_raises(self) -> None:
        """The runtime re-initializes on every load, and the arena is one.

        et_init_buffer makes the emitter serialize the buffer's stored bytes,
        and the runtime copies them into the planned allocation while it loads
        each method that has them. With the allocation shared, loading `peek`
        after `forward` has run puts the initial value back over the live one --
        measured: `forward` writes 2.0 and `peek` then reads 1.0.
        """
        model = InitializedStateModel().eval()
        forward_ep = export(model, (torch.ones(4),))
        with patch_forward(model, model.peek):
            peek_ep = export(model, (torch.ones(4),))

        with self.assertRaises(ValueError) as cm:
            self._to_executorch(
                {"forward": forward_ep, "peek": peek_ep},
                MemoryPlanningPass(
                    share_mutable_buffers=True,
                    shared_buffer_fqns=frozenset({"state"}),
                ),
                passes=[InitializedMutableBufferPass(["state"])],
            )
        message = str(cm.exception)
        self.assertIn("'state'", message)
        # Wording only this check uses: the never-mutated refusal also names
        # the buffer and explains et_init_buffer, so matching those two
        # separately would leave this test green on it.
        self.assertIn("carry an initializer", message)

    def test_an_initialized_buffer_used_by_one_method_raises_too(self) -> None:
        """One owner is not a safe case, so it is refused with the rest.

        A single method is not loaded a single time: PyProgram::load_method
        builds a fresh Method on every call, and each build copies the stored
        bytes back over whatever the last run left in the shared arena. So the
        refusal is on the initializer, not on the number of methods that have
        the buffer.
        """
        model = InitializedStateModel().eval()
        with self.assertRaises(ValueError) as cm:
            self._to_executorch(
                {"forward": export(model, (torch.ones(4),))},
                MemoryPlanningPass(
                    share_mutable_buffers=True,
                    shared_buffer_fqns=frozenset({"state"}),
                ),
                passes=[InitializedMutableBufferPass(["state"])],
            )
        self.assertIn("carry an initializer", str(cm.exception))

    def test_shared_buffer_leaves_no_hole_in_the_regular_arena(self) -> None:
        """Withholding the buffer must shrink the arena it would have occupied,
        not leave a gap where it would have sat.
        """
        model = PeekStateModel().eval()
        plain = self._to_executorch(
            {"forward": export(model, (torch.ones(4),))}, MemoryPlanningPass()
        )
        shared = self._to_executorch(
            {"forward": export(model, (torch.ones(4),))},
            MemoryPlanningPass(
                share_mutable_buffers=True,
                shared_buffer_fqns=frozenset({"state"}),
            ),
        )
        plain_sizes = plain.executorch_program.execution_plan[0].non_const_buffer_sizes
        shared_sizes = shared.executorch_program.execution_plan[
            0
        ].non_const_buffer_sizes
        buffer_bytes = shared_sizes[2]
        self.assertEqual(buffer_bytes, 16)
        self.assertEqual(shared_sizes[1], plain_sizes[1] - buffer_bytes)

    def test_undeclared_program_layout_is_untouched(self) -> None:
        """A pass with no sharing configured still emits the default layout.

        apply_algo writes non_const_buffer_device on every program, this one
        included, so the empty list it gets here has to serialize as null.
        """
        model = PeekStateModel().eval()
        et = self._to_executorch(
            {"forward": export(model, (torch.ones(4),))}, MemoryPlanningPass()
        )
        plan = et.executorch_program.execution_plan[0]
        self.assertEqual(len(plan.non_const_buffer_sizes), 2)
        # apply_algo writes the key whatever the program, empty when every
        # arena is CPU:0; turning an empty list into null is the emitter's.
        gm = et.exported_program("forward").graph_module
        self.assertEqual(gm.meta["non_const_buffer_device"], [])
        self.assertIsNone(plan.non_const_buffer_device)

    def test_a_view_over_a_device_tensor_stays_out_of_the_shared_arena(self) -> None:
        """The arena renumbering must reach one tensor's mem_id exactly once.

        `remove_view_copy` and `enable_non_cpu_memory_planning` are both on by
        default, and a view over a device tensor is ordinary in a real model.
        The view's `_ViewSpec` is a second object holding the same mem_id, so a
        walk keyed on object identity would renumber that id twice and land a
        scratch activation on top of the buffer the arena exists to protect.
        """
        et = self._to_executorch(
            {"forward": export(ViewedDeviceTensorModel().eval(), (torch.ones(4),))},
            _DeviceTaggingPass(
                torch.ops.aten.mul.out,
                share_mutable_buffers=True,
                shared_buffer_fqns=frozenset({"state"}),
            ),
            enable_non_cpu_memory_planning=True,
        )

        plan = et.executorch_program.execution_plan[0]
        placed = [
            (v.val.allocation_info.memory_id, v.val.allocation_info.memory_offset_low)
            for v in plan.values
            if getattr(v.val, "allocation_info", None) is not None
        ]
        shared = self._state_placements(et, "state")[0]
        cuda_arenas = self._cuda_arenas(plan)
        self.assertEqual(len(cuda_arenas), 1)
        self.assertNotIn(shared[0], cuda_arenas)

        # Nothing but the declared buffer may sit in its arena.
        self.assertEqual([p for p in placed if p[0] == shared[0]], [shared])
        # The `mul` output and the view of it are the two, both in the device
        # arena the double mapping would otherwise leave sized and empty.
        in_device_arena = [p for p in placed if p[0] == cuda_arenas[0]]
        self.assertEqual(in_device_arena, [(3, 0), (3, 0)])

    def test_a_view_spec_carried_by_another_node_is_still_skipped(self) -> None:
        """The walk recognizes a _ViewSpec by type, not by the node it sits on.

        `_alias_inplace_result_specs` puts a mutated input's spec object on the
        in-place op's node, so an in-place op over a view would carry the view's
        _ViewSpec on a node whose target is the in-place op: past the
        `memory.view` skip, and past the id() dedup. Whether functionalization
        leaves that shape is unproven, so this pins the walk's contract rather
        than covering a defect anything has reached, and the graph is built by
        hand for the same reason.
        """
        base = TensorSpec(dtype=torch.float32, shape=torch.Size([4]))
        viewed = _ViewSpec(base, [2, 2])

        graph = Graph()
        x = graph.placeholder("x")
        x.meta["spec"] = base
        view_node = graph.call_function(view, (x, [2, 2]))
        view_node.meta["spec"] = viewed
        in_place = graph.call_function(torch.ops.aten.add_.Tensor, (view_node, x))
        in_place.meta["spec"] = viewed
        graph.output(in_place)

        reached = _iter_unique_specs(GraphModule(torch.nn.Module(), graph))
        # The base once, on the node that owns it, and the _ViewSpec never: a
        # renumbering walk therefore writes the base's mem_id exactly once.
        self.assertEqual([(node.name, spec) for node, spec in reached], [("x", base)])

    def test_an_undeclared_mutable_buffer_warns_that_it_is_not_shared(self) -> None:
        """Naming one buffer leaves the rest to the algorithm, and says so.

        `share_mutable_buffers` on its own shares every mutable buffer, so
        adding `shared_buffer_fqns` to an existing caller takes cross-method
        sharing away from everything it does not name. Nothing at runtime
        reports that, so export does.
        """
        model = TwoMutableBufferModel().eval()
        forward_ep = export(model, (torch.ones(4),))
        with patch_forward(model, model.second):
            second_ep = export(model, (torch.ones(4),))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            et = self._to_executorch(
                {"forward": forward_ep, "second": second_ep},
                MemoryPlanningPass(
                    share_mutable_buffers=True,
                    shared_buffer_fqns=frozenset({"kept"}),
                ),
            )
        said = [str(w.message) for w in caught if "'dropped'" in str(w.message)]
        self.assertEqual(len(said), 1, [str(w.message) for w in caught])
        self.assertIn("is not shared across methods", said[0])

        kept = self._state_placements(et, "kept")
        self.assertEqual(kept[0], kept[1])
        # What the warning is about: the buffer nobody named is placed by each
        # method's own plan, so the two methods do not agree on where it is.
        self.assertNotEqual(*self._state_placements(et, "dropped"))

    def test_declaring_every_mutable_buffer_does_not_warn(self) -> None:
        """The warning above must not fire on the shape it does not describe."""
        model = TwoMutableBufferModel().eval()
        forward_ep = export(model, (torch.ones(4),))
        with patch_forward(model, model.second):
            second_ep = export(model, (torch.ones(4),))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            et = self._to_executorch(
                {"forward": forward_ep, "second": second_ep},
                MemoryPlanningPass(
                    share_mutable_buffers=True,
                    shared_buffer_fqns=frozenset({"kept", "dropped"}),
                ),
            )
        self.assertEqual(
            [
                str(w.message)
                for w in caught
                if "is not shared across methods" in str(w.message)
            ],
            [],
        )
        # And it is the buffer being shared, not the warning being missed.
        self.assertEqual(*self._state_placements(et, "dropped"))

    def test_a_one_method_program_is_not_warned_about_cross_method_sharing(
        self,
    ) -> None:
        """The same undeclared buffer, in a program with nobody to share with.

        Every method of a program is planned before run_multimethod, so one
        record means one method: there is no second method to disagree about
        where `dropped` sits, so no agreement is lost.
        """
        model = TwoMutableBufferModel().eval()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._to_executorch(
                {"forward": export(model, (torch.ones(4),))},
                MemoryPlanningPass(
                    share_mutable_buffers=True,
                    shared_buffer_fqns=frozenset({"kept"}),
                ),
            )
        self.assertEqual(
            [
                str(w.message)
                for w in caught
                if "is not shared across methods" in str(w.message)
            ],
            [],
        )

    def test_an_undeclared_buffer_only_one_method_touches_is_not_warned_about(
        self,
    ) -> None:
        """Two methods, but only one of them reaches the undeclared buffer.

        `touch_b` is the only method whose graph reads or writes `cache_b`;
        `forward` carries its placeholder unused, as export gives every method
        every registered buffer. There is no second placement for the one
        placement to disagree with, so nothing is lost.
        """
        model = TwoBufferModel().eval()
        forward_ep = export(model, (torch.ones(4),))
        with patch_forward(model, model.touch_b):
            touch_b_ep = export(model, (torch.ones(4),))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            et = self._to_executorch(
                {"forward": forward_ep, "touch_b": touch_b_ep},
                MemoryPlanningPass(
                    share_mutable_buffers=True,
                    shared_buffer_fqns=frozenset({"cache_a"}),
                ),
            )
        self.assertEqual(
            [
                str(w.message)
                for w in caught
                if "is not shared across methods" in str(w.message)
            ],
            [],
        )
        # Saying nothing is not sharing nothing: the declared buffer still
        # holds one address across both methods.
        self.assertEqual(self._state_placements(et, "cache_a"), [(2, 0), (2, 0)])

    def test_a_method_that_only_reads_a_declared_buffer_warns(self) -> None:
        """The reader-before-writer hazard, reported where it can be acted on.

        A declared buffer is emitted without its `state_dict` data, so a method
        that only reads it serves whatever the arena holds until some method has
        written it. That is a supported shape -- it is the point of the feature
        for a prefill/decode pair -- so it is warned about rather than refused.
        """
        model = PeekStateModel().eval()
        forward_ep = export(model, (torch.ones(4),))
        with patch_forward(model, model.peek):
            peek_ep = export(model, (torch.ones(4),))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._to_executorch(
                {"forward": forward_ep, "peek": peek_ep},
                MemoryPlanningPass(
                    share_mutable_buffers=True,
                    shared_buffer_fqns=frozenset({"state"}),
                ),
            )
        said = [
            str(w.message) for w in caught if "some method only reads" in str(w.message)
        ]
        self.assertEqual(len(said), 1, [str(w.message) for w in caught])
        # `peek` is the method planned second and is the only one that reads
        # without writing.
        self.assertIn("'state' in method(s) 1", said[0])

    def test_a_method_that_writes_the_declared_buffer_does_not_warn(self) -> None:
        """The warning above must not fire on the shape it does not describe."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._to_executorch(
                {"forward": export(_AddOnlyStateModel().eval(), (torch.ones(4),))},
                MemoryPlanningPass(
                    share_mutable_buffers=True,
                    shared_buffer_fqns=frozenset({"state"}),
                ),
            )
        self.assertEqual(
            [
                str(w.message)
                for w in caught
                if "some method only reads" in str(w.message)
            ],
            [],
        )

    def test_a_method_that_never_touches_a_declared_buffer_does_not_warn(self) -> None:
        """A lifted placeholder with no users is not a reader.

        Export puts every registered buffer in every method's signature, so a
        method that ignores one carries its placeholder all the same. Neither
        method here reads the buffer the other writes, and the hazard the
        warning describes takes a method that reads it.
        """
        model = TwoBufferModel().eval()
        forward_ep = export(model, (torch.ones(4),))
        with patch_forward(model, model.touch_b):
            touch_b_ep = export(model, (torch.ones(4),))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            et = self._to_executorch(
                {"forward": forward_ep, "touch_b": touch_b_ep},
                MemoryPlanningPass(
                    share_mutable_buffers=True,
                    shared_buffer_fqns=frozenset({"cache_a", "cache_b"}),
                ),
            )
        self.assertEqual(
            [
                str(w.message)
                for w in caught
                if "some method only reads" in str(w.message)
            ],
            [],
        )
        # Saying nothing is not placing nothing: both buffers still hold one
        # address across both methods.
        self.assertEqual(self._state_placements(et, "cache_a"), [(2, 0), (2, 0)])
        self.assertEqual(self._state_placements(et, "cache_b"), [(2, 16), (2, 16)])

    def _write_only_state_program(
        self,
    ) -> Tuple[GraphModule, ExportGraphSignature, TensorSpec]:
        """A one-buffer program plus the spec of `state`."""

        class WriteOnlyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("state", torch.zeros(4))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.state.add_(x)
                return x * x

        model = WriteOnlyModel().eval()
        edge = to_edge(export(model, (torch.ones(4),), strict=True))
        gm = edge.exported_program().graph_module
        gs = edge.exported_program().graph_signature
        gm = PassManager(passes=[SpecPropPass(), ToOutVarPass()])(gm).graph_module

        state_spec = None
        for node in gm.graph.nodes:
            if node.op == "placeholder" and isinstance(node.target, str):
                if gs.inputs_to_buffers.get(node.target) == "state":
                    state_spec = get_node_tensor_specs(node)[0]
        assert state_spec is not None
        return gm, gs, state_spec

    def test_an_alias_of_a_declared_buffer_is_refused(self) -> None:
        """A spec placed inside a declared buffer's storage has no address.

        The declared buffers are withheld from the planning algorithm and given
        their arena after it has returned, so an alias the algorithm is asked to
        fit inside one has no base allocation to be an offset into, and greedy
        raises on a storage base it cannot resolve. Refusing names the tensor
        and says what to do about it instead.
        """
        gm, gs, state_spec, _, mul_spec = self._two_buffer_alias_program()
        mul_spec.storage_base = state_spec
        mul_spec.storage_base_offset = 0

        mem_pass = MemoryPlanningPass(
            share_mutable_buffers=True,
            shared_buffer_fqns=frozenset({"state"}),
        )
        with self.assertRaises(NotImplementedError) as cm:
            mem_pass.run(gm, gs)

        message = str(cm.exception)
        self.assertIn("storage_base", message)
        self.assertIn("shared_buffer_fqns", message)
        # Refused before anything is placed, so no method is left half-planned.
        self.assertIsNone(mul_spec.mem_offset)
        self.assertIsNone(state_spec.mem_offset)

    def test_hand_placed_specs_pass_the_storage_reuse_verifier(self) -> None:
        """The buffers this path places itself carry a mem_obj_id.

        verify_storage_reuse raises as soon as one spec has a mem_obj_id and
        another does not, so a hand-placed spec without one makes every program
        of this feature fail the verifier -- which run() itself invokes under
        DEBUG logging.
        """
        gm, gs, state_spec = self._write_only_state_program()

        mem_pass = MemoryPlanningPass(
            share_mutable_buffers=True,
            shared_buffer_fqns=frozenset({"state"}),
        )
        mem_pass.run(gm, gs)
        mem_pass.run_multimethod()

        self.assertIsNotNone(state_spec.mem_obj_id)
        Verifier(
            gm,
            alloc_graph_input=True,
            alloc_graph_output=True,
            alloc_mutable_buffers=True,
            graph_signature=gs,
        ).verify_storage_reuse()

    def _two_buffer_alias_program(
        self,
    ) -> Tuple[GraphModule, ExportGraphSignature, TensorSpec, TensorSpec, TensorSpec]:
        """A two-buffer program plus the specs of both and of the `mul` out-var.

        `scratch` is twice the size of `state`, so it is roomy enough to hold
        the 16-byte `mul` out-var at a non-zero offset.
        """

        class TwoBufferModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("state", torch.zeros(4))
                self.register_buffer("scratch", torch.zeros(8))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.state.add_(x)
                self.scratch.add_(1.0)
                return x * x

        model = TwoBufferModel().eval()
        edge = to_edge(export(model, (torch.ones(4),), strict=True))
        gm = edge.exported_program().graph_module
        gs = edge.exported_program().graph_signature
        gm = PassManager(passes=[SpecPropPass(), ToOutVarPass()])(gm).graph_module

        specs_by_fqn = {}
        for node in gm.graph.nodes:
            if node.op == "placeholder" and isinstance(node.target, str):
                fqn = gs.inputs_to_buffers.get(node.target)
                if fqn is not None:
                    specs_by_fqn[fqn] = get_node_tensor_specs(node)[0]
        mul_spec = None
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.mul.out:
                mul_spec = node.meta["spec"]
        assert mul_spec is not None
        return gm, gs, specs_by_fqn["state"], specs_by_fqn["scratch"], mul_spec

    def test_an_alias_of_an_undeclared_tensor_is_refused_too(self) -> None:
        """The refusal is on the alias, not on which tensor it reaches.

        Whether an alias reaches a declared buffer is a question about the whole
        `storage_base` chain, and no walk of one is left here, so an alias of a
        tensor no declared name mentions is refused with the rest. This is the
        one shape the refusal costs: the planning algorithm places it, inside
        its base at the offset it asked for, in a program that declares nothing.
        """
        gm, gs, state_spec, scratch_spec, mul_spec = self._two_buffer_alias_program()
        mul_spec.storage_base = scratch_spec
        mul_spec.storage_base_offset = 16

        mem_pass = MemoryPlanningPass(
            share_mutable_buffers=True,
            shared_buffer_fqns=frozenset({"state"}),
        )
        with self.assertRaises(NotImplementedError):
            mem_pass.run(gm, gs)
        self.assertIsNone(mul_spec.mem_offset)
        self.assertIsNone(state_spec.mem_offset)

    def test_size_mismatch_across_methods_raises(self) -> None:
        mem_pass = MemoryPlanningPass(
            share_mutable_buffers=True,
            shared_buffer_fqns=frozenset({"state"}),
        )
        for size in (4, 8):

            class StateModel(nn.Module):
                def __init__(self, n: int = size) -> None:
                    super().__init__()
                    self.register_buffer("state", torch.zeros(n))

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    self.state.add_(x)
                    return self.state * 2

            edge = to_edge(
                export(StateModel().eval(), (torch.ones(size),), strict=True)
            )
            gm = edge.exported_program().graph_module
            gs = edge.exported_program().graph_signature
            gm = PassManager(passes=[SpecPropPass(), ToOutVarPass()])(gm).graph_module
            mem_pass.run(gm, gs)

        with self.assertRaises(ValueError) as cm:
            mem_pass.run_multimethod()
        message = str(cm.exception)
        self.assertIn("'state' is described differently", message)
        # Only the fields that differ are named, and both values with each.
        self.assertIn("allocated_memory 16 against 32", message)

    def test_two_methods_disagreeing_only_on_device_raises(self) -> None:
        """The cross-method check compares the device as well as the size.

        Both methods here declare `state` as the same tensor; only the device
        differs, so a comparison of sizes alone would accept them and put the
        two in one arena on whichever device was seen first.
        """
        mem_pass = MemoryPlanningPass(
            share_mutable_buffers=True,
            shared_buffer_fqns=frozenset({"state"}),
        )
        mem_pass.enable_non_cpu_memory_planning = True
        for on_device in (False, True):
            model = PeekStateModel().eval()
            edge = to_edge(export(model, (torch.ones(4),), strict=True))
            gm = edge.exported_program().graph_module
            gs = edge.exported_program().graph_signature
            gm = PassManager(passes=[SpecPropPass(), ToOutVarPass()])(gm).graph_module
            if on_device:
                for node in gm.graph.nodes:
                    if node.op == "placeholder" and isinstance(node.target, str):
                        if gs.inputs_to_buffers.get(node.target) == "state":
                            get_node_tensor_specs(node)[0].device = DeviceType.CUDA
            mem_pass.run(gm, gs)

        with self.assertRaises(ValueError) as cm:
            mem_pass.run_multimethod()
        message = str(cm.exception)
        self.assertIn("'state' is described differently", message)
        self.assertIn("device CPU:0 against CUDA:0", message)

    def test_two_methods_disagreeing_only_below_the_alignment_raises(self) -> None:
        """Two tensors that round to the same aligned size are still two tensors.

        float32[3] and float32[4] both occupy 16 bytes once aligned, so a check
        on allocated size alone accepts them; the method holding the wider one
        would then read four elements out of a slot the other only ever fills
        three of.
        """
        mem_pass = MemoryPlanningPass(
            share_mutable_buffers=True,
            shared_buffer_fqns=frozenset({"state"}),
        )
        for n in (3, 4):

            class StateModel(nn.Module):
                def __init__(self, size: int = n) -> None:
                    super().__init__()
                    self.register_buffer("state", torch.zeros(size))

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    self.state.add_(x)
                    return self.state * 2

            edge = to_edge(export(StateModel().eval(), (torch.ones(n),), strict=True))
            gm = edge.exported_program().graph_module
            gs = edge.exported_program().graph_signature
            gm = PassManager(passes=[SpecPropPass(), ToOutVarPass()])(gm).graph_module
            mem_pass.run(gm, gs)

        with self.assertRaises(ValueError) as cm:
            mem_pass.run_multimethod()
        message = str(cm.exception)
        self.assertIn("'state' is described differently", message)
        # The aligned size is equal, so shape is the only field named.
        self.assertIn("shape (3,) against (4,)", message)
        self.assertNotIn("allocated_memory", message)

    def test_two_methods_disagreeing_only_on_dtype_raises(self) -> None:
        """float32[4] and int32[4] are the same sixteen bytes, not one tensor.

        Everything the arena is laid out from -- the aligned size, the shape,
        the stride -- agrees, so a check that left dtype out would give the two
        one slot and let each method read the other's bits as its own type.
        """
        mem_pass = MemoryPlanningPass(
            share_mutable_buffers=True,
            shared_buffer_fqns=frozenset({"state"}),
        )
        for dtype in (torch.float32, torch.int32):

            class StateModel(nn.Module):
                def __init__(self, buffer_dtype: torch.dtype = dtype) -> None:
                    super().__init__()
                    self.register_buffer("state", torch.zeros(4, dtype=buffer_dtype))

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    self.state.add_(x)
                    return self.state * 2

            edge = to_edge(
                export(StateModel().eval(), (torch.ones(4, dtype=dtype),), strict=True)
            )
            gm = edge.exported_program().graph_module
            gs = edge.exported_program().graph_signature
            gm = PassManager(passes=[SpecPropPass(), ToOutVarPass()])(gm).graph_module
            mem_pass.run(gm, gs)

        with self.assertRaises(ValueError) as cm:
            mem_pass.run_multimethod()
        message = str(cm.exception)
        self.assertIn("'state' is described differently", message)
        self.assertIn("dtype torch.float32 against torch.int32", message)
        # Both occupy sixteen bytes, so dtype is the only field named.
        self.assertNotIn("allocated_memory", message)

    def test_two_methods_disagreeing_only_on_stride_raises(self) -> None:
        """One buffer read down the columns and across the rows is two tensors.

        Shape, dtype and aligned size all agree, so nothing but the stride
        separates a method writing element (0, 1) at byte 4 from one writing it
        at byte 8. No exporter gives one buffer name two strides, so the second
        method's spec is restrided by hand, the way the device case is
        device-tagged by hand.
        """
        mem_pass = MemoryPlanningPass(
            share_mutable_buffers=True,
            shared_buffer_fqns=frozenset({"state"}),
        )

        class StateModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("state", torch.zeros(2, 2))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.state.add_(x)
                return self.state * 2.0

        for restride in (False, True):
            edge = to_edge(
                export(StateModel().eval(), (torch.ones(2, 2),), strict=True)
            )
            gm = edge.exported_program().graph_module
            gs = edge.exported_program().graph_signature
            gm = PassManager(passes=[SpecPropPass(), ToOutVarPass()])(gm).graph_module
            if restride:
                for node in gm.graph.nodes:
                    if node.op == "placeholder" and isinstance(node.target, str):
                        if gs.inputs_to_buffers.get(node.target) == "state":
                            get_node_tensor_specs(node)[0].stride = (1, 2)
            mem_pass.run(gm, gs)

        with self.assertRaises(ValueError) as cm:
            mem_pass.run_multimethod()
        message = str(cm.exception)
        self.assertIn("'state' is described differently", message)
        self.assertIn("stride (2, 1) against (1, 2)", message)
        self.assertNotIn("shape", message)

    def test_two_methods_disagreeing_only_on_layout_raises(self) -> None:
        """A strided buffer and a sparse one are not one buffer either.

        `allocated_memory` is computed from shape and dtype, so it reports the
        same number for both, and an arena sized from it would hand the sparse
        tensor the dense one's bytes. Set by hand because export has no route to
        a sparse buffer here.
        """
        mem_pass = MemoryPlanningPass(
            share_mutable_buffers=True,
            shared_buffer_fqns=frozenset({"state"}),
        )
        for sparse in (False, True):
            edge = to_edge(
                export(PeekStateModel().eval(), (torch.ones(4),), strict=True)
            )
            gm = edge.exported_program().graph_module
            gs = edge.exported_program().graph_signature
            gm = PassManager(passes=[SpecPropPass(), ToOutVarPass()])(gm).graph_module
            if sparse:
                for node in gm.graph.nodes:
                    if node.op == "placeholder" and isinstance(node.target, str):
                        if gs.inputs_to_buffers.get(node.target) == "state":
                            get_node_tensor_specs(node)[0].layout = torch.sparse_coo
            mem_pass.run(gm, gs)

        with self.assertRaises(ValueError) as cm:
            mem_pass.run_multimethod()
        message = str(cm.exception)
        self.assertIn("'state' is described differently", message)
        self.assertIn("layout torch.strided against torch.sparse_coo", message)
        self.assertNotIn("allocated_memory", message)

    def test_a_reused_pass_instance_starts_each_program_afresh(self) -> None:
        """Two programs planned by one pass instance do not see each other.

        The per-method records the cross-method numbering is built from
        accumulate on the pass. The two programs here name a buffer `state` at
        different sizes, which within one program is refused, so laying the
        second one out alongside the first would be visible.

        What clears the records here is the `finally` inside `run_multimethod`,
        not the reset `to_executorch` runs, so this passes on a build with no
        `to_executorch` reset at all.
        `test_a_refusal_from_run_does_not_move_the_next_program_s_arena` and
        `TestMap.test_a_reused_pass_instance_replans_shared_mutable_buffers`
        are the two shapes the `finally` does not reach.
        """
        mem_pass = MemoryPlanningPass(
            share_mutable_buffers=True,
            shared_buffer_fqns=frozenset({"state"}),
        )
        arenas = []
        for size in (4, 8):

            class StateModel(nn.Module):
                def __init__(self, n: int = size) -> None:
                    super().__init__()
                    self.register_buffer("state", torch.zeros(n))

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    self.state.add_(x)
                    return self.state * 2

            edge = to_edge(
                export(StateModel().eval(), (torch.ones(size),), strict=True)
            )
            gm = edge.exported_program().graph_module
            gs = edge.exported_program().graph_signature
            gm = PassManager(passes=[SpecPropPass(), ToOutVarPass()])(gm).graph_module
            mem_pass.run(gm, gs)
            mem_pass.run_multimethod()
            arenas.append(list(gm.meta["non_const_buffer_sizes"]))

        # One activation arena and the shared arena, holding that program's own
        # buffer: 16 bytes then 32.
        self.assertEqual([len(sizes) for sizes in arenas], [3, 3])
        self.assertEqual([sizes[2] for sizes in arenas], [16, 32])

    def test_a_refused_program_does_not_poison_the_next_one(self) -> None:
        """A rejection is what a caller fixes and retries on the same instance.

        Every cross-method check raises before the records are cleared, so
        clearing on the success path alone would leave the refused program's
        methods on the pass and lay the retry out alongside a dead program.

        The refusal raised here comes from inside `run_multimethod`, so it is
        that method's own `finally` this holds to.
        `test_a_refusal_from_run_does_not_move_the_next_program_s_arena`
        covers the refusal that never reaches `run_multimethod`.
        """
        mem_pass = MemoryPlanningPass(
            share_mutable_buffers=True,
            shared_buffer_fqns=frozenset({"state"}),
        )

        def plan(n: int) -> torch.fx.GraphModule:
            class StateModel(nn.Module):
                def __init__(self, size: int = n) -> None:
                    super().__init__()
                    self.register_buffer("state", torch.zeros(size))

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    self.state.add_(x)
                    return self.state * 2

            edge = to_edge(export(StateModel().eval(), (torch.ones(n),), strict=True))
            gm = edge.exported_program().graph_module
            gs = edge.exported_program().graph_signature
            gm = PassManager(passes=[SpecPropPass(), ToOutVarPass()])(gm).graph_module
            mem_pass.run(gm, gs)
            return gm

        # A program whose two methods disagree about `state` is refused.
        plan(4)
        plan(8)
        with self.assertRaises(ValueError):
            mem_pass.run_multimethod()
        self.assertEqual(mem_pass.state.shared_arena_methods, [])

        # The retry is a valid single-method program and must compile.
        gm = plan(4)
        mem_pass.run_multimethod()
        self.assertEqual(gm.meta["non_const_buffer_sizes"][2], 16)

    def test_a_refusal_from_run_does_not_move_the_next_program_s_arena(self) -> None:
        """The per-method checks raise from run(), which run_multimethod never sees.

        to_executorch holds one pass instance for the whole program and a caller
        reuses it after fixing a refusal, so the methods recorded before the
        refusal have to be dropped somewhere that is still reached when run()
        raises rather than returns. Here the refused program's first method
        carries a custom pool, so keeping its record would widen the retry's CPU
        block and move the shared arena off the index a fresh instance gives it.
        """

        def pass_for(fqn: str) -> _PoolOrPinPass:
            return _PoolOrPinPass(
                fqn,
                share_mutable_buffers=True,
                shared_buffer_fqns=frozenset({fqn}),
            )

        def refused_program(mem_pass: _PoolOrPinPass) -> None:
            model = PeekStateModel().eval()
            forward_ep = export(model, (torch.ones(4),))
            with patch_forward(model, model.peek):
                peek_ep = export(model, (torch.ones(4),))
            with self.assertRaises(ValueError):
                self._to_executorch({"forward": forward_ep, "peek": peek_ep}, mem_pass)

        def retry(mem_pass: _PoolOrPinPass) -> Any:  # pyre-ignore[3]
            model = _AddOnlyStateModel().eval()
            return self._to_executorch(
                {"forward": export(model, (torch.ones(4),))}, mem_pass
            )

        mem_pass = pass_for("state")
        refused_program(mem_pass)
        retried = retry(mem_pass)
        fresh = retry(pass_for("state"))

        # No custom pool in this program, so the CPU block is one arena wide and
        # the shared arena closes it at 2. The refused program's pool widened
        # that block to three.
        self.assertEqual(self._state_placements(retried, "state"), [(2, 0)])
        self.assertEqual(
            self._state_placements(retried, "state"),
            self._state_placements(fresh, "state"),
        )
        self.assertEqual(
            retried.executorch_program.execution_plan[0].non_const_buffer_sizes,
            fresh.executorch_program.execution_plan[0].non_const_buffer_sizes,
        )

    def test_buffer_declared_in_no_method_raises(self) -> None:
        model = PeekStateModel().eval()
        mem_pass = MemoryPlanningPass(
            share_mutable_buffers=True,
            shared_buffer_fqns=frozenset({"state", "nonexistent"}),
        )
        with self.assertRaises(ValueError) as cm:
            self._to_executorch({"forward": export(model, (torch.ones(4),))}, mem_pass)
        self.assertIn("nonexistent", str(cm.exception))

    def test_planning_without_a_graph_signature_warns(self) -> None:
        """Which placeholders are buffers is knowable only from the signature.

        `ExirExportedProgram.to_executorch` and the Vulkan preprocess pass both
        call the pass without one, and the whole dedicated-arena path then
        switches off: the named buffers are planned as ordinary tensors and the
        checks that run across methods -- including the one that would report a
        name no buffer has -- never run. The program is built exactly as it
        would be with shared_buffer_fqns=None, so this warns rather than
        refusing.
        """
        gm, gs, state_spec = self._write_only_state_program()

        mem_pass = MemoryPlanningPass(
            share_mutable_buffers=True,
            shared_buffer_fqns=frozenset({"state", "a_name_no_buffer_has"}),
        )
        with self.assertWarnsRegex(UserWarning, "no graph signature") as caught:
            mem_pass.run(gm)
        mem_pass.run_multimethod()

        message = str(caught.warning)
        self.assertIn("'state'", message)
        self.assertIn("a_name_no_buffer_has", message)
        # No third arena, and `state` sits in the ordinary one.
        self.assertEqual(len(gm.meta["non_const_buffer_sizes"]), 2)
        self.assertEqual(state_spec.mem_id, 1)

    def test_per_method_pass_dict_is_rejected(self) -> None:
        """Each method would get its own pass instance and its own layout.

        The assertion names wording only this check uses. `peek` never writes
        `state`, so its own instance would raise "no method mutates" -- also a
        ValueError, and also naming the argument -- if this check were gone.
        """
        model = PeekStateModel().eval()
        forward_ep = export(model, (torch.ones(4),))
        with patch_forward(model, model.peek):
            peek_ep = export(model, (torch.ones(4),))

        def make() -> MemoryPlanningPass:
            return MemoryPlanningPass(
                share_mutable_buffers=True,
                shared_buffer_fqns=frozenset({"state"}),
            )

        with self.assertRaises(ValueError) as cm:
            to_edge({"forward": forward_ep, "peek": peek_ep}).to_executorch(
                ExecutorchBackendConfig(
                    memory_planning_pass={"forward": make(), "peek": make()},
                    emit_mutable_buffer_names=True,
                )
            )
        self.assertIn(
            "one memory planning pass instance for the whole program",
            str(cm.exception),
        )

    def test_per_method_pass_dict_is_rejected_before_any_method_is_planned(
        self,
    ) -> None:
        """The check reads only the config, so it must not run after planning.

        By the time the methods have been planned every graph module has been
        rewritten and each pass instance is holding per-program state, and a
        failure from inside the planning run would surface first and hide this.
        """
        model = PeekStateModel().eval()
        forward_ep = export(model, (torch.ones(4),))
        with patch_forward(model, model.peek):
            peek_ep = export(model, (torch.ones(4),))

        class _CountingPass(MemoryPlanningPass):
            runs = 0

            def run(
                self,
                graph_module: torch.fx.GraphModule,
                graph_signature: Optional[ExportGraphSignature] = None,
            ) -> PassResult:
                type(self).runs += 1
                return super().run(graph_module, graph_signature)

        def make() -> MemoryPlanningPass:
            return _CountingPass(
                share_mutable_buffers=True,
                shared_buffer_fqns=frozenset({"state"}),
            )

        with self.assertRaises(ValueError):
            to_edge({"forward": forward_ep, "peek": peek_ep}).to_executorch(
                ExecutorchBackendConfig(
                    memory_planning_pass={"forward": make(), "peek": make()},
                    emit_mutable_buffer_names=True,
                )
            )
        self.assertEqual(_CountingPass.runs, 0)

    def test_a_per_method_dict_of_legacy_passes_is_rejected(self) -> None:
        """`share_mutable_buffers` needs one instance for the same reason
        `shared_buffer_fqns` does.

        Each pass in a dict sees one method, so the legacy path hand-places
        that method's mutable buffers on arena 2 without any of them agreeing
        a size with the others. `peek` only reads `state`, so its arena 2 comes
        out zero-sized: the export succeeds and the value `forward` wrote is
        not there to read.
        """
        model = PeekStateModel().eval()

        def programs() -> dict[str, Any]:  # pyre-ignore[3]
            forward_ep = export(model, (torch.ones(4),))
            with patch_forward(model, model.peek):
                peek_ep = export(model, (torch.ones(4),))
            return {"forward": forward_ep, "peek": peek_ep}

        with self.assertRaises(ValueError) as cm:
            to_edge(programs()).to_executorch(
                ExecutorchBackendConfig(
                    memory_planning_pass={
                        "forward": MemoryPlanningPass(share_mutable_buffers=True),
                        "peek": MemoryPlanningPass(share_mutable_buffers=True),
                    },
                    emit_mutable_buffer_names=True,
                )
            )
        message = str(cm.exception)
        self.assertIn("share_mutable_buffers", message)
        self.assertIn("shared_buffer_fqns", message)

        single = to_edge(programs()).to_executorch(
            ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(share_mutable_buffers=True),
                emit_mutable_buffer_names=True,
            )
        )
        self.assertEqual(
            [
                plan.non_const_buffer_sizes
                for plan in single.executorch_program.execution_plan
            ],
            [[0, 32, 16], [0, 32, 16]],
        )

    def test_a_per_method_dict_that_asks_for_no_sharing_is_accepted(self) -> None:
        """The guard refuses a sharing request, not the per-method dict.

        Every in-tree caller that passes a dict varies only
        alloc_graph_input/alloc_graph_output, so refusing dicts outright would
        break them and both refusal tests above would still pass.
        """
        model = PeekStateModel().eval()
        forward_ep = export(model, (torch.ones(4),))
        with patch_forward(model, model.peek):
            peek_ep = export(model, (torch.ones(4),))

        program = to_edge({"forward": forward_ep, "peek": peek_ep}).to_executorch(
            ExecutorchBackendConfig(
                memory_planning_pass={
                    "forward": MemoryPlanningPass(alloc_graph_output=False),
                    "peek": MemoryPlanningPass(),
                },
            )
        )
        # No arena beyond the one activation arena: nothing is being shared.
        self.assertEqual(
            [
                len(plan.non_const_buffer_sizes)
                for plan in program.executorch_program.execution_plan
            ],
            [2, 2],
        )

    def test_shared_buffer_fqns_requires_sharing(self) -> None:
        with self.assertRaises(ValueError):
            MemoryPlanningPass(shared_buffer_fqns=frozenset({"state"}))

    def test_empty_shared_buffer_fqns_raises(self) -> None:
        """Empty but non-None gets neither legacy nor dedicated-arena sharing."""
        with self.assertRaises(ValueError):
            MemoryPlanningPass(
                share_mutable_buffers=True, shared_buffer_fqns=frozenset()
            )

    def test_a_bare_string_of_shared_buffer_fqns_raises(self) -> None:
        """A single name passed as a string is a set of characters.

        It is non-empty and iterable, so every other check passes it; the name
        lookup then becomes a substring test and the diagnostics name single
        letters.
        """
        with self.assertRaises(TypeError) as cm:
            MemoryPlanningPass(
                share_mutable_buffers=True,
                shared_buffer_fqns="state",  # pyre-ignore[6]
            )
        self.assertIn("frozenset({'state'})", str(cm.exception))

    def test_a_generator_of_shared_buffer_fqns_lays_out_like_a_set(self) -> None:
        """A one-shot iterable of names must not change what is emitted.

        Kept as given, the names would be membership-tested once per
        placeholder per method, so a generator would be empty from the second
        test on and nothing downstream would notice: the program would emit
        with no shared arena and no message.
        """
        names = ("cache_a", "cache_b")

        def build(fqns: Any) -> Any:  # pyre-ignore[2,3]
            model = TwoBufferModel().eval()
            forward_ep = export(model, (torch.ones(4),))
            with patch_forward(model, model.touch_b):
                touch_b_ep = export(model, (torch.ones(4),))
            return self._to_executorch(
                {"forward": forward_ep, "touch_b": touch_b_ep},
                MemoryPlanningPass(share_mutable_buffers=True, shared_buffer_fqns=fqns),
            )

        def declared_placements(et: Any) -> List[Any]:  # pyre-ignore[2,3]
            found = []
            for plan in et.executorch_program.execution_plan:
                for value in plan.values:
                    info = getattr(value.val, "extra_tensor_info", None)
                    allocation = getattr(value.val, "allocation_info", None)
                    if info is None or allocation is None:
                        continue
                    if info.fully_qualified_name in names:
                        found.append(
                            (
                                plan.name,
                                info.fully_qualified_name,
                                allocation.memory_id,
                                allocation.memory_offset_low,
                            )
                        )
            return sorted(found)

        expected = build(frozenset(names))
        actual = build(name for name in names)

        expected_placements = declared_placements(expected)
        # Both buffers really are placed, so the comparison below is against a
        # layout rather than against two empty lists.
        self.assertEqual({p[1] for p in expected_placements}, set(names))
        self.assertEqual(declared_placements(actual), expected_placements)
        self.assertEqual(
            [
                plan.non_const_buffer_sizes
                for plan in actual.executorch_program.execution_plan
            ],
            [
                plan.non_const_buffer_sizes
                for plan in expected.executorch_program.execution_plan
            ],
        )

    def test_declared_buffers_get_disjoint_ranges_in_one_arena(self) -> None:
        """Buffers sharing a device's dedicated arena must not overlap.

        run() only calls verify_storage_reuse under DEBUG logging or for the
        greedy algorithm by name, and this test runs under neither, so the
        offsets the pass hands out are asserted directly.
        """
        model = ThreeBufferModel().eval()
        edge = to_edge(export(model, (torch.ones(8),), strict=True))
        gm = edge.exported_program().graph_module
        gs = edge.exported_program().graph_signature
        gm = PassManager(passes=[SpecPropPass(), ToOutVarPass()])(gm).graph_module

        fqns = frozenset({"cache_x", "cache_y", "cache_z"})
        mem_pass = MemoryPlanningPass(
            share_mutable_buffers=True, shared_buffer_fqns=fqns
        )
        mem_pass.run(gm, gs)
        mem_pass.run_multimethod()

        specs = {}
        for node in gm.graph.nodes:
            if node.op == "placeholder" and isinstance(node.target, str):
                fqn = gs.inputs_to_buffers.get(node.target)
                if fqn in fqns:
                    specs[fqn] = get_node_tensor_specs(node)[0]
        self.assertEqual(set(specs), set(fqns))

        arenas = {spec.mem_id for spec in specs.values()}
        self.assertEqual(len(arenas), 1, f"one device, so one arena: {arenas}")
        arena = arenas.pop()

        ranges = {
            fqn: (spec.mem_offset, spec.mem_offset + spec.allocated_memory)
            for fqn, spec in specs.items()
        }

        for lhs, rhs in itertools.combinations(sorted(ranges), 2):
            low, high = ranges[lhs], ranges[rhs]
            self.assertTrue(
                low[1] <= high[0] or high[1] <= low[0],
                f"{lhs} {low} overlaps {rhs} {high}",
            )

        # Disjointness alone is satisfied by any layout that leaves gaps or
        # orders the buffers differently, so the offsets are named: the buffers
        # are packed in name order, each taking the 16-byte-aligned size of its
        # own tensor and nothing more. cache_x is 12 bytes of a 16-byte slot.
        self.assertEqual(
            ranges,
            {"cache_x": (0, 16), "cache_y": (16, 48), "cache_z": (48, 80)},
        )

        arena_size = gm.meta["non_const_buffer_sizes"][arena]
        self.assertEqual(arena_size, 80)

    def test_a_declared_buffer_with_no_elements_raises(self) -> None:
        """A zero-element buffer sizes its dedicated arena to nothing.

        There is no state in it for two methods to share, and it is not free to
        declare: the only declared buffer of a device being empty appends an
        arena of zero bytes to `non_const_buffer_sizes` -- measured as
        `[0, 32, 0]`, with `state` at `(2, 0)`.
        """
        mem_pass = MemoryPlanningPass(
            share_mutable_buffers=True,
            shared_buffer_fqns=frozenset({"state"}),
        )
        with self.assertRaises(ValueError) as cm:
            self._to_executorch(
                {"forward": export(EmptyStateModel().eval(), (torch.ones(4),))},
                mem_pass,
            )
        message = str(cm.exception)
        self.assertIn("'state'", message)
        # Wording only this check uses. Every refusal here names the buffer, so
        # the name alone would leave this green on any of them.
        self.assertIn("no elements", message)

    def test_buffer_no_method_mutates_raises(self) -> None:
        """A declared buffer that nothing writes to must not get a placement.

        The emitter reads a mem_id and mem_offset on a buffer placeholder as
        proof the buffer is mutable, and emits a mutable buffer without its
        state_dict data, so this buffer would be served from uninitialized
        planned memory instead of the values it was registered with. A buffer
        mutated in one method and only read in another is a different case and
        is accepted -- see
        test_one_method_reads_what_another_wrote_to_the_shared_buffer.
        """
        model = PeekStateModel().eval()
        with patch_forward(model, model.peek):
            peek_ep = export(model, (torch.ones(4),))

        mem_pass = MemoryPlanningPass(
            share_mutable_buffers=True,
            shared_buffer_fqns=frozenset({"state"}),
        )
        with self.assertRaises(ValueError) as cm:
            self._to_executorch({"peek": peek_ep}, mem_pass)
        message = str(cm.exception)
        self.assertIn("'state'", message)
        self.assertIn("mutates", message)

    def test_a_control_flow_submodule_is_refused(self) -> None:
        """A method with a `cond` in it, on the plainest possible program.

        A submodule's tensors are planned by a recursive apply_algo into arena
        indices nothing renumbers afterwards, so the cross-method numbering
        would have to hand every one of them back unchanged. Rather than work
        out whether a particular method's would survive, any method that has a
        submodule at all is refused.

        Nothing here is unusual -- one CPU method, one declared buffer -- so
        the refusal is NotImplementedError with a sentence saying what to
        change, rather than an internal assertion.
        """
        model = CondStateModel().eval()
        with self.assertRaises(NotImplementedError) as cm:
            self._to_executorch(
                {"forward": export(model, (torch.full((4,), 2.0),))},
                MemoryPlanningPass(
                    share_mutable_buffers=True,
                    shared_buffer_fqns=frozenset({"state"}),
                ),
            )
        message = str(cm.exception)
        self.assertIn("control-flow submodule", message)
        self.assertIn("shared_buffer_fqns", message)

    def test_a_host_arena_at_a_non_zero_index_is_refused(self) -> None:
        """A non-zero host device index is the one host key the arenas skip.

        Every host index other than zero is out of this argument's scope, so the
        shape is declined rather than laid out; CPU:1 is the case built here.

        The control is `state` on CPU:0, which the same pass lays out.
        """
        model = PeekStateModel().eval()

        def plan(index: int) -> Any:  # pyre-ignore[3]
            return self._to_executorch(
                {"forward": export(model, (torch.ones(4),))},
                _BufferDeviceIndexPass(
                    index,
                    share_mutable_buffers=True,
                    shared_buffer_fqns=frozenset({"state"}),
                ),
                enable_non_cpu_memory_planning=True,
            )

        self.assertEqual(self._state_placements(plan(0), "state"), [(2, 0)])
        with self.assertRaises(NotImplementedError) as cm:
            plan(1)
        self.assertIn("CPU:1", str(cm.exception))

    def test_a_refusal_leaves_no_earlier_method_renumbered(self) -> None:
        """The submodule-arena refusal, reached on the second of two methods.

        The cross-method numbering rewrites one method at a time, so a refusal
        part way through would leave the methods before it in the common
        numbering and the rest in their own -- a program in two numbering
        schemes, which the caller can read off the edge program. Every refusal
        is a function of the records alone, so all of them are asked first.
        """
        plain_ep = export(PeekStateModel().eval(), (torch.ones(4),))
        cond_ep = export(CondStateModel().eval(), (torch.full((4,), 2.0),))

        mem_pass = _PlannedMetaRecordingPass(
            share_mutable_buffers=True,
            shared_buffer_fqns=frozenset({"state"}),
        )
        with self.assertRaises(NotImplementedError):
            self._to_executorch(
                {"plain": plain_ep, "condy": cond_ep},
                mem_pass,
                enable_non_cpu_memory_planning=True,
            )

        self.assertEqual(len(mem_pass.planned), 2)
        for graph_module, planned in mem_pass.planned:
            self.assertEqual(graph_module.meta["non_const_buffer_sizes"], planned)
