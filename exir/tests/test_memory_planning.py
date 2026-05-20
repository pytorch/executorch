# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import itertools
import unittest
from typing import Any, Callable, List, Optional, Tuple, Type

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
from executorch.exir.memory_planning import (
    _do_user_inputs_exist,
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
from executorch.exir.passes.reinplace import DEFAULT_INPLACEABLE_OPS, reinplace_pass
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
        """CPU-only specs produce bufsizes = [0, X] with no device metadata."""
        gm, gs = self._prepare_model()

        algo = MemoryPlanningAlgorithmSuite(algo_list=[greedy])
        bufsizes = apply_algo(algo, gm, 16, gs, enable_non_cpu_memory_planning=True)

        # The CUDA spec is the only tensor in its buffer
        self.assertEqual(bufsizes[0], 0)  # constants
        self.assertGreater(bufsizes[1], 0)  # CPU activations
        self.assertNotIn("non_const_buffer_device", gm.meta)

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
        planned into CPU memory — no device-specific buffers are created."""
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
        self.assertNotIn("non_const_buffer_device", gm.meta)
