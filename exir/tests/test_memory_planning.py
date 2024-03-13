# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import itertools
import unittest
from typing import Callable, List, Optional, Tuple, Type

import executorch.exir as exir
import executorch.exir.schema as schema

import torch
import torch.utils._pytree as pytree
from executorch.backends.fb.qnnpack.partition.qnnpack_partitioner import (
    QnnpackPartitioner,
)
from executorch.exir.backend.backend_api import to_backend, validation_disabled
from executorch.exir.memory_planning import filter_nodes, Verifier
from executorch.exir.pass_base import PassResult
from executorch.exir.pass_manager import PassManager
from executorch.exir.passes import (  # noqa
    ConstPropPass,
    DebugPass,
    MemoryPlanningPass,
    QuantFusionPass,
    SpecPropPass,
    ToOutVarPass,
)
from executorch.exir.print_program import print_program
from executorch.exir.tests.asr_joiner import ASRJoiner
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
from torch.fx import Graph, GraphModule, Node
from torch.nn import functional as F

torch.ops.load_library("//executorch/kernels/portable:custom_ops_generated_lib")


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


class ModelWithDifferentTensorSizes(torch.nn.Module):
    def __init__(self) -> None:
        super(ModelWithDifferentTensorSizes, self).__init__()
        self.linears: List[torch.nn.Linear] = []
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

        return super().call(graph_module)


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
    criteria: Optional[List[Tuple[str, bool]]] = None,
    extra_check: Optional[Callable[..., None]] = None,
    use_functionalization: bool = True,
    alloc_graph_input: bool = True,
    alloc_graph_output: bool = True,
    has_unused_graph_input: bool = False,
) -> Callable[..., None]:
    # parameterized.expand is not compatible with maketest. I'll just loop thru
    # the test setups in the wrapper.
    def wrapper(self: "TestMemoryPlanning") -> None:
        nonlocal criteria
        if not criteria:
            criteria = [
                # naive algorithm does not reuse tensor storages
                ("naive", False),
                # greedy algorithm should reuse tensor storages in the testing model
                ("greedy", True),
            ]

        for algo, expect_reuse in criteria:
            print(f"algo {algo}, expect_reuse {expect_reuse}")
            eager_module = module_cls().eval()
            inputs = eager_module.get_random_inputs()
            graph_module = (
                exir.capture(
                    eager_module,
                    inputs,
                    exir.CaptureConfig(),
                )
                # torch._ops.aten.t.default
                .to_edge(
                    exir.EdgeCompileConfig(_check_ir_validity=False)
                ).exported_program.graph_module
            )

            graph_module = PassManager(
                passes=[
                    SpecPropPass(),
                    ToOutVarPass(),
                    MemoryPlanningPass(
                        algo,
                        alloc_graph_input=alloc_graph_input,
                        alloc_graph_output=alloc_graph_output,
                    ),
                ],
            )(graph_module).graph_module

            self.verify_reuse(
                graph_module, expect_reuse, alloc_graph_input, alloc_graph_output
            )
            self.verify_graph_input_output(
                graph_module, alloc_graph_input, alloc_graph_output
            )

            self.verify_overlap_placeholders(has_unused_graph_input, graph_module)

            # print(f"Final code: {graph_module.code}")
            # print(f"Final graph: {graph_module.graph}")

            if extra_check:
                extra_check(self, graph_module)

    return wrapper


class TestMemoryPlanning(unittest.TestCase):
    def verify_reuse(
        self,
        graph_module: torch.fx.GraphModule,
        expect_reuse: bool,
        alloc_graph_input: bool,
        alloc_graph_output: bool,
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
    ) -> None:
        Verifier(
            graph_module, alloc_graph_input, alloc_graph_output
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
            ("naive", False),
            ("greedy", True),
        ],
    )

    # greedy algorithm will reuse memory if we let the algorithm allocate
    # memory for both graph input and output.
    test_list_arg: Callable[..., None] = maketest(
        ModuleListArg,
        criteria=[
            ("naive", False),
            ("greedy", True),
        ],
        extra_check=ModuleListArg.extra_check,
    )

    def test_graph_input_output(self) -> None:
        for alloc_graph_input, alloc_graph_output in itertools.product(
            [True, False], [True, False]
        ):
            case = maketest(
                ModelWithDifferentTensorSizes,
                alloc_graph_input=alloc_graph_input,
                alloc_graph_output=alloc_graph_output,
            )
            case(self)


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

    def test_asr_joiner(self) -> None:
        eager_model = self.quantize(ASRJoiner())
        inputs = eager_model.get_random_inputs()
        edge_program = (
            exir.capture(
                eager_model,
                inputs,
                exir.CaptureConfig(
                    enable_dynamic_shape=True,
                ),
            )
            .to_edge(
                exir.EdgeCompileConfig(
                    _check_ir_validity=False,
                )
            )
            .transform(ConstPropPass())
        )
        with validation_disabled():
            backend_module = to_backend(
                edge_program.exported_program, QnnpackPartitioner()
            )

        debug_pass = DebugPass(show_spec=True)
        debug_pass(edge_program.exported_program.graph_module)
        config = exir.ExecutorchBackendConfig(
            passes=[QuantFusionPass(), ConstPropPass(propogate_quant=True)],
        )
        edge_program.exported_program = backend_module
        program = edge_program.to_executorch(config)
        gm = program.dump_graph_module()
        gm.print_readable()
        print_program(program.program, mark_dynamic_shape_tensor=True)

        *_, out_node = gm.graph.nodes
        ncheck = 0
        for i, arg in enumerate(pytree.tree_flatten(out_node.args)[0]):
            spec = arg.meta["spec"]
            ncheck += 1
            if i == 2:
                self.assertEqual(spec.shape_dynamism, schema.TensorShapeDynamism.STATIC)
            else:
                self.assertEqual(
                    spec.shape_dynamism, schema.TensorShapeDynamism.DYNAMIC_BOUND
                )

        self.assertEqual(3, ncheck)

    # pyre-ignore
    @parameterized.expand(
        [
            (
                "naive",
                [(1, 0), (3, 0), (1, 4), (3, 4), (1, 8)],
                [0, 12, 0, 8],
            ),
            (
                "greedy",
                [(1, 0), (3, 0), (1, 4), (3, 4), (1, 0)],
                [0, 8, 0, 8],
            ),
        ]
    )
    def test_multiple_pools(
        self,
        algo: str,
        expected_allocs: List[Tuple[int, int]],
        expected_bufsizes: List[int],
    ) -> None:
        edge_program = exir.capture(
            MultiplePoolsToyModel(),
            (torch.ones(1),),
        ).to_edge(exir.EdgeCompileConfig(_check_ir_validity=False))

        program = edge_program.to_executorch(
            exir.ExecutorchBackendConfig(
                memory_planning_pass=CustomPoolMemoryPlanningPass(
                    memory_planning_algo=algo,
                    alignment=1,
                )
            )
        )
        graph_module = program.dump_graph_module()

        verifier = Verifier(
            graph_module,
            alloc_graph_input=True,
            alloc_graph_output=True,
        )
        verifier.verify_storage_reuse()
        verifier.verify_graph_input_output()

        idx = 0
        for node in graph_module.graph.nodes:
            if node.op == "placeholder" or (
                node.op == "call_function"
                and node.target in (torch.ops.aten.add.out, torch.ops.aten.mul.out)
            ):
                mem_id, mem_offset = expected_allocs[idx]
                self.assertEqual(node.meta["spec"].mem_id, mem_id)
                self.assertEqual(node.meta["spec"].mem_offset, mem_offset)
                idx += 1
        self.assertEqual(graph_module.meta["non_const_buffer_sizes"], expected_bufsizes)
