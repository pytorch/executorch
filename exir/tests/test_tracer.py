# pyre-strict

import copy
import unittest
from typing import Dict, List, Tuple

import executorch.exir as exir
import executorch.exir.schema as schema
import executorch.exir.tests.control_flow_models as control_flow_models
import executorch.exir.tests.models as models

import torch

import torch.utils._pytree as pytree
from executorch.exir.error import ExportError
from executorch.exir.experimental.funktionalize import FunktionalizationPass
from executorch.exir.passes import DebugPass
from executorch.exir.tests.common import register_additional_test_aten_ops
from executorch.exir.tracer import dynamo_trace, ExirDynamoConfig, using_dynamo

from parameterized import parameterized
from torch._export.verifier import SpecViolationError
from torch.fx.experimental.symbolic_shapes import is_concrete_int
from torch.testing import FileCheck


class TestTorchDispatchFXTracer(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        register_additional_test_aten_ops()

    def test_simple(self) -> None:
        f = models.BasicSinMax()
        f = (
            exir.capture(f, f.get_random_inputs(), exir.CaptureConfig(pt2_mode=True))
            .to_edge()
            .graph_module
        )

        FileCheck().check("torch.ops.aten.sin").check("torch.ops.aten.max").run(f.code)

    def test_static_control_flow(self) -> None:
        def f(pred: bool, x: torch.Tensor) -> torch.Tensor:
            if pred:
                return torch.sin(x).max()
            else:
                return torch.sin(x)

        pred = True
        x = torch.randn(100)
        f_true = (
            exir.capture(f, (pred, x), exir.CaptureConfig(pt2_mode=True))
            .to_edge()
            .graph_module
        )

        FileCheck().check("torch.ops.aten.sin").check("torch.ops.aten.max").run(
            f_true.code
        )

        pred = False
        f_false = (
            exir.capture(f, (pred, x), exir.CaptureConfig(pt2_mode=True))
            .to_edge()
            .graph_module
        )
        FileCheck().check("torch.ops.aten.sin").check_not("torch.ops.aten.max").run(
            f_false.code
        )

    def test_copy(self) -> None:
        f = models.BasicSinMax()
        f = (
            exir.capture(f, f.get_random_inputs(), exir.CaptureConfig(pt2_mode=True))
            .to_edge()
            .graph_module
        )

        self.assertTrue(isinstance(f, torch.fx.GraphModule))
        g = copy.deepcopy(f)
        self.assertTrue(isinstance(g, torch.fx.GraphModule))

    def test_stacktrace(self) -> None:
        def f(x: torch.Tensor) -> torch.Tensor:
            return x + x

        traced_f = (
            exir.capture(f, (torch.rand(2, 2),), exir.CaptureConfig(pt2_mode=True))
            .to_edge()
            .graph_module
        )
        # Check that stacktrace is populated and retained (by checking twice)
        self.assertTrue(
            any([node.meta.get("stack_trace", None) for node in traced_f.graph.nodes])
        )
        self.assertTrue(
            any([node.meta.get("stack_trace", None) for node in traced_f.graph.nodes])
        )

    def test_possible_input_mutation(self) -> None:
        def f(x: torch.Tensor) -> torch.Tensor:
            return torch.add(torch.ones(5), torch.ones(5), out=x)

        with self.assertRaisesRegex(
            SpecViolationError,
            r"operator .* is not functional",
        ):
            exir.capture(
                f, (torch.zeros(5),), exir.CaptureConfig(pt2_mode=True)
            ).to_edge()

    def test_tensor_spec_for_const_tensors(self) -> None:
        class Module(torch.nn.Module):
            def __init__(self) -> None:
                super(Module, self).__init__()
                self.linear = torch.nn.Linear(2, 3)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

            def get_random_inputs(self) -> Tuple[torch.Tensor, ...]:
                return (torch.randn(2),)

        model = Module()
        graph_module = (
            exir.capture(
                model, model.get_random_inputs(), exir.CaptureConfig(pt2_mode=True)
            )
            # torch._ops.aten.t.default
            .to_edge(exir.EdgeCompileConfig(_check_ir_validity=False)).graph_module
        )
        num_get_attr_node = 0
        num_get_attr_node_with_tensorspec = 0
        for nd in graph_module.graph.nodes:
            if nd.op == "get_attr":
                num_get_attr_node += 1
                if nd.meta.get("val") is not None:
                    num_get_attr_node_with_tensorspec += 1

        self.assertEqual(2, num_get_attr_node)
        self.assertEqual(2, num_get_attr_node_with_tensorspec)

    def test_multiple_returns_spec(self) -> None:
        def f(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            return torch.ops.aten.max.dim(x, 0, False)

        cnt = 0
        module = (
            exir.capture(f, (torch.zeros(1, 2, 3),), exir.CaptureConfig(pt2_mode=True))
            .to_edge()
            .graph_module
        )
        for node in module.graph.nodes:
            if node.target == torch.ops.aten.max.dim:
                cnt += 1
                self.assertIsInstance(node.meta["val"], tuple)
        self.assertEqual(cnt, 1)

    def test_multiple_returns_pt2_mode(self) -> None:
        def f(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            a = x * x
            b = x + a
            return a, b

        inputs = (torch.ones(1, 2, 3),)
        orig_res = f(*inputs)
        module = (
            exir.capture(
                f,
                inputs,
                exir.CaptureConfig(pt2_mode=True),
            )
            .to_edge()
            .graph_module
        )
        new_res = module(*inputs)
        for node in module.graph.nodes:
            if node.op == "output":
                self.assertIsInstance(node.meta["val"], list)
                self.assertEqual(len(node.meta["val"]), 2)

        self.assertTrue(torch.allclose(orig_res[0], new_res[0]))
        self.assertTrue(torch.allclose(orig_res[1], new_res[1]))

    def test_toggle_dynamo_capture_scalar_outputs(self) -> None:
        def f(x: torch.Tensor) -> float:
            return x.item()

        with self.assertRaisesRegex(
            ExportError,
            "The user code is using a feature we don't support.",
        ):
            dynamo_trace(
                f,
                (torch.ones(1),),
                False,
                "real",
                ExirDynamoConfig(capture_scalar_outputs=False),
            )
        dynamo_trace(
            f,
            (torch.ones(1),),
            False,
            "real",
            ExirDynamoConfig(capture_scalar_outputs=True),
        )

    # pyre-ignore
    @parameterized.expand([("stock_tensor",)])
    def test_embedding_dynamic_shape(self, input_type: str) -> None:
        class Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return x + x

        example_input = torch.ones(10, dtype=torch.int64)
        m = Module()
        gm = (
            exir.capture(
                m,
                (example_input,),
                exir.CaptureConfig(
                    pt2_mode=True,
                    enable_functionalization=False,
                    enable_dynamic_shape=True,
                ),
            )
            .to_edge()
            .graph_module
        )

        print(gm.graph)

    def test_dynamic_shape(self) -> None:
        def forward(x: torch.Tensor) -> torch.Tensor:
            x = x.view(x.shape[0] - 1, -1)
            return torch.cat([x, x])

        gm = (
            exir.capture(
                forward,
                (torch.ones(3, 2, dtype=torch.int64),),
                exir.CaptureConfig(
                    pt2_mode=True,
                    enable_functionalization=False,
                    enable_dynamic_shape=True,
                ),
                # sym_size is not reg op
            )
            .to_edge(exir.EdgeCompileConfig(_check_ir_validity=False))
            .graph_module
        )

        for node in gm.graph.nodes:
            if node.op in ("placeholder", "call_function"):
                self.assertIn("val", node.meta)

    def test_dynamo_frontend_container_input(self) -> None:
        class Module(torch.nn.Module):
            def __init__(self) -> None:
                super(Module, self).__init__()

            def forward(
                self, x: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            ) -> torch.Tensor:
                a = x[0]
                b = x[1]
                cum = 0
                for i in b:
                    cum += i.sum()
                return a.cos() + cum.sin()

        with using_dynamo(True):
            inp = ((torch.ones(6), (torch.ones(6), torch.ones(6))),)
            gm = exir.capture(Module(), inp, exir.CaptureConfig(pt2_mode=True))
            self.assertTrue(torch.allclose(Module()(*inp), gm(*inp)))

    # TODO (tmanlaibaatar) remove this test
    def test_pt2_mode_with_dynamo_config(self) -> None:
        def f(x: torch.Tensor) -> torch.Tensor:
            return x[: x.shape[0] - 1]

        inp = (torch.randn(4, 5),)
        prog = exir.capture(
            f,
            inp,
            config=exir.CaptureConfig(
                pt2_mode=True, _dynamo_config=ExirDynamoConfig(dynamic_shapes=False)
            ),
            # missing dispatch key
        ).to_edge()
        self.assertTrue(prog(torch.randn(6, 5)).shape[0], 3)

    def test_input_container_type(self) -> None:
        def f(x: torch.Tensor, y: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
            # pyre-ignore
            return {"a": x.sum() + sum(y).sum()}

        inp = (torch.randn(6, 5), [torch.randn(6, 5), torch.randn(6, 5)])

        gm, _ = torch._dynamo.export(f, *inp, aten_graph=True, tracing_mode="symbolic")
        prog = exir.capture(f, inp, config=exir.CaptureConfig(pt2_mode=True)).to_edge()

        self.assertEqual(prog(*inp), f(*inp))

    def test_aot_buffer_mutation(self) -> None:
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "_bin_num_examples",
                    torch.empty([42]).fill_(0.0),
                )

            def forward(self, x, y, z):
                self._bin_num_examples.index_copy_(
                    dim=0,
                    index=y,
                    source=z,
                )
                self._bin_num_examples.index_add_(
                    dim=0, index=torch.arange(4), source=x
                )
                return self._bin_num_examples - 1, x * z

        model = Module()
        example_inputs = (
            torch.randn(4, requires_grad=True),
            torch.tensor(0),
            torch.tensor(3.14),
        )

        ep = exir.capture(
            model,
            example_inputs,
            exir.CaptureConfig(
                pt2_mode=True,
                enable_aot=True,
            ),
        )

        test_inputs = (
            torch.randn(4),
            torch.tensor(0),
            torch.tensor(2.1),
        )
        graph_outputs = ep(*test_inputs)
        eager_outputs = model(*test_inputs)
        self.assertEqual(len(graph_outputs), 2)
        self.assertEqual(len(eager_outputs), 2)
        self.assertTrue(torch.allclose(graph_outputs[0], eager_outputs[0]))
        self.assertTrue(torch.allclose(graph_outputs[1], eager_outputs[1]))

    def test_retain_original_inputs(self) -> None:
        """
        The way we setup config here minics what we do for DPE right now.

        The tests makes sure the upperbound shape information is retained
        after tracing and functionalization.
        """
        eager_model = models.ModelWithUnusedArg()
        inputs = tuple(eager_model.get_random_inputs())
        gm = (
            exir.capture(
                eager_model,
                inputs,
                exir.CaptureConfig(
                    pt2_mode=True,
                    enable_functionalization=False,
                    enable_dynamic_shape=True,
                ),
            )
            .transform(FunktionalizationPass(inputs))
            .to_edge(
                exir.EdgeCompileConfig(
                    _check_ir_validity=False,
                )
            )
            .to_executorch()
        ).dump_graph_module()
        DebugPass(show_spec=True)(gm)
        print(f"gm meta {gm.meta}")
        ncheck = 0
        for node in gm.graph.nodes:
            for spec in pytree.tree_flatten(node.meta.get("spec", []))[0]:
                ncheck += 1
                self.assertTrue(
                    spec.shape_dynamism == schema.TensorShapeDynamism.DYNAMIC_BOUND
                )
        self.assertTrue(ncheck > 0)

    def test_retain_original_inputs_with_submodule(self) -> None:
        """
        For a GraphModule has a submodule, make sure StopIteration is caught when
        visiting submodule's placeholder nodes in SpecPropPass.
        The passing criteria for this test is no exception raised in top level.
        """
        eager_model = control_flow_models.FTCondBasic()
        inputs = eager_model.get_random_inputs()
        (
            exir.capture(
                eager_model,
                inputs,
                exir.CaptureConfig(
                    pt2_mode=True,
                    enable_functionalization=False,
                    enable_dynamic_shape=True,
                ),
            )
            .transform(FunktionalizationPass(inputs))
            .to_edge(
                exir.EdgeCompileConfig(
                    _check_ir_validity=False,
                )
            )
            .to_executorch()
        )
        # pass the test if no exception

    def test_assume_constant_by_default_prop(self) -> None:
        def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            if x.shape[0] > 3:
                return x.cos()
            return x.sin()

        dynamo_config = ExirDynamoConfig(assume_static_by_default=True)
        capture_config = exir.CaptureConfig(
            pt2_mode=True, enable_dynamic_shape=True, _dynamo_config=dynamo_config
        )
        captured = exir.capture(
            foo, (torch.ones(6, 2), torch.ones(6, 3)), capture_config
        ).graph_module
        found = False
        for node in captured.graph.nodes:
            # at least one input needs to have concrete dims
            if "val" in node.meta:
                fake_val = node.meta["val"]
                for dim in fake_val.shape:
                    if is_concrete_int(dim):
                        found = True

        self.assertTrue(found)

    def test_aot_config(self) -> None:
        class FooWithBuffer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer", torch.zeros(42))

            def forward(self, x):
                return x.cos() + self.buffer.sum()

        capture_config = exir.CaptureConfig(pt2_mode=True, enable_aot=True)
        captured_gm = exir.capture(
            FooWithBuffer(), (torch.ones(6, 2),), capture_config
        ).graph_module

        placeholder_nodes = set()
        print(captured_gm.graph)
        for node in captured_gm.graph.nodes:
            self.assertFalse(node.op == "get_attr")
            if node.op == "placeholder":
                placeholder_nodes.add(node)
            if node.op == "call_function" and node.target == torch.ops.aten.add.Tensor:
                # make sure the placeholders are used
                arg_0, arg_1 = node.args
                self.assertEqual(
                    placeholder_nodes,
                    {
                        list(arg_0._input_nodes.keys())[0],
                        list(arg_1._input_nodes.keys())[0],
                    },
                )

        self.assertEqual(len(placeholder_nodes), 2)
