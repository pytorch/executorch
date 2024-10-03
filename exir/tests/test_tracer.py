# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import unittest
from typing import Dict, List, Tuple

import executorch.exir as exir
import executorch.exir.tests.models as models

import torch

from executorch.exir import CaptureConfig
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.tests.common import register_additional_test_aten_ops
from executorch.exir.tracer import dynamo_trace, ExirDynamoConfig, using_dynamo
from functorch.experimental.control_flow import cond, map

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
            exir.capture(f, f.get_random_inputs(), exir.CaptureConfig())
            .to_edge()
            .exported_program.graph_module
        )

        FileCheck().check("executorch_exir_dialects_edge__ops_aten_sin").run(f.code)

    def test_static_control_flow(self) -> None:
        def f(pred: bool, x: torch.Tensor) -> torch.Tensor:
            if pred:
                return torch.sin(x).max()
            else:
                return torch.sin(x)

        pred = True
        x = torch.randn(100)
        f_true = (
            exir.capture(f, (pred, x), exir.CaptureConfig())
            .to_edge()
            .exported_program.graph_module
        )

        FileCheck().check("executorch_exir_dialects_edge__ops_aten_sin").check(
            "executorch_exir_dialects_edge__ops_aten_max"
        ).run(f_true.code)

        pred = False
        f_false = (
            exir.capture(f, (pred, x), exir.CaptureConfig())
            .to_edge()
            .exported_program.graph_module
        )
        FileCheck().check("executorch_exir_dialects_edge__ops_aten_sin").check_not(
            "executorch_exir_dialects_edge__ops_aten_max"
        ).run(f_false.code)

    def test_copy(self) -> None:
        f = models.BasicSinMax()
        f = (
            exir.capture(f, f.get_random_inputs(), exir.CaptureConfig())
            .to_edge()
            .exported_program.graph_module
        )

        self.assertTrue(isinstance(f, torch.fx.GraphModule))
        g = copy.deepcopy(f)
        self.assertTrue(isinstance(g, torch.fx.GraphModule))

    def test_stacktrace(self) -> None:
        def f(x: torch.Tensor) -> torch.Tensor:
            return x + x

        traced_f = (
            exir.capture(f, (torch.rand(2, 2),), exir.CaptureConfig())
            .to_edge()
            .exported_program.graph_module
        )
        # Check that stacktrace is populated and retained (by checking twice)
        self.assertTrue(
            any(node.meta.get("stack_trace", None) for node in traced_f.graph.nodes)
        )
        self.assertTrue(
            any(node.meta.get("stack_trace", None) for node in traced_f.graph.nodes)
        )

    def test_ones(self) -> None:
        class M(torch.nn.Module):
            def forward(self, x):
                y = torch.ones(x.shape[0])
                return x + y

        ep = torch.export.export(
            M(), (torch.ones(3),), dynamic_shapes={"x": {0: torch.export.Dim("x")}}
        )
        exir.to_edge(ep)

    def test_possible_input_mutation(self) -> None:
        def f(x: torch.Tensor) -> torch.Tensor:
            return torch.add(torch.ones(5), torch.ones(5), out=x)

        with self.assertRaisesRegex(
            SpecViolationError,
            r"operator .* is not functional",
        ):
            exir.capture(f, (torch.zeros(5),), exir.CaptureConfig()).to_edge()

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
            exir.capture(model, model.get_random_inputs(), exir.CaptureConfig())
            # torch._ops.aten.t.default
            .to_edge(
                exir.EdgeCompileConfig(_check_ir_validity=False)
            ).exported_program.graph_module
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
            exir.capture(f, (torch.zeros(1, 2, 3),), exir.CaptureConfig())
            .to_edge()
            .exported_program.graph_module
        )
        for node in module.graph.nodes:
            if node.target == exir_ops.edge.aten.max.dim:
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
                exir.CaptureConfig(),
            )
            .to_edge()
            .exported_program.graph_module
        )
        new_res = module(*inputs)
        for node in module.graph.nodes:
            if node.op == "output":
                self.assertIsInstance(node.meta["val"], list)
                self.assertEqual(len(node.meta["val"]), 2)

        self.assertTrue(torch.allclose(orig_res[0], new_res[0]))
        self.assertTrue(torch.allclose(orig_res[1], new_res[1]))

    def test_dynamo_capture_scalar_outputs(self) -> None:
        def f(x: torch.Tensor) -> float:
            return x.item()

        gm, guards = dynamo_trace(
            f,
            (torch.ones(1),),
            False,
            "real",
            ExirDynamoConfig(),
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
                    enable_functionalization=False,
                    enable_dynamic_shape=True,
                ),
            )
            .to_edge()
            .exported_program.graph_module
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
                    enable_functionalization=False,
                    enable_dynamic_shape=True,
                    _dynamo_config=ExirDynamoConfig(assume_static_by_default=True),
                ),
                # sym_size is not reg op
            )
            .to_edge(exir.EdgeCompileConfig(_check_ir_validity=False))
            .exported_program.graph_module
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
            gm = exir.capture(Module(), inp, exir.CaptureConfig())
            self.assertTrue(torch.allclose(Module()(*inp), gm(*inp)))

    # TODO (tmanlaibaatar) remove this test
    def test_pt2_mode_with_dynamo_config(self) -> None:
        def f(x: torch.Tensor) -> torch.Tensor:
            return x[: x.shape[0] - 1]

        inp = (torch.randn(4, 5),)
        prog = exir.capture(
            f,
            inp,
            # missing dispatch key
        ).to_edge()
        self.assertTrue(prog(torch.randn(4, 5)).shape[0], 3)

    def test_input_container_type(self) -> None:
        def f(x: torch.Tensor, y: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
            # pyre-ignore
            return {"a": x.sum() + sum(y).sum()}

        inp = (torch.randn(6, 5), [torch.randn(6, 5), torch.randn(6, 5)])

        # pyre-fixme[23]: Unable to unpack `(...) -> Tuple[GraphModule,
        #  Set[torch._guards.Guard]]` into 2 values.
        gm, _ = torch._dynamo.export(f, *inp, aten_graph=True, tracing_mode="symbolic")
        prog = exir.capture(f, inp, config=exir.CaptureConfig()).to_edge()

        self.assertEqual(prog(*inp), f(*inp))

    def test_aot_buffer_mutation(self) -> None:
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "_bin_num_examples",
                    torch.empty([42]).fill_(
                        0.0,
                    ),
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

        with self.assertRaisesRegex(
            RuntimeError,
            "Found a graph input that requires gradients, and received a mutation.",
        ):
            _ = exir.capture(
                model,
                example_inputs,
                exir.CaptureConfig(
                    enable_aot=True,
                ),
            )

        # Note that model._bin_num_examples is mutated during exir.capture
        # We need to create a new_model
        new_model = Module()
        example_inputs = (
            torch.randn(4),
            torch.tensor(0),
            torch.tensor(3.14),
        )

        ep = exir.capture(
            new_model,
            example_inputs,
            exir.CaptureConfig(
                enable_aot=True,
            ),
        )

        test_inputs = (
            torch.randn(4),
            torch.tensor(0),
            torch.tensor(2.1),
        )
        graph_outputs = ep(*test_inputs)
        eager_outputs = Module()(*test_inputs)
        self.assertEqual(len(graph_outputs), 2)
        self.assertEqual(len(eager_outputs), 2)
        self.assertTrue(torch.allclose(graph_outputs[0], eager_outputs[0]))
        self.assertTrue(torch.allclose(graph_outputs[1], eager_outputs[1]))

    def test_assume_constant_by_default_prop(self) -> None:
        def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            if x.shape[0] > 3:
                return x.cos()
            return x.sin()

        dynamo_config = ExirDynamoConfig(assume_static_by_default=True)
        capture_config = exir.CaptureConfig(
            enable_dynamic_shape=True, _dynamo_config=dynamo_config
        )
        captured = exir.capture(
            foo, (torch.ones(6, 2), torch.ones(6, 3)), capture_config
        ).exported_program.graph_module
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

        capture_config = exir.CaptureConfig(enable_aot=True)
        captured_ep = exir.capture(FooWithBuffer(), (torch.ones(6, 2),), capture_config)
        captured_gm = captured_ep.exported_program.graph_module

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
        captured_ep.to_edge()

    def test_export_unlift(self) -> None:
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer", torch.ones(6, 4))

            def forward(self, x):
                return x.cos() + self.buffer.sin()

        ep = exir.capture(
            Foo(),
            (torch.ones(6, 4),),
            exir.CaptureConfig(enable_aot=True, _unlift=True),
        )

        self.assertTrue(torch.allclose(ep(torch.ones(6, 4)), Foo()(torch.ones(6, 4))))

    def test_export_container_unlift(self) -> None:
        class FooContainerInputOutput(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer", torch.ones(6, 4))

            def forward(self, x):
                return x[0][0].cos() + x[0][1].sin() + self.buffer.sin()

        inp = ((torch.ones(6, 4), torch.ones(6, 4)),)
        ep = exir.capture(
            FooContainerInputOutput(),
            (inp,),
            CaptureConfig(enable_aot=True, _unlift=True),
        )
        self.assertTrue(torch.allclose(ep(inp), FooContainerInputOutput()(inp)))

    def test_export_container_input_unlift(self) -> None:
        class FooContainerInputOutputV2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer", torch.ones(6, 4))

            def forward(self, x, y):
                return x[0].cos() + y[0].sin() + self.buffer.sin()

        inp = ((torch.ones(6, 4),), (torch.ones(6, 4),))
        ep = exir.capture(
            FooContainerInputOutputV2(),
            inp,
            CaptureConfig(enable_aot=True, _unlift=True),
        )
        self.assertTrue(torch.allclose(ep(*inp), FooContainerInputOutputV2()(*inp)))

    def test_export_cond(self) -> None:
        class A(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer", torch.ones(6, 4))

            def forward(self):
                return self.buffer.cos()

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = A()

            def forward(self, x):
                def true_fn(x):
                    return x.cos() + self.a().sum()

                def false_fn(x):
                    return x.sin()

                return cond(x.shape[0] > 4, true_fn, false_fn, [x])

        inp = torch.ones(6, 4)
        ep = exir.capture(
            Foo(),
            (inp,),
            CaptureConfig(enable_aot=True, _unlift=True),
        )
        self.assertTrue(torch.allclose(ep(torch.ones(6, 4)), Foo()(torch.ones(6, 4))))

    def test_export_cond_map(self) -> None:
        class A(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer", torch.ones(6, 4))

            def forward(self):
                return self.buffer.sum()

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = A()

            def inner(self, x, pred):
                def true_fn(x):
                    return x + x + self.a()

                def false_fn(x):
                    return x * x - self.a()

                return cond(pred, true_fn, false_fn, [x])

            def forward(self, pred, xs):
                def body(x, pred):
                    return self.inner(x, pred) + self.a()

                return map(body, xs, pred)

        inp = torch.randn(3, 2, 1)
        ep = exir.capture(
            Module(),
            (torch.tensor(True), inp),
            CaptureConfig(enable_aot=True, _unlift=True),
        )

        inp_test = torch.randn(3, 2, 1)
        self.assertTrue(
            torch.allclose(
                ep(torch.tensor(True), inp_test),
                Module()(torch.tensor(True), inp_test),
            )
        )
        self.assertTrue(
            torch.allclose(
                ep(torch.tensor(False), inp_test),
                Module()(torch.tensor(False), inp_test),
            )
        )
