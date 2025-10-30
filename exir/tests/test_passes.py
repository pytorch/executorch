# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import copy
import os
import tempfile
import unittest
from typing import Callable, List, Optional, Tuple

import executorch.exir as exir

# Import passes
import executorch.exir.memory_planning  # noqa
import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer_utils import (
    QuantizationConfig,
)
from executorch.backends.xnnpack.utils.configs import (
    get_xnnpack_executorch_backend_config,
)

from executorch.exir import (
    EdgeCompileConfig,
    EdgeProgramManager,
    memory,
    to_edge,
    to_edge_transform_and_lower,
)
from executorch.exir.dialects._ops import bind_pattern_to_op, ops, ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.emit import emit_program
from executorch.exir.graph_module import get_control_flow_submodules
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import (
    dead_code_elimination_pass,
    DebugPass,
    HintBasedSymShapeEvalPass,
    MemoryPlanningPass,
    propagate_dynamic_shape,
    RemoveNoopPass,
    ReplaceSymSizeOpPass,
    ToOutVarPass,
)
from executorch.exir.passes.constant_prop_pass import constant_prop_pass
from executorch.exir.passes.debug_handle_generator_pass import (
    DebugHandleGeneratorPass,
    generate_missing_debug_handles,
)
from executorch.exir.passes.insert_write_back_for_buffers_pass import (
    insert_write_back_for_buffers_pass,
)

from executorch.exir.passes.memory_format_ops_pass import DimOrderOpsRevertPass
from executorch.exir.passes.normalize_view_copy_base_pass import (
    NormalizeViewCopyBasePass,
)
from executorch.exir.passes.remove_graph_asserts_pass import RemoveGraphAssertsPass
from executorch.exir.passes.remove_mixed_type_operators import RemoveMixedTypeOperators
from executorch.exir.passes.replace_edge_with_backend_pass import EdgeToBackendOpsPass
from executorch.exir.passes.replace_view_copy_with_view_pass import (
    ReplaceViewCopyWithViewPass,
)
from executorch.exir.passes.scalar_to_tensor_pass import ScalarToTensorPass
from executorch.exir.passes.spec_prop_pass import SpecPropPass
from executorch.exir.passes.sym_to_tensor_pass import SymToTensorPass
from executorch.exir.program._program import lift_constant_tensor_pass
from executorch.exir.schema import TensorShapeDynamism
from executorch.exir.tensor import TensorSpec
from executorch.exir.tests.common import register_additional_test_aten_ops
from executorch.exir.tests.control_flow_models import FTCondDeadCode, FTMapBasic
from executorch.exir.tests.models import FeedForwardBlock, MLP, Mul
from functorch.experimental import control_flow

from torch import nn
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND
from torch.export import export
from torch.export.graph_signature import InputKind, InputSpec, TensorArgument
from torch.fx import GraphModule, subgraph_rewriter
from torch.fx.experimental.proxy_tensor import make_fx
from torch.library import impl, Library
from torch.testing import FileCheck
from torch.utils import _pytree as pytree

from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchao.quantization.pt2e.quantizer import QuantizationSpec


# pyre-ignore
def collect_ops(gm: torch.fx.GraphModule):
    """
    Collect all targets for call_function nodes from the graph module recursively.
    """
    ops = set()
    for subgm in gm.modules():
        if not isinstance(subgm, torch.fx.GraphModule):
            continue
        for node in subgm.graph.nodes:
            if node.op == "call_function":
                ops.add(node.target)
    return ops


lib = Library("DO_NOT_USE_TEST_ONLY", "DEF")

lib.define("foo(Tensor self) -> (Tensor, Tensor)")
lib.define("add_relu(Tensor self, Tensor other) -> Tensor")


@impl(lib, "foo", "CompositeExplicitAutograd")
def foo(a: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    return a + 1, None


lib.define(
    "foo.out(Tensor self, *, Tensor(a!) out1, Tensor(b!) out2) -> (Tensor(a!), Tensor(b!))"
)


@impl(lib, "foo.out", "CompositeExplicitAutograd")
def foo_out(
    a: torch.Tensor, out1: torch.Tensor, out2: torch.Tensor
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    return a + 1, None


def simple_promote_dtype(
    dtype: torch.dtype, promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND
) -> torch.dtype:
    if promotion_kind == ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT:
        return dtype
    if promotion_kind == ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT:
        return dtype if dtype.is_floating_point else torch.float
    else:
        raise Exception(f"Unsupported promotion kind {promotion_kind}")


def count_nodes_with_target_asserting_arguments_have_dtype(
    self, module, target, arg_dtype
) -> int:
    count = 0
    for node in module.graph.nodes:
        if node.op == "call_function" and node.target == target:
            count += 1
            for arg in node.args:
                self.assertEqual(arg.meta["val"].dtype, arg_dtype)
    return count


class TestPasses(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        register_additional_test_aten_ops()

    def test_remove_mixed_type_operators(self) -> None:
        def make_module(fwd: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
            class Module(torch.nn.Module):
                def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                    return fwd(x, y)

            return Module

        Add = make_module(lambda x, y: (x + y) + x)
        Sub = make_module(lambda x, y: (x - y) - x)
        Mult = make_module(lambda x, y: x * y)
        Minimum = make_module(torch.minimum)
        DivWithoutMode = make_module(torch.div)
        DivWithNoneMode = make_module(lambda x, y: torch.div(x, y, rounding_mode=None))
        DivWithTruncMode = make_module(
            lambda x, y: torch.div(x, y, rounding_mode="trunc")
        )
        DivWithFloorMode = make_module(
            lambda x, y: torch.div(x, y, rounding_mode="floor")
        )

        for module, op, expected_count, promotion_kind in (
            (
                Add,
                exir_ops.edge.aten.add.Tensor,
                2,
                ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
            ),
            (
                Sub,
                exir_ops.edge.aten.sub.Tensor,
                2,
                ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
            ),
            (
                Mult,
                exir_ops.edge.aten.mul.Tensor,
                1,
                ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
            ),
            (
                Minimum,
                exir_ops.edge.aten.minimum.default,
                1,
                ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
            ),
            (
                DivWithoutMode,
                exir_ops.edge.aten.div.Tensor,
                1,
                ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
            ),
            (
                DivWithNoneMode,
                exir_ops.edge.aten.div.Tensor_mode,
                1,
                ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
            ),
            (
                DivWithTruncMode,
                exir_ops.edge.aten.div.Tensor_mode,
                1,
                ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
            ),
            (
                DivWithFloorMode,
                exir_ops.edge.aten.div.Tensor_mode,
                1,
                ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
            ),
        ):
            for second_arg_dtype in (torch.int64, torch.float, torch.double):
                int_tensor = torch.tensor([[1, 2, 3]], dtype=torch.int64)
                float_tensor = torch.tensor([[1.0, 2.0, 3.0]], dtype=second_arg_dtype)
                edge_prog = to_edge(
                    export(module(), (int_tensor, float_tensor), strict=True)
                )

                new_prog = edge_prog.transform([RemoveMixedTypeOperators()])
                new_graph_module = new_prog.exported_program().graph_module
                self.assertIsNotNone(new_graph_module)

                promoted_type = simple_promote_dtype(second_arg_dtype, promotion_kind)
                count = count_nodes_with_target_asserting_arguments_have_dtype(
                    self, new_graph_module, op, promoted_type
                )
                self.assertEqual(count, expected_count)

    def test_remove_noop_pass(self) -> None:
        class Foo(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.to(dtype=torch.float32)

        foo = Foo()

        # Turn off functionalization so that we can get the actual to.dtype op
        edge_prog = to_edge(
            export(foo, (torch.ones(1, dtype=torch.float32),), strict=True)
        )
        edge_prog = edge_prog.transform([RemoveNoopPass()])
        self.assertIsNotNone(edge_prog.exported_program().graph_module)
        new_graph_module = edge_prog.exported_program().graph_module
        for node in new_graph_module.graph.nodes:
            if node.op == "call_function":
                self.assertNotEqual(node.target, torch.ops.aten.to.dtype)

    def test_redundant_slice_copy_removal(self) -> None:
        class FooWithNoSlice(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x[:, :, :]

        foo_with_no_slice = FooWithNoSlice()

        class FooWithOneSlice(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x[:1, :, :]

        foo_with_one_slice = FooWithOneSlice()

        class FooWithAllSlices(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x[:1, :2, 2:4]

        foo_with_all_slices = FooWithAllSlices()

        # Turn off functionalization so that we can get the actual to.dtype op
        x = torch.ones((3, 8, 8))
        prog = to_edge(export(foo_with_no_slice, (x,), strict=True))
        prog = prog.transform([RemoveNoopPass()])
        new_graph_module = prog.exported_program().graph_module
        FileCheck().check_count(
            "executorch_exir_dialects_edge__ops_aten_slice_copy_Tensor", 0, exactly=True
        ).run(new_graph_module.code)

        prog = to_edge(export(foo_with_one_slice, (x,), strict=True))
        prog = prog.transform([RemoveNoopPass()])
        new_graph_module = prog.exported_program().graph_module
        FileCheck().check_count(
            "executorch_exir_dialects_edge__ops_aten_slice_copy_Tensor", 1, exactly=True
        ).run(new_graph_module.code)

        prog = to_edge(export(foo_with_all_slices, (x,), strict=True))
        prog = prog.transform([RemoveNoopPass()])
        new_graph_module = prog.exported_program().graph_module
        FileCheck().check_count(
            "executorch_exir_dialects_edge__ops_aten_slice_copy_Tensor", 3, exactly=True
        ).run(new_graph_module.code)

    def test_compile_to_edge(self) -> None:
        class Foo(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * 2

        f = Foo()

        x = (torch.randn(2, 3),)

        to_edge(export(f, x, strict=True)).exported_program().graph_module
        # TODO(angelayi): Add a utility function that verifies a model is in
        # the edge dialect

    def test_to_out_variant_none_output(self) -> None:
        class CompositeModel(torch.nn.Module):
            def __init__(self, _weight):
                super().__init__()
                self.weight = _weight
                self.lstm = torch.nn.LSTM(
                    input_size=32,
                    hidden_size=32,
                    num_layers=1,
                )

            def forward(self, x_raw, h, c):
                output, (hn, cn) = self.lstm(x_raw, (h, c))
                return output

        # Prepare input and trace it
        input_x = torch.ones([1, 32])
        input_h = torch.ones([1, 32])
        input_c = torch.ones([1, 32])
        inputs = (input_x, input_h, input_c)

        composite_m = CompositeModel(3)

        edge_prog = to_edge(
            export(composite_m, inputs, strict=True),
            # torch._ops.aten.t.default
            compile_config=exir.EdgeCompileConfig(_check_ir_validity=False),
        )

        new_prog = edge_prog.transform([SpecPropPass()])

        new_gm_res = ToOutVarPass()(new_prog.exported_program().graph_module)
        self.assertIsNotNone(new_gm_res)
        new_gm = new_gm_res.graph_module
        for node in new_gm.graph.nodes:
            if node.op == "call_function" and node.target in [
                torch.ops.DO_NOT_USE_TEST_ONLY.foo.out,
                torch.ops.my_awesome_3rdparty_ns.awesome_op.out,
            ]:
                self.assertEqual(len(node.kwargs), 2)
                out1_node = node.kwargs["out1"]
                self.assertEqual(out1_node.op, "call_function")
                self.assertIs(out1_node.target, memory.alloc)
                self.assertIs(node.kwargs["out2"], None)

        new_gm_res = MemoryPlanningPass()(new_gm)
        self.assertIsNotNone(new_gm_res)
        new_gm = new_gm_res.graph_module
        new_prog.exported_program().graph_module.graph = new_gm.graph
        emit_program(new_prog.exported_program())

    def test_to_out_variant_singleon_tensor_list(self) -> None:
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.split(x, 10)

            def get_random_inputs(self):
                return (torch.randn(10),)

        model = MyModel()
        inputs = model.get_random_inputs()
        prog = to_edge(
            export(model, inputs, strict=True),
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )  # TODO(larryliu): fix split_copy
        new_gm_res = ToOutVarPass()(prog.exported_program().graph_module)
        self.assertIsNotNone(new_gm_res)
        new_gm = new_gm_res.graph_module

        for nd in new_gm.graph.nodes:
            if nd.target is exir_ops.edge.aten.split_copy.Tensor_out:
                break

        val = nd.meta["val"]

        # We must return a spec which is a list of a signle TensorSpec item.
        # Returning the TensorSpec item directly cause future getitem op fails.
        self.assertTrue(isinstance(val, (tuple, list)))
        self.assertEqual(1, len(val))

    def test_to_out_variant_multiple_out(self) -> None:
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.topk(x, 5)

            def get_random_inputs(self):
                return (torch.randn(10),)

        model = MyModel()
        inputs = model.get_random_inputs()
        prog = to_edge(
            export(model, inputs, strict=True),
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )  # TODO(larryliu): fix topk
        new_gm_res = ToOutVarPass()(prog.exported_program().graph_module)
        self.assertIsNotNone(new_gm_res)
        new_gm = new_gm_res.graph_module

        for nd in new_gm.graph.nodes:
            if nd.target is torch.ops.aten.topk.values:
                break

        val = nd.meta["val"]

        # We must return a spec which is a list of a signle TensorSpec item.
        # Returning the TensorSpec item directly cause future getitem op fails.
        self.assertTrue(isinstance(val, (tuple, list)))
        self.assertEqual(2, len(val))

    def test_to_out_variant_to_copy(self) -> None:
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.to(torch.int32)

        model = Module()

        inputs = torch.tensor(1.0, dtype=torch.float)
        model_res = model(inputs)

        edge_dialect = to_edge(export(model, (inputs,), strict=True))
        edge_res = edge_dialect.exported_program().module()(inputs)
        self.assertTrue(torch.allclose(model_res, edge_res))

    def test_export_pass(self) -> None:
        class Foo(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
                y = torch.cat([x, x])
                return torch.ops.aten.tensor_split.sections(y, 2)

        f = Foo()

        class NullPass(ExportPass):
            pass

        prog = to_edge(
            export(f, (torch.ones(3, 2),), strict=True),
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )  # TODO(larryliu): fix cat
        new_prog = prog.transform([NullPass()])
        new_nodes = new_prog.exported_program().graph_module.graph.nodes
        for node in new_nodes:
            if node.op != "call_function":
                continue
            self.assertTrue(hasattr(node, "stack_trace"))
            self.assertIsNotNone(node.stack_trace)

        old_nodes = prog.exported_program().graph_module.graph.nodes
        self.assertEqual(len(new_nodes), len(old_nodes))
        for new_node, old_node in zip(new_nodes, old_nodes):
            self.assertEqual(new_node.op, old_node.op)
            self.assertEqual(new_node.target, old_node.target)

    def test_export_pass_pt2(self) -> None:
        class Foo(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
                y = torch.cat([x, x])
                return torch.ops.aten.tensor_split.sections(y, 2)

        f = Foo()

        class NullPass(ExportPass):
            pass

        prog = to_edge(
            export(f, (torch.ones(3, 2),), strict=True),
            compile_config=exir.EdgeCompileConfig(_check_ir_validity=False),
        )
        new_prog = prog.transform([NullPass()])
        new_nodes = new_prog.exported_program().graph_module.graph.nodes
        for node in new_nodes:
            if node.op != "call_function":
                continue
            self.assertTrue(hasattr(node, "stack_trace"))
            self.assertIsNotNone(node.stack_trace)

        old_nodes = prog.exported_program().graph_module.graph.nodes
        self.assertEqual(len(new_nodes), len(old_nodes))
        for new_node, old_node in zip(new_nodes, old_nodes):
            self.assertEqual(new_node.op, old_node.op)
            self.assertEqual(new_node.target, old_node.target)

    def test_export_scalar_to_tensor_pass(self) -> None:
        class Mul(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * 3.14

        mul = Mul()

        expo_prog = to_edge(export(mul, (torch.ones(1),), strict=True))
        new_prog = expo_prog.transform([ScalarToTensorPass()])
        self.assertIsNotNone(new_prog.exported_program().graph_module)
        new_graph_module = new_prog.exported_program().graph_module

        inp = torch.zeros(1)
        self.assertTrue(
            torch.allclose(
                expo_prog.exported_program().module()(inp),
                new_prog.exported_program().module()(inp),
            )
        )
        for node in new_graph_module.graph.nodes:
            if node.op == "call_function":
                for arg in node.args + tuple(node.kwargs.values()):
                    self.assertFalse(isinstance(arg, float))

    def test_remove_mixed_types_symfloats(self) -> None:
        class Foo(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.nn.functional.interpolate(
                    x,
                    size=(x.shape[2] * 2, x.shape[3] * 3),
                    mode="bilinear",
                    align_corners=False,
                    antialias=False,
                )

        f = Foo()

        example_inputs = (torch.randn(2, 3, 4, 5),)

        gm = to_edge(export(f, example_inputs, strict=True))
        new_gm = gm.transform(
            [ReplaceSymSizeOpPass(), ScalarToTensorPass(), RemoveMixedTypeOperators()]
        )
        self.assertIsNotNone(new_gm.exported_program().graph_module)

        self.assertTrue(
            torch.allclose(
                gm.exported_program().module()(*example_inputs),
                new_gm.exported_program().module()(*example_inputs),
            )
        )

    def test_spec_prop_pass(self) -> None:
        class Foo(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + x

        f = Foo()

        gm = (
            to_edge(export(f, (torch.ones(3, 2),), strict=True))
            .exported_program()
            .graph_module
        )
        new_gm = SpecPropPass()(gm)
        self.assertIsNotNone(new_gm)
        new_nodes = new_gm.graph_module.graph.nodes
        counter = 0
        for node in new_nodes:
            if node.op != "output":
                continue
            counter += 1
            self.assertIs(node.meta["spec"][0], node.args[0][0].meta["spec"])

        self.assertEqual(counter, 1)

    def test_spec_prop_pass_tuple_output(self) -> None:
        class Foo(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
                return (x + x,)

        f = Foo()

        gm = (
            to_edge(export(f, (torch.ones(3, 2),), strict=True))
            .exported_program()
            .graph_module
        )
        new_gm = SpecPropPass()(gm)
        self.assertIsNotNone(new_gm)
        new_nodes = new_gm.graph_module.graph.nodes
        counter = 0
        for node in new_nodes:
            if node.op != "output":
                continue
            counter += 1
            self.assertIs(node.meta["spec"][0], node.args[0][0].meta["spec"])

        self.assertEqual(counter, 1)

    def test_compile_fix_broken_ops(self) -> None:
        class ExportableLoop(nn.Module):
            def __init__(self, hidden_size, out_channels):
                super().__init__()
                self.hidden_size = hidden_size
                self.B = nn.Parameter(torch.randn(hidden_size, 1))  # (H, in_channels)
                self.C = nn.Parameter(
                    torch.randn(out_channels, hidden_size)
                )  # (C_out, H)
                A = torch.randn(2, hidden_size)
                self.A_real = nn.Parameter(A[0].clone())
                self.A_imag = nn.Parameter(A[1].clone())

            def update_state(self, h, x_t):
                # h: [B, 2, H], x_t: [B, H]
                hr, hi = h[:, 0, :], h[:, 1, :]  # [B, H]
                hrn = hr * self.A_real - hi * self.A_imag + x_t  # [B, H]
                hin = hi * self.A_real + hr * self.A_imag  # [B, H]
                hn = torch.stack([hrn, hin], dim=1)  # [B, 2, H]
                return hn, hrn

            def forward(self, u):
                # u: [B, 1, T]
                x = torch.matmul(self.B, u)  # (B, H, T)
                B, H, T = x.shape

                h = torch.zeros(B, 2, H, device=x.device, dtype=x.dtype)  # [B, 2, H]
                h_accum = torch.zeros(
                    B, H, T, device=x.device, dtype=x.dtype
                )  # [B, H, T]
                i = torch.tensor(0, device=x.device, dtype=torch.int64)
                one = torch.tensor(1, device=x.device, dtype=torch.int64)

                def cond(i, h, h_accum):
                    return i < T

                def body(i, h, h_accum):
                    x_t = x.index_select(-1, i.unsqueeze(0)).squeeze(
                        -1
                    )  # ✅ safe for export
                    h, hr = self.update_state(h, x_t)  # h: [B, 2, H], hr: [B, H]
                    h_accum = h_accum.index_copy(
                        -1, i.unsqueeze(0), hr.unsqueeze(-1)
                    )  # [B, H, T]
                    i_next = i + one
                    return i_next, h, h_accum

                _, h, h_accum = torch._higher_order_ops.while_loop(
                    cond, body, (i, h, h_accum)
                )
                y = torch.matmul(self.C, h_accum).transpose(0, 1)  # (B, C_out, T)
                return y

        # Instantiate and export
        model = ExportableLoop(hidden_size=128, out_channels=10)
        inp = torch.randn(1, 1, 32)  # (B, in_channels=1, T=32)
        ep = export(model, (inp,))
        prog = to_edge(
            ep,
            compile_config=exir.EdgeCompileConfig(_check_ir_validity=False),
        )
        gm = prog.exported_program().graph_module
        count_after = 0
        for node in gm.graph.nodes:
            if (
                node.target == torch.ops.aten.squeeze.dims
                or node.target == torch.ops.aten.select.int
            ):
                count_after += 1
        self.assertEqual(count_after, 0)
        self.assertTrue(
            torch.allclose(prog.exported_program().module()(inp), model(inp))
        )

    def test_convert_symb_ops(self) -> None:
        class Foo(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.add(x, x.shape[0] - 1)

        f = Foo()

        # Mark the 0th dimension of X as dynamic with a max value of 3.
        dim_x = torch.export.Dim("dim_x", max=3)

        prog = to_edge(
            export(
                f, (torch.ones(3, 2),), dynamic_shapes={"x": {0: dim_x}}, strict=True
            ),
            compile_config=exir.EdgeCompileConfig(_check_ir_validity=False),
        )
        new_prog = prog.transform([EdgeToBackendOpsPass()])
        self.assertIsNotNone(new_prog.exported_program().graph_module)
        converted_gm = new_prog.exported_program().graph_module

        FileCheck().check("torch.ops.aten.sym_size.int").check(
            "executorch_exir_dialects_backend__ops_executorch_prim_sub_Scalar"
        ).check_not("operator.sub").run(converted_gm.code)

    def test_alloc_node_spec(self) -> None:
        """
        Make sure every memory.alloc node including those in sub graph modules
        have a TensorSpec.
        """
        eager_model = FTMapBasic()
        inputs = eager_model.get_random_inputs()
        prog = to_edge(
            export(eager_model, inputs, strict=True),
            compile_config=exir.EdgeCompileConfig(_check_ir_validity=False),
        )
        passes = [
            SpecPropPass(),
            HintBasedSymShapeEvalPass(),
        ]
        new_prog = prog.transform(passes)

        new_gm_res = ToOutVarPass()(new_prog.exported_program().graph_module)
        self.assertIsNotNone(new_gm_res)
        new_gm = new_gm_res.graph_module

        new_gm_res = MemoryPlanningPass()(new_gm)
        self.assertIsNotNone(new_gm_res)
        new_gm = new_gm_res.graph_module

        alloc_nodes = []
        for subgm in new_gm.modules():
            if isinstance(subgm, torch.fx.GraphModule):
                for node in subgm.graph.nodes:
                    if node.target == memory.alloc:
                        alloc_nodes.append(node)
        self.assertTrue(len(alloc_nodes) > 0)
        for node in alloc_nodes:
            self.assertTrue(isinstance(node.meta.get("spec", None), TensorSpec))

    def test_debug_pass_file_log(self) -> None:
        eager_model = Mul()
        inputs = eager_model.get_random_inputs()

        # the debug pass works with a graph generated with make_fx directly
        gm = make_fx(eager_model)(*inputs)

        try:
            fd, path = tempfile.mkstemp()

            print(f"Write DebugPass output to {path}")
            DebugPass(log_filename=path)(gm)
            with open(path) as f:
                file_cont = f.read()
            self.assertTrue("torch.ops.aten.mul" in file_cont)
        finally:
            os.close(fd)
            os.unlink(path)

    def test_dce_recursive(self) -> None:
        eager_model = FTCondDeadCode()
        inputs = eager_model.get_random_inputs()
        gm = export(eager_model, inputs, strict=True).graph_module

        dead_code_elimination_pass(gm)
        gm.print_readable()
        self.assertFalse(torch.ops.aten.sub.Tensor in collect_ops(gm))

    def test_propagate_dynamic_shape(self) -> None:
        class Foo(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = x
                for _ in range(2):
                    y = y + x
                return y

        f = Foo()

        prog = to_edge(
            export(f, (torch.rand(5),), strict=True),
            # missing dispatch key
            compile_config=exir.EdgeCompileConfig(_check_ir_validity=False),
        ).transform(propagate_dynamic_shape())
        gm = prog.exported_program().graph_module
        nspec = 0
        for n in gm.graph.nodes:
            for spec in pytree.tree_flatten(n.meta["spec"])[0]:
                self.assertTrue(all(isinstance(x, int) for x in spec.shape))
                nspec += 1

        self.assertTrue(nspec > 0)

    def test_losing_symbolic_info(self) -> None:
        """
        Guard against an issue that after calling ConvertSymbolicOpsPass(),
        future ExportPass will encounter symbolic information loss.
        """

        class Foo(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.add(x, x.shape[0] - 1)

        f = Foo()

        dim_x = torch.export.Dim("dim_x", max=3)
        prog = to_edge(
            export(
                f, (torch.ones(3, 2),), dynamic_shapes={"x": {0: dim_x}}, strict=True
            ),
            compile_config=exir.EdgeCompileConfig(_check_ir_validity=False),
        )

        new_prog = prog.transform([EdgeToBackendOpsPass()])
        gm = new_prog.exported_program().graph_module
        gm.print_readable()
        *_, ones, out = gm.graph.nodes
        print(f"Before ExportPass: {ones.format_node()}")
        self.assertTrue(isinstance(ones.meta["val"].shape[0], torch.SymInt))
        self.assertTrue(len(ones.meta["val"].shape[0].node.expr.free_symbols) > 0)

        new_prog = new_prog.transform([ExportPass()])
        gm = new_prog.exported_program().graph_module
        gm.print_readable()
        *_, ones, out = gm.graph.nodes
        print(f"After ExportPass: {ones.format_node()}")
        self.assertTrue(isinstance(ones.meta["val"].shape[0], torch.SymInt))
        self.assertTrue(len(ones.meta["val"].shape[0].node.expr.free_symbols) > 0)

    def test_to_edge_with_edge_ops(self) -> None:
        x = torch.randn([2, 3, 4, 5])

        class Foo(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + x

        f = Foo()

        gm = to_edge(export(f, (x,), strict=True)).exported_program().graph_module
        for node in gm.graph.nodes:
            if node.op == "call_function":
                self.assertEqual(type(node.target), EdgeOpOverload)

    # TODO(T143084047)
    @unittest.expectedFailure
    def test_backend_fused_op_retraceable(self) -> None:
        """This test makes sure the backend op is still retraceable, with the pattern being registered as kernel."""

        class Foo(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                z = x + y
                return torch.ops.aten.relu.default(z)

        f = Foo()

        gm = export(
            f,
            (
                torch.randn(2, 2),
                torch.randn(2, 2),
            ),
            strict=True,
        )
        # should look like:
        # graph():
        #     %ph_0 : [#users=1] = placeholder[target=ph_0]
        #     %ph_1 : [#users=1] = placeholder[target=ph_1]
        #     %add_tensor : [#users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%ph_0, %ph_1), kwargs = {})
        #     %relu_default : [#users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor,), kwargs = {})
        #     return [relu_default]
        FileCheck().check("torch.ops.aten.add.Tensor").check(
            "torch.ops.aten.relu.default"
        ).run(gm.graph_module.code)

        class AddReluFusionPass(ExportPass):
            def call(self, graph_module: GraphModule) -> PassResult:
                # decorator registers this pattern as a CompositeExplicitAutograd kernel, since there's no kernel registered before.
                @bind_pattern_to_op(lib, "add_relu")
                def pattern(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                    z = torch.ops.aten.add.Tensor(x, y)
                    out = torch.ops.aten.relu.default(z)
                    return out

                def replacement(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                    return ops.backend.DO_NOT_USE_TEST_ONLY.add_relu.default(x, y)

                subgraph_rewriter.replace_pattern(graph_module, pattern, replacement)
                return PassResult(graph_module, True)

        # TODO: larryliu this pass needs to be in to_executorch()
        class OpReplacePass(ExportPass):
            def call_operator(self, op, args, kwargs, meta):
                if op == torch.ops.DO_NOT_USE_TEST_ONLY.add_relu.default:
                    return super().call_operator(
                        ops.backend.DO_NOT_USE_TEST_ONLY.add_relu.default,
                        args,
                        kwargs,
                        meta,
                    )
                return super().call_operator(op, args, kwargs, meta)

        gm_lowered = to_edge(
            gm,
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
            ),
        ).transform([AddReluFusionPass(), OpReplacePass()])

        FileCheck().check(
            "executorch_exir_dialects_backend__ops_DO_NOT_USE_TEST_ONLY_add_relu_default"
        ).run(gm_lowered.exported_program().graph_module.code)
        # lowered module:
        # def forward(self, ph_0, ph_1):
        #     do_not_use_test_only_add_relu_default = executorch_exir_dialects_backend__ops_DO_NOT_USE_TEST_ONLY_add_relu_default(ph_0, ph_1);  ph_0 = ph_1 = None
        #     return [do_not_use_test_only_add_relu_default]

        # Retrace:
        # If not backend op retrace will error out because no CPU/CompositeExplicitAutograd kernel registered.
        gm_retraced = to_edge(
            export(
                gm_lowered.exported_program().module(),
                (
                    torch.randn(2, 2),
                    torch.randn(2, 2),
                ),
                strict=True,
            )
        )
        # Retrace-able, the graph "promote" back to ATen dialect, showing up add and relu, which is expected.
        FileCheck().check("torch.ops.aten.add.Tensor").check(
            "torch.ops.aten.relu.default"
        ).run(gm_retraced.exported_program().graph_module.code)

    def test_debug_handle_generator_pass(self) -> None:
        eager_model = MLP(2, output_size=4)
        inputs = eager_model.get_random_inputs()

        graph_module = (
            to_edge(export(eager_model, inputs, strict=True))
            .exported_program()
            .graph_module
        )

        # Every node except input and output should have debug handle
        for node in graph_module.graph.nodes:
            if node.op != "placeholder" and node.op != "output":
                self.assertIn("debug_handle", node.meta)
        ScalarToTensorPass()(graph_module)

        for node in graph_module.graph.nodes:
            if node.op != "placeholder" and node.op != "output":
                self.assertIn("debug_handle", node.meta)

    def test_debug_handle_generator_pass_generate_same_debug_handle_on_ops_sharing_same_source(
        self,
    ) -> None:
        eager_model = FeedForwardBlock(256, 512)
        inputs = (torch.randn(12, 256),)

        graph_module = (
            to_edge(export(eager_model, inputs, strict=True))
            .exported_program()
            .graph_module
        )

        same_source_nodes = {
            "aten_native_layer_norm_default": (
                "aten_native_layer_norm_default",
                "getitem",
            ),
            "getitem": ("aten_native_layer_norm_default", "getitem"),
            "aten_permute_copy_default": (
                "aten_permute_copy_default",
                "aten_addmm_default",
            ),
            "aten_addmm_default": ("aten_permute_copy_default", "aten_addmm_default"),
            "aten_native_dropout_default": ("aten_native_dropout_default", "getitem_1"),
            "getitem_1": ("aten_native_dropout_default", "getitem_1"),
            "aten_relu_default": ("aten_relu_default",),
            "aten_permute_copy_default_1": (
                "aten_permute_copy_default_1",
                "aten_addmm_default_1",
            ),
            "aten_addmm_default_1": (
                "aten_permute_copy_default_1",
                "aten_addmm_default_1",
            ),
            "aten_native_dropout_default_1": (
                "aten_native_dropout_default_1",
                "getitem_2",
            ),
            "getitem_2": ("aten_native_dropout_default_1", "getitem_2"),
        }

        node_name_to_debug_handle = {}

        # Node having same source should have same debug handle
        for node in graph_module.graph.nodes:
            if node.op != "placeholder" and node.op != "output":
                self.assertIn("debug_handle", node.meta)
                if node.name in node_name_to_debug_handle:
                    for node_name_with_same_debug_handle in same_source_nodes[
                        node.name
                    ]:
                        self.assertEqual(
                            node_name_to_debug_handle[node_name_with_same_debug_handle],
                            node.meta["debug_handle"],
                        )
                else:
                    for node_name_with_same_debug_handle in same_source_nodes[
                        node.name
                    ]:
                        node_name_to_debug_handle[node_name_with_same_debug_handle] = (
                            node.meta["debug_handle"]
                        )

    def test_generate_missing_debug_handles(self) -> None:
        eager_model = MLP(2, output_size=4)
        inputs = eager_model.get_random_inputs()

        ep = to_edge(export(eager_model, inputs, strict=True)).exported_program()

        # get the first non-placeholder node
        first_non_placeholder_node = [
            n for n in ep.graph.nodes if n.op != "placeholder"
        ][0]

        first_non_placeholder_node.meta.pop("debug_handle")
        self.assertTrue(first_non_placeholder_node.meta.get("debug_handle") is None)
        generate_missing_debug_handles(ep)
        self.assertTrue(first_non_placeholder_node.meta.get("debug_handle") is not None)

    def test_debug_handle_generator_pass_with_control_flow(self) -> None:
        def true_nested(y: torch.Tensor) -> torch.Tensor:
            y = y + y
            y = torch.mm(y, y)
            return y

        def false_nested(y: torch.Tensor) -> torch.Tensor:
            return torch.mm(y, y)

        def true_fn(x: torch.Tensor, pred2: torch.Tensor) -> torch.Tensor:
            z = control_flow.cond(pred2, true_nested, false_nested, [x])
            return x + z

        def false_fn(x: torch.Tensor, _) -> torch.Tensor:
            return x.cos()

        def map_fn(
            x: torch.Tensor, pred1: torch.Tensor, pred2: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            x = x.cos()
            y = control_flow.cond(pred1, true_fn, false_fn, [y, pred2])
            x = x + y
            return x.sin()

        class Foo(torch.nn.Module):
            def forward(
                self,
                xs: torch.Tensor,
                pred1: torch.Tensor,
                pred2: torch.Tensor,
                y: torch.Tensor,
            ) -> torch.Tensor:
                y = torch.mm(y, y)
                return control_flow.map(map_fn, xs, pred1, pred2, y)

        f = Foo()

        inputs = (
            torch.ones(2, 2),
            torch.tensor([False]),
            torch.tensor([False]),
            torch.ones(2, 2),
        )

        ep = to_edge(export(f, inputs, strict=True)).exported_program()
        graph_module = ep.graph_module

        def check_debug_handle_metadata(graph_module: torch.fx.GraphModule) -> None:
            queue = [graph_module]
            while queue:
                current_graph_module = queue.pop(0)
                for node in current_graph_module.graph.nodes:
                    if node.op != "placeholder" and node.op != "output":
                        self.assertIn("debug_handle", node.meta)
                control_flow_submodules = [
                    submodule
                    for _, submodule, _ in get_control_flow_submodules(
                        current_graph_module
                    )
                ]
                queue.extend(control_flow_submodules)

        DebugHandleGeneratorPass()(graph_module)
        check_debug_handle_metadata(graph_module)

        # Check debug handle still preserved after ScalarToTensorPass
        ScalarToTensorPass()(graph_module)
        check_debug_handle_metadata(graph_module)

    def test_symint_conversion(self) -> None:
        class Foo(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.add(x, x.shape[0] - 1)

        f = Foo()

        dim_x = torch.export.Dim("dim_x", max=3)
        prog = to_edge(
            export(
                f, (torch.ones(3, 2),), dynamic_shapes={"x": {0: dim_x}}, strict=True
            ),
            compile_config=exir.EdgeCompileConfig(_check_ir_validity=False),
        )
        prog = prog.transform([SymToTensorPass()])

        FileCheck().check("torch.ops.aten.scalar_tensor.default").run(
            prog.exported_program().graph_module.code
        )
        self.assertTrue(
            torch.allclose(
                f(torch.ones(3, 2)), prog.exported_program().module()(torch.ones(3, 2))
            )
        )
        self.assertTrue(
            torch.allclose(
                f(torch.zeros(3, 2)),
                prog.exported_program().module()(torch.zeros(3, 2)),
            )
        )

    def test_remove_assert_pass(self) -> None:
        class Foo(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                assert x.shape[0] == 5
                return x * x

        f = Foo()

        gm = to_edge(
            export(f, (torch.randn(5),), strict=True),
            compile_config=exir.EdgeCompileConfig(_check_ir_validity=False),
        )
        new_gm = gm.transform([RemoveGraphAssertsPass()])
        num_asserts = [
            node
            for node in new_gm.exported_program().graph.nodes
            if node.op == "call_function"
            and node.target == torch.ops.aten._assert_async.msg
        ]
        self.assertEqual(len(num_asserts), 0)

    def test_arange(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.ones(2)

            def forward(self, x):
                return torch.arange(start=0, end=2) + x

        _ = to_edge(export(M(), (torch.randn(2),), strict=True)).to_executorch()

    def test_replace_slice(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.ones(10)

            def forward(self, x):
                return self.a[:2] + x

        gm = (
            to_edge(export(M(), (torch.randn(2),), strict=True))
            .exported_program()
            .graph_module
        )
        FileCheck().check(
            "executorch_exir_dialects_edge__ops_aten_slice_copy_Tensor"
        ).run(gm.code)

    def test_constant_prop_for_output(self) -> None:
        class Add(torch.nn.Module):
            def forward(self) -> torch.Tensor:
                return torch.add(torch.tensor(3), torch.tensor(5))

        add = Add()

        edge = to_edge(
            export(add, (), strict=True),
            compile_config=EdgeCompileConfig(_skip_dim_order=False),
        )
        # Check there is a lifted tensor followed by a to_copy node
        FileCheck().check("c_lifted_tensor_0").check("c_lifted_tensor_1").run(
            edge.exported_program().graph_module.code
        )

        edge._edge_programs["forward"] = constant_prop_pass(
            edge.exported_program("forward"), _skip_dim_order=False
        )

        # Check (c_lifted_tensor_*) nodes are all replaced by _prop_tensor_constant.
        FileCheck().check_not("c_lifted_tensor_").check("_prop_tensor_constant").run(
            edge.exported_program().graph_module.code
        )
        # Validate that the program successfully passes validation to executorch:
        edge.exported_program()._validate()
        edge.to_executorch()

    def test_constant_prop_pass_for_add(self) -> None:
        class Add(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + 3

        add = Add()

        edge = to_edge(
            export(add, (torch.ones(1),), strict=True),
            compile_config=EdgeCompileConfig(_skip_dim_order=False),
        )
        edge = edge.transform([ScalarToTensorPass(), RemoveMixedTypeOperators()])
        exported_program = lift_constant_tensor_pass(edge.exported_program())

        # Check there is a lifted tensor followed by a to_copy node
        FileCheck().check("_lifted_tensor_constant0").check(
            "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default"
        ).run(exported_program.graph_module.code)

        new_ep = constant_prop_pass(exported_program)

        # Check (_lifted_tensor_constant + to_copy) node is replaced by prop tensor
        FileCheck().check_not("_lifted_tensor_constant").check(
            "_prop_tensor_constant0"
        ).check_not(
            "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default"
        ).run(
            new_ep.graph_module.code
        )

    def test_pass_no_user_inputs(self) -> None:
        class NoUserInputs(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("a", torch.ones(1))

            def forward(self) -> torch.Tensor:
                return 3 + self.a

        mod = NoUserInputs()
        exported_program = export(mod, (), strict=True)
        edge = to_edge(
            exported_program,
            compile_config=EdgeCompileConfig(_skip_dim_order=False),
        )
        ep = edge.exported_program()
        # because there is no user input, the lifted constant should be the first input.
        FileCheck().check("_lifted_tensor_constant1").check(
            "b_a"  # followed by the buffer input.
        ).run(ep.graph_module.code)

        # the graph signature should also be the same:
        self.assertEqual(
            ep.graph_signature.input_specs[0].arg.name, "_lifted_tensor_constant1"
        )
        self.assertEqual(ep.graph_signature.input_specs[1].arg.name, "b_a")

        # Validate that the program successfully passes validation to executorch:
        edge.to_executorch()

    def test_constant_prop_pass_for_parameter(self) -> None:
        def count_additions(gm: torch.fx.GraphModule) -> int:
            return sum(
                (node.target == torch.ops.aten.add.Tensor) for node in gm.graph.nodes
            )

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Parameter(torch.ones(1, 2, 3))

            def forward(self, x):
                b = self.a + self.a
                c = torch.cat([self.a, b])
                return (c + c) + x

        aten = export(M(), (torch.zeros(2, 2, 3),), strict=True)
        self.assertEqual(count_additions(aten.graph_module), 3)
        new_ep = constant_prop_pass(aten)
        self.assertEqual(count_additions(new_ep.graph_module), 1)

    def test_constant_prop_pass_graph_signature(self) -> None:
        def count_additions(gm: torch.fx.GraphModule) -> int:
            return sum(
                (node.target == torch.ops.aten.add.Tensor) for node in gm.graph.nodes
            )

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Parameter(torch.ones(1, 2, 3))

            def forward(self, x):
                b = self.a + self.a
                c = torch.cat([self.a, b])
                return (c + c) + x

        aten = export(M(), (torch.zeros(2, 2, 3),), strict=True)
        # Input signature will have two entries:
        # (1) parameter `a` and (2) user input `x`.
        self.assertEqual(len(aten.graph_signature.input_specs), 2)
        new_ep = constant_prop_pass(aten)
        # Check that there are exactly two propagated tensors - (1) propagated
        # constant and (2) user input.
        self.assertEqual(
            new_ep.graph_signature.input_specs,
            [
                InputSpec(
                    kind=InputKind.CONSTANT_TENSOR,
                    arg=TensorArgument(name="_prop_tensor_constant0"),
                    target="_prop_tensor_constant0",
                    persistent=True,
                ),
                # User input graph signature.
                aten.graph_signature.input_specs[-1],
            ],
        )

    def test_constant_prop_pass_after_delegation(self) -> None:
        class M(torch.nn.Module):
            def __init__(self, dim=32):
                super().__init__()
                self.linear = torch.nn.Linear(dim, dim)

            def forward(self, query, key, value):
                query = self.linear(query)
                key = self.linear(key)
                value = self.linear(value)
                return torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=True
                )

        query = torch.randn(32, 32, 32, 32)
        key = torch.randn(32, 32, 32, 32)
        value = torch.randn(32, 32, 32, 32)

        # Capture the model
        m = torch.export.export(M(32), (query, key, value), strict=True).module()

        # 8w16a quantization
        from torchao.quantization.pt2e import MinMaxObserver, PerChannelMinMaxObserver

        activation_qspec = QuantizationSpec(
            dtype=torch.int16,
            quant_min=-32768,
            quant_max=32767,
            qscheme=torch.per_tensor_affine,
            is_dynamic=False,
            observer_or_fake_quant_ctr=MinMaxObserver,
        )
        weight_qspec = QuantizationSpec(
            dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0,
            is_dynamic=False,
            observer_or_fake_quant_ctr=PerChannelMinMaxObserver,
        )
        custom_qconfig = QuantizationConfig(
            input_activation=activation_qspec,
            output_activation=activation_qspec,
            weight=weight_qspec,
            bias=None,
            is_qat=False,
        )
        quantizer = XNNPACKQuantizer()
        quantizer.set_global(custom_qconfig)
        m = prepare_pt2e(m, quantizer)  # pyre-fixme[6]
        m = convert_pt2e(m)

        # export, perform constant propagation to make weights const
        aten_prog = export(m, (query, key, value), strict=True)
        aten_prog = constant_prop_pass(aten_prog)

        # lower to edge dialect
        edge_prog = to_edge(
            aten_prog,
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False, _use_edge_ops=True
            ),
        )
        edge_prog = edge_prog.to_backend(XnnpackPartitioner())

        # Perform constant propagation on the decomposed ops from sdpa
        aten_prog = constant_prop_pass(edge_prog.exported_program())
        # There should be at least one const due to spda op
        self.assertGreaterEqual(len(aten_prog.constants), 1)

    def test_constant_prop_pass_for_parameter_slice(self) -> None:
        def count_slice(gm: torch.fx.GraphModule) -> int:
            return sum(
                (node.target == torch.ops.aten.slice_copy.Tensor)
                for node in gm.graph.nodes
            )

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Parameter(torch.ones(3, 2, 2))

            def forward(self, x):
                # Create slice of shape (1, 2, 2)
                slice_tensor = torch.slice_copy(self.a, dim=0, start=0, end=1)
                return torch.cat([x, slice_tensor])

        aten = export(M(), (torch.zeros(2, 2, 2),), strict=True)
        self.assertIn("a", aten.state_dict)
        self.assertEqual(count_slice(aten.graph_module), 1)

        new_ep = constant_prop_pass(aten)
        # Check there is a propagated tensor.
        FileCheck().check("_prop_tensor_constant0").run(aten.graph_module.code)
        self.assertIn("_prop_tensor_constant0", new_ep.constants)
        self.assertNotIn("a", new_ep.state_dict)
        # No more slice copy.
        self.assertEqual(count_slice(new_ep.graph_module), 0)

    def test_constant_prop_pass_no_propagate(self) -> None:
        def count_placeholder(gm: torch.fx.GraphModule) -> int:
            return sum((node.op == "placeholder") for node in gm.graph.nodes)

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Parameter(torch.ones(3, 2, 4))

            def forward(self, x, y):
                # y is unused.
                return x + self.a

        aten = export(M(), (torch.zeros(3, 2, 4), torch.zeros(3, 2, 4)), strict=True)
        self.assertIn("a", aten.state_dict)
        self.assertEqual(count_placeholder(aten.graph_module), 3)

        new_ep = constant_prop_pass(aten)
        # Check there is no propagated tensor.
        FileCheck().check("p_a").check("x").check("y").run(aten.graph_module.code)
        self.assertNotIn("_prop_tensor_constant0", new_ep.constants)
        self.assertIn("a", new_ep.state_dict)
        self.assertEqual(count_placeholder(new_ep.graph_module), 3)

    def test_constant_prop_pass_for_control_flow(self) -> None:
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)
                self.w = torch.randn(3, 3)

            def t(self, val):
                return val + 1

            def f(self, val):
                return val - 1

            def true_fn(self, val):
                return self.linear(val) + self.t(val)

            def false_fn(self, val):
                return self.linear(val) - self.f(val)

            def forward(self, pred, x):
                out = torch.nn.functional.linear(
                    x, self.w.to(torch.float16).to(torch.float32)
                )
                return torch.cond(pred, self.true_fn, self.false_fn, [out])

        mod = Module()
        x = torch.randn([3, 3])
        pred = torch.tensor(x[0][0].item() < 0)
        edge = to_edge(
            export(mod, (pred, x), strict=True),
            compile_config=exir.EdgeCompileConfig(_check_ir_validity=False),
        )
        expected_out = edge.exported_program().module()(pred, x)

        warn_log = (
            "constant_prop_pass does not constant propagate in control flow modules"
        )
        with self.assertLogs(level="WARNING") as log:
            program = constant_prop_pass(edge.exported_program())
            self.assertIn(warn_log, log.output[0])

        out = program.module()(pred, x)
        self.assertTrue(torch.allclose(expected_out, out))

        # dtype casts in parent module are const propagated
        FileCheck().check(
            "executorch_exir_dialects_edge__ops_aten_mm_default(x, _prop_tensor_constant"
        ).run(program.graph_module.code)

    def test_constant_prop_pass_quant_primitives(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w_int = torch.ones(3, 3, dtype=torch.int8)
                self.w_scale = 3.0
                self.w_zero_point = 3

            def forward(self, x):
                w_dq = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                    self.w_int, self.w_scale, self.w_zero_point, -127, 128, torch.int8
                )
                return torch.nn.functional.linear(x, w_dq)

        mod = M()
        x = torch.randn([3])
        mod(x)
        edge = to_edge(
            export(mod, (x,), strict=True),
            compile_config=exir.EdgeCompileConfig(_check_ir_validity=False),
        )
        constant_prop_pass(edge.exported_program())
        FileCheck().check(
            "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default"
        ).run(edge.exported_program().graph_module.code)

    def test_mutable_buffers(self) -> None:
        def count_copies(gm: torch.fx.GraphModule) -> int:
            return sum(
                (node.target == torch.ops.aten.copy_.default) for node in gm.graph.nodes
            )

        class MutableStateModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(1))
                self.register_buffer("direct_copy_from_input", torch.zeros(1))

            def forward(self, x):
                y = x + self.state
                self.state.add_(1)
                self.direct_copy_from_input.copy_(x)
                return y

        model = to_edge(export(MutableStateModule(), (torch.zeros(1),), strict=True))
        self.assertEqual(count_copies(model.exported_program().graph_module), 0)
        # Before
        # graph():
        #     %b_state : [num_users=2] = placeholder[target=b_state]
        #     %b_direct_copy_from_input : [num_users=0] = placeholder[target=b_direct_copy_from_input]
        #     %_lifted_tensor_constant2 : [num_users=1] = placeholder[target=_lifted_tensor_constant2]
        #     %x : [num_users=2] = placeholder[target=x]
        #     %aten_add_tensor : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.add.Tensor](args = (%x, %b_state), kwargs = {})
        #     %dim_order_ops__to_dim_order_copy_default : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.dim_order_ops._to_dim_order_copy.default](args = (%_lifted_tensor_constant2,), kwargs = {dtype: torch.float32, dim_order: []})
        #     %aten_add_tensor_1 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.add.Tensor](args = (%b_state, %dim_order_ops__to_dim_order_copy_default), kwargs = {})
        #     return (aten_add_tensor_1, x, aten_add_tensor)
        gm, _ = insert_write_back_for_buffers_pass(model.exported_program())

        # After
        # graph():
        #     %b_state : [num_users=3] = placeholder[target=b_state]
        #     %b_direct_copy_from_input : [num_users=1] = placeholder[target=b_direct_copy_from_input]
        #     %_lifted_tensor_constant2 : [num_users=1] = placeholder[target=_lifted_tensor_constant2]
        #     %x : [num_users=2] = placeholder[target=x]
        #     %aten_add_tensor : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.add.Tensor](args = (%x, %b_state), kwargs = {})
        #     %dim_order_ops__to_dim_order_copy_default : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.dim_order_ops._to_dim_order_copy.default](args = (%_lifted_tensor_constant2,), kwargs = {dtype: torch.float32, dim_order: []})
        #     %aten_add_tensor_1 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.add.Tensor](args = (%b_state, %dim_order_ops__to_dim_order_copy_default), kwargs = {})
        #     %copy__default : [num_users=1] = call_function[target=torch.ops.aten.copy_.default](args = (%b_state, %aten_add_tensor_1), kwargs = {})
        #     %copy__default_1 : [num_users=1] = call_function[target=torch.ops.aten.copy_.default](args = (%b_direct_copy_from_input, %x), kwargs = {})
        #     return (copy__default, copy__default_1, aten_add_tensor)
        self.assertEqual(count_copies(gm), 2)

    def test_remove_quantized_op_noop_pass(self) -> None:
        class TestAddSliceNoop(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x = x + x
                x = x + x[:]
                return x

        class TestAddSliceNotNoop(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x = x + x
                x = x + x[:1]
                return x

        def count_dq_nodes(gm: torch.fx.GraphModule) -> int:
            return sum(
                (
                    node.target
                    in (
                        torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
                    )
                )
                for node in gm.graph.nodes
            )

        def count_q_nodes(gm: torch.fx.GraphModule) -> int:
            return sum(
                (
                    node.target
                    in (
                        torch.ops.quantized_decomposed.quantize_per_tensor.default,
                        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
                    )
                )
                for node in gm.graph.nodes
            )

        def quantize_model(
            m_eager: torch.nn.Module, example_inputs: Tuple[torch.Tensor]
        ) -> Tuple[EdgeProgramManager, int, int]:
            # program capture
            m = torch.export.export(m_eager, example_inputs, strict=True).module()

            quantizer = XNNPACKQuantizer()
            quantization_config = get_symmetric_quantization_config()
            quantizer.set_global(quantization_config)
            m = prepare_pt2e(m, quantizer)  # pyre-fixme[6]
            m = convert_pt2e(m, fold_quantize=True)
            ep = torch.export.export(m, example_inputs, strict=True)
            dq_nodes_pre = count_dq_nodes(ep.graph_module)
            q_nodes_pre = count_q_nodes(ep.graph_module)
            edge = to_edge(
                ep, compile_config=EdgeCompileConfig(_check_ir_validity=False)
            )
            return edge, dq_nodes_pre, q_nodes_pre

        example_inputs = (torch.randn(9, 8),)
        model = TestAddSliceNoop()
        m_eager = model.eval()
        edge, dq_nodes_pre, q_nodes_pre = quantize_model(m_eager, example_inputs)

        dq_nodes_post = count_dq_nodes(edge.exported_program().graph_module)
        q_nodes_post = count_q_nodes(edge.exported_program().graph_module)
        # One dq and one q node around the slice copy should have been removed.
        self.assertEqual(dq_nodes_pre - dq_nodes_post, 1)
        self.assertEqual(q_nodes_pre - q_nodes_post, 1)

        # Check that the slice_copy is removed by the RemoveNoopPass.
        for node in edge.exported_program().graph_module.graph.nodes:
            self.assertFalse("slice" in str(node.target))

        model = TestAddSliceNotNoop()
        m_eager = model.eval()
        edge, dq_nodes_pre, q_nodes_pre = quantize_model(m_eager, example_inputs)

        dq_nodes_post = count_dq_nodes(edge.exported_program().graph_module)
        q_nodes_post = count_q_nodes(edge.exported_program().graph_module)
        # One dq and one q node around the slice copy should have been removed.
        self.assertEqual(dq_nodes_pre, dq_nodes_post)
        self.assertEqual(q_nodes_pre, q_nodes_post)

        # Check that the slice_copy is not removed by the RemoveNoopPass.
        self.assertTrue(
            any(
                "slice" in str(node.target)
                for node in edge.exported_program().graph_module.graph.nodes
            )
        )

    def test_dq_q_no_op_pass(self) -> None:
        class TestDqQ(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                dq = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                    x, 1.0, 0, -128, 127, torch.int8
                )
                q = torch.ops.quantized_decomposed.quantize_per_tensor.default(
                    dq, 1.0, 0, -128, 127, torch.int8
                )
                return q

        model = TestDqQ()
        m_eager = model.eval()
        ep = torch.export.export(m_eager, (torch.randn(9, 8),), strict=True)
        edge = to_edge(ep)
        # Check that the dq and q nodes are not touched by the RemoveNoopPass.
        self.assertTrue(
            any(
                "dequantize" in str(node.target)
                for node in edge.exported_program().graph_module.graph.nodes
            )
        )
        self.assertTrue(
            any(
                "quantize" in str(node.target)
                for node in edge.exported_program().graph_module.graph.nodes
            )
        )

    def test_dq_q_different_qparams(self) -> None:
        class TestDqQDifferentQParam(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                dq = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                    x, 1.0, 0, -128, 127, torch.int8
                )
                slice_copy_output = torch.ops.aten.slice_copy.Tensor(dq, 0, 0)
                q = torch.ops.quantized_decomposed.quantize_per_tensor.default(
                    slice_copy_output, 1.0, 0, -127, 127, torch.int8
                )
                return q

        model = TestDqQDifferentQParam()
        m_eager = model.eval()
        ep = torch.export.export(m_eager, (torch.randn(9, 8),), strict=True)
        edge = to_edge(ep)
        print(edge.exported_program().graph_module.graph)
        # Check that the dq and q nodes are not touched by the RemoveNoopPass.
        self.assertTrue(
            any(
                "dequantize" in str(node.target)
                for node in edge.exported_program().graph_module.graph.nodes
            )
        )
        self.assertTrue(
            any(
                "quantize" in str(node.target)
                for node in edge.exported_program().graph_module.graph.nodes
            )
        )
        self.assertFalse(
            any(
                "slice" in str(node.target)
                for node in edge.exported_program().graph_module.graph.nodes
            )
        )

    def test_normalize_view_copy_base_pass(self) -> None:
        class ViewChain(torch.nn.Module):
            def forward(self, x):
                x = torch.ops.aten.view_copy.default(x, [30, 1])
                x = torch.ops.aten.view_copy.default(x, [5, 6])
                x = torch.ops.aten.view_copy.default(x, [2, 15])
                x = torch.ops.aten.view_copy.default(x, [3, -1])
                return x

        def is_view_copy(node: torch.fx.Node) -> bool:
            return (
                node.op == "call_function"
                and node.target == torch.ops.aten.view_copy.default
            )

        gm = export(ViewChain(), (torch.ones(30),), strict=True).graph_module

        # Check before transformation
        n_view_copy_before = 0
        n_view_copy_bases_before = 0
        for node in gm.graph.nodes:
            if is_view_copy(node):
                n_view_copy_before += 1
                base = node.args[0]
                if is_view_copy(base):
                    n_view_copy_bases_before += 1

        self.assertEqual(n_view_copy_before, 4)
        self.assertEqual(n_view_copy_bases_before, 3)

        # Do transformation
        p = NormalizeViewCopyBasePass()
        gm_res = p(gm)
        assert gm_res is not None
        gm = gm_res.graph_module

        # Check after transformation
        n_view_copy_after = 0
        n_view_copy_bases_after = 0
        for node in gm.graph.nodes:
            if is_view_copy(node):
                n_view_copy_after += 1
                base = node.args[0]
                if is_view_copy(base):
                    n_view_copy_bases_after += 1

        self.assertEqual(n_view_copy_after, 4)
        self.assertEqual(n_view_copy_bases_after, 0)

    def test_replace_view_copy_with_view_pass(self) -> None:  # noqa: C901
        # Helper functions
        def is_view_copy(node: torch.fx.Node) -> bool:
            return (
                node.op == "call_function"
                and node.target == torch.ops.aten.view_copy.default
            )

        def is_memory_view(node: torch.fx.Node) -> bool:
            return node.op == "call_function" and node.target == memory.view

        # Test example set up
        class TestViewCopies(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.parameter = torch.nn.Parameter(torch.ones(1))

            def forward(self, x):
                o1 = torch.ops.aten.view_copy.default(x, [1])
                o2 = torch.ops.aten.view_copy.default(self.parameter, [1])
                # view_copys at the end of a function are not replaced, so add
                # a computation before the end of the graph.
                return torch.ops.aten.add.Tensor(o1, o2)

        ep = torch.export.export(TestViewCopies(), args=(torch.ones(1),), strict=True)
        for node in ep.graph.nodes:
            if node.op == "placeholder":
                node.meta["spec"] = TensorSpec.from_tensor(torch.empty(1))
                node.meta["spec"].shape_dynamism = TensorShapeDynamism.STATIC

        # Run tests
        gm = ep.graph_module

        # Check before transformation
        FileCheck().check_count(
            "torch.ops.aten.view_copy.default", 2, exactly=True
        ).run(gm.code)
        FileCheck().check_count("executorch_exir_memory_view", 0, exactly=True).run(
            gm.code
        )

        # Do transformation
        p = ReplaceViewCopyWithViewPass()
        gm_res = p(gm)
        assert gm_res is not None
        gm = gm_res.graph_module

        # Check after transformation
        FileCheck().check_count(
            "torch.ops.aten.view_copy.default", 0, exactly=True
        ).run(gm.code)
        FileCheck().check_count("executorch_exir_memory_view", 2, exactly=True).run(
            gm.code
        )

    def test_constant_prop_pass_for_mutable_buffers(self) -> None:
        def count_adds(gm: torch.fx.GraphModule) -> int:
            return len(
                gm.graph.find_nodes(
                    op="call_function", target=exir_ops.edge.aten.add.Tensor
                )
            )

        class MutableStateModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(1))

            def forward(self, x):
                x = x + self.state
                # Add 1 (constant) to state.
                self.state.add_(1)
                return x

        edge_manager = to_edge(
            export(MutableStateModule(), (torch.zeros(1),), strict=True)
        )
        self.assertEqual(count_adds(edge_manager.exported_program().graph_module), 2)
        edge_manager._edge_programs["forward"] = constant_prop_pass(
            edge_manager._edge_programs["forward"]
        )
        self.assertEqual(count_adds(edge_manager.exported_program().graph_module), 2)

    def test_constant_prop_pass_for_no_grad(self) -> None:
        class LSTM(torch.nn.Module):
            def __init__(self, input_size, hidden_size, num_layers):
                super(LSTM, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.lstm = torch.nn.LSTM(
                    input_size, hidden_size, num_layers, batch_first=True
                )

            def forward(self, text_tokens):
                # input: (seq_len, batch, input_size)
                lstm_out, (new_hidden_state, new_cell_state) = self.lstm(
                    input=text_tokens, hx=None
                )
                return lstm_out

        lstm = LSTM(input_size=200, hidden_size=203, num_layers=2)
        example_input = (torch.rand(2, 10, 200),)

        aten = torch.export.export(lstm, example_input, strict=False)
        _EDGE_COMPILE_CONFIG = exir.EdgeCompileConfig(
            _check_ir_validity=True,
            _skip_dim_order=True,  # TODO(T189114319): Reuse dim order op after solving the ios oss issue
        )

        edge_manager: EdgeProgramManager = to_edge(
            aten,
            compile_config=_EDGE_COMPILE_CONFIG,
        )
        new_ep = constant_prop_pass(edge_manager._edge_programs["forward"])
        _ = copy.deepcopy(new_ep.module_call_graph)

    def test_dim_order_revert_pass(self) -> None:
        aten_op_str = "torch.ops.aten._to_copy.default"
        edge_aten_op_str = "executorch_exir_dialects_edge__ops_aten__to_copy_default"
        edge_dim_order_op_str = "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default"

        class Module(torch.nn.Module):
            """
            A simple module that has a single to op that converts to channels last and then back to contiguous.
            Assuming contiguous input.
            """

            def __init__(self):
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.to(memory_format=torch.channels_last).to(
                    memory_format=torch.contiguous_format
                ) + x.to(memory_format=torch.channels_last).to(
                    memory_format=torch.contiguous_format
                )

            @staticmethod
            def to_copy_count():
                return 4

        def _do_checks(
            test_str: str, allowed: str, allowed_count: int, not_allowed_list: List[str]
        ) -> None:
            for not_allowed in not_allowed_list:
                FileCheck().check_count(allowed, allowed_count, exactly=True).check_not(
                    not_allowed
                ).run(test_str)

        m = Module()
        n = m.to_copy_count()
        input = torch.randn([2, 3, 4, 5]).to(memory_format=torch.contiguous_format)

        # 1. vanilla export, no edge ops
        ep = export(m, (input,), strict=True).run_decompositions({})
        _do_checks(
            ep.graph_module.code,
            aten_op_str,
            n,
            [edge_aten_op_str, edge_dim_order_op_str],
        )

        # 2a. to edge without dim orders, we should see edge aten ops but not dim order ops
        edge_prog = to_edge(
            ep, compile_config=exir.EdgeCompileConfig(_skip_dim_order=True)
        )._edge_programs["forward"]
        _do_checks(
            edge_prog.graph_module.code,
            edge_aten_op_str,
            n,
            [aten_op_str, edge_dim_order_op_str],
        )

        # 3a. expect no change after the pass, we should see edge aten ops but not dim order ops
        new_res = DimOrderOpsRevertPass()(edge_prog.graph_module)
        self.assertIsNotNone(new_res)
        _do_checks(
            new_res.graph_module.code,
            edge_aten_op_str,
            n,
            [aten_op_str, edge_dim_order_op_str],
        )

        # 2b. let's try with dim order enabled, we should see edge dim order ops but not edge aten ops
        edge_prog_dim_order = to_edge(
            ep, compile_config=exir.EdgeCompileConfig(_skip_dim_order=False)
        )._edge_programs["forward"]
        _do_checks(
            edge_prog_dim_order.graph_module.code,
            edge_dim_order_op_str,
            n,
            [aten_op_str, edge_aten_op_str],
        )

        # 3b. expect edge aten ops after the pass, we should see not see the edge dim order ops
        new_res_dim_order = DimOrderOpsRevertPass()(edge_prog_dim_order.graph_module)
        self.assertIsNotNone(new_res_dim_order)
        _do_checks(
            new_res_dim_order.graph_module.code,
            edge_aten_op_str,
            n,
            [aten_op_str, edge_dim_order_op_str],
        )

        output_no_dim_order = new_res.graph_module(input)
        output_no_dim_order_revert = new_res_dim_order.graph_module(input)
        self.assertTrue(
            torch.allclose(output_no_dim_order[0], output_no_dim_order_revert[0])
        )

    def test_constant_prop_pass_none(self) -> None:
        """
        This checks that None arguments are treated as constants in constant_prop_pass.
        """

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cst = torch.ones(3, 3, 3, dtype=torch.int8)
                self.w = torch.ones(3, 3, 3, dtype=torch.int8)

            def forward(self, x):
                # Note: using e.g aten.linear would not work as None is not in the graph
                a = torch.ops.aten.convolution.default(
                    self.cst, self.w, None, [1], [0], [1], False, [0], 1
                )
                return a + x

        mod = M()
        x = torch.randn([3, 3, 3])
        mod(x)
        edge = to_edge(
            export(mod, (x,), strict=True),
            compile_config=exir.EdgeCompileConfig(_check_ir_validity=False),
        )
        # 2 constants: self.w and self.cst
        self.assertEqual(2, len(edge.exported_program().constants))
        pass_result = constant_prop_pass(edge.exported_program())
        # 1 constant: a (= self.w @ self.cst)
        self.assertEqual(1, len(pass_result.constants))

    def test_constant_prop_pass_zero_stride_tensors(self) -> None:
        """
        Test that constant propagation correctly handles tensors with zero strides
        by converting them to contiguous tensors. Zero-stride tensors can be created
        by operations like expand() and are not supported by ExecuTorch.
        """

        class ZeroStrideModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.const_param = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))

            def forward(self, x):
                unsqueezed = self.const_param.unsqueeze(
                    1
                )  # Shape: (3, 1), strides: (1, 1)
                # expand creates zero-stride tensor
                expanded = unsqueezed.expand(3, 5)  # Shape: (3, 5), strides: (1, 0)

                # Use the expanded tensor with the input to prevent elimination
                result = x + expanded.sum()
                return result

        model = ZeroStrideModel()
        x = torch.randn(3, 5)
        exported = torch.export.export(model, (x,))

        # Before constant prop: verify we have the parameter
        self.assertIn("const_param", exported.state_dict)

        const_prop_result = constant_prop_pass(exported)
        lowered = to_edge_transform_and_lower(
            const_prop_result,
            partitioner=[XnnpackPartitioner()],
        )

        # Should go through
        lowered.to_executorch(get_xnnpack_executorch_backend_config([SpecPropPass()]))
        self.assertGreater(len(const_prop_result.constants), 0)

        # Find the propagated constant tensor
        prop_tensor = None
        for constant_name, constant_tensor in const_prop_result.constants.items():
            if constant_name.startswith("_prop_tensor_constant"):
                prop_tensor = constant_tensor
                break

        # Verify the propagated tensor exists and has no zero strides
        self.assertIsNotNone(prop_tensor)
        self.assertNotIn(
            0,
            prop_tensor.stride(),
            f"Propagated tensor still has zero stride: {prop_tensor.stride()}",
        )

        # Verify the tensor is contiguous
        self.assertTrue(
            prop_tensor.is_contiguous(),
            f"Propagated tensor is not contiguous: {prop_tensor.stride()}",
        )
