# Copyright © 2023 Apple Inc. All rights reserved.
#
# Please refer to the license found in the LICENSE file in the root directory of the source tree.

import copy
import io
import logging
import sys
import unittest

import coremltools as ct

import executorch.exir

import torch
import torchvision

from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.backends.apple.coreml.test.test_coreml_utils import (
    IS_VALID_TEST_RUNTIME,
)
from executorch.exir.backend.utils import format_delegated_graph


@torch.library.custom_op("unsupported::linear", mutates_args=())
def _(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.aten.linear.default(x, w, b)


@torch.library.register_fake("unsupported::linear")
def _(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.aten.linear.default(x, w, b)


def is_fbcode():
    return not hasattr(torch.version, "git_version")


_TEST_RUNTIME = (sys.platform == "darwin") and not is_fbcode()
if _TEST_RUNTIME:
    from executorch.runtime import Runtime


class TestCoreMLPartitioner(unittest.TestCase):
    edge_compile_config = executorch.exir.EdgeCompileConfig()

    def test_add_sub_skip_mm(self):
        class Model(torch.nn.Module):
            def forward(self, a, x, b):
                y = torch.mm(a, x)
                z = y + b
                a = z - a
                y = torch.mm(a, x)
                z = y + b
                return z

        model = Model()
        model.eval()

        example_inputs = (torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2))
        exir_program_aten = torch.export.export(model, example_inputs, strict=True)

        edge_program_manager = executorch.exir.to_edge(
            exir_program_aten, compile_config=self.edge_compile_config
        )
        delegated_program_manager = edge_program_manager.to_backend(
            CoreMLPartitioner(skip_ops_for_coreml_delegation=["aten.mm.default"])
        )

        assert [
            node.target.__name__
            for node in delegated_program_manager.exported_program().graph.nodes
            if node.op == "call_function"
        ] == [
            "aten.mm.default",
            "executorch_call_delegate",
            "getitem",
            "aten.mm.default",
            "executorch_call_delegate",
            "getitem",
        ]

    def test_vit_skip_conv(self):
        model = torchvision.models.vit_b_16(weights="IMAGENET1K_V1")
        model.eval()

        example_inputs = (torch.randn(1, 3, 224, 224),)
        exir_program_aten = torch.export.export(model, example_inputs, strict=True)
        edge_program_manager = executorch.exir.to_edge(
            exir_program_aten, compile_config=self.edge_compile_config
        )
        delegated_program_manager = edge_program_manager.to_backend(
            CoreMLPartitioner(
                skip_ops_for_coreml_delegation=["aten.convolution.default"]
            )
        )

        assert [
            node.target.__name__
            for node in delegated_program_manager.exported_program().graph.nodes
            if node.op == "call_function"
        ] == [
            "aten.convolution.default",
            "executorch_call_delegate",
            "getitem",
        ]

    def test_ops_to_not_decompose(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, q, k, v, mask):
                out = torch.ops.aten.scaled_dot_product_attention.default(
                    q, k, v, attn_mask=mask
                )

                # triu/tril should be ignored by do_not_decompose
                # because otherwise they fail during CoreML lowering
                offset1 = torch.triu(mask, diagonal=1)
                offset2 = torch.tril(mask)
                offset = offset1 + offset2
                offset = torch.sum(offset)

                # Add non-functional and alias ops
                # These will be removed by ExecuTorch in non-decomposition
                # table because they cannot be functionalized
                out = out.transpose(1, 2)
                out = out.view(1, -1)
                out = out.permute(0, 1)
                out = out.add_(1.0)
                out = out.mul_(2.0)
                out = out.div_(3.0)
                out = out.sub_(4.0)
                out = torch.ops.aten.view_copy.default(out, (-1,))
                out = out.select(0, 0)
                return out + offset

        model = Model()
        model.eval()

        batch_size = 1
        n_heads = 12
        seq_len = 1
        max_seq_length = 32
        embedding_dim = 16
        q = torch.randn(batch_size, n_heads, seq_len, embedding_dim)
        k = torch.randn(batch_size, n_heads, max_seq_length, embedding_dim)
        v = torch.randn(batch_size, n_heads, max_seq_length, embedding_dim)
        mask = torch.randn(seq_len, max_seq_length)
        example_inputs = (q, k, v, mask)
        ep = torch.export.export(model, example_inputs, strict=True)
        self.assertTrue(
            "torch.ops.aten.triu.default" in ep.graph_module.code,
        )
        self.assertTrue(
            "torch.ops.aten.tril.default" in ep.graph_module.code,
        )

        coreml_partitioner = CoreMLPartitioner()

        # Using to_edge_transform_and_lower, we expect SDPA will be preserved and show up in delegated graph
        edge_program_manager = executorch.exir.to_edge_transform_and_lower(
            ep, partitioner=[coreml_partitioner]
        )
        self.assertTrue(
            "executorch.exir.dialects.edge._ops.aten.scaled_dot_product_attention.default"
            in format_delegated_graph(
                edge_program_manager.exported_program().graph_module
            )
        )

        # Using to_edge flow, we expect SDPA will be decomposed and not show up in delegated graph
        edge_program_manager2 = executorch.exir.to_edge(ep)
        edge_program_manager2.to_backend(coreml_partitioner)
        self.assertTrue(
            "executorch.exir.dialects.edge._ops.aten.scaled_dot_product_attention.default"
            not in format_delegated_graph(
                edge_program_manager2.exported_program().graph_module
            )
        )

    def test_buffer(self):
        embedding_dim = 3
        max_seq_len = 2

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "cache",
                    torch.zeros((max_seq_len, embedding_dim), dtype=torch.float32),
                )

            def forward(self, q, k_val, input_pos):
                q_T = q.transpose(0, 1)
                k = torch.ops.aten.index_put_(self.cache, [input_pos, None], k_val)
                attn = k.mm(q_T)
                return attn

        model = Model()
        model.eval()

        q = torch.randn((1, embedding_dim))
        k_val = torch.randn((1, embedding_dim))
        input_pos = torch.tensor([0])
        example_inputs = (q, k_val, input_pos)
        exir_program_aten = torch.export.export(model, example_inputs, strict=True)

        compile_specs = CoreMLBackend.generate_compile_specs(
            minimum_deployment_target=ct.target.iOS18
        )
        partitioner = CoreMLPartitioner(compile_specs=compile_specs)
        edge_program_manager = executorch.exir.to_edge(
            exir_program_aten, compile_config=self.edge_compile_config
        )
        delegated_program_manager = edge_program_manager.to_backend(partitioner)

        assert [
            node.target.__name__
            for node in delegated_program_manager.exported_program().graph.nodes
            if node.op == "call_function"
        ] == [
            "executorch_call_delegate",
            "getitem",
        ]

    def test_lower_full_graph(self):
        class Model(torch.nn.Module):
            def forward(self, a, x, b):
                out = torch.ops.aten.linear.default(a, x, b)
                out2 = torch.ops.unsupported.linear.default(out, x, b)
                return out2

        model = Model()
        model.eval()

        example_inputs = (torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2))
        exir_program_aten = torch.export.export(model, example_inputs, strict=True)
        edge_program_manager = executorch.exir.to_edge(exir_program_aten)
        edge_program_manager2 = copy.deepcopy(edge_program_manager)

        delegated_program_manager = edge_program_manager.to_backend(CoreMLPartitioner())

        for node in delegated_program_manager.exported_program().graph.nodes:
            if node.op == "call_function":
                assert node.target.__name__ in [
                    "unsupported.linear.default",
                    "executorch_call_delegate",
                    "getitem",
                ], node.target.__name__

        with self.assertRaises(NotImplementedError):
            edge_program_manager2.to_backend(CoreMLPartitioner(lower_full_graph=True))

    # TODO: enable this after bugs are fixed in ExecuTorch's partitioner
    # def test_symint_arg(self):
    #     class Model(torch.nn.Module):
    #         def forward(self, x, w, b, y):
    #             val = y.item()
    #             torch._check(val >= 0)
    #             torch._check(val < 2)
    #             out = torch.ops.aten.linear.default(x, w, b)
    #             out2 = out.relu()[val]
    #             return out2

    #     model = Model()
    #     model.eval()
    #     example_inputs = (
    #         torch.randn(2, 2),
    #         torch.randn(2, 2),
    #         torch.randn(2, 2),
    #         torch.tensor(2),
    #     )
    #     exir_program_aten = torch.export.export(model, example_inputs)

    #     edge_program_manager = executorch.exir.to_edge(exir_program_aten)

    #     delegated_program_manager = edge_program_manager.to_backend(CoreMLPartitioner(skip_ops_for_coreml_delegation=["aten.scalar_tensor.default"]))

    #     # This op has symbolic args
    #     assert (
    #         "torch.ops.aten._assert_scalar.default"
    #         in delegated_program_manager.exported_program().graph_module.code
    #     )

    #     if _TEST_RUNTIME:
    #         et_prog = delegated_program_manager.to_executorch()
    #         runtime = Runtime.get()
    #         program = runtime.load_program(et_prog.buffer)
    #         method = program.load_method("forward")
    #         et_outputs = method.execute(*example_inputs)[0]
    #         eager_outputs = model(*example_inputs)
    #         self.assertTrue(torch.allclose(et_outputs, eager_outputs, atol=1e-02, rtol=1e-02))

    def test_take_over_constant_data_false(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(50, 100)

            def forward(self, x):
                return self.linear(x)

        model = Model()
        model.eval()
        example_inputs = (torch.randn(2, 50),)
        exir_program_aten = torch.export.export(model, example_inputs)

        edge_program_manager = executorch.exir.to_edge_transform_and_lower(
            exir_program_aten,
            partitioner=[CoreMLPartitioner(take_over_constant_data=False)],
        )
        for node in edge_program_manager.exported_program().graph.nodes:
            if (
                node.op == "call_function"
                and node.target.__name__ == "executorch_call_delegate"
            ):
                break

        # lowered_module_0, x, p_linear_weight, p_linear_bias
        assert len(node.args) == 4

        if _TEST_RUNTIME:
            et_prog = edge_program_manager.to_executorch()
            runtime = Runtime.get()
            program = runtime.load_program(et_prog.buffer)
            method = program.load_method("forward")
            et_outputs = method.execute(*example_inputs)[0]
            eager_outputs = model(*example_inputs)
            self.assertTrue(
                torch.allclose(et_outputs, eager_outputs, atol=1e-02, rtol=1e-02)
            )

    def test_aten_rand_default_falls_back_to_portable(self):
        """
        Regression test for https://github.com/pytorch/executorch/issues/11722.

        coremltools 9.0's ``aten.rand.default`` handler hits an unimplemented
        branch (``not enough values to unpack (expected 5, got 1)``).  The
        partitioner must reject it so the op falls back to the portable
        backend instead of crashing the export.  Sibling ops like
        ``aten.randn``, ``aten.rand_like``, etc. lower cleanly and are
        intentionally still delegated.
        """

        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.rand(x.shape) + x

        model = Model().eval()
        example_inputs = (torch.zeros(5, 5),)
        exir_program_aten = torch.export.export(model, example_inputs, strict=True)
        edge_program_manager = executorch.exir.to_edge_transform_and_lower(
            exir_program_aten, partitioner=[CoreMLPartitioner()]
        )
        op_names = [
            node.target.__name__
            for node in edge_program_manager.exported_program().graph.nodes
            if node.op == "call_function"
        ]
        self.assertIn("aten.rand.default", op_names)

    def test_aten_randn_is_still_delegated(self):
        """``aten.randn`` is *not* in the deny list — it lowers cleanly."""

        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.randn(x.shape) + x

        ep = torch.export.export(Model().eval(), (torch.zeros(5, 5),), strict=True)
        edge = executorch.exir.to_edge_transform_and_lower(
            ep, partitioner=[CoreMLPartitioner()]
        )
        op_names = [
            n.target.__name__
            for n in edge.exported_program().graph.nodes
            if n.op == "call_function"
        ]
        self.assertIn("executorch_call_delegate", op_names)
        self.assertNotIn("aten.randn.default", op_names)

    def test_ops_above_rank_5_are_skipped(self):
        """
        Core ML tensors have at most rank 5.  coremltools only rewrites an
        unbranched view, permute, view chain through a higher rank (a window
        partition), and raises on any other op above rank 5.  The partitioner
        must leave those ops to the portable backend and keep such a layout
        chain delegated.
        """

        class Model(torch.nn.Module):
            def forward(self, x):
                windows = x.view(2, 2, 2, 2, 2, 8).permute(0, 1, 3, 2, 4, 5)
                windows = windows.reshape(-1, 4, 8).relu()
                branched = x.view(2, 2, 2, 2, 2, 8).permute(0, 1, 3, 2, 4, 5)
                branched = (
                    branched.reshape(-1, 4, 8).sum() + branched.reshape(32, 8).sum()
                )
                stacked = torch.stack([x.unsqueeze(0), x.unsqueeze(0) * 2])
                return windows.sum() + branched + stacked.sum(0).relu().sum()

        model = Model().eval()
        x = torch.randn(2, 4, 4, 8)
        ep = torch.export.export(model, (x,), strict=True)
        edge = executorch.exir.to_edge_transform_and_lower(
            ep, partitioner=[CoreMLPartitioner()]
        )
        op_names = [
            n.target.__name__
            for n in edge.exported_program().graph.nodes
            if n.op == "call_function"
        ]
        self.assertIn("executorch_call_delegate", op_names)
        self.assertIn("aten.sum.dim_IntList", op_names)
        self.assertEqual(op_names.count("aten.permute_copy.default"), 1)
        if _TEST_RUNTIME:
            program = Runtime.get().load_program(edge.to_executorch().buffer)
            out = program.load_method("forward").execute((x,))[0]
            self.assertTrue(torch.allclose(out, model(x), atol=5e-2, rtol=5e-2))

        # A skipped permute leaves the chain's views with no rank 6 rewrite.
        edge = executorch.exir.to_edge_transform_and_lower(
            ep,
            partitioner=[
                CoreMLPartitioner(
                    skip_ops_for_coreml_delegation=["aten.permute_copy.default"]
                )
            ],
        )
        op_names = [
            n.target.__name__
            for n in edge.exported_program().graph.nodes
            if n.op == "call_function"
        ]
        self.assertEqual(op_names.count("aten.permute_copy.default"), 2)

    def test_argmax_argmin_dim_none_is_skipped(self):
        """
        Regression test for https://github.com/pytorch/executorch/issues/11715.

        argmax/argmin with dim=None reduces over the flattened tensor, which
        CoreML does not support; the resulting model intermittently crashes
        the process at runtime.  The partitioner must reject these so they
        fall back to the portable backend, while still delegating the
        ordinary dim=int form.
        """

        class FlatModel(torch.nn.Module):
            def forward(self, x):
                return torch.argmax(x, dim=None, keepdim=False) + torch.argmin(
                    x, dim=None
                )

        ep = torch.export.export(
            FlatModel().eval(), (torch.randn(10, 10),), strict=True
        )
        edge = executorch.exir.to_edge_transform_and_lower(
            ep, partitioner=[CoreMLPartitioner()]
        )
        op_names = [
            n.target.__name__
            for n in edge.exported_program().graph.nodes
            if n.op == "call_function"
        ]
        self.assertIn("aten.argmax.default", op_names)
        self.assertIn("aten.argmin.default", op_names)

        class DimModel(torch.nn.Module):
            def forward(self, x):
                return torch.argmax(x, dim=1)

        ep = torch.export.export(DimModel().eval(), (torch.randn(10, 10),), strict=True)
        edge = executorch.exir.to_edge_transform_and_lower(
            ep, partitioner=[CoreMLPartitioner()]
        )
        op_names = [
            n.target.__name__
            for n in edge.exported_program().graph.nodes
            if n.op == "call_function"
        ]
        self.assertIn("executorch_call_delegate", op_names)
        self.assertNotIn("aten.argmax.default", op_names)

    def test_bfloat16_ops_are_skipped(self):
        """
        coremltools has no bfloat16 type and raises on ops that use it.  The
        partitioner must leave those ops to the portable backend and still
        delegate the rest, also when the same op has a float32 call that Core
        ML takes (as under bfloat16 autocast).
        """

        class Model(torch.nn.Module):
            def forward(self, q):
                a = torch.nn.functional.scaled_dot_product_attention(q, q, q)
                b = q.to(torch.bfloat16)
                c = torch.nn.functional.scaled_dot_product_attention(b, b, b)
                return a + c.float()

        model = Model().eval()
        x = torch.randn(1, 2, 4, 8)
        ep = torch.export.export(model, (x,), strict=True)
        edge = executorch.exir.to_edge_transform_and_lower(
            ep, partitioner=[CoreMLPartitioner()]
        )
        op_names = [
            n.target.__name__
            for n in edge.exported_program().graph.nodes
            if n.op == "call_function"
        ]
        self.assertIn("executorch_call_delegate", op_names)
        self.assertNotIn("aten.scaled_dot_product_attention.default", op_names)
        if _TEST_RUNTIME:
            program = Runtime.get().load_program(edge.to_executorch().buffer)
            out = program.load_method("forward").execute((x,))[0]
            self.assertTrue(torch.allclose(out, model(x), atol=5e-2, rtol=5e-2))

    def test_min_max_of_bool_is_skipped(self):
        """
        coremltools raises on every form of a min or max of bool tensors, such
        as the all-true check torch.distributions exports.  The partitioner
        must leave those ops to the portable backend and still delegate the
        rest, including a min or max with a float input.
        """

        class Model(torch.nn.Module):
            def forward(self, x):
                ok = (x.tril() == x).view(1, -1)
                no = ok.flip(-1)
                reduced = [ok.min(), ok.max(), ok.min(-1)[0], ok.max(-1)[0]]
                reduced += [ok.amin(-1), ok.amax(-1)]
                paired = [torch.min(ok, no), torch.max(ok, no)]
                paired += [torch.minimum(ok, no), torch.maximum(ok, no)]
                flags = sum(r.to(x.dtype).sum() for r in reduced)
                both = sum(p.to(x.dtype) for p in paired).view(x.shape)
                floats = x.amax(-1, keepdim=True) + torch.maximum(x > 0.5, x)
                return x * 2 + flags + both + floats

        model = Model().eval()
        # Core ML computes in float16, so use values it holds exactly, or
        # x > 0.5 can differ from eager for x just above 0.5.
        x = torch.rand(14, 14).half().float()
        ep = torch.export.export(model, (x,), strict=True)
        edge = executorch.exir.to_edge_transform_and_lower(
            ep, partitioner=[CoreMLPartitioner()]
        )
        op_names = [
            n.target.__name__
            for n in edge.exported_program().graph.nodes
            if n.op == "call_function"
        ]
        self.assertIn("executorch_call_delegate", op_names)
        for name in ("min.default", "max.default", "min.dim", "max.dim"):
            self.assertIn(f"aten.{name}", op_names)
        self.assertIn("aten.amin.default", op_names)
        # The float amax and maximum stay in Core ML, and lowering turns the
        # min and max of two tensors into minimum and maximum.
        self.assertEqual(op_names.count("aten.amax.default"), 1)
        self.assertEqual(op_names.count("aten.minimum.default"), 2)
        self.assertEqual(op_names.count("aten.maximum.default"), 2)
        if _TEST_RUNTIME:
            program = Runtime.get().load_program(edge.to_executorch().buffer)
            out = program.load_method("forward").execute((x,))[0]
            self.assertTrue(torch.allclose(out, model(x), atol=1e-2))

    def test_index_put_forms_coremltools_gets_wrong(self):
        """
        coremltools fails some forms of ``index_put``, such as
        ``y[:, idx] = v`` with two entries in ``idx``, or writes them to the
        wrong place, so the partitioner refuses those, and they run on the
        portable backend.  With lower_full_graph=True, lowering fails and says
        why.
        """

        class Scatter(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cols", torch.tensor([0, 2]))
                self.register_buffer("col", torch.tensor([1]))
                self.register_buffer("last", torch.tensor(3))
                self.register_buffer("rows", torch.tensor([[1], [2]]))
                self.register_buffer("flag", torch.tensor([2]))
                self.register_buffer("rows_mask", torch.tensor([True, True, True]))
                self.register_buffer("trio", torch.tensor([0, 2, 3]))
                self.register_buffer("grid", torch.tensor([[1]]))
                self.register_buffer("yes", torch.tensor([True]))
                self.register_buffer("pair", torch.tensor([0, 1]))

            def forward(self, x, a, b, c):
                y = x * 2
                y[:, self.cols] = a
                y[:, self.grid] = a[:, :, None]
                y[self.rows, self.cols] = b
                # Same-shape indices without a slice stay in Core ML, also when
                # a refused write reads the same index.
                y[self.cols, self.cols] = a[:2, 0]
                y[self.rows_mask, self.trio] = c
                torch.ops.aten.index_put_(y, [None, self.col], a, True)
                torch.ops.aten.index_put_(y, [None, self.last], c)
                mask = y > 0.5
                mask[:, self.flag] = a > 0.5
                # Core ML would return this bool write as int32, and ~ would
                # then read it wrong.
                flags = x > 0.5
                flags[self.pair, self.pair] = a[:2, 0] > 0.5
                z = a * 2
                z[:, self.yes] = c[:, None]
                return y + mask + z + ~flags

        class Scatter3d(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("last", torch.tensor(2))
                self.register_buffer("mid", torch.tensor(1))
                self.register_buffer("row", torch.tensor([1]))
                self.register_buffer("col", torch.tensor([2]))

            def forward(self, x, v):
                y = x.clone()
                torch.ops.aten.index_put_(y, [None, None, self.last], v)
                # Values that vary only after the index stay in Core ML.
                torch.ops.aten.index_put_(y, [None, self.mid], v[0] * torch.arange(3.0))
                torch.ops.aten.index_put_(y, [None, self.row, self.col], v)
                return y * 2

        ios18 = CoreMLBackend.generate_compile_specs(
            minimum_deployment_target=ct.target.iOS18
        )
        # Values that float16 holds exactly, so that the comparisons with 0.5
        # in Core ML agree with eager.
        scatter_inputs = tuple(
            torch.rand(shape).half().float() for shape in ((3, 4), (3, 1), (2, 2), (3,))
        )
        scatter3d_inputs = (torch.rand(2, 2, 3), torch.rand(2, 1))
        cases = [
            (Scatter(), scatter_inputs, 9, None),
            # From iOS 18 the bool write next to a slice stays in Core ML.
            (Scatter(), scatter_inputs, 8, ios18),
            (Scatter3d(), scatter3d_inputs, 2, None),
        ]
        for model, inputs, cpu_calls, compile_specs in cases:
            with self.subTest(
                model=type(model).__name__, ios18=compile_specs is not None
            ):
                ep = torch.export.export(model.eval(), inputs, strict=True)
                edge = executorch.exir.to_edge_transform_and_lower(
                    ep, partitioner=[CoreMLPartitioner(compile_specs=compile_specs)]
                )
                op_names = [
                    n.target.__name__
                    for n in edge.exported_program().graph.nodes
                    if n.op == "call_function"
                ]
                self.assertIn("executorch_call_delegate", op_names)
                self.assertEqual(op_names.count("aten.index_put.default"), cpu_calls)
                # An iOS 18 program needs macOS 15 to run.
                if _TEST_RUNTIME and (compile_specs is None or IS_VALID_TEST_RUNTIME):
                    program = Runtime.get().load_program(edge.to_executorch().buffer)
                    out = program.load_method("forward").execute(inputs)[0]
                    self.assertTrue(torch.allclose(out, model(*inputs), atol=1e-2))

        class BoolWrite(torch.nn.Module):
            def __init__(self, index, sliced=True):
                super().__init__()
                self.register_buffer("index", index)
                self.sliced = sliced

            def forward(self, x, a):
                indices = [None, self.index] if self.sliced else [self.index]
                return torch.ops.aten.index_put(x > 0.5, indices, a > 0.5)

        for model, inputs, reason in (
            (Scatter3d(), scatter3d_inputs, "values that vary before the last index"),
            (
                BoolWrite(torch.tensor([2])),
                scatter_inputs[:2],
                "below minimum_deployment_target=ct.target.iOS18",
            ),
            # A bool write that a later target would not fix gives its own reason.
            (BoolWrite(torch.tensor([0, 2])), scatter_inputs[:2], "other than an int"),
            (
                BoolWrite(torch.tensor([0, 2]), sliced=False),
                (scatter_inputs[0], scatter_inputs[1][:2]),
                "on a bool tensor without a slice",
            ),
        ):
            with self.assertRaisesRegex(NotImplementedError, reason) as error:
                executorch.exir.to_edge_transform_and_lower(
                    torch.export.export(model, inputs, strict=True),
                    partitioner=[CoreMLPartitioner(lower_full_graph=True)],
                )
            self.assertNotIn("does not support the op", str(error.exception))
        # With a batch size known only at run time, the write that stays in
        # Core ML above goes to the CPU too.
        batch = torch.export.Dim("batch", min=2, max=8)
        ep = torch.export.export(
            Scatter3d(),
            scatter3d_inputs,
            dynamic_shapes=({0: batch}, {0: batch}),
            strict=True,
        )
        edge = executorch.exir.to_edge_transform_and_lower(
            ep, partitioner=[CoreMLPartitioner()]
        )
        op_names = [
            n.target.__name__
            for n in edge.exported_program().graph.nodes
            if n.op == "call_function"
        ]
        self.assertEqual(op_names.count("aten.index_put.default"), 3)

    def test_reduction_of_0d_tensor_is_skipped(self):
        """
        Core ML cannot reduce a 0-d tensor.  The partitioner must leave such
        reductions to the CPU and still delegate the rest.
        """

        class Model(torch.nn.Module):
            def forward(self, x):
                total = x.sum()
                flag = x[0, 0] > 0.5
                zero_d = [total.sum(), total.mean(), total.mean(0), total.amax()]
                zero_d += [total.amin(), total.max(), total.max(0)[0], total.min()]
                zero_d += [total.min(0)[0], total.argmax(0), total.argmin(0)]
                zero_d += [x[0, 0].sum(), flag.any(), flag.any(0), flag.any((0,))]
                zero_d += [flag.all(), flag.all(0), flag.all((0,))]
                zero_d += [total.var(correction=0), total.var(unbiased=False)]
                zero_d += [total.var(0, False)]
                return x * 2 + sum(r.to(x.dtype) for r in zero_d)

        model = Model().eval()
        # Core ML computes in float16, so use values it holds exactly, or
        # x[0, 0] > 0.5 can differ from eager for x just above 0.5.
        x = torch.rand(2, 3).half().float()
        ep = torch.export.export(model, (x,), strict=True)
        edge = executorch.exir.to_edge_transform_and_lower(
            ep, partitioner=[CoreMLPartitioner()]
        )
        op_names = [
            n.target.__name__
            for n in edge.exported_program().graph.nodes
            if n.op == "call_function"
        ]
        self.assertIn("executorch_call_delegate", op_names)
        # One per reduction, in the order of the model. torch makes all() a
        # logical_not of any() of a logical_not, which Core ML takes, and every
        # var a var.correction.
        reductions = [
            "sum.dim_IntList",
            "mean.default",
            "mean.dim",
            "amax.default",
            "amin.default",
            "max.default",
            "max.dim",
            "min.default",
            "min.dim",
            "argmax.default",
            "argmin.default",
            "sum.dim_IntList",
            "any.default",
            "any.dim",
            "any.dims",
            "any.dims",
            "any.dim",
            "any.dims",
        ] + ["var.correction"] * 3
        self.assertEqual(
            sorted(
                name
                for name in op_names
                if name not in ("executorch_call_delegate", "getitem")
            ),
            sorted(f"aten.{name}" for name in reductions),
        )
        if _TEST_RUNTIME:
            program = Runtime.get().load_program(edge.to_executorch().buffer)
            out = program.load_method("forward").execute((x,))[0]
            self.assertTrue(torch.allclose(out, model(x), atol=1e-2, rtol=1e-2))

        # A sum of a 0-d tensor over a dim stays in Core ML, where lowering
        # fails as before, since ExecuTorch's optimized sum kernel aborts on it.
        class SumOverDim(torch.nn.Module):
            def forward(self, x):
                return x * x.sum().sum(0)

        ep = torch.export.export(SumOverDim(), (x,), strict=True)
        result = CoreMLPartitioner().partition(
            executorch.exir.to_edge(ep).exported_program()
        )
        sums = [
            n
            for n in result.tagged_exported_program.graph.nodes
            if getattr(n.target, "__name__", "") == "aten.sum.dim_IntList"
        ]
        self.assertEqual(len(sums), 2)
        self.assertTrue(all("delegation_tag" in n.meta for n in sums))

    def test_tied_embedding_is_quantized_with_its_linear(self):
        """A weight that is both an embedding table and an output projection lowers.

        The quantizer config opts gathers out so that embedding tables are not compressed
        by it. When the table is tied to a linear's weight, that opt-out leaves one
        constant with two configurations, which coremltools rejects — so the opt-out has
        to stop at the tie rather than fail the whole lowering.
        """

        class Tied(torch.nn.Module):
            def __init__(self, vocab=512, dim=64, tie=True):
                super().__init__()
                self.embedding = torch.nn.Embedding(vocab, dim)
                self.output = torch.nn.Linear(dim, vocab, bias=False)
                if tie:
                    self.output.weight = self.embedding.weight

            def forward(self, ids):
                return self.output(self.embedding(ids))

        ids = torch.zeros(1, 4, dtype=torch.long)
        for tie in (True, False):
            compile_specs = CoreMLBackend.generate_compile_specs(
                minimum_deployment_target=ct.target.iOS18,
                compute_precision=ct.precision(ct.precision.FLOAT16.value),
                op_linear_quantizer_config={
                    "mode": "linear_symmetric",
                    "dtype": "int4",
                    "granularity": "per_channel",
                },
            )
            exported = torch.export.export(Tied(tie=tie).eval(), (ids,), strict=True)
            delegated = executorch.exir.to_edge_transform_and_lower(
                exported,
                partitioner=[CoreMLPartitioner(compile_specs=compile_specs)],
            )
            # Reaching a program at all is the assertion: the failure this covers is
            # raised during lowering, not carried in the result.
            self.assertIsNotNone(delegated.to_executorch())

    def test_deprecation_warning_for_to_backend_workflow(self):
        """
        Test that the deprecated to_edge + to_backend workflow shows a deprecation warning.
        """

        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()
        model.eval()
        x = torch.randn(1, 10)

        exported_model = torch.export.export(model, (x,), strict=True)

        # Capture log output to check for deprecation warning
        log_capture_string = io.StringIO()
        ch = logging.StreamHandler(log_capture_string)
        ch.setLevel(logging.WARNING)

        partitioner_logger = logging.getLogger(
            "executorch.backends.apple.coreml.partition.coreml_partitioner"
        )
        partitioner_logger.addHandler(ch)
        partitioner_logger.setLevel(logging.WARNING)

        edge = executorch.exir.to_edge(
            exported_model, compile_config=self.edge_compile_config
        )
        partitioner = CoreMLPartitioner()

        edge.to_backend(partitioner)

        log_contents = log_capture_string.getvalue()
        self.assertIn("DEPRECATION WARNING", log_contents)
        self.assertIn("to_edge() + to_backend()", log_contents)
        self.assertIn("to_edge_transform_and_lower()", log_contents)

        # Clean up handler
        partitioner_logger.removeHandler(ch)

    def test_no_warning_for_to_edge_transform_and_lower_workflow(self):
        """
        Test that the recommended to_edge_transform_and_lower workflow does NOT show a deprecation warning.
        """

        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()
        model.eval()
        x = torch.randn(1, 10)

        exported_model = torch.export.export(model, (x,), strict=True)

        # Capture log output to check for deprecation warning
        log_capture_string = io.StringIO()
        ch = logging.StreamHandler(log_capture_string)
        ch.setLevel(logging.WARNING)

        partitioner_logger = logging.getLogger(
            "executorch.backends.apple.coreml.partition.coreml_partitioner"
        )
        partitioner_logger.addHandler(ch)
        partitioner_logger.setLevel(logging.WARNING)

        partitioner = CoreMLPartitioner()

        executorch.exir.to_edge_transform_and_lower(
            exported_model, partitioner=[partitioner]
        )

        log_contents = log_capture_string.getvalue()
        self.assertNotIn("DEPRECATION WARNING", log_contents)

        # Clean up handler
        partitioner_logger.removeHandler(ch)


if __name__ == "__main__":
    test_runner = TestCoreMLPartitioner()
    test_runner.test_add_sub_skip_mm()
    test_runner.test_vit_skip_conv()
    test_runner.test_ops_to_not_decompose()
    test_runner.test_buffer()
    test_runner.test_lower_full_graph()
    # test_runner.test_symint_arg()
    test_runner.test_take_over_constant_data_false()
    test_runner.test_aten_rand_default_falls_back_to_portable()
    test_runner.test_aten_randn_is_still_delegated()
    test_runner.test_tied_embedding_is_quantized_with_its_linear()
    test_runner.test_deprecation_warning_for_to_backend_workflow()
    test_runner.test_no_warning_for_to_edge_transform_and_lower_workflow()
