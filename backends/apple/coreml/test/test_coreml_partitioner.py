# Copyright © 2023 Apple Inc. All rights reserved.
#
# Please refer to the license found in the LICENSE file in the root directory of the source tree.

import copy
import io
import logging
import sys
import unittest
from unittest import mock

import coremltools as ct

import executorch.exir

import torch
import torchvision

from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.apple.coreml.partition import (
    coreml_partitioner,
    CoreMLPartitioner,
)
from executorch.backends.apple.coreml.test.test_coreml_utils import (
    IS_VALID_TEST_RUNTIME,
)
from executorch.exir.backend.utils import format_delegated_graph
from executorch.exir.passes.init_mutable_pass import InitializedMutableBufferPass


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

    def _lower_with_one_search(self, ep, partitioner):
        """Lowers ep, checking that partitions are searched for once, since
        each search is slow on a large model."""
        with mock.patch.object(
            coreml_partitioner,
            "CapabilityBasedPartitioner",
            wraps=coreml_partitioner.CapabilityBasedPartitioner,
        ) as search:
            edge = executorch.exir.to_edge_transform_and_lower(
                ep, partitioner=[partitioner]
            )
        self.assertEqual(search.call_count, 1)
        return edge

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

    def test_op_made_by_decomposing_a_rejected_op_is_not_kept_whole(self):
        """
        A float32 matmul keeps aten.matmul whole.  A bfloat16 linear is
        rejected and decomposes into another matmul, which Core ML rejects too.
        That matmul must be decomposed as well, since the portable kernels
        have no matmul, or the program fails to load.
        """

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(8, 8, bias=False).to(torch.bfloat16)
                self.register_buffer("inv_freq", torch.rand(1, 4, 1))

            def forward(self, x, pos):
                freqs = self.inv_freq @ pos[:, None, :].float()
                return self.proj(x).float(), freqs.cos()

        model = Model().eval()
        inputs = (torch.rand(1, 6, 8, dtype=torch.bfloat16), torch.arange(6)[None])
        ep = torch.export.export(model, inputs, strict=True)
        edge = executorch.exir.to_edge_transform_and_lower(
            ep, partitioner=[CoreMLPartitioner()]
        )
        op_names = [
            n.target.__name__
            for n in edge.exported_program().graph.nodes
            if n.op == "call_function"
        ]
        self.assertIn("executorch_call_delegate", op_names)
        self.assertNotIn("aten.matmul.default", op_names)
        if _TEST_RUNTIME:
            program = Runtime.get().load_program(edge.to_executorch().buffer)
            outputs = program.load_method("forward").execute(inputs)
            for out, ref in zip(outputs, model(*inputs)):
                self.assertTrue(torch.allclose(out, ref, atol=1e-2))

    def test_int8_output(self):
        """
        The Core ML delegate has no int8 data type to hand an int8 tensor back
        to ExecuTorch.  So an op that makes one that is read outside Core ML,
        here the model output, runs on the CPU.  So does an op that hands one
        out through a getitem, which always stays with the op.
        """

        class Model(torch.nn.Module):
            def forward(self, x):
                return (x * 2).to(torch.int8)

        class Split(torch.nn.Module):
            def forward(self, x):
                return (x * 2).to(torch.int8).split(2, dim=1)[0]

        # The cast to int8 reads a uint8 tensor that Core ML makes, which is no
        # input from outside, so the error must not name it.
        class FromUint8(torch.nn.Module):
            def forward(self, x):
                return (x.abs() * 2).to(torch.uint8).to(torch.int8)

        # Small integers, which Core ML's float16 math keeps exact.
        x = torch.randint(-50, 50, (1, 4, 4, 4)).float()
        for model in (Model().eval(), Split().eval(), FromUint8().eval()):
            with self.subTest(model=type(model).__name__):
                ep = torch.export.export(model, (x,), strict=True)
                edge = self._lower_with_one_search(ep, CoreMLPartitioner())
                op_names = [
                    n.target.__name__
                    for n in edge.exported_program().graph.nodes
                    if n.op == "call_function"
                ]
                self.assertIn("executorch_call_delegate", op_names)
                self.assertIn("dim_order_ops._to_dim_order_copy.default", op_names)
                buffer = edge.to_executorch().buffer
                if _TEST_RUNTIME:
                    program = Runtime.get().load_program(buffer)
                    out = program.load_method("forward").execute((x,))[0]
                    self.assertTrue(torch.equal(out, model(x)))
                with self.assertRaisesRegex(
                    NotImplementedError, "makes an int8"
                ) as error:
                    executorch.exir.to_edge_transform_and_lower(
                        ep, partitioner=[CoreMLPartitioner(lower_full_graph=True)]
                    )
                self.assertNotIn("cannot take as an input", str(error.exception))

        # Core ML makes this int8 tensor in one partition and reads it in
        # another, past the asinh, which runs on the CPU, so the cast to int8
        # runs on the CPU too.
        class AcrossPartitions(torch.nn.Module):
            def forward(self, x):
                a = (x * 2).to(torch.int8)
                return a.to(torch.int32).float() * torch.asinh(a.float())

        model = AcrossPartitions().eval()
        x = torch.randint(-20, 20, (2, 4)).float()
        ep = torch.export.export(model, (x,), strict=True)
        edge = executorch.exir.to_edge_transform_and_lower(
            ep, partitioner=[CoreMLPartitioner()]
        )
        delegate_outputs = [
            n.meta["val"].dtype
            for n in edge.exported_program().graph.nodes
            if n.op == "call_function" and n.target.__name__ == "getitem"
        ]
        self.assertTrue(delegate_outputs)
        self.assertNotIn(torch.int8, delegate_outputs)
        buffer = edge.to_executorch().buffer
        if _TEST_RUNTIME:
            program = Runtime.get().load_program(buffer)
            out = program.load_method("forward").execute((x,))[0]
            self.assertTrue(torch.allclose(out, model(x), atol=1e-2, rtol=1e-3))

        # Here the cast to int8 reads a uint8 tensor from another partition,
        # and only ops on the CPU read its int8, so the log names that tensor.
        class Uint8AcrossPartitions(torch.nn.Module):
            def forward(self, x):
                t = (x.abs() * 2).to(torch.uint8)
                return t.to(torch.int8).to(torch.int32).float() * torch.asinh(t.float())

        # It names that tensor too when the op that reads it, here maximum,
        # reads an int8 tensor as well, made on the CPU only because maximum
        # runs there.
        class Int8MadeForTheReader(torch.nn.Module):
            def forward(self, x):
                t = (x.abs() * 2).to(torch.uint8)
                q = (x * 2).to(torch.int8)
                return torch.maximum(q, t).float() * torch.asinh(t.float())

        for model in (Uint8AcrossPartitions().eval(), Int8MadeForTheReader().eval()):
            with self.subTest(model=type(model).__name__):
                ep = torch.export.export(model, (x,), strict=True)
                with self.assertLogs(coreml_partitioner.logger, logging.INFO) as logs:
                    executorch.exir.to_edge_transform_and_lower(
                        ep, partitioner=[CoreMLPartitioner()]
                    )
                self.assertIn("cannot take as an input", "\n".join(logs.output))

    def test_int8_state(self):
        """
        Writing int8 state that Core ML keeps is no int8 output: Core ML keeps
        the state as float16 state, so the whole model stays in Core ML.
        When the partitioner does not take state over, Core ML cannot read the
        int8 state, so the cast and the add that read it run on the CPU, with
        the cast that makes the int8 tensor the add reads. When a CPU op reads
        the state, or no op does, it stays in the program, and writing it is an
        int8 output, so the cast that makes its new value runs on the CPU.
        """

        class Kept(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "state", torch.tensor([[1, -2, 3, -4]], dtype=torch.int8)
                )

            def forward(self, x):
                y = self.state.float() * x
                self.state.add_((x > 0).to(torch.int8))
                return y

        class KeptFromZero(Kept):
            # The Core ML delegate zeroes the state it holds when it loads.
            def __init__(self):
                super().__init__()
                self.state.zero_()

        class NeverRead(Kept):
            def forward(self, x):
                self.state.copy_((x * 4).to(torch.int8))
                return x + 1

        class ReadOnCpu(Kept):
            def forward(self, x):
                y = torch.asinh(self.state) * x
                self.state.copy_((x > 0).to(torch.int8))
                return y

        x = torch.tensor([[0.3, -0.2, 0.4, -0.1]])
        compile_specs = CoreMLBackend.generate_compile_specs(
            minimum_deployment_target=ct.target.iOS18
        )
        cast = "dim_order_ops._to_dim_order_copy.default"
        for model_class, take_over, cpu_ops in (
            (KeptFromZero, True, set()),
            (Kept, False, {cast, "aten.add.Tensor"}),
            (ReadOnCpu, True, {cast, "aten.asinh.default"}),
            (NeverRead, True, {cast}),
        ):
            with self.subTest(model=model_class.__name__, take_over=take_over):
                ep = torch.export.export(model_class().eval(), (x,), strict=True)
                partitioner = CoreMLPartitioner(
                    compile_specs=compile_specs, take_over_mutable_buffer=take_over
                )
                edge = executorch.exir.to_edge_transform_and_lower(
                    ep, partitioner=[partitioner]
                )
                op_names = {
                    n.target.__name__
                    for n in edge.exported_program().graph.nodes
                    if n.op == "call_function"
                }
                self.assertEqual(
                    op_names - {"executorch_call_delegate", "getitem"}, cpu_ops
                )
                # Mutable buffers start uninitialized without this pass.
                config = executorch.exir.ExecutorchBackendConfig(
                    passes=[InitializedMutableBufferPass(["state"])]
                )
                buffer = edge.to_executorch(config).buffer
                # An iOS 18 program needs macOS 15 to run.
                if IS_VALID_TEST_RUNTIME:
                    method = Runtime.get().load_program(buffer).load_method("forward")
                    eager = model_class().eval()
                    for _ in range(2):
                        out = method.execute((x,))[0]
                        self.assertTrue(torch.allclose(out, eager(x), atol=1e-2))

    def test_int8_input_is_read_outside_the_delegate(self):
        """
        Core ML cannot take a uint8 or int8 input, so the ops that read an int8
        image stay on the CPU up to its conversion to float, and the float ops
        after it are still delegated.  With lower_full_graph=True, lowering
        fails and names the input, but not a uint8 tensor that Core ML makes
        and an op on the CPU reads.
        """

        class Model(torch.nn.Module):
            def forward(self, image):
                return torch.relu(image.permute(2, 0, 1).float() / 127.0 - 0.25)

        model = Model().eval()
        x = torch.randint(-128, 128, (8, 8, 3), dtype=torch.int8)
        ep = torch.export.export(model, (x,), strict=True)
        edge = executorch.exir.to_edge_transform_and_lower(
            ep, partitioner=[CoreMLPartitioner()]
        )
        op_names = [
            n.target.__name__
            for n in edge.exported_program().graph.nodes
            if n.op == "call_function"
        ]
        self.assertIn("aten.permute_copy.default", op_names)
        self.assertIn("dim_order_ops._to_dim_order_copy.default", op_names)
        self.assertIn("executorch_call_delegate", op_names)
        for name in ("aten.div.Tensor", "aten.sub.Tensor", "aten.relu.default"):
            self.assertNotIn(name, op_names)
        buffer = edge.to_executorch().buffer
        if _TEST_RUNTIME:
            program = Runtime.get().load_program(buffer)
            out = program.load_method("forward").execute((x,))[0]
            self.assertTrue(torch.allclose(out, model(x), atol=1e-2))
        with self.assertRaisesRegex(NotImplementedError, "image \\(torch\\.int8, "):
            executorch.exir.to_edge_transform_and_lower(
                ep, partitioner=[CoreMLPartitioner(lower_full_graph=True)]
            )

        class Mixed(torch.nn.Module):
            def forward(self, image, x):
                return (image + (x * 255).to(torch.uint8)).float() * 2

        class Chain(torch.nn.Module):
            def forward(self, image, x):
                return ((image + 1) + (x * 255).to(torch.uint8)).float() * 2

        inputs = (torch.randint(0, 100, (2, 4), dtype=torch.uint8), torch.rand(2, 4))
        for model in (Mixed(), Chain()):
            with self.subTest(model=type(model).__name__):
                ep = torch.export.export(model.eval(), inputs, strict=True)
                with self.assertRaises(NotImplementedError) as error:
                    executorch.exir.to_edge_transform_and_lower(
                        ep, partitioner=[CoreMLPartitioner(lower_full_graph=True)]
                    )
                self.assertIn(
                    "partition: image (torch.uint8, (2, 4)).", str(error.exception)
                )

    def test_uint8_input_read_by_an_op_without_a_cpu_kernel(self):
        """
        A reflect pad that reads a uint8 input runs on the CPU.  Core ML keeps
        pad whole, but pad is not a core ATen op and has no portable kernel, so
        it is decomposed there.  Below iOS 18 the index math it decomposes into
        runs on the CPU too, since it would leave Core ML a partition with no
        inputs.  So does a pad that would be such a partition itself, also
        once lowering decomposes the ops before it.
        """

        class Model(torch.nn.Module):
            def forward(self, image):
                image = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")
                return image.float() / 255.0

        class PadOfBuffer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("w", torch.rand(1, 3, 8))

            def forward(self, image):
                p = torch.nn.functional.pad(self.w, (1, 1), mode="reflect")
                return torch.asinh(p) + image.float()

        class PadOfFilledBuffer(PadOfBuffer):
            def __init__(self):
                super().__init__()
                # One entry: Core ML refuses an index_put with two entries next to
                # a slice.
                self.register_buffer("cols", torch.tensor([2]))

            def forward(self, image):
                w = self.w.index_fill(2, self.cols, 0.0)
                p = torch.nn.functional.pad(w, (1, 1), mode="reflect")
                return torch.asinh(p) + image.float()

        image = torch.randint(0, 256, (1, 3, 10), dtype=torch.uint8)
        for model, x in (
            (Model(), torch.randint(0, 256, (1, 3, 8, 8), dtype=torch.uint8)),
            # The pad reads only a buffer Core ML takes over, and the asinh
            # after it runs on the CPU, so it would be a partition with no
            # inputs.
            (PadOfBuffer(), image),
            # Core ML rejects index_fill, so the pad reads only buffers once
            # lowering decomposes it, into a scalar_tensor and an index_put.
            (PadOfFilledBuffer(), image),
        ):
            with self.subTest(model=type(model).__name__):
                ep = torch.export.export(model.eval(), (x,), strict=True)
                edge = executorch.exir.to_edge_transform_and_lower(
                    ep, partitioner=[CoreMLPartitioner()]
                )
                op_names = [
                    n.target.__name__
                    for n in edge.exported_program().graph.nodes
                    if n.op == "call_function"
                ]
                self.assertIn("aten.arange.start_step", op_names)
                self.assertNotIn("aten.pad.default", op_names)
                self.assertIn("executorch_call_delegate", op_names)
                buffer = edge.to_executorch().buffer
                if _TEST_RUNTIME:
                    program = Runtime.get().load_program(buffer)
                    out = program.load_method("forward").execute((x,))[0]
                    self.assertTrue(torch.allclose(out, model(x), atol=1e-2, rtol=1e-3))

    def test_pad_of_buffer_after_a_decomposition_that_reads_a_size(self):
        """
        channel_shuffle and repeat_interleave of a 5-D tensor with a dynamic
        size decompose into a rank 6 view or expand, which runs on the CPU and
        reads a size Core ML would compute, so that size moves to the CPU too.
        Then, below iOS 18, the pad of a buffer is a partition without inputs,
        since the asinh after it runs on the CPU, so it moves out.  Only the
        decomposed program shows this, so the program is decomposed to find
        it, and pad is decomposed too.  A uint8 value made and read inside Core
        ML changes nothing.
        """

        class Model(torch.nn.Module):
            def __init__(self, op):
                super().__init__()
                self.op = op
                self.register_buffer("w", torch.rand(1, 3, 8))

            def forward(self, x):
                p = torch.nn.functional.pad(self.w, (1, 1), mode="reflect")
                x = x + torch.asinh(p).mean()
                if self.op == "uint8":
                    x = x + (x * 0).to(torch.uint8).float()
                if self.op == "repeat_interleave":
                    return x.repeat_interleave(2, dim=1)
                return torch.nn.functional.channel_shuffle(x, 2)

        x = torch.rand(1, 4, 2, 4, 6)
        for op in ("channel_shuffle", "uint8", "repeat_interleave"):
            with self.subTest(op=op):
                model = Model(op).eval()
                ep = torch.export.export(
                    model,
                    (x,),
                    dynamic_shapes=({3: torch.export.Dim("height", min=2, max=32)},),
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
                self.assertNotIn("aten.pad.default", op_names)
                self.assertIn("executorch_call_delegate", op_names)
                buffer = edge.to_executorch().buffer
                if _TEST_RUNTIME:
                    program = Runtime.get().load_program(buffer)
                    out = program.load_method("forward").execute((x,))[0]
                    self.assertTrue(torch.allclose(out, model(x), atol=1e-2, rtol=1e-3))

    def test_pad_of_buffer_next_to_uint8_input_stays_whole(self):
        """
        A pad that reads only a buffer Core ML takes over would be a partition
        without inputs if the ops after it moved to the CPU.  In a model with a
        uint8 input some ops do move, so the search for partitions decides
        whether pad stays whole.  Here the partition of the pad also reads x,
        so pad stays whole.
        """

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("w", torch.rand(1, 3, 4))

            def forward(self, image, x):
                return image.float() + torch.nn.functional.pad(self.w, (1, 1)) * x

        model = Model().eval()
        inputs = (
            torch.randint(0, 256, (1, 3, 6), dtype=torch.uint8),
            torch.rand(1, 3, 6),
        )
        ep = torch.export.export(model, inputs, strict=True)
        keep, _ = CoreMLPartitioner().ops_to_not_decompose(ep)
        self.assertIn(torch.ops.aten.pad.default, keep)
        edge = executorch.exir.to_edge_transform_and_lower(
            ep, partitioner=[CoreMLPartitioner()]
        )
        buffer = edge.to_executorch().buffer
        if _TEST_RUNTIME:
            program = Runtime.get().load_program(buffer)
            out = program.load_method("forward").execute(inputs)[0]
            self.assertTrue(torch.allclose(out, model(*inputs), atol=1e-2, rtol=1e-3))

    def test_uint8_input_with_dynamic_shape(self):
        """
        With a dynamic height, the index math of a decomposed roll reads only
        a symbolic size, which Core ML cannot take as an input, so it runs on
        the CPU too, also from iOS 18, where a partition with no inputs at all
        may stay in Core ML.  So do a reshape to a size read from the uint8
        input, a scale computed from it, and the add after the reshape, since
        Core ML cannot take a tensor sized by an expression of the height, such
        as 3 * height, as an input.
        """

        class Model(torch.nn.Module):
            def forward(self, image):
                rolled = torch.roll(image, 1, 2).float() / 255.0
                scaled = rolled * (image.shape[2] / 8)
                return scaled.reshape(1, 3 * image.shape[2], 8) + 1

        model = Model().eval()
        x = torch.randint(0, 256, (1, 3, 8, 8), dtype=torch.uint8)
        ep = torch.export.export(
            model,
            (x,),
            dynamic_shapes=({2: torch.export.Dim("height", min=4, max=64)},),
            strict=True,
        )
        compile_specs = CoreMLBackend.generate_compile_specs(
            minimum_deployment_target=ct.target.iOS18
        )
        edge = self._lower_with_one_search(
            ep, CoreMLPartitioner(compile_specs=compile_specs)
        )
        op_names = [
            n.target.__name__
            for n in edge.exported_program().graph.nodes
            if n.op == "call_function"
        ]
        for name in (
            "aten.arange.start_step",
            "aten.view_copy.default",
            "aten.add.Tensor",
        ):
            self.assertIn(name, op_names)
        self.assertIn("executorch_call_delegate", op_names)
        buffer = edge.to_executorch().buffer
        # An iOS 18 program needs macOS 15 to run.
        if IS_VALID_TEST_RUNTIME:
            y = torch.randint(0, 256, (1, 3, 12, 8), dtype=torch.uint8)
            program = Runtime.get().load_program(buffer)
            out = program.load_method("forward").execute((y,))[0]
            self.assertTrue(torch.allclose(out, model(y), atol=1e-2))

    def test_symbolic_value_read_outside_the_delegate(self):
        """
        The Core ML delegate returns only tensors, so the item of a flag, a
        symbolic value that sym_ite reads on the CPU, runs on the CPU too,
        below and from iOS 18, where a partition with no inputs at all may
        stay in Core ML.  So does a size the model returns, and with
        lower_full_graph=True lowering fails and names it.  The reshape to a
        size computed from it runs on the CPU too, and so does the op after it,
        since Core ML cannot take a tensor sized by an expression of a dynamic
        size, such as 3 * rows, as an input.  The other ops are still
        delegated.
        """

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("flag", torch.tensor(True))

            def forward(self, x):
                k = torch.sym_ite(self.flag.item(), 2, 3)
                return (torch.relu(x) + 1) * k

        model = Model().eval()
        x = torch.rand(2, 3)
        for target in (ct.target.iOS15, ct.target.iOS18):
            with self.subTest(target=target):
                ep = torch.export.export(model, (x,), strict=False)
                compile_specs = CoreMLBackend.generate_compile_specs(
                    minimum_deployment_target=target
                )
                edge = self._lower_with_one_search(
                    ep, CoreMLPartitioner(compile_specs=compile_specs)
                )
                op_names = [
                    n.target.__name__
                    for n in edge.exported_program().graph.nodes
                    if n.op == "call_function"
                ]
                self.assertIn("_local_scalar_dense.default", op_names)
                self.assertIn("executorch_call_delegate", op_names)
                for name in ("aten.relu.default", "aten.add.Tensor", "aten.mul.Tensor"):
                    self.assertNotIn(name, op_names)
                buffer = edge.to_executorch().buffer
                # An iOS 18 program needs macOS 15 to run.
                if _TEST_RUNTIME and (
                    target < ct.target.iOS18 or IS_VALID_TEST_RUNTIME
                ):
                    program = Runtime.get().load_program(buffer)
                    out = program.load_method("forward").execute((x,))[0]
                    self.assertTrue(torch.allclose(out, model(x), atol=1e-2))

        class ReturnsSize(torch.nn.Module):
            def forward(self, x):
                rows = x.shape[0]
                return (torch.relu(x) + 1).reshape(1, rows * 3) * 2, rows

        ep = torch.export.export(
            ReturnsSize().eval(),
            (x,),
            dynamic_shapes=({0: torch.export.Dim("rows", min=2, max=8)},),
            strict=True,
        )
        edge = self._lower_with_one_search(ep, CoreMLPartitioner())
        op_names = [
            n.target.__name__
            for n in edge.exported_program().graph.nodes
            if n.op == "call_function"
        ]
        for name in ("sym_size.int", "aten.view_copy.default", "aten.mul.Tensor"):
            self.assertIn(name, op_names)
        for name in ("aten.relu.default", "aten.add.Tensor"):
            self.assertNotIn(name, op_names)
        self.assertIn("executorch_call_delegate", op_names)
        buffer = edge.to_executorch().buffer
        if _TEST_RUNTIME:
            y = torch.rand(5, 3)
            program = Runtime.get().load_program(buffer)
            out, rows = program.load_method("forward").execute((y,))
            expected = (torch.relu(y) + 1).reshape(1, 15) * 2
            self.assertTrue(torch.allclose(out, expected, atol=1e-2))
            self.assertEqual(rows, 5)
        with self.assertRaisesRegex(NotImplementedError, "symbolic value.*sym_size"):
            executorch.exir.to_edge_transform_and_lower(
                ep, partitioner=[CoreMLPartitioner(lower_full_graph=True)]
            )

        # The size of a uint8 input runs on the CPU only because it reads that
        # input, so the error names the input and not the size.
        class SizeOfUint8(torch.nn.Module):
            def forward(self, image):
                return image.float().reshape(image.shape[0], -1) * 2

        ep = torch.export.export(
            SizeOfUint8().eval(),
            (torch.randint(0, 100, (4, 6), dtype=torch.uint8),),
            dynamic_shapes=({0: torch.export.Dim("rows", min=2, max=8)},),
            strict=True,
        )
        with self.assertRaises(NotImplementedError) as error:
            executorch.exir.to_edge_transform_and_lower(
                ep, partitioner=[CoreMLPartitioner(lower_full_graph=True)]
            )
        self.assertIn("image (torch.uint8, ", str(error.exception))
        self.assertNotIn("sym_size", str(error.exception))

        # Partitions on both sides of the asinh, which Core ML rejects, read the
        # same size, so it runs on the CPU, and so do the reshapes, which the
        # log explains with the symbolic value they read.
        class SharedSize(torch.nn.Module):
            def forward(self, x):
                rows = x.shape[0]
                return torch.asinh((x * 2).reshape(rows, 2, 2)).reshape(rows, 4) + 1

        ep = torch.export.export(
            SharedSize().eval(),
            (torch.rand(4, 4),),
            dynamic_shapes=({0: torch.export.Dim("rows", min=2, max=8)},),
            strict=True,
        )
        with self.assertLogs(coreml_partitioner.logger, logging.INFO) as logs:
            executorch.exir.to_edge_transform_and_lower(
                ep, partitioner=[CoreMLPartitioner()]
            )
        self.assertIn("makes or reads a symbolic value", "\n".join(logs.output))

    def test_input_sized_by_a_declared_expression_stays_in_the_delegate(self):
        """
        coremltools takes an input sized by an expression of a dynamic size
        when the exported program declares it, such as 2 * n here, so the ops
        that read it stay in Core ML.
        """

        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x.reshape(1, -1, 2).sum(-1) + y

        size = torch.export.Dim("n", min=2, max=32)
        ep = torch.export.export(
            Model().eval(),
            (torch.rand(1, 8), torch.rand(1, 4)),
            dynamic_shapes={"x": {1: 2 * size}, "y": {1: size}},
            strict=True,
        )
        edge = executorch.exir.to_edge_transform_and_lower(
            ep, partitioner=[CoreMLPartitioner()]
        )
        op_names = {
            n.target.__name__
            for n in edge.exported_program().graph.nodes
            if n.op == "call_function"
        }
        self.assertEqual(op_names, {"executorch_call_delegate", "getitem"})
        buffer = edge.to_executorch().buffer
        if _TEST_RUNTIME:
            x, y = torch.rand(1, 12), torch.rand(1, 6)
            program = Runtime.get().load_program(buffer)
            out = program.load_method("forward").execute((x, y))[0]
            self.assertTrue(torch.allclose(out, Model()(x, y), atol=1e-2))

    def test_program_is_decomposed_only_when_it_may_matter(self):
        """
        Finding the ops partition() runs on the CPU decomposes the whole
        program, as lowering will, only when that may change the answer, since
        that takes minutes on a large model.  Here index_fill, which Core ML
        rejects, decomposes into ops the program does not keep.  It reads only
        buffers, so its decomposition could make a partition without inputs,
        but such a partition moves out only once some other op does, and no op
        does here.  The item of a buffer runs on the CPU, since the model
        returns its value, but it makes no tensor, so decomposing it leaves
        every partition as it is.
        """

        class Fill(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("table", torch.rand(2, 4))
                self.register_buffer("cols", torch.tensor([0, 2]))

            def forward(self, x):
                return self.table.index_fill(1, self.cols, 0.0) + x

        class ItemOfBuffer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("n", torch.tensor(3))

            def forward(self, x):
                return x + 1, self.n.item()

        decompose = torch.export.ExportedProgram.run_decompositions
        for model in (Fill(), ItemOfBuffer()):
            with self.subTest(model=type(model).__name__):
                ep = torch.export.export(model.eval(), (torch.rand(2, 4),), strict=True)
                with mock.patch.object(
                    torch.export.ExportedProgram,
                    "run_decompositions",
                    autospec=True,
                    side_effect=decompose,
                ) as run_decompositions:
                    keep, _ = CoreMLPartitioner().ops_to_not_decompose(ep)
                self.assertIn(torch.ops.aten.add.Tensor, keep)
                self.assertEqual(run_decompositions.call_count, 0)

    def test_skip_names_in_a_program_as_exported(self):
        """
        ops_to_not_decompose can take a program as exported, which can still
        have a view op that has no copy, such as reshape.  Lowering turns it
        into views first, so a skip name never names it.
        """

        class Model(torch.nn.Module):
            def forward(self, x):
                return x.reshape(2, -1) * 2

        ep = torch.export.export(Model().eval(), (torch.rand(2, 4),), strict=True)
        keep, _ = CoreMLPartitioner().ops_to_not_decompose(ep)
        partitioner = CoreMLPartitioner(
            skip_ops_for_coreml_delegation=["aten.mul.tensor"]
        )
        self.assertEqual(partitioner.ops_to_not_decompose(ep)[0], keep)

    def test_uint8_input_read_by_a_kept_op_once_decomposed(self):
        """
        isposinf of an int tensor decomposes into zeros_like, and the portable
        kernels have no zeros_like.  The zeros_like that Core ML runs keeps
        zeros_like whole, so the one that isposinf of a uint8 input decomposes
        into would run whole on the CPU.  Tracing the decomposition of
        isposinf by itself shows that it reads the uint8 input, so the program
        is decomposed to find that call, and zeros_like is decomposed too.
        """

        class Model(torch.nn.Module):
            def forward(self, image, x):
                return torch.zeros_like(x) + x, torch.isposinf(image)

        model = Model().eval()
        inputs = (
            torch.randint(0, 256, (1, 3, 4), dtype=torch.uint8),
            torch.rand(1, 3, 4),
        )
        ep = torch.export.export(model, inputs, strict=True)
        edge = executorch.exir.to_edge_transform_and_lower(
            ep, partitioner=[CoreMLPartitioner()]
        )
        op_names = [
            n.target.__name__
            for n in edge.exported_program().graph.nodes
            if n.op == "call_function"
        ]
        self.assertIn("executorch_call_delegate", op_names)
        self.assertNotIn("aten.zeros_like.default", op_names)
        buffer = edge.to_executorch().buffer
        if _TEST_RUNTIME:
            program = Runtime.get().load_program(buffer)
            outputs = program.load_method("forward").execute(inputs)
            for out, ref in zip(outputs, model(*inputs)):
                self.assertTrue(torch.allclose(out.float(), ref.float(), atol=1e-2))

    def test_uint8_input_with_state(self):
        """
        A CPU op that reads a uint8 input can sit between the Core ML ops that
        read and write state, or make its new value.  Such state stays outside
        of Core ML, which reads it as an input.
        """

        class Split(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(1, 3, 2, 2))

            def forward(self, image):
                y = (image + (self.state * 2).to(torch.uint8)).float()
                self.state.add_(y)
                return y + 1

        class CopyIn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(1, 3, 2, 2))

            def forward(self, image):
                y = self.state * 2 + 1
                self.state.copy_(image.float())
                return y

        x = torch.randint(0, 3, (1, 3, 2, 2), dtype=torch.uint8)
        for model_class in (Split, CopyIn):
            with self.subTest(model=model_class.__name__):
                ep = torch.export.export(model_class().eval(), (x,), strict=True)
                edge = executorch.exir.to_edge_transform_and_lower(
                    ep, partitioner=[CoreMLPartitioner()]
                )
                op_names = [
                    n.target.__name__
                    for n in edge.exported_program().graph.nodes
                    if n.op == "call_function"
                ]
                # Below iOS 18 Core ML rejects a partition without inputs, but
                # the state is an input here, so Core ML takes state * 2 too.
                self.assertNotIn("aten.mul.Tensor", op_names)
                self.assertIn(
                    "state",
                    edge.exported_program().graph_signature.buffers_to_mutate.values(),
                )
                buffer = edge.to_executorch().buffer
                if _TEST_RUNTIME:
                    method = Runtime.get().load_program(buffer).load_method("forward")
                    eager = model_class().eval()
                    for _ in range(2):
                        out = method.execute((x,))[0]
                        self.assertTrue(torch.allclose(out, eager(x), atol=1e-2))

    def test_uint8_buffer_not_taken_over_is_read_outside_the_delegate(self):
        """
        A uint8 buffer that Core ML does not take over is a program input, so
        the op that reads it runs on the CPU.  So is uint8 state that stays in
        the program, because a CPU op reads it too, also through its new value,
        or because the model returns its new value.  A pad of such state runs on
        the CPU decomposed, since the portable kernels have no pad.  With
        lower_full_graph=True, lowering fails and names that state.
        """

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("lut", torch.arange(0, 256, 32, dtype=torch.uint8))

            def forward(self, x):
                return x * (self.lut.float() / 255.0)

        class ReadOnCpu(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "state", torch.tensor([[1, 2, 3, 4]], dtype=torch.uint8)
                )

            def forward(self, image, x):
                y = torch.nn.functional.pad(self.state, (1, 1)).float() * x
                y = y + (image + self.state).float().sum()
                self.state.add_(1)
                return y

        class NewValueOnCpu(ReadOnCpu):
            def forward(self, x):
                y = self.state.float() * x
                self.state.copy_((x * 10).to(torch.uint8))
                return y, torch.diagonal(self.state).float()

        class Returned(ReadOnCpu):
            def forward(self, x):
                self.state.add_(1)
                return self.state.float() * x, self.state

        state_inputs = (
            torch.randint(0, 3, (1, 4), dtype=torch.uint8),
            torch.rand(1, 6),
        )
        x = torch.rand(1, 4)
        for model_class, inputs, partitioner in (
            (
                Model,
                (torch.randn(2, 8),),
                CoreMLPartitioner(take_over_constant_data=False),
            ),
            (ReadOnCpu, state_inputs, CoreMLPartitioner()),
            (NewValueOnCpu, (x,), CoreMLPartitioner()),
            (Returned, (x,), CoreMLPartitioner()),
        ):
            with self.subTest(model=model_class.__name__):
                ep = torch.export.export(model_class().eval(), inputs, strict=True)
                edge = executorch.exir.to_edge_transform_and_lower(
                    ep, partitioner=[partitioner]
                )
                op_names = [
                    n.target.__name__
                    for n in edge.exported_program().graph.nodes
                    if n.op == "call_function"
                ]
                self.assertIn("dim_order_ops._to_dim_order_copy.default", op_names)
                self.assertIn("executorch_call_delegate", op_names)
                # Mutable buffers start uninitialized without this pass.
                config = executorch.exir.ExecutorchBackendConfig(
                    passes=[InitializedMutableBufferPass(["state"])]
                )
                buffer = edge.to_executorch(config).buffer
                if _TEST_RUNTIME:
                    method = Runtime.get().load_program(buffer).load_method("forward")
                    eager = model_class().eval()
                    for _ in range(2):
                        outputs = method.execute(inputs)
                        refs = eager(*inputs)
                        refs = refs if isinstance(refs, tuple) else (refs,)
                        for out, ref in zip(outputs, refs, strict=True):
                            self.assertTrue(
                                torch.allclose(
                                    out.float(), ref.float(), atol=1e-2, rtol=1e-3
                                )
                            )
        with self.assertRaisesRegex(NotImplementedError, "b_state \\(torch\\.uint8, "):
            executorch.exir.to_edge_transform_and_lower(
                torch.export.export(ReadOnCpu().eval(), state_inputs, strict=True),
                partitioner=[CoreMLPartitioner(lower_full_graph=True)],
            )

    def test_uint8_made_inside_the_delegate_keeps_ops_whole(self):
        """
        A uint8 tensor made inside the delegate is not an input, so the pad
        that reads it stays in the delegate, whole.
        """

        class Model(torch.nn.Module):
            def forward(self, x):
                image = torch.nn.functional.pad((x * 255).to(torch.uint8), (1, 1))
                return torch.nn.functional.pad(
                    image.float() / 255.0, (1, 1), mode="reflect"
                )

        model = Model().eval()
        x = torch.rand(1, 3, 8)
        ep = torch.export.export(model, (x,), strict=True)
        edge = executorch.exir.to_edge_transform_and_lower(
            ep, partitioner=[CoreMLPartitioner()]
        )
        op_names = {
            n.target.__name__
            for n in edge.exported_program().graph.nodes
            if n.op == "call_function"
        }
        self.assertEqual(op_names, {"executorch_call_delegate", "getitem"})
        buffer = edge.to_executorch().buffer
        if _TEST_RUNTIME:
            program = Runtime.get().load_program(buffer)
            out = program.load_method("forward").execute((x,))[0]
            self.assertTrue(torch.allclose(out, model(x), atol=1e-2))

    def test_ops_after_a_cpu_op_on_a_tensor_made_inside(self):
        """
        A uint8 tensor made in Core ML and passed through an op that stays on
        the CPU, such as a skipped view or cast, or an add that reads it from
        another partition past acosh, sends the pad after it to the CPU too.
        The pad is decomposed there, since the portable kernels have no pad,
        and so is a pad that makes an int8 model output.  The log names the
        uint8 tensors that ops read from the CPU or from another partition.
        """

        class FlipPad(torch.nn.Module):
            def forward(self, x):
                d = (x * 255).to(torch.uint8).flip(-1)
                return torch.nn.functional.pad(d, (1, 1)).float() / 255.0

        class ViewPad(torch.nn.Module):
            def forward(self, x):
                d = (x * 255).to(torch.uint8).view(1, 3, 8)
                return torch.nn.functional.pad(d, (1, 1)).float() / 255.0

        class AcoshPad(torch.nn.Module):
            def forward(self, x):
                d = (x * 255).to(torch.uint8)
                d = d + torch.acosh(d.float() + 2).to(torch.uint8)
                return torch.nn.functional.pad(d, (1, 1)).float() / 255.0

        class PadInt8(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.pad((x * 2).to(torch.int8), (1, 1))

        for model, x, skip in (
            (ViewPad(), torch.rand(1, 3, 2, 4), ["aten.view_copy.default"]),
            (
                FlipPad(),
                torch.rand(1, 3, 8),
                ["dim_order_ops._to_dim_order_copy.default"],
            ),
            (AcoshPad(), torch.rand(1, 3, 8), None),
            (PadInt8(), torch.randint(-25, 25, (1, 3, 8)).float(), None),
        ):
            with self.subTest(model=type(model).__name__, skip=skip):
                ep = torch.export.export(model.eval(), (x,), strict=True)
                with self.assertLogs(coreml_partitioner.logger, logging.INFO) as logs:
                    edge = executorch.exir.to_edge_transform_and_lower(
                        ep,
                        partitioner=[
                            CoreMLPartitioner(skip_ops_for_coreml_delegation=skip)
                        ],
                    )
                self.assertEqual(
                    "cannot take as an input" in "\n".join(logs.output),
                    not isinstance(model, PadInt8),
                )
                op_names = [
                    n.target.__name__
                    for n in edge.exported_program().graph.nodes
                    if n.op == "call_function"
                ]
                self.assertIn("aten.constant_pad_nd.default", op_names)
                self.assertIn("executorch_call_delegate", op_names)
                buffer = edge.to_executorch().buffer
                if _TEST_RUNTIME:
                    program = Runtime.get().load_program(buffer)
                    out = program.load_method("forward").execute((x,))[0]
                    self.assertTrue(
                        torch.allclose(out.float(), model(x).float(), atol=1e-2)
                    )

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
