# Copyright (c) 2026 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the A16W4/A16W8 quantization configs and the LLM custom
annotation helpers in ``backends/samsung/quantizer/custom_annotation.py``.

A silent regression in these annotations only surfaces as an accuracy drop
downstream, so these tests pin the configs and the annotation behavior
explicitly (which ops get quantized, what dtype/qscheme each qspec carries,
and which inputs must keep independent quantization params).
"""

import unittest

import torch
from executorch.backends.samsung.quantizer import (
    custom_annotation as ca,
    EnnQuantizer,
    Precision,
)
from executorch.backends.samsung.quantizer.observers import PerChannelClippingObserver
from executorch.backends.samsung.quantizer.qconfig import (
    get_a16w4_enn_quant_config,
    get_a16w8_enn_quant_config,
    get_quant_config,
)
from executorch.backends.test.graph_builder import GraphBuilder
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchao.quantization.pt2e.quantizer import (
    QuantizationAnnotation,
    SharedQuantizationSpec,
)

MUL = torch.ops.aten.mul.Tensor
MATMUL = torch.ops.aten.matmul.default
TRANSPOSE = torch.ops.aten.transpose.int
LINEAR = torch.ops.aten.linear.default
CAT = torch.ops.aten.cat.default


def _annotated(gm: torch.fx.GraphModule, target):
    return [n for n in gm.graph.nodes if n.op == "call_function" and n.target == target]


def _first(gm: torch.fx.GraphModule, target) -> torch.fx.Node:
    nodes = _annotated(gm, target)
    assert len(nodes) == 1, f"Expected one {target} node, found {len(nodes)}"
    return nodes[0]


def _input_specs(node: torch.fx.Node):
    return list(node.meta["quantization_annotation"].input_qspec_map.values())


class TestA16Configs(unittest.TestCase):
    def test_a16w4_activation_is_int16(self):
        cfg = get_a16w4_enn_quant_config(
            is_per_channel=True, is_qat=False, act_symmetric=False, wei_symmetric=True
        )
        self.assertEqual(cfg.input_activation.dtype, torch.int16)
        self.assertEqual(cfg.output_activation.dtype, torch.int16)

    def test_a16w4_weight_is_int4_range_per_channel(self):
        cfg = get_a16w4_enn_quant_config(
            is_per_channel=True, is_qat=False, act_symmetric=False, wei_symmetric=True
        )
        w = cfg.weight
        self.assertEqual(w.dtype, torch.int8)
        self.assertEqual(w.quant_min, -8)
        self.assertEqual(w.quant_max, 7)
        self.assertEqual(w.qscheme, torch.per_channel_symmetric)
        self.assertEqual(w.ch_axis, 0)

    def test_a16w8_activation_is_int16(self):
        cfg = get_a16w8_enn_quant_config(
            is_per_channel=True, is_qat=False, act_symmetric=False, wei_symmetric=True
        )
        self.assertEqual(cfg.input_activation.dtype, torch.int16)
        self.assertEqual(cfg.output_activation.dtype, torch.int16)

    def test_get_quant_config_a16w4_dispatch(self):
        cfg = get_quant_config(Precision.A16W4, is_per_channel=True, is_qat=False)
        self.assertEqual(cfg.input_activation.dtype, torch.int16)
        self.assertEqual(cfg.weight.quant_min, -8)
        self.assertEqual(cfg.weight.quant_max, 7)


class TestCustomAnnotation(unittest.TestCase):
    def _llama_like_module(self):
        class LlamaLike(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(8, 8)

            def forward(self, x, w):
                y = x * x
                wt = w.t()
                z = torch.matmul(y, wt)
                return self.proj(z)

        ep = torch.export.export(
            LlamaLike().eval(),
            (torch.randn(2, 8), torch.randn(8, 8)),
            strict=True,
        )
        return ep.graph_module

    def test_llama_matmul_weight_gets_shared_spec(self):
        gm = self._llama_like_module()
        ca.custom_annotate_llama_a16w4(gm)

        matmul = _first(gm, MATMUL)
        self.assertTrue(matmul.meta["quantization_annotation"]._annotated)
        specs = _input_specs(matmul)
        # 2nd input (weight path) reuses the default annotator's shared spec,
        # while the 1st input gets a fresh int16 activation qspec.
        self.assertTrue(
            any(isinstance(s, SharedQuantizationSpec) for s in specs), specs
        )
        self.assertTrue(
            any(
                not isinstance(s, SharedQuantizationSpec) and s.dtype == torch.int16
                for s in specs
            ),
            specs,
        )

    def test_llama_binary_ops_keep_independent_params(self):
        gm = self._llama_like_module()
        ca.custom_annotate_llama_a16w4(gm)

        mul = _first(gm, MUL)
        self.assertTrue(mul.meta["quantization_annotation"]._annotated)
        specs = _input_specs(mul)
        # Non-shared binary ops must NOT reuse a SharedQuantizationSpec.
        self.assertTrue(specs, "mul inputs were not annotated")
        self.assertTrue(
            all(not isinstance(s, SharedQuantizationSpec) for s in specs), specs
        )
        self.assertTrue(all(s.dtype == torch.int16 for s in specs), specs)

    def _gemma3_like_module(self):
        class Gemma3Like(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(16, 8)
                self.hidden_proj = torch.nn.Linear(8, 8)
                self.lm_head = torch.nn.Linear(8, 8, bias=False)

            def forward(self, ids, w):
                h = self.embed(ids)
                h = self.hidden_proj(h)
                y = self.lm_head(h)
                z = torch.matmul(y, w.t())
                return z * z

        ep = torch.export.export(
            Gemma3Like().eval(),
            (torch.randint(0, 16, (2, 4)), torch.randn(8, 8)),
            strict=True,
        )
        return ep.graph_module

    def test_gemma3_lm_head_annotated_with_high_precision(self):
        gm = self._gemma3_like_module()
        ca.custom_annotate_gemma3_a16w4(gm)

        linears = _annotated(gm, LINEAR)
        self.assertTrue(linears, "expected linear ops in graph")
        lm_head = linears[-1]
        self.assertIs(ca._find_lm_head(gm), lm_head)
        others = [n for n in linears if n is not lm_head]

        lm_annot = lm_head.meta["quantization_annotation"]
        self.assertTrue(lm_annot._annotated, "lm_head was not re-annotated")
        weight_spec = lm_annot.input_qspec_map[lm_head.args[1]]
        # lm_head is quantized with the high-precision per-channel A16W8 config.
        self.assertEqual(weight_spec.dtype, torch.int8)
        self.assertEqual(weight_spec.qscheme, torch.per_channel_symmetric)

        # Non-lm_head linears are left untouched by the custom annotation.
        for n in others:
            self.assertNotIn("quantization_annotation", n.meta)


class TestTiedLmHead(unittest.TestCase):
    def test_lm_head_tied_to_the_embedding_gets_high_precision(self):
        class Tied(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(16, 8)
                self.hidden_proj = torch.nn.Linear(8, 8)
                self.lm_head = torch.nn.Linear(8, 16, bias=False)
                self.lm_head.weight = self.embed.weight

            def forward(self, ids):
                return self.lm_head(self.hidden_proj(self.embed(ids)))

        gm = torch.export.export(
            Tied().eval(), (torch.randint(0, 16, (2, 4)),), strict=True
        ).graph_module
        ca.custom_annotate_gemma3_a16w4(gm)

        hidden_proj, lm_head = _annotated(gm, LINEAR)
        self.assertNotIn("quantization_annotation", hidden_proj.meta)
        annotation = lm_head.meta["quantization_annotation"]
        self.assertTrue(annotation._annotated)
        weight_spec = annotation.input_qspec_map[lm_head.args[1]]
        self.assertEqual(weight_spec.dtype, torch.int8)
        self.assertEqual(weight_spec.qscheme, torch.per_channel_symmetric)


class TestHelpers(unittest.TestCase):
    def test_matmul_with_non_node_args_does_not_crash(self):
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(2, 8))
        y = builder.placeholder("y", torch.randn(8, 8))
        mm = builder.call_operator(MATMUL, (x, y))
        builder.output([mm])
        gm = builder.get_graph_module()
        mm_node = next(n for n in gm.graph.nodes if n.target == MATMUL)
        # Mutate one matmul input to a non-node; the guard must not raise.
        mm_node.args = (mm_node.args[0], 7)
        self.assertFalse(ca._is_matmul_with_call_function_inputs(mm_node))

    def test_is_matmul_with_call_function_inputs(self):
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(2, 8))
        y = builder.placeholder("y", torch.randn(8, 8))
        mm = builder.call_operator(MATMUL, (x, y))
        builder.output([mm])
        gm = builder.get_graph_module()
        mm_node = next(n for n in gm.graph.nodes if n.target == MATMUL)
        # Both inputs are placeholders, not call_function nodes.
        self.assertFalse(ca._is_matmul_with_call_function_inputs(mm_node))

    def test_find_lm_head_is_the_last_linear(self):
        class Two(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4)
                self.head = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.head(self.proj(x))

        gm = torch.export.export(Two(), (torch.randn(2, 4),), strict=True).graph_module
        linears = _annotated(gm, LINEAR)
        self.assertEqual(len(linears), 2)
        self.assertIs(ca._find_lm_head(gm), linears[1])

        no_linear = torch.export.export(
            torch.nn.ReLU(), (torch.randn(2, 4),), strict=True
        ).graph_module
        self.assertIsNone(ca._find_lm_head(no_linear))

    def test_annotate_cat_accepts_tuple_inputs(self):
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(1, 4))
        y = builder.placeholder("y", torch.randn(1, 4))
        cat = builder.call_operator(CAT, ((x, y),))
        builder.output([cat])
        gm = builder.get_graph_module()
        cat_node = next(n for n in gm.graph.nodes if n.target == CAT)
        self.assertIsInstance(cat_node.args[0], tuple)

        cfg = get_a16w8_enn_quant_config(
            is_per_channel=False, is_qat=False, act_symmetric=False, wei_symmetric=False
        )
        # Must not assert-fail on a tuple-shaped inputs argument.
        ca.annotate_cat(cat_node, cfg)
        self.assertTrue(cat_node.meta["quantization_annotation"]._annotated)


class TestAnnotateCat(unittest.TestCase):
    def test_existing_annotation_of_the_first_input_is_kept(self):
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(1, 4))
        w = builder.placeholder("w", torch.randn(4, 4))
        y = builder.placeholder("y", torch.randn(1, 4))
        linear = builder.call_operator(LINEAR, (x, w))
        cat = builder.call_operator(CAT, ([linear, y],))
        builder.output([cat])
        gm = builder.get_graph_module()
        linear_node = _first(gm, LINEAR)
        cat_node = _first(gm, CAT)
        x_node = linear_node.args[0]

        cfg = get_a16w8_enn_quant_config(
            is_per_channel=False, is_qat=False, act_symmetric=False, wei_symmetric=False
        )
        default_spec = get_a16w4_enn_quant_config().input_activation
        linear_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map={x_node: default_spec}, _annotated=True
        )

        ca.annotate_cat(cat_node, cfg)

        annotation = linear_node.meta["quantization_annotation"]
        self.assertTrue(annotation._annotated)
        self.assertEqual(annotation.input_qspec_map, {x_node: default_spec})
        self.assertEqual(annotation.output_qspec, cfg.input_activation)


class TestPerChannelClippingObserver(unittest.TestCase):
    @staticmethod
    def _observer():
        return PerChannelClippingObserver(
            ch_axis=0,
            dtype=torch.int8,
            qscheme=torch.per_channel_symmetric,
            quant_min=-8,
            quant_max=7,
        )

    def test_range_is_searched_once_until_reset(self):
        observer = self._observer()
        weight = torch.randn(4, 16)
        observer(weight)
        first_scale, _ = observer.calculate_qparams()

        observer(weight * 10)
        scale, _ = observer.calculate_qparams()
        self.assertTrue(torch.equal(scale, first_scale))

        observer.reset_min_max_vals()
        observer(weight * 10)
        scale, _ = observer.calculate_qparams()
        self.assertFalse(torch.equal(scale, first_scale))

    def test_zero_channel_has_a_finite_scale(self):
        observer = self._observer()
        weight = torch.randn(4, 16)
        weight[1] = 0
        observer(weight)
        scale, _ = observer.calculate_qparams()
        self.assertTrue(bool(torch.isfinite(scale).all()))
        self.assertTrue(bool((scale > 0).all()))


class TestA16W4EndToEnd(unittest.TestCase):
    def test_prepare_convert_a16w4(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(16, 8)

            def forward(self, x):
                return self.lin(x)

        m = M().eval()
        x = torch.randn(2, 16)
        ep = torch.export.export(m, (x,), strict=True)

        quantizer = EnnQuantizer()
        quantizer.setup_quant_params(Precision.A16W4, is_per_channel=True, is_qat=False)
        prepared = prepare_pt2e(ep.module(), quantizer)
        prepared(x)
        converted = convert_pt2e(prepared)
        with torch.inference_mode():
            out = converted(x)
        self.assertEqual(tuple(out.shape), (2, 8))

        quant_ops = torch.ops.quantized_decomposed
        activation_dtypes = {
            n.args[-1]
            for n in converted.graph.nodes
            if n.target == quant_ops.quantize_per_tensor.default
        }
        self.assertEqual(activation_dtypes, {torch.int16})
        weight_ranges = {
            (n.args[-3], n.args[-2])
            for n in converted.graph.nodes
            if n.target == quant_ops.dequantize_per_channel.default
        }
        self.assertEqual(weight_ranges, {(-8, 7)})


if __name__ == "__main__":
    unittest.main()
