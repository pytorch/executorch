# Copyright (c) 2026 Software Mansion
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import math
import os
import tempfile
import unittest
from collections import Counter
from unittest import mock

import executorch.backends.test.harness.stages as BaseStages
import torch
from executorch.backends.test.harness.stages import StageType
from executorch.backends.xnnpack._passes import XNNPACKPassManager
from executorch.backends.xnnpack._passes.decompose_conv_transpose import (
    DecomposeConvTransposePass,
)
from executorch.backends.xnnpack._passes.fuse_activation_pass import FuseActivationPass
from executorch.backends.xnnpack._passes.fuse_pad_into_conv_pass import (
    FusePadIntoConvPass,
)
from executorch.backends.xnnpack.serialization.xnnpack_graph_serialize import (
    _get_schema_bytes,
    XNNHeader,
)
from executorch.backends.xnnpack.test.tester import RunPasses, Tester
from executorch.exir._serialize import _deserialize_pte_binary
from executorch.exir._serialize._flatbuffer import _flatc_decompile
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.passes.external_constants_pass import (
    delegate_external_constants_pass_unlifted,
)
from executorch.extension.flat_tensor.serialize.serialize import (
    _deserialize_to_flat_tensor,
)


class TestDecomposeDepthwiseConvTransposePass(unittest.TestCase):
    PassStage = RunPasses([DecomposeConvTransposePass])

    conv = "executorch_exir_dialects_edge__ops_aten_convolution_default"
    pad = "executorch_exir_dialects_edge__ops_aten_constant_pad_nd_default"
    cat = "executorch_exir_dialects_edge__ops_aten_cat_default"
    d2s = "executorch_exir_dialects_edge__ops_et_xnnpack_depth_to_space_2d_default"

    def setUp(self):
        torch._dynamo.reset()

    class DwDeconv(torch.nn.Module):
        def __init__(
            self, kernel, stride, padding, output_padding=0, bias=True, channels=8
        ):
            super().__init__()
            self.deconv = torch.nn.ConvTranspose2d(
                channels,
                channels,
                kernel,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                groups=channels,
                bias=bias,
            )
            with torch.no_grad():
                self.deconv.weight.normal_()
                if bias:
                    self.deconv.bias.normal_()

        def forward(self, x):
            return self.deconv(x)

    class QuantizedDwDeconv(torch.nn.Module):
        # The XNNPACK quantizer never annotates grouped transposed convs, so
        # build the dequantized weight by hand.
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "weight", torch.randint(-128, 127, (8, 1, 4, 4), dtype=torch.int8)
            )

        def forward(self, x):
            weight = torch.ops.quantized_decomposed.dequantize_per_tensor(
                self.weight, 0.02, 0, -128, 127, torch.int8
            )
            return torch.nn.functional.conv_transpose2d(
                x, weight, stride=2, padding=1, groups=8
            )

    def _check_pass(self, module, inputs, counts, tol=1e-3, dynamic_shapes=None):
        return (
            Tester(module.eval(), inputs, dynamic_shapes=dynamic_shapes)
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .check_count(counts)
            .run_method_and_compare_outputs(atol=tol, rtol=tol)
        )

    @staticmethod
    def _xnnpack_node_types(tester) -> Counter:
        """Node types in the serialized XNNPACK graph of the single delegate
        produced by to_edge_transform_and_lower."""
        graph_module = (
            tester.get_artifact(StageType.TO_EDGE_TRANSFORM_AND_LOWER)
            .exported_program()
            .graph_module
        )
        (delegate,) = [
            node
            for node in graph_module.graph.nodes
            if node.target == torch.ops.higher_order.executorch_call_delegate
        ]
        payload = getattr(graph_module, delegate.args[0].target).processed_bytes
        header = XNNHeader.from_bytes(payload[: XNNHeader.EXPECTED_LENGTH])
        flatbuffer = payload[
            header.flatbuffer_offset : header.flatbuffer_offset + header.flatbuffer_size
        ]
        with tempfile.TemporaryDirectory() as d:
            schema_path = os.path.join(d, "schema.fbs")
            with open(schema_path, "wb") as f:
                f.write(_get_schema_bytes())
            bin_path = os.path.join(d, "graph.bin")
            with open(bin_path, "wb") as f:
                f.write(flatbuffer)
            _flatc_decompile(d, schema_path, bin_path)
            with open(os.path.join(d, "graph.json")) as f:
                # flatc writes unbounded clamp limits as a bare inf
                text = (
                    f.read().replace("-inf", "-Infinity").replace(" inf", " Infinity")
                )
            graph = json.loads(text)
        return Counter(node["xnode_union_type"] for node in graph["xnodes"])

    def _check_lowered(
        self, module, inputs, tol=1e-3, dynamic_shapes=None, num_deconvs=1
    ):
        # XNNPACK delegates an fp32 depthwise transposed conv on its own, so
        # one delegate and matching outputs do not prove the pass ran during
        # lowering: count its calls and check the serialized graph.
        tester = Tester(module.eval(), inputs, dynamic_shapes=dynamic_shapes).export()
        with mock.patch.object(
            DecomposeConvTransposePass,
            "_decompose",
            autospec=True,
            side_effect=DecomposeConvTransposePass._decompose,
        ) as decompose:
            tester.to_edge_transform_and_lower()
        self.assertEqual(decompose.call_count, num_deconvs)
        tester.check_count(
            {"torch.ops.higher_order.executorch_call_delegate": 1}
        ).check_not([self.conv])
        node_types = self._xnnpack_node_types(tester)
        self.assertEqual(node_types["XNNConvTranspose2d"], 0, node_types)
        self.assertEqual(node_types["XNNDepthToSpace2d"], num_deconvs, node_types)
        (
            tester.to_executorch()
            .serialize()
            .run_method_and_compare_outputs(num_runs=3, atol=tol, rtol=tol)
        )
        return tester

    def test_upfirdn_4x4_stride2(self):
        # StyleGAN-style x2 upsampling filter: 4 phases of 2x2 taps, each with
        # a one-sided pad.
        module = self.DwDeconv(kernel=4, stride=2, padding=1)
        inputs = (torch.randn(1, 8, 6, 10),)
        self._check_pass(
            module, inputs, {self.conv: 4, self.pad: 4, self.cat: 1, self.d2s: 1}
        )
        self._check_lowered(module, inputs)

    def test_no_bias_fp16(self):
        module = self.DwDeconv(kernel=4, stride=2, padding=1, bias=False).half()
        inputs = (torch.randn(1, 8, 5, 7).half(),)
        # fp16 accumulation order differs from the reference: 1 ulp at |y| ~ 10
        self._check_pass(module, inputs, {self.conv: 4, self.d2s: 1}, tol=1e-2)
        self._check_lowered(module, inputs, tol=1e-2)

    def test_3x3_stride2_output_padding(self):
        # k=3, p=1, output_padding=1 keeps the x2 scaling; the phases have 1
        # and 2 taps per axis.
        module = self.DwDeconv(kernel=3, stride=2, padding=1, output_padding=1)
        inputs = (torch.randn(1, 8, 6, 6),)
        self._check_pass(module, inputs, {self.conv: 4, self.cat: 1, self.d2s: 1})
        self._check_lowered(module, inputs)

    def test_2x2_stride2_no_padding(self):
        # Every phase is a single tap, no pads at all.
        module = self.DwDeconv(kernel=2, stride=2, padding=0)
        inputs = (torch.randn(1, 8, 4, 6),)
        self._check_pass(
            module, inputs, {self.conv: 4, self.pad: 0, self.cat: 1, self.d2s: 1}
        )
        self._check_lowered(module, inputs)

    def test_stride3(self):
        module = self.DwDeconv(kernel=3, stride=3, padding=0, channels=4)
        inputs = (torch.randn(1, 4, 4, 5),)
        self._check_pass(module, inputs, {self.conv: 9, self.cat: 1, self.d2s: 1})
        self._check_lowered(module, inputs)

    def test_conv_before_deconv(self):
        # The phase pads read the conv's output rather than a graph input.
        module = torch.nn.Sequential(
            torch.nn.Conv2d(8, 8, 3, padding=1),
            self.DwDeconv(kernel=4, stride=2, padding=1),
        )
        inputs = (torch.randn(1, 8, 6, 10),)
        tester = self._check_pass(
            module, inputs, {self.conv: 5, self.pad: 4, self.cat: 1, self.d2s: 1}
        )
        graph = (
            tester.get_artifact(StageType.RUN_PASSES)
            .exported_program()
            .graph_module.graph
        )
        for node in graph.nodes:
            if node.target == exir_ops.edge.aten.constant_pad_nd.default:
                self.assertEqual(
                    node.args[0].target, exir_ops.edge.aten.convolution.default
                )
        self._check_lowered(module, inputs)

    def test_two_deconvs_in_a_row(self):
        # The second deconv's input is the first one's depth_to_space_2d.
        module = torch.nn.Sequential(
            self.DwDeconv(kernel=4, stride=2, padding=1),
            self.DwDeconv(kernel=4, stride=2, padding=1),
        )
        inputs = (torch.randn(1, 8, 4, 6),)
        self._check_pass(
            module, inputs, {self.conv: 8, self.pad: 8, self.cat: 2, self.d2s: 2}
        )
        self._check_lowered(module, inputs, num_deconvs=2)

    class SharedWeightDeconvs(torch.nn.Module):
        def __init__(self, kernel, stride, paddings, output_paddings):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(8, 1, kernel, kernel))
            self.stride = stride
            self.configs = list(zip(paddings, output_paddings))

        def forward(self, x):
            return sum(
                torch.nn.functional.conv_transpose2d(
                    x,
                    self.weight,
                    stride=self.stride,
                    padding=padding,
                    output_padding=output_padding,
                    groups=8,
                )
                for padding, output_padding in self.configs
            )

    def _count_phase_placeholders(self, tester):
        graph = (
            tester.get_artifact(StageType.RUN_PASSES)
            .exported_program()
            .graph_module.graph
        )
        return sum(
            1
            for node in graph.nodes
            if node.op == "placeholder" and "_phase_" in node.name
        )

    def test_shared_weight_different_padding(self):
        # Both deconvs are clean x3 upscales of the same weight, but their
        # phase taps differ, so they must not share the phase placeholders.
        module = self.SharedWeightDeconvs(
            kernel=9, stride=3, paddings=(3, 4), output_paddings=(0, 2)
        )
        inputs = (torch.randn(1, 8, 5, 6),)
        tester = self._check_pass(
            module, inputs, {self.conv: 18, self.cat: 2, self.d2s: 2}
        )
        self.assertEqual(self._count_phase_placeholders(tester), 18)
        self._check_lowered(module, inputs, num_deconvs=2)

    def test_shared_weight_same_params(self):
        # Identical deconvs on one weight share one set of phase placeholders.
        module = self.SharedWeightDeconvs(
            kernel=9, stride=3, paddings=(3, 3), output_paddings=(0, 0)
        )
        inputs = (torch.randn(1, 8, 5, 6),)
        tester = self._check_pass(
            module, inputs, {self.conv: 18, self.cat: 2, self.d2s: 2}
        )
        self.assertEqual(self._count_phase_placeholders(tester), 9)
        self._check_lowered(module, inputs, num_deconvs=2)

    def test_batch_gt_1(self):
        module = self.DwDeconv(kernel=4, stride=2, padding=1)
        inputs = (torch.randn(3, 8, 6, 10),)
        self._check_pass(
            module, inputs, {self.conv: 4, self.pad: 4, self.cat: 1, self.d2s: 1}
        )
        self._check_lowered(module, inputs)

    def test_single_element_spatial_params(self):
        class Deconv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(8, 1, 4, 4))

            def forward(self, x):
                return torch.ops.aten.convolution.default(
                    x, self.weight, None, [2], [1], [1], True, [0], 8
                )

        inputs = (torch.randn(1, 8, 6, 10),)
        self._check_pass(Deconv(), inputs, {self.conv: 4, self.pad: 4, self.d2s: 1})
        self._check_lowered(Deconv(), inputs)

    def test_deconv_relu(self):
        # The activation follows depth_to_space_2d instead of fusing into the
        # removed transposed conv, which relies on the pass running before
        # FuseActivationPass in the default pass list. The phase pads read the
        # NHWC output of the leading conv and fold into the phase convs through
        # FusePadIntoConvPass at the end of that list.
        module = torch.nn.Sequential(
            torch.nn.Conv2d(8, 8, 3, padding=1),
            self.DwDeconv(kernel=4, stride=2, padding=1),
            torch.nn.ReLU(),
        )
        inputs = (torch.randn(1, 8, 6, 10),)
        default_passes = XNNPACKPassManager(
            Tester(module.eval(), inputs).export().get_artifact()
        ).passes
        self.assertLess(
            default_passes.index(DecomposeConvTransposePass),
            default_passes.index(FuseActivationPass),
        )
        tester = (
            Tester(module.eval(), inputs)
            .export()
            .to_edge()
            .run_passes(RunPasses(default_passes))
            .check_count({self.conv: 5, self.pad: 4, self.d2s: 1})
        )
        graph = (
            tester.get_artifact(StageType.RUN_PASSES)
            .exported_program()
            .graph_module.graph
        )
        (relu,) = [
            n for n in graph.nodes if n.target == exir_ops.edge.aten.relu.default
        ]
        self.assertEqual(
            relu.args[0].target, exir_ops.edge.et_xnnpack.depth_to_space_2d.default
        )
        convs = [
            n for n in graph.nodes if n.target == exir_ops.edge.aten.convolution.default
        ]
        self.assertEqual(
            [FuseActivationPass.get_fused_activation(n) for n in convs], [None] * 5
        )
        self.assertEqual(
            sum(FusePadIntoConvPass.get_fused_padding(n) is not None for n in convs), 4
        )
        self.assertEqual(
            sum(
                FusePadIntoConvPass.is_folded_pad(n)
                for n in graph.nodes
                if n.target == exir_ops.edge.aten.constant_pad_nd.default
            ),
            4,
        )
        self._check_lowered(module, inputs)

    class ExportWithExternalConstants(BaseStages.Export):
        def run(self, artifact, inputs):
            tagged = torch.export.export(artifact, inputs, strict=True).module()
            delegate_external_constants_pass_unlifted(
                module=tagged, gen_tag_fn=lambda _: "model"
            )
            self.exported_program = torch.export.export(tagged, inputs, strict=True)

    def test_external_constants(self):
        # The phase weights inherit the original weight's external-constant
        # tag, so every constant ends up in model.ptd and none in the .pte.
        module = self.DwDeconv(kernel=4, stride=2, padding=1)
        inputs = (torch.randn(1, 8, 6, 10),)
        tester = Tester(module.eval(), inputs)
        tester.export(self.ExportWithExternalConstants())
        tester.to_edge_transform_and_lower().check_count(
            {"torch.ops.higher_order.executorch_call_delegate": 1}
        ).to_executorch()
        exec = tester.get_artifact()
        self.assertIsNone(_deserialize_pte_binary(exec.buffer).named_data)
        self.assertEqual(list(exec._tensor_data), ["model"])
        ptd = _deserialize_to_flat_tensor(bytes(exec._tensor_data["model"]))
        self.assertEqual(len(ptd.named_data), 5)  # 4 phase weights + bias

        from executorch.extension.pybindings import portable_lib as runtime

        loaded = runtime._load_for_executorch_from_buffer(
            exec.buffer, bytes(exec._tensor_data["model"])
        )
        self.assertTrue(
            torch.allclose(loaded.forward(inputs)[0], module(*inputs), atol=1e-3)
        )

    def test_dynamic_shape(self):
        module = self.DwDeconv(kernel=4, stride=2, padding=1)
        inputs = (torch.randn(1, 8, 6, 10),)
        dynamic_shapes = (
            {
                2: torch.export.Dim("height", min=2, max=16),
                3: torch.export.Dim("width", min=2, max=16),
            },
        )
        self._check_pass(
            module,
            inputs,
            {self.conv: 4, self.pad: 4, self.cat: 1, self.d2s: 1},
            dynamic_shapes=dynamic_shapes,
        )
        self._check_lowered(module, inputs, dynamic_shapes=dynamic_shapes)

    def test_quantized_is_left_alone(self):
        inputs = (torch.randn(1, 8, 6, 6),)
        self._check_pass(self.QuantizedDwDeconv(), inputs, {self.conv: 1, self.d2s: 0})

    def test_not_a_clean_upscale(self):
        # k=4, p=0: output is 2H + 2; every phase has 2 taps and a symmetric
        # pad of 1, which the phase convs take as their padding.
        module = self.DwDeconv(kernel=4, stride=2, padding=0)
        inputs = (torch.randn(1, 8, 6, 6),)
        self._check_pass(
            module, inputs, {self.conv: 4, self.pad: 0, self.cat: 1, self.d2s: 1}
        )
        self._check_lowered(module, inputs)

    def test_negative_pad_is_left_alone(self):
        # k=3, p=2: phase 1 only reads inputs past its own step.
        module = self.DwDeconv(kernel=3, stride=2, padding=2)
        inputs = (torch.randn(1, 8, 6, 6),)
        self._check_pass(module, inputs, {self.conv: 1, self.d2s: 0})

    def test_grouped_but_not_depthwise_is_left_alone(self):
        module = torch.nn.ConvTranspose2d(8, 8, 4, stride=2, padding=1, groups=4)
        inputs = (torch.randn(1, 8, 6, 6),)
        self._check_pass(module, inputs, {self.conv: 1, self.d2s: 0})

    def test_phase_math_matches_conv_transpose2d(self):
        # The decomposition in plain torch, independent of the lowering.
        import executorch.backends.xnnpack.custom_ops_lib as lib

        for k, s, p, o in [
            (4, 2, 1, 0),
            (3, 2, 1, 1),
            (2, 2, 0, 0),
            (3, 3, 0, 0),
            (5, 2, 2, 1),
        ]:
            c, h, w = 3, 5, 6
            x = torch.randn(1, c, h, w)
            weight = torch.randn(c, 1, k, k)
            ref = torch.nn.functional.conv_transpose2d(
                x, weight, stride=s, padding=p, output_padding=o, groups=c
            )
            phases = []
            for ay in range(s):
                for ax in range(s):
                    taps = []
                    for a in (ay, ax):
                        t_min = math.ceil(-(a + p) / s)
                        t_max = math.floor((k - 1 - a - p) / s)
                        taps.append((t_min, t_max))
                    (ty_min, ty_max), (tx_min, tx_max) = taps
                    iy = torch.tensor(
                        [s * (ty_max - q) + ay + p for q in range(ty_max - ty_min + 1)]
                    )
                    ix = torch.tensor(
                        [s * (tx_max - q) + ax + p for q in range(tx_max - tx_min + 1)]
                    )
                    wp = weight[:, :, iy][:, :, :, ix]
                    xp = torch.nn.functional.pad(x, (tx_max, -tx_min, ty_max, -ty_min))
                    phases.append(torch.nn.functional.conv2d(xp, wp, groups=c))
            out = lib.depth_to_space_2d_op(torch.cat(phases, 1), s)
            self.assertEqual(out.shape, ref.shape, (k, s, p, o))
            self.assertTrue(torch.allclose(out, ref, atol=1e-5), (k, s, p, o))
