# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest import mock

import executorch.backends.test.harness.stages as BaseStages
import executorch.backends.xnnpack.test.passes.test_decompose_depthwise_conv_transpose as dw_tests
import torch
from executorch.backends.test.harness.stages import StageType
from executorch.backends.xnnpack._passes import XNNPACKPassManager
from executorch.backends.xnnpack._passes.decompose_conv_transpose import (
    DecomposeConvTransposePass,
)
from executorch.backends.xnnpack._passes.fuse_activation_pass import FuseActivationPass
from executorch.backends.xnnpack.test.tester import RunPasses, Tester
from executorch.exir._serialize import _deserialize_pte_binary
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.passes.external_constants_pass import (
    delegate_external_constants_pass_unlifted,
)
from executorch.extension.flat_tensor.serialize.serialize import (
    _deserialize_to_flat_tensor,
)


class TestDecomposeConvTransposePass(unittest.TestCase):
    PassStage = RunPasses([DecomposeConvTransposePass])

    conv = "executorch_exir_dialects_edge__ops_aten_convolution_default"
    view = "executorch_exir_dialects_edge__ops_aten_view_copy_default"
    permute = "executorch_exir_dialects_edge__ops_aten_permute_copy_default"
    slice = "executorch_exir_dialects_edge__ops_aten_slice_copy_Tensor"
    pad = "executorch_exir_dialects_edge__ops_aten_constant_pad_nd_default"
    cat = "executorch_exir_dialects_edge__ops_aten_cat_default"
    d2s = "executorch_exir_dialects_edge__ops_et_xnnpack_depth_to_space_2d_default"

    def setUp(self):
        torch._dynamo.reset()
        torch.manual_seed(0)

    @staticmethod
    def _deconv(dims, c_in, c_out, kernel, stride, **kwargs):
        cls = torch.nn.ConvTranspose1d if dims == 1 else torch.nn.ConvTranspose2d
        return cls(c_in, c_out, kernel, stride=stride, **kwargs)

    @staticmethod
    def _transposed_convs(tester) -> int:
        graph = (
            tester.get_artifact(StageType.RUN_PASSES).exported_program().graph_module
        ).graph
        return sum(
            node.target == exir_ops.edge.aten.convolution.default and node.args[6]
            for node in graph.nodes
        )

    def _run_pass(self, module, inputs, dynamic_shapes=None):
        return (
            Tester(module.eval(), inputs, dynamic_shapes=dynamic_shapes)
            .export()
            .to_edge()
            .run_passes(self.PassStage)
        )

    def _check_rewritten(self, module, inputs, counts, tol=1e-4):
        tester = self._run_pass(module, inputs).check_count(counts)
        self.assertEqual(self._transposed_convs(tester), 0)
        tester.run_method_and_compare_outputs(atol=tol, rtol=tol)
        return tester

    def _check_left_alone(self, module, inputs, dynamic_shapes=None):
        tester = self._run_pass(module, inputs, dynamic_shapes)
        self.assertEqual(self._transposed_convs(tester), 1)

    def _check_lowered(self, module, inputs, tol=1e-4, num_deconvs=1):
        # XNNPACK delegates these transposed convs on their own, so one
        # delegate and matching outputs do not prove the pass ran during
        # lowering: count its calls and check the serialized graph.
        tester = Tester(module.eval(), inputs).export()
        with mock.patch.object(
            DecomposeConvTransposePass,
            "_decompose",
            autospec=True,
            side_effect=DecomposeConvTransposePass._decompose,
        ) as decompose:
            tester.to_edge_transform_and_lower()
        self.assertEqual(decompose.call_count, num_deconvs)
        tester.check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
        node_types = (
            dw_tests.TestDecomposeDepthwiseConvTransposePass._xnnpack_node_types(tester)
        )
        self.assertEqual(node_types["XNNConvTranspose2d"], 0, node_types)
        self.assertGreaterEqual(
            node_types["XNNConv2d"] + node_types["XNNDepthwiseConv2d"],
            num_deconvs,
            node_types,
        )
        (
            tester.to_executorch()
            .serialize()
            .run_method_and_compare_outputs(num_runs=3, atol=tol, rtol=tol)
        )

    def test_1d_kernel_twice_stride(self):
        for stride in (2, 4, 5, 6):
            for bias in (True, False):
                with self.subTest(stride=stride, bias=bias):
                    module = self._deconv(1, 6, 3, 2 * stride, stride, bias=bias)
                    inputs = (torch.randn(1, 6, 7),)
                    self._check_rewritten(
                        module,
                        inputs,
                        {self.conv: 1, self.view: 2, self.permute: 1, self.slice: 0},
                    )
        self._check_lowered(module, inputs)

    def test_1d_single_step(self):
        # Streaming decoders run one input step per call.
        module = self._deconv(1, 8, 4, 8, 4)
        inputs = (torch.randn(1, 8, 1),)
        self._check_rewritten(module, inputs, {self.conv: 1, self.view: 2})
        self._check_lowered(module, inputs)

    def test_1d_padding_and_output_padding(self):
        for kernel, stride, padding, output_padding in (
            (8, 4, 2, 1),
            (16, 8, 4, 0),
            (4, 2, 1, 1),
        ):
            with self.subTest(kernel=kernel, stride=stride, padding=padding):
                module = self._deconv(
                    1,
                    5,
                    4,
                    kernel,
                    stride,
                    padding=padding,
                    output_padding=output_padding,
                )
                inputs = (torch.randn(2, 5, 6),)
                self._check_rewritten(module, inputs, {self.conv: 1, self.slice: 1})
        self._check_lowered(module, inputs)

    def test_1d_kernel_multiple_of_stride(self):
        # m = 1 needs no padding, m = 3 pads the conv by 2.
        for kernel, stride in ((3, 3), (6, 2), (9, 3)):
            with self.subTest(kernel=kernel, stride=stride):
                module = self._deconv(1, 4, 6, kernel, stride)
                inputs = (torch.randn(1, 4, 5),)
                self._check_rewritten(module, inputs, {self.conv: 1, self.view: 2})
        self._check_lowered(module, inputs)

    def test_2d(self):
        for kernel, stride, padding, output_padding, slices in (
            (4, 2, 1, 0, 2),
            ((4, 2), 2, 0, 0, 0),
            (6, 3, (1, 2), (1, 0), 2),
        ):
            with self.subTest(kernel=kernel, stride=stride, padding=padding):
                module = self._deconv(
                    2,
                    4,
                    6,
                    kernel,
                    stride,
                    padding=padding,
                    output_padding=output_padding,
                )
                inputs = (torch.randn(1, 4, 5, 7),)
                self._check_rewritten(
                    module, inputs, {self.conv: 1, self.d2s: 1, self.slice: slices}
                )
                self._check_lowered(module, inputs)

    def test_fp16(self):
        module = self._deconv(1, 8, 4, 8, 4).half()
        inputs = (torch.randn(1, 8, 6).half(),)
        self._check_rewritten(module, inputs, {self.conv: 1, self.view: 2}, tol=1e-2)
        self._check_lowered(module, inputs, tol=1e-2)

    def test_two_deconvs_with_activation(self):
        # The second deconv's input is the first one's interleave.
        module = torch.nn.Sequential(
            self._deconv(1, 16, 8, 12, 6),
            torch.nn.ELU(),
            self._deconv(1, 8, 4, 10, 5),
        )
        inputs = (torch.randn(1, 16, 3),)
        self._check_rewritten(module, inputs, {self.conv: 2, self.view: 4})
        self._check_lowered(module, inputs, num_deconvs=2)

    class SharedDeconv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.deconv = torch.nn.ConvTranspose1d(6, 4, 8, stride=4)

        def forward(self, x, z):
            return self.deconv(x) + self.deconv(z)

    def test_shared_weight(self):
        # Both deconvs reuse one phase weight and one phase bias.
        module = self.SharedDeconv()
        inputs = (torch.randn(1, 6, 5), torch.randn(1, 6, 5))
        tester = self._check_rewritten(module, inputs, {self.conv: 2, self.view: 4})
        placeholders = [
            node.name
            for node in tester.get_artifact(StageType.RUN_PASSES)
            .exported_program()
            .graph_module.graph.nodes
            if node.op == "placeholder"
        ]
        self.assertEqual(sum(n.endswith("_s4_phases") for n in placeholders), 1)
        self.assertEqual(sum(n.endswith("_x4") for n in placeholders), 1)

    class DeconvBiasReused(torch.nn.Module):
        """The overlap-add of a streaming decoder reads the bias again."""

        def __init__(self):
            super().__init__()
            self.deconv = torch.nn.ConvTranspose1d(6, 4, 8, stride=4)

        def forward(self, x):
            return self.deconv(x) - self.deconv.bias.view(1, -1, 1)

    def test_reused_bias(self):
        module = self.DeconvBiasReused()
        inputs = (torch.randn(1, 6, 5),)
        # The interleave and the bias view.
        self._check_rewritten(module, inputs, {self.conv: 1, self.view: 3})
        self._check_lowered(module, inputs)

    def test_pass_order(self):
        module = self._deconv(1, 6, 3, 4, 2)
        inputs = (torch.randn(1, 6, 7),)
        passes = XNNPACKPassManager(
            Tester(module.eval(), inputs).export().get_artifact()
        ).passes
        self.assertLess(
            passes.index(DecomposeConvTransposePass),
            passes.index(FuseActivationPass),
        )

    class ExportWithExternalConstants(BaseStages.Export):
        def run(self, artifact, inputs):
            tagged = torch.export.export(artifact, inputs, strict=True).module()
            delegate_external_constants_pass_unlifted(
                module=tagged, gen_tag_fn=lambda _: "model"
            )
            self.exported_program = torch.export.export(tagged, inputs, strict=True)

    def test_external_constants(self):
        # The phase weight and bias inherit the external-constant tag of the
        # constants they replace.
        module = self._deconv(1, 6, 3, 8, 4).eval()
        inputs = (torch.randn(1, 6, 7),)
        tester = Tester(module, inputs)
        tester.export(self.ExportWithExternalConstants())
        tester.to_edge_transform_and_lower().check_count(
            {"torch.ops.higher_order.executorch_call_delegate": 1}
        ).to_executorch()
        program = tester.get_artifact()
        self.assertIsNone(_deserialize_pte_binary(program.buffer).named_data)
        ptd = _deserialize_to_flat_tensor(bytes(program._tensor_data["model"]))
        self.assertEqual(len(ptd.named_data), 2)

        from executorch.extension.pybindings import portable_lib as runtime

        loaded = runtime._load_for_executorch_from_buffer(
            program.buffer, bytes(program._tensor_data["model"])
        )
        self.assertTrue(
            torch.allclose(loaded.forward(inputs)[0], module(*inputs), atol=1e-4)
        )

    def test_depthwise_1d_unpadded(self):
        # The upsampler of a streaming audio decoder: every phase has 2 taps
        # and a symmetric pad of 1, so no pads and no slice.
        for stride in (2, 16):
            for steps in (1, 5):
                with self.subTest(stride=stride, steps=steps):
                    module = self._deconv(
                        1, 8, 8, 2 * stride, stride, groups=8, bias=False
                    )
                    inputs = (torch.randn(1, 8, steps),)
                    self._check_rewritten(
                        module,
                        inputs,
                        {
                            self.conv: stride,
                            self.pad: 0,
                            self.cat: 1,
                            self.view: 2,
                            self.slice: 0,
                        },
                    )
                    self._check_lowered(module, inputs)

    def test_depthwise_1d_padded(self):
        # k=4, p=1, output_padding=1: phase 1 reads its input through an
        # asymmetric pad, and the odd output size needs a slice.
        module = self._deconv(1, 8, 8, 4, 2, padding=1, output_padding=1, groups=8)
        inputs = (torch.randn(2, 8, 5),)
        self._check_rewritten(
            module, inputs, {self.conv: 2, self.pad: 1, self.cat: 1, self.slice: 1}
        )
        self._check_lowered(module, inputs)

    def test_depthwise_2d_odd_output(self):
        # k=5, p=0: 2H + 3 outputs, sliced from 2H + 4 on both axes.
        module = self._deconv(2, 8, 8, 5, 2, groups=8)
        inputs = (torch.randn(1, 8, 4, 6),)
        self._check_rewritten(
            module,
            inputs,
            {self.conv: 4, self.pad: 3, self.cat: 1, self.d2s: 1, self.slice: 2},
        )
        self._check_lowered(module, inputs)

    def test_left_alone(self):
        for name, module, shape in (
            ("grouped", self._deconv(1, 4, 4, 4, 2, groups=2), (1, 4, 5)),
            ("dilated", self._deconv(1, 4, 4, 4, 2, dilation=2), (1, 4, 5)),
            ("kernel not a multiple", self._deconv(1, 4, 4, 3, 2), (1, 4, 5)),
            ("stride 1", self._deconv(1, 4, 4, 2, 1), (1, 4, 5)),
            (
                "output_padding > padding",
                self._deconv(1, 4, 4, 4, 2, output_padding=1),
                (1, 4, 5),
            ),
            ("unequal strides", self._deconv(2, 4, 4, (4, 2), (2, 1)), (1, 4, 5, 5)),
            (
                "depthwise with a negative pad",
                self._deconv(1, 4, 4, 3, 2, padding=2, groups=4),
                (1, 4, 5),
            ),
        ):
            with self.subTest(name):
                self._check_left_alone(module, (torch.randn(*shape),))

    def test_dynamic_shape_is_left_alone(self):
        module = self._deconv(1, 4, 4, 4, 2)
        self._check_left_alone(
            module,
            (torch.randn(1, 4, 6),),
            dynamic_shapes=({2: torch.export.Dim("steps", min=2, max=16)},),
        )

    def test_dequantized_weight_is_left_alone(self):
        class QuantizedDeconv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "weight", torch.randint(-128, 127, (4, 4, 4), dtype=torch.int8)
                )

            def forward(self, x):
                weight = torch.ops.quantized_decomposed.dequantize_per_tensor(
                    self.weight, 0.02, 0, -128, 127, torch.int8
                )
                return torch.nn.functional.conv_transpose1d(x, weight, stride=2)

        self._check_left_alone(QuantizedDeconv(), (torch.randn(1, 4, 6),))
