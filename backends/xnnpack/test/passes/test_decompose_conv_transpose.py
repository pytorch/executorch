# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import unittest
from unittest import mock

import executorch.backends.test.harness.stages as BaseStages
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

    # Delegated nodes of one deconv on an NCHW input: the layout conversions
    # into and out of the conv, and the 1-D interleave's permute.
    dense_1d = {"XNNConv2d": 1, "XNNStaticTranspose": 3}
    dense_2d = {"XNNConv2d": 1, "XNNStaticTranspose": 2}

    @staticmethod
    def twice(nodes):
        return {k: 2 * v for k, v in nodes.items()}

    @staticmethod
    def depthwise(phases, transposes=2, **nodes):
        return {
            "XNNDepthwiseConv2d": phases,
            "XNNStaticTranspose": transposes,
            **nodes,
        }

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

    def _check_rewritten(self, module, inputs, counts, tol=1e-4, dynamic_shapes=None):
        tester = self._run_pass(module, inputs, dynamic_shapes).check_count(counts)
        self.assertEqual(self._transposed_convs(tester), 0)
        tester.run_method_and_compare_outputs(atol=tol, rtol=tol)
        return tester

    def _check_left_alone(self, module, inputs, dynamic_shapes=None):
        tester = self._run_pass(module, inputs, dynamic_shapes)
        self.assertEqual(self._transposed_convs(tester), 1)

    @staticmethod
    def _count_decompose():
        return mock.patch.object(
            DecomposeConvTransposePass,
            "_decompose",
            autospec=True,
            side_effect=DecomposeConvTransposePass._decompose,
        )

    def _check_lowered(
        self, module, inputs, nodes, tol=1e-4, num_deconvs=1, dynamic_shapes=None
    ):
        # XNNPACK delegates these transposed convs on their own, so one
        # delegate and matching outputs do not prove the pass ran during
        # lowering: count its calls and check the serialized graph.
        tester = Tester(module.eval(), inputs, dynamic_shapes=dynamic_shapes).export()
        with self._count_decompose() as decompose:
            tester.to_edge_transform_and_lower()
        self.assertEqual(decompose.call_count, num_deconvs)
        tester.check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
        node_types = tester.xnnpack_node_types()
        expected = {
            "XNNConvTranspose2d": 0,
            "XNNConv2d": 0,
            "XNNDepthwiseConv2d": 0,
            "XNNClamp": 0,
            **nodes,
        }
        self.assertEqual(
            {k: node_types[k] for k in expected}, expected, dict(node_types)
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
                    self._check_lowered(module, inputs, self.dense_1d)

    def test_1d_single_step(self):
        # Streaming decoders run one input step per call.
        module = self._deconv(1, 8, 4, 8, 4)
        inputs = (torch.randn(1, 8, 1),)
        self._check_rewritten(module, inputs, {self.conv: 1, self.view: 2})
        self._check_lowered(module, inputs, self.dense_1d)

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
                self._check_lowered(module, inputs, self.dense_1d)

    def test_1d_kernel_multiple_of_stride(self):
        # m = 1 needs no padding, m = 3 pads the conv by 2.
        for kernel, stride in ((3, 3), (6, 2), (9, 3)):
            with self.subTest(kernel=kernel, stride=stride):
                module = self._deconv(1, 4, 6, kernel, stride)
                inputs = (torch.randn(1, 4, 5),)
                self._check_rewritten(module, inputs, {self.conv: 1, self.view: 2})
                self._check_lowered(module, inputs, self.dense_1d)

    def test_2d(self):
        for kernel, stride, padding, output_padding, slices in (
            (4, 2, 1, 0, 2),
            ((4, 2), 2, 0, 0, 0),
            (6, 3, (1, 2), (1, 0), 2),
            (2, 2, 0, 0, 0),
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
                self._check_lowered(module, inputs, self.dense_2d)

    def test_fp16(self):
        for name, module, shape, counts, nodes in (
            (
                "1d",
                self._deconv(1, 8, 4, 8, 4),
                (6,),
                {self.conv: 1, self.view: 2},
                self.dense_1d,
            ),
            (
                "2d",
                self._deconv(2, 4, 6, 4, 2, padding=1),
                (5, 7),
                {self.conv: 1, self.d2s: 1, self.slice: 2},
                self.dense_2d,
            ),
        ):
            with self.subTest(name):
                module = module.half()
                inputs = (torch.randn(1, module.in_channels, *shape).half(),)
                self._check_rewritten(module, inputs, counts, tol=1e-2)
                self._check_lowered(module, inputs, nodes, tol=1e-2)

    def test_two_deconvs_in_a_row(self):
        # The second deconv's input is the first one's interleave.
        module = torch.nn.Sequential(
            self._deconv(1, 16, 8, 12, 6), self._deconv(1, 8, 4, 10, 5)
        )
        inputs = (torch.randn(1, 16, 3),)
        self._check_rewritten(module, inputs, {self.conv: 2, self.view: 4})
        self._check_lowered(module, inputs, self.twice(self.dense_1d), num_deconvs=2)

    def test_two_deconvs_with_activation(self):
        module = torch.nn.Sequential(
            self._deconv(1, 16, 8, 12, 6),
            torch.nn.ELU(),
            self._deconv(1, 8, 4, 10, 5),
        )
        inputs = (torch.randn(1, 16, 3),)
        self._check_rewritten(module, inputs, {self.conv: 2, self.view: 4})
        self._check_lowered(module, inputs, self.twice(self.dense_1d), num_deconvs=2)

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
        self._check_lowered(module, inputs, self.twice(self.dense_1d), num_deconvs=2)

    class DeconvTwoStrides(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.deconv = torch.nn.ConvTranspose1d(6, 4, 8)

        def forward(self, x):
            w, b = self.deconv.weight, self.deconv.bias
            return (
                torch.nn.functional.conv_transpose1d(x, w, b, stride=2),
                torch.nn.functional.conv_transpose1d(x, w, b, stride=4),
            )

    def test_shared_weight_at_two_strides(self):
        # Each stride gets its own phase weight and phase bias.
        module = self.DeconvTwoStrides()
        inputs = (torch.randn(1, 6, 5),)
        tester = self._check_rewritten(module, inputs, {self.conv: 2, self.view: 4})
        placeholders = [
            node.name
            for node in tester.get_artifact(StageType.RUN_PASSES)
            .exported_program()
            .graph_module.graph.nodes
            if node.op == "placeholder"
        ]
        for suffix in ("_s2_phases", "_s4_phases", "_x2", "_x4"):
            self.assertEqual(sum(n.endswith(suffix) for n in placeholders), 1, suffix)
        self._check_lowered(module, inputs, self.twice(self.dense_1d), num_deconvs=2)

    def test_fused_activation(self):
        # The activation moves onto every phase conv, so XNNPACK fuses it
        # there instead of running a clamp over the upsampled output.
        for name, deconv, activation, shape, nodes in (
            (
                "1d relu",
                self._deconv(1, 6, 3, 8, 4, padding=2),
                torch.nn.ReLU(),
                (7,),
                self.dense_1d,
            ),
            (
                "2d hardtanh",
                self._deconv(2, 4, 6, 4, 2, padding=1),
                torch.nn.Hardtanh(-0.5, 0.5),
                (5, 7),
                self.dense_2d,
            ),
            (
                "2d relu6",
                self._deconv(2, 4, 6, 2, 2),
                torch.nn.ReLU6(),
                (5, 7),
                self.dense_2d,
            ),
            (
                "depthwise 1d relu",
                self._deconv(1, 8, 8, 4, 2, padding=1, groups=8),
                torch.nn.ReLU(),
                (5,),
                self.depthwise(2),
            ),
            (
                "depthwise 2d hardtanh",
                self._deconv(2, 8, 8, 5, 2, groups=8),
                torch.nn.Hardtanh(-0.5, 0.5),
                (4, 6),
                self.depthwise(4, transposes=5),
            ),
        ):
            with self.subTest(name):
                module = torch.nn.Sequential(deconv, activation)
                inputs = (torch.randn(1, deconv.in_channels, *shape),)
                phases = nodes.get("XNNConv2d", 0) + nodes.get("XNNDepthwiseConv2d", 0)
                tester = self._check_rewritten(module, inputs, {self.conv: phases})
                acts = [
                    n
                    for n in tester.get_artifact(StageType.RUN_PASSES)
                    .exported_program()
                    .graph_module.graph.nodes
                    if n.target in FuseActivationPass.FUSEABLE_ACTIVATIONS
                ]
                self.assertEqual(
                    [act.args[0].target for act in acts],
                    [exir_ops.edge.aten.convolution.default] * phases,
                )
                self._check_lowered(module, inputs, nodes)

    def test_fused_activation_chain(self):
        module = torch.nn.Sequential(
            self._deconv(1, 6, 3, 8, 4, padding=2),
            torch.nn.ReLU(),
            torch.nn.Hardtanh(-0.5, 0.5),
        )
        inputs = (torch.randn(1, 6, 7),)
        tester = self._check_rewritten(module, inputs, {self.conv: 1})
        relu, hardtanh = [
            n
            for n in tester.get_artifact(StageType.RUN_PASSES)
            .exported_program()
            .graph_module.graph.nodes
            if n.target in FuseActivationPass.FUSEABLE_ACTIVATIONS
        ]
        self.assertEqual(relu.args[0].target, exir_ops.edge.aten.convolution.default)
        self.assertIs(hardtanh.args[0], relu)
        self._check_lowered(module, inputs, self.dense_1d)

    def test_activation_with_other_users_is_not_moved(self):
        class DeconvReluAndSkip(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.deconv = torch.nn.ConvTranspose1d(6, 4, 8, stride=4)

            def forward(self, x):
                y = self.deconv(x)
                return torch.relu(y) + y

        module = DeconvReluAndSkip()
        inputs = (torch.randn(1, 6, 5),)
        self._check_rewritten(module, inputs, {self.conv: 1, self.view: 2})
        self._check_lowered(module, inputs, {**self.dense_1d, "XNNClamp": 1})

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
        self._check_lowered(module, inputs, self.dense_1d)

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
        # constants they replace. The original weight and bias would make two
        # entries as well, so check that the pass replaced them.
        module = self._deconv(1, 6, 3, 8, 4).eval()
        inputs = (torch.randn(1, 6, 7),)
        tester = Tester(module, inputs)
        tester.export(self.ExportWithExternalConstants())
        with self._count_decompose() as decompose:
            tester.to_edge_transform_and_lower()
        self.assertEqual(decompose.call_count, 1)
        self.assertEqual(tester.xnnpack_node_types()["XNNConvTranspose2d"], 0)
        tester.check_count(
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

    def test_depthwise_1d(self):
        # Zero taps make every phase pad symmetric, so the phases take it as
        # their padding and need no pad of the NCHW input.
        for kernel, stride, padding, output_padding, steps, slices in (
            # The upsampler of a streaming audio decoder.
            (4, 2, 0, 0, 1, 0),
            (32, 16, 0, 0, 5, 0),
            (32, 16, 8, 0, 5, 0),
            (10, 5, 3, 1, 4, 0),
            # The odd output sizes need a slice.
            (4, 2, 1, 1, 5, 1),
            (4, 2, 0, 1, 5, 1),
        ):
            with self.subTest(
                kernel=kernel, stride=stride, padding=padding, steps=steps
            ):
                module = self._deconv(
                    1,
                    8,
                    8,
                    kernel,
                    stride,
                    padding=padding,
                    output_padding=output_padding,
                    groups=8,
                )
                inputs = (torch.randn(2, 8, steps),)
                self._check_rewritten(
                    module,
                    inputs,
                    {
                        self.conv: stride,
                        self.pad: 0,
                        self.cat: 1,
                        self.view: 1,
                        self.permute: 0,
                        self.slice: slices,
                    },
                )
                self._check_lowered(module, inputs, self.depthwise(stride))

    class DepthwiseDeconvTwoOutputPaddings(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(8, 1, 4))

        def forward(self, x):
            return tuple(
                torch.nn.functional.conv_transpose1d(
                    x, self.weight, stride=2, padding=1, output_padding=o, groups=8
                )
                for o in (0, 1)
            )

    def test_depthwise_1d_shared_weight_at_two_output_paddings(self):
        # The zero taps depend on the output padding, so each deconv gets its
        # own phase weights.
        module = self.DepthwiseDeconvTwoOutputPaddings()
        inputs = (torch.randn(1, 8, 5),)
        tester = self._check_rewritten(module, inputs, {self.conv: 4, self.pad: 0})
        placeholders = [
            node.name
            for node in tester.get_artifact(StageType.RUN_PASSES)
            .exported_program()
            .graph_module.graph.nodes
            if node.op == "placeholder" and "_phase_" in node.name
        ]
        self.assertEqual(len(placeholders), 4, placeholders)
        self._check_lowered(module, inputs, self.depthwise(4, 4), num_deconvs=2)

    def test_depthwise_2d(self):
        # The NCHW input is converted once for the unpadded phase and once
        # for each pad.
        for kernel, padding, output_padding, pads in (
            # 2H + 3 outputs, sliced from 2H + 4
            (5, 0, 0, 3),
            (4, 0, 1, 4),
        ):
            with self.subTest(
                kernel=kernel, padding=padding, output_padding=output_padding
            ):
                module = self._deconv(
                    2,
                    8,
                    8,
                    kernel,
                    2,
                    padding=padding,
                    output_padding=output_padding,
                    groups=8,
                )
                inputs = (torch.randn(1, 8, 4, 6),)
                self._check_rewritten(
                    module,
                    inputs,
                    {
                        self.conv: 4,
                        self.pad: pads,
                        self.cat: 1,
                        self.d2s: 1,
                        self.slice: 2,
                    },
                )
                self._check_lowered(module, inputs, self.depthwise(4, 5))

    def test_single_channel_2d(self):
        # k=3 gives the dense plan unequal phase pads, so it is lowered as
        # depthwise; k=4 stays dense.
        for kernel, counts, nodes in (
            (3, {self.conv: 4, self.cat: 1, self.d2s: 1}, self.depthwise(4, 5)),
            # XNNPACK takes a conv with one input channel as depthwise.
            (4, {self.conv: 1, self.cat: 0, self.d2s: 1}, self.depthwise(1)),
        ):
            with self.subTest(kernel=kernel):
                module = self._deconv(
                    2, 1, 1, kernel, 2, padding=1, output_padding=kernel % 2
                )
                inputs = (torch.randn(1, 1, 6, 6),)
                self._check_rewritten(module, inputs, counts)
                self._check_lowered(module, inputs, nodes)

    def test_2d_dynamic_shape(self):
        # Neither the 2-D interleave nor a phase pad needs explicit sizes.
        dynamic_shapes = (
            {
                2: torch.export.Dim("height", min=2, max=16),
                3: torch.export.Dim("width", min=2, max=16),
            },
        )
        for name, module, counts, nodes in (
            # A U-Net upsampler
            ("dense", self._deconv(2, 8, 4, 2, 2), {self.conv: 1}, self.dense_2d),
            (
                "depthwise",
                self._deconv(2, 8, 8, 4, 2, groups=8),
                {self.conv: 4, self.pad: 0},
                self.depthwise(4),
            ),
        ):
            with self.subTest(name):
                inputs = (torch.randn(1, 8, 6, 10),)
                self._check_rewritten(
                    module,
                    inputs,
                    {**counts, self.d2s: 1, self.slice: 0},
                    dynamic_shapes=dynamic_shapes,
                )
                self._check_lowered(
                    module, inputs, nodes, dynamic_shapes=dynamic_shapes
                )

    def test_phase_math_matches_conv_transpose2d(self):
        # The plan's taps, pads and crops in plain torch, independent of the
        # lowering.
        import executorch.backends.xnnpack.custom_ops_lib as lib

        for groups, k, s, p, o in [
            (1, 4, 2, 1, 0),
            (1, 4, 2, 1, 1),
            (1, 6, 3, 2, 1),
            (1, 2, 2, 0, 0),
            (3, 4, 2, 1, 0),
            (3, 3, 2, 1, 1),
            (3, 3, 3, 0, 0),
            (3, 5, 2, 2, 1),
            # extra > 0
            (3, 4, 2, 0, 0),
            # a crop
            (3, 5, 2, 0, 0),
            (3, 4, 2, 0, 1),
            (3, 6, 4, 1, 3),
        ]:
            x = torch.randn(1, 3, 5, 6)
            weight = torch.randn(3, 3 // groups, k, k)
            ref = torch.nn.functional.conv_transpose2d(
                x, weight, stride=s, padding=p, output_padding=o, groups=groups
            )
            axes = [
                DecomposeConvTransposePass._axis(n, k, s, p, o, dense=groups == 1)
                for n in x.shape[2:]
            ]
            assert axes[0] is not None and axes[1] is not None
            phases = []
            for ay, ax in itertools.product(range(s), repeat=2):
                w = weight[:, :, axes[0].taps[ay]][:, :, :, axes[1].taps[ax]]
                xp = torch.nn.functional.pad(x, (*axes[1].pads[ax], *axes[0].pads[ay]))
                phases.append(
                    torch.nn.functional.conv2d(
                        xp, w if groups > 1 else w.transpose(0, 1), groups=groups
                    )
                )
            out = lib.depth_to_space_2d_op(torch.cat(phases, 1), s)
            for dim, axis in enumerate(axes, 2):
                if axis.crop is not None:
                    start, end = axis.crop
                    out = out.narrow(dim, start, end - start)
            self.assertEqual(out.shape, ref.shape, (groups, k, s, p, o))
            self.assertTrue(torch.allclose(out, ref, atol=1e-5), (groups, k, s, p, o))

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
            ("kernel < stride", self._deconv(1, 4, 4, 2, 4), (1, 4, 5)),
            (
                "depthwise kernel < stride",
                self._deconv(1, 4, 4, 2, 3, groups=4),
                (1, 4, 5),
            ),
        ):
            with self.subTest(name):
                self._check_left_alone(module, (torch.randn(*shape),))

    def test_bias_input_is_left_alone(self):
        class DeconvBiasInput(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(4, 4, 4))

            def forward(self, x, bias):
                return torch.nn.functional.conv_transpose1d(
                    x, self.weight, bias, stride=2
                )

        self._check_left_alone(
            DeconvBiasInput(), (torch.randn(1, 4, 5), torch.randn(4))
        )

    def test_dynamic_shape_needing_explicit_sizes_is_left_alone(self):
        # The 1-D interleave and the slice take explicit sizes.
        for name, module, shape in (
            ("1d", self._deconv(1, 4, 4, 4, 2), (1, 4, 6)),
            ("1d depthwise", self._deconv(1, 4, 4, 4, 2, groups=4), (1, 4, 6)),
            ("2d slice", self._deconv(2, 4, 4, 4, 2, padding=1), (1, 4, 6, 6)),
            ("2d depthwise slice", self._deconv(2, 4, 4, 5, 2, groups=4), (1, 4, 6, 6)),
        ):
            with self.subTest(name):
                dims = len(shape) - 2
                dynamic_shapes = (
                    {
                        2 + i: torch.export.Dim(f"d{i}", min=2, max=16)
                        for i in range(dims)
                    },
                )
                self._check_left_alone(module, (torch.randn(*shape),), dynamic_shapes)

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
