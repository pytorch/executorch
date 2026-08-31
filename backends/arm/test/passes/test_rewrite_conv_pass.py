# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os
from typing import Any, cast

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from executorch.backends.arm._passes import (
    ConvertToClampPass,
    FoldAndAnnotateQParamsPass,
    FuseQuantizedActivationPass,
    InsertRescaleInt32Pass,
    QuantizeClampArgumentsPass,
)
from executorch.backends.arm._passes.rewrite_conv_pass import RewriteConvPass
from executorch.backends.arm.quantizer.arm_quantizer import (
    get_symmetric_a16w8_quantization_config,
    get_symmetric_quantization_config,
    VgfQuantizer,
)
from executorch.backends.arm.test.misc.test_dw_convs_with_shared_weights import (
    DWConvsModule,
)
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.tosa.mapping import TosaSpecialDtype
from executorch.backends.arm.tosa.partitioner import TOSAPartitioner
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.backends.arm.vgf import VgfCompileSpec, VgfPartitioner
from executorch.exir import EdgeCompileConfig, to_edge, to_edge_transform_and_lower
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import Dim, export
from torch.export.exported_program import _get_shape_env

_VGF_ENABLED = "LAVAPIPE_LIB_PATH" in os.environ


class TinyConvReluCat(nn.Module):
    def __init__(self, conv1_bias: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(4, 4, 3, padding=1, bias=conv1_bias)
        self.conv2 = nn.Conv2d(8, 4, 1)
        with torch.no_grad():
            for param in self.parameters():
                param.uniform_(-0.1, 0.1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        relu_out = F.relu(self.conv1(x))
        merged = torch.cat((relu_out, y), dim=1)
        return self.conv2(merged)


class A16W8MixedConsumerChain(nn.Module):
    """Exercise a shared A16W8 convolution output with mixed consumers."""

    def __init__(self, following: nn.Module) -> None:
        super().__init__()
        self.expand = nn.Conv2d(4, 8, 1)
        self.depthwise = nn.Conv2d(8, 8, 3, padding=1, groups=8)
        self.project = nn.Conv2d(8, 4, 1)
        self.following = following

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Use a projection in both an addition and another operator."""
        expanded = F.relu(self.expand(x))
        filtered = F.relu(self.depthwise(expanded))
        projected = self.project(filtered)
        residual = projected + x
        return self.following(projected) + residual


class A16W8Int32Consumer(nn.Module):
    """Exercise an A16W8 convolution consumed only by an INT32 add."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed the convolution output directly to a residual addition."""
        return self.conv(x) + x


class A16W8MixedConsumer(nn.Module):
    """Exercise a shared A16W8 convolution output with mixed consumers."""

    def __init__(self, producer: nn.Module, following: nn.Module) -> None:
        super().__init__()
        self.producer = producer
        self.following = following

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Feed the convolution output to both INT16 and INT32 consumers."""
        produced = self.producer(x)
        return self.following(produced) + produced + x


class A16W8PermutedInt32Consumer(nn.Module):
    """Exercise an indirect INT32 consumer behind a permutation."""

    def __init__(self) -> None:
        super().__init__()
        self.producer = nn.Conv2d(4, 4, 1)
        self.following = nn.Conv2d(4, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Share the convolution between INT16 and permuted INT32 paths."""
        produced = self.producer(x)
        permuted = produced.permute(0, 1, 3, 2)
        return self.following(produced) + permuted + x


def _example_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.rand(1, 4, 16, 16)
    y = torch.rand(1, 4, 16, 16) - 0.065
    return x, y


def _compile_spec() -> VgfCompileSpec:
    return VgfCompileSpec("TOSA-1.0+INT+FP")


def _compile_spec_int16() -> VgfCompileSpec:
    return VgfCompileSpec("TOSA-1.0+INT+FP+int16")


def _quantizer() -> VgfQuantizer:
    quantizer = VgfQuantizer(_compile_spec())
    quantizer.set_global(
        get_symmetric_quantization_config(
            is_per_channel=True,
            act_qmin=-127,
            act_qmax=127,
            weight_qmin=-127,
            weight_qmax=127,
        )
    )
    return quantizer


def _export_quantized(model: nn.Module):
    inputs = _example_inputs()
    exported = torch.export.export(model.eval(), inputs).module(check_guards=False)
    quantized = _quantizer()._quantize_with_submodules(exported, [inputs])
    return torch.export.export(quantized, inputs)


def _export_quantized_a16w8(model: nn.Module, inputs: tuple[torch.Tensor, ...]):
    exported = torch.export.export(model.eval(), inputs).module(check_guards=False)
    quantizer = VgfQuantizer(_compile_spec_int16())
    quantizer.set_global(get_symmetric_a16w8_quantization_config())
    quantized = quantizer._quantize_with_submodules(exported, [inputs])
    return torch.export.export(quantized, inputs)


def _lower_a16w8_to_tosa(model: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
    """Lower a VGF-quantized A16W8 model through the TOSA partitioner."""
    compile_spec = TosaCompileSpec("TOSA-1.0+INT+int16")
    to_edge_transform_and_lower(
        _export_quantized_a16w8(model, inputs),
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
        partitioner=[TOSAPartitioner(compile_spec)],
    )


def _run_pre_rewrite_passes(exported_program: torch.export.ExportedProgram):
    gm = exported_program.graph_module
    for pass_ in (
        FuseQuantizedActivationPass(),
        ConvertToClampPass(),
        FoldAndAnnotateQParamsPass(exported_program),
        QuantizeClampArgumentsPass(),
    ):
        result = pass_(gm)
        assert result is not None
        gm = result.graph_module
    return gm


def _get_expected_int32_scales(
    gm: torch.fx.GraphModule, rewrite_pass: RewriteConvPass
) -> list[list[float]]:
    """Calculate expected scales for direct INT32 convolution consumers.

    Args:
        gm (torch.fx.GraphModule): Graph containing rescaled convolutions.
        rewrite_pass (RewriteConvPass): Pass used to identify INT32 consumers.

    Returns:
        list[list[float]]: Expected per-consumer rescale values.

    """
    expected_int32_scales = []
    for node in gm.graph.nodes:
        if node.target != exir_ops.edge.aten.convolution.default:
            continue
        for int32_user in rewrite_pass._get_direct_int32_rescale_users(node):
            input_qparams = node.meta["input_qparams"]
            output_qparams = node.meta["output_qparams"][0]
            activation_qparams = input_qparams[0]
            weight_qparams = input_qparams[1]
            weight_scales = (
                weight_qparams.get_scale_per_channel()
                if weight_qparams.per_channel
                else [weight_qparams.get_scale_per_tensor()]
            )
            activation_scale = activation_qparams.get_scale_per_tensor()
            output_scale = output_qparams.get_scale_per_tensor()
            assert output_qparams.get_zp_per_tensor() == int32_user.args[3]
            user_scales = cast(list[float], int32_user.args[2])
            assert len(user_scales) in (1, len(weight_scales))
            # Derive this independently from the rewrite helper:
            #
            #   accumulator -> declared output -> INT32 consumer
            #   (input * weight / output) * consumer
            expected_int32_scales.append(
                [
                    activation_scale * weight_scale / output_scale * user_scale
                    for weight_scale, user_scale in zip(
                        weight_scales,
                        itertools.cycle(user_scales),
                    )
                ]
            )
    return expected_int32_scales


def _rewrite_a16w8_convs(
    model: nn.Module, inputs: tuple[torch.Tensor, ...]
) -> tuple[torch.fx.GraphModule, list[list[float]]]:
    """Run the passes needed to inspect rewritten A16W8 convolutions."""
    exported_program = _export_quantized_a16w8(model, inputs)
    edge_program = to_edge(
        exported_program, compile_config=EdgeCompileConfig(_check_ir_validity=False)
    ).exported_program()
    gm = _run_pre_rewrite_passes(edge_program)
    rewrite_pass = RewriteConvPass(edge_program)
    with TosaLoweringContext(_compile_spec_int16().tosa_spec):
        rescale_result = InsertRescaleInt32Pass()(gm)
        assert rescale_result is not None
        expected_int32_scales = _get_expected_int32_scales(
            rescale_result.graph_module, rewrite_pass
        )
        rewrite_result = rewrite_pass(rescale_result.graph_module)
        assert rewrite_result is not None
    return rewrite_result.graph_module, expected_int32_scales


def _get_call_function_node(gm: torch.fx.GraphModule, target):
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == target:
            return node
    raise AssertionError(f"Node with target {target} not found")


class ConvModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=3, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv2dBiasModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 6, kernel_size=3, stride=1, padding=1, bias=True)

    def get_inputs(self) -> tuple[torch.Tensor]:
        return (torch.randn(1, 4, 8, 8),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DepthwiseConv2dBiasModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, kernel_size=3, padding=1, groups=4, bias=True)

    def get_inputs(self) -> tuple[torch.Tensor]:
        return (torch.randn(1, 4, 8, 8),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv3dBiasModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv3d(3, 5, kernel_size=3, stride=1, padding=1, bias=True)

    def get_inputs(self) -> tuple[torch.Tensor]:
        return (torch.randn(1, 3, 6, 6, 6),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TransposeConv2dBiasModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(
            3,
            4,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=True,
        )

    def get_inputs(self) -> tuple[torch.Tensor]:
        return (torch.randn(1, 3, 6, 6),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _make_rewrite_pass(
    example_inputs: tuple[torch.Tensor, ...],
    dynamic_shapes: dict[int, object] | None = None,
) -> tuple[RewriteConvPass, object, int | torch.SymInt]:
    if dynamic_shapes is None:
        ep = export(ConvModule(), example_inputs)
    else:
        ep = export(ConvModule(), example_inputs, dynamic_shapes={"x": dynamic_shapes})
    edge_model = to_edge(ep)
    gm = edge_model.exported_program().graph_module
    conv_node = next(
        n for n in gm.graph.nodes if n.target == exir_ops.edge.aten.convolution.default
    )
    input_len = conv_node.args[0].meta["val"].shape[2]
    return RewriteConvPass(edge_model.exported_program()), _get_shape_env(gm), input_len


def _multiples_of_three_dynamic_shapes() -> dict[int, object]:
    return {
        2: Dim("height", min=2, max=6) * 3,
        3: Dim("width", min=2, max=6) * 3,
    }


def test_rewrite_conv_tosa_FP():
    module = DWConvsModule()
    pipeline = PassPipeline(
        module, module.get_inputs(), passes_with_exported_program=[RewriteConvPass]
    )
    pipeline.run()


@pytest.mark.parametrize(
    ("model", "inputs", "tosa_target"),
    (
        (
            A16W8MixedConsumerChain(nn.Conv2d(4, 4, 1)),
            (torch.randn(1, 4, 8, 8),),
            exir_ops.backend.tosa.CONV2D.default,
        ),
        (
            A16W8MixedConsumerChain(nn.Conv2d(4, 4, 3, padding=1, groups=4)),
            (torch.randn(1, 4, 8, 8),),
            exir_ops.backend.tosa.CONV2D.default,
        ),
        (
            A16W8MixedConsumerChain(nn.AvgPool2d(3, stride=1, padding=1)),
            (torch.randn(1, 4, 8, 8),),
            exir_ops.backend.tosa.CONV2D.default,
        ),
        (
            A16W8MixedConsumerChain(nn.MaxPool2d(3, stride=1, padding=1)),
            (torch.randn(1, 4, 8, 8),),
            exir_ops.backend.tosa.CONV2D.default,
        ),
        (
            A16W8MixedConsumer(
                nn.Conv3d(4, 4, 3, padding=1),
                nn.Conv3d(4, 4, 1),
            ),
            (torch.randn(1, 4, 4, 8, 8),),
            exir_ops.backend.tosa.CONV3D.default,
        ),
        (
            A16W8MixedConsumer(
                nn.ConvTranspose2d(4, 4, 3, padding=1),
                nn.Conv2d(4, 4, 1),
            ),
            (torch.randn(1, 4, 8, 8),),
            exir_ops.backend.tosa.TRANSPOSE_CONV2D.default,
        ),
    ),
    ids=(
        "conv",
        "depthwise_conv",
        "avg_pool",
        "max_pool",
        "conv3d",
        "transpose_conv2d",
    ),
)
def test_rewrite_conv_a16w8_mixed_consumers_restore_int16(
    model: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    tosa_target: Any,
) -> None:
    r"""Test that mixed consumers branch at the INT48 accumulator.

                              +-> combined RESCALE (INT32) -> PERMUTE
                              |                              -> residual ADD
        A16W8 CONV ----------+
        accumulator (INT48)  +-> RESCALE (INT16) -> PERMUTE
                                                             -> following op

    The INT32 branch must consume the convolution output directly. Routing it
    through the INT16 rescale first would irreversibly lose accumulator range.
    The parameterized following operation still receives INT16. This applies
    to all convolution variants supported by the rewrite pass.

    Args:
        model (nn.Module): Model containing the convolution under test.
        inputs (tuple[torch.Tensor, ...]): Inputs used to export the model.
        tosa_target (Any): Expected rewritten TOSA convolution target.

    """
    _lower_a16w8_to_tosa(model, inputs)
    gm, expected_int32_scales = _rewrite_a16w8_convs(model, inputs)

    mixed_output_convs = [
        node
        for node in gm.graph.nodes
        if node.target == tosa_target
        and len(node.users) == 2
        and {
            user.args[1]
            for user in node.users
            if user.target == exir_ops.backend.tosa.RESCALE.default
        }
        == {torch.int16, torch.int32}
    ]
    assert len(mixed_output_convs) == 1
    int32_rescale = next(
        user for user in mixed_output_convs[0].users if user.args[1] == torch.int32
    )
    assert any(
        int32_rescale.args[2] == pytest.approx(expected)
        for expected in expected_int32_scales
    )


def test_rewrite_conv_a16w8_preserves_int32_for_int32_consumers() -> None:
    r"""Test that an exclusively INT32 consumer keeps the widened path.

        x (INT16) -> RESCALE (INT32) -------------------------+
                                                              |
        x (INT16) -> CONV (INT48)                              ADD
                          -> combined RESCALE (INT32)           |
                          -> PERMUTE ---------------------------+

    No INT16 rescale is needed because every convolution consumer accepts
    INT32. Combining both rescale factors converts directly from INT48 to the
    addition's INT32 domain without changing the represented values.

    """
    inputs = (torch.randn(1, 4, 8, 8),)
    gm, expected_int32_scales = _rewrite_a16w8_convs(A16W8Int32Consumer(), inputs)

    direct_int32_rescales = [
        node
        for node in gm.graph.nodes
        if node.op == "call_function"
        and node.target == exir_ops.backend.tosa.RESCALE.default
        and node.args[1] == torch.int32
        and node.all_input_nodes[0].target
        in {
            exir_ops.backend.tosa.CONV2D.default,
            exir_ops.backend.tosa.DEPTHWISE_CONV2D.default,
        }
    ]
    assert len(direct_int32_rescales) == len(expected_int32_scales) == 1
    assert direct_int32_rescales[0].args[2] == pytest.approx(expected_int32_scales[0])


def test_rewrite_conv_a16w8_preserves_int32_after_permute() -> None:
    r"""Test that an indirect INT32 consumer keeps a widened branch.

        CONV (INT48) -+-> RESCALE (INT16) -> PERMUTE -> following CONV
                      |
                      +-> RESCALE (INT32) -> layout PERMUTE
                          -> source PERMUTE -> RESCALE (INT32) -> ADD

    The source permutation cannot operate on INT48. The first rescale widens
    to the declared convolution domain in INT32, avoiding INT16 rounding while
    leaving the following rescale on the correct side of the permutation.

    """
    inputs = (torch.randn(1, 4, 8, 8),)
    model = A16W8PermutedInt32Consumer()
    _lower_a16w8_to_tosa(model, inputs)
    gm, _ = _rewrite_a16w8_convs(model, inputs)

    widened_paths: list[torch.fx.Node] = []
    for node in gm.graph.nodes:
        if (
            node.target != exir_ops.backend.tosa.RESCALE.default
            or node.args[1] != torch.int32
            or node.all_input_nodes[0].target != exir_ops.backend.tosa.CONV2D.default
        ):
            continue
        for layout_permute in node.users:
            if layout_permute.target != exir_ops.edge.aten.permute_copy.default:
                continue
            for source_permute in layout_permute.users:
                if source_permute.target != exir_ops.edge.aten.permute_copy.default:
                    continue
                widened_paths.extend(
                    consumer
                    for consumer in source_permute.users
                    if consumer.target == exir_ops.backend.tosa.RESCALE.default
                    and consumer.args[1] == torch.int32
                )

    assert len(widened_paths) == 1


@pytest.mark.skipif(not _VGF_ENABLED, reason="VGF not enabled")
def test_fold_and_annotate_q_params_vgf_quant_tracks_fused_relu_qparams() -> None:
    exported_program = _export_quantized(TinyConvReluCat())
    gm = _run_pre_rewrite_passes(to_edge(exported_program).exported_program())

    conv = _get_call_function_node(gm, exir_ops.edge.aten.convolution.default)
    output_qparams = conv.meta["output_qparams"][0]

    assert conv.meta["input_qparams"]
    assert output_qparams.qmin == output_qparams.zp
    assert not any(
        node.target == exir_ops.edge.aten.clamp.default for node in gm.graph.nodes
    )


@pytest.mark.skipif(not _VGF_ENABLED, reason="VGF not enabled")
def test_rewrite_conv_vgf_quant_handles_fused_conv_relu_cat_branch() -> None:
    exported_program = _export_quantized(TinyConvReluCat())
    compile_spec = _compile_spec()

    to_edge_transform_and_lower(
        exported_program,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
        partitioner=[VgfPartitioner(compile_spec)],
    )


@pytest.mark.skipif(not _VGF_ENABLED, reason="VGF not enabled")
def test_rewrite_conv_vgf_quant_infers_quantized_bias_dtype_from_inputs() -> None:
    exported_program = _export_quantized(TinyConvReluCat(conv1_bias=False))
    edge_program = to_edge(
        exported_program, compile_config=EdgeCompileConfig(_check_ir_validity=False)
    ).exported_program()
    gm = _run_pre_rewrite_passes(edge_program)
    with TosaLoweringContext(_compile_spec().tosa_spec):
        result = RewriteConvPass(edge_program)(gm)
        assert result is not None
        gm = result.graph_module

    bias_nodes = [
        node
        for node in gm.graph.nodes
        if node.op == "placeholder" and node.name.endswith("_bias")
    ]

    assert len(bias_nodes) == 1
    assert bias_nodes[0].meta["val"].dtype == torch.int32


@pytest.mark.parametrize(
    "module,target_op",
    [
        (Conv2dBiasModule(), exir_ops.backend.tosa.CONV2D.default),
        (DepthwiseConv2dBiasModule(), exir_ops.backend.tosa.DEPTHWISE_CONV2D.default),
        (Conv3dBiasModule(), exir_ops.backend.tosa.CONV3D.default),
        (TransposeConv2dBiasModule(), exir_ops.backend.tosa.TRANSPOSE_CONV2D.default),
    ],
)
def test_rewrite_conv_int16_bias_lowers_to_single_tosa_conv(
    module: (
        Conv2dBiasModule
        | DepthwiseConv2dBiasModule
        | Conv3dBiasModule
        | TransposeConv2dBiasModule
    ),
    target_op,
) -> None:
    exported_program = _export_quantized_a16w8(module, module.get_inputs())
    edge_program = to_edge(
        exported_program, compile_config=EdgeCompileConfig(_check_ir_validity=False)
    ).exported_program()
    gm = _run_pre_rewrite_passes(edge_program)

    with TosaLoweringContext(_compile_spec_int16().tosa_spec):
        result = RewriteConvPass(edge_program)(gm)
        assert result is not None
        gm = result.graph_module

    tosa_conv_nodes = [
        node
        for node in gm.graph.nodes
        if node.op == "call_function" and node.target == target_op
    ]
    assert len(tosa_conv_nodes) == 1
    assert all(node.target != exir_ops.edge.aten.add.Tensor for node in gm.graph.nodes)

    bias_node = tosa_conv_nodes[0].args[2]
    assert isinstance(bias_node, torch.fx.Node)
    assert bias_node.meta.get(TosaSpecialDtype.meta_key()) == TosaSpecialDtype.INT48


def test_rewrite_conv_dynamic_keeps_static_padding_when_symbolic_remainder_is_zero():
    model = ConvModule()
    example_inputs = (torch.randn(1, 3, 9, 12),)
    ep = export(
        model,
        example_inputs,
        dynamic_shapes={"x": _multiples_of_three_dynamic_shapes()},
    )
    edge_model = to_edge(ep)
    shape_env = _get_shape_env(edge_model.exported_program().graph_module)
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"), shape_env=shape_env
    ):
        edge_model = edge_model.transform(
            [RewriteConvPass(edge_model.exported_program())]
        )

    conv_node = next(
        n
        for n in edge_model.exported_program().graph.nodes
        if n.target == exir_ops.backend.tosa.CONV2D.default
    )
    padding = conv_node.args[4]
    assert padding == [0, 0, 0, 0]
    assert all(not isinstance(p, torch.SymInt) for p in padding)


def test_rewrite_conv_adjust_pad_if_needed_static_allows_negative_padding_until_later_validation():
    rewrite_pass, _, _ = _make_rewrite_pass((torch.randn(1, 3, 9, 12),))

    try:
        rewrite_pass._adjust_pad_if_needed(6, 2, 3, 0, 1)
    except RuntimeError as e:
        assert "SizeAdjustInputPass" in str(e)
    else:
        pytest.fail("Expected RuntimeError was not raised")


def test_rewrite_conv_adjust_pad_if_needed_static_positive_padding_stays_non_negative():
    rewrite_pass, _, _ = _make_rewrite_pass((torch.randn(1, 3, 9, 12),))

    adjusted_pad = rewrite_pass._adjust_pad_if_needed(8, 2, 3, 2, 1)

    assert adjusted_pad == 1


def test_rewrite_conv_adjust_pad_if_needed_static_exact_remainder_matches_pad():
    rewrite_pass, _, _ = _make_rewrite_pass((torch.randn(1, 3, 9, 12),))

    adjusted_pad = rewrite_pass._adjust_pad_if_needed(6, 1, 3, 1, 1)

    assert adjusted_pad == 0


def test_rewrite_conv_adjust_pad_if_needed_symbolic_exact_zero_keeps_zero_pad():
    rewrite_pass, shape_env, input_len = _make_rewrite_pass(
        (torch.randn(1, 3, 9, 12),),
        dynamic_shapes=_multiples_of_three_dynamic_shapes(),
    )

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"), shape_env=shape_env
    ):
        adjusted_pad = rewrite_pass._adjust_pad_if_needed(input_len, 3, 3, 0, 1)

    assert adjusted_pad == 0


def test_rewrite_conv_adjust_pad_if_needed_symbolic_exact_zero_keeps_positive_pad():
    rewrite_pass, shape_env, input_len = _make_rewrite_pass(
        (torch.randn(1, 3, 9, 12),),
        dynamic_shapes=_multiples_of_three_dynamic_shapes(),
    )

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"), shape_env=shape_env
    ):
        adjusted_pad = rewrite_pass._adjust_pad_if_needed(input_len, 2, 3, 1, 1)

    assert adjusted_pad == 1


def test_rewrite_conv_adjust_pad_if_needed_symbolic_positive_padding_range_returns_symbolic_padding():
    rewrite_pass, shape_env, input_len = _make_rewrite_pass(
        (torch.randn(1, 3, 8, 8),),
        dynamic_shapes={
            2: Dim("height", min=6, max=10),
            3: Dim("width", min=6, max=10),
        },
    )

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"), shape_env=shape_env
    ):
        adjusted_pad = rewrite_pass._adjust_pad_if_needed(input_len, 2, 3, 1, 1)

    assert isinstance(adjusted_pad, torch.SymInt)


def test_rewrite_conv_symbolic_comparison_with_int_specializes_to_hint():
    rewrite_pass, shape_env, input_len = _make_rewrite_pass(
        (torch.randn(1, 3, 8, 8),),
        dynamic_shapes={
            2: Dim("height", min=6, max=10),
            3: Dim("width", min=6, max=10),
        },
    )

    def unsafe_adjust(input_len, input_weight, stride, pad, dilation):
        mod_remainder = (
            input_len + 2 * pad - dilation * (input_weight - 1) - 1
        ) % stride
        if mod_remainder == 0:
            return pad
        if mod_remainder > pad:
            raise RuntimeError("SizeAdjustInputPass")
        return pad - mod_remainder

    mod_remainder = (input_len - 2) % 3
    value_ranges = shape_env.bound_sympy(mod_remainder.node.expr)

    assert value_ranges.lower == 0
    assert value_ranges.upper == 2
    assert len(shape_env.guards) == 0
    assert unsafe_adjust(input_len, 2, 3, 0, 1) == 0
    assert len(shape_env.guards) == 1
    assert shape_env.guards[-1].expr in {
        (mod_remainder == 0).node.expr,
        (mod_remainder <= 0).node.expr,
    }

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"), shape_env=shape_env
    ):
        adjusted_pad = rewrite_pass._adjust_pad_if_needed(input_len, 2, 3, 0, 1)

    assert isinstance(adjusted_pad, torch.SymInt)


def test_rewrite_conv_adjust_pad_if_needed_symbolic_zero_padding_range_returns_symbolic_padding():
    rewrite_pass, shape_env, input_len = _make_rewrite_pass(
        (torch.randn(1, 3, 8, 8),),
        dynamic_shapes={
            2: Dim("height", min=6, max=10),
            3: Dim("width", min=6, max=10),
        },
    )

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"), shape_env=shape_env
    ):
        adjusted_pad = rewrite_pass._adjust_pad_if_needed(input_len, 2, 3, 0, 1)

    assert isinstance(adjusted_pad, torch.SymInt)


def test_rewrite_conv_adjust_pad_if_needed_symbolic_singleton_overflow_still_raises():
    rewrite_pass, shape_env, input_len = _make_rewrite_pass(
        (torch.randn(1, 3, 9, 12),),
        dynamic_shapes=_multiples_of_three_dynamic_shapes(),
    )

    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+FP+shape"), shape_env=shape_env
    ):
        with pytest.raises(RuntimeError, match="SizeAdjustInputPass"):
            rewrite_pass._adjust_pad_if_needed(input_len, 3, 3, 1, 1)
