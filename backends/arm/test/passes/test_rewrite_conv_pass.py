# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from executorch.backends.arm._passes import (
    ConvertToClampPass,
    FoldAndAnnotateQParamsPass,
    FuseQuantizedActivationPass,
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
from executorch.backends.arm.tosa.mapping import TosaSpecialDtype
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.backends.arm.vgf import VgfCompileSpec, VgfPartitioner
from executorch.exir import EdgeCompileConfig, to_edge, to_edge_transform_and_lower
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import Dim, export
from torch.export.exported_program import _get_shape_env


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
    # We cannot run TOSA backend dialect operators in eager mode.
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()


def test_fold_and_annotate_q_params_vgf_quant_preserves_output_qparams_on_non_fuseable_clamp() -> (
    None
):
    exported_program = _export_quantized(TinyConvReluCat())
    gm = _run_pre_rewrite_passes(to_edge(exported_program).exported_program())

    conv = _get_call_function_node(gm, exir_ops.edge.aten.convolution.default)
    clamp = _get_call_function_node(gm, exir_ops.edge.aten.clamp.default)

    assert conv.meta["input_qparams"]
    assert not conv.meta["output_qparams"]
    assert clamp.meta["output_qparams"]


def test_rewrite_conv_vgf_quant_handles_non_fuseable_conv_clamp_cat_branch() -> None:
    exported_program = _export_quantized(TinyConvReluCat())
    compile_spec = _compile_spec()

    to_edge_transform_and_lower(
        exported_program,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
        partitioner=[VgfPartitioner(compile_spec)],
    )


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


def test_rewrite_conv_adjust_pad_if_needed_static_raises_before_negative_padding():
    rewrite_pass, _, _ = _make_rewrite_pass((torch.randn(1, 3, 9, 12),))

    with pytest.raises(RuntimeError, match="SizeAdjustInputPass"):
        rewrite_pass._adjust_pad_if_needed(6, 2, 3, 0, 1)


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


def test_rewrite_conv_adjust_pad_if_needed_symbolic_positive_padding_range_raises_before_negative_padding():
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
        with pytest.raises(RuntimeError, match="SizeAdjustInputPass"):
            rewrite_pass._adjust_pad_if_needed(input_len, 2, 3, 1, 1)


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
        with pytest.raises(RuntimeError, match="SizeAdjustInputPass"):
            rewrite_pass._adjust_pad_if_needed(input_len, 2, 3, 0, 1)


def test_rewrite_conv_adjust_pad_if_needed_symbolic_zero_padding_range_raises_before_negative_padding():
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
        with pytest.raises(RuntimeError, match="SizeAdjustInputPass"):
            rewrite_pass._adjust_pad_if_needed(input_len, 2, 3, 0, 1)
