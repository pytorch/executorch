# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
    get_symmetric_quantization_config,
    VgfQuantizer,
)
from executorch.backends.arm.test.misc.test_dw_convs_with_shared_weights import (
    DWConvsModule,
)
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline
from executorch.backends.arm.tosa.specification import TosaLoweringContext
from executorch.backends.arm.vgf import VgfCompileSpec, VgfPartitioner
from executorch.exir import EdgeCompileConfig, to_edge, to_edge_transform_and_lower
from executorch.exir.dialects._ops import ops as exir_ops


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


def test_rewrite_conv_tosa_FP():
    module = DWConvsModule()
    pipeline = PassPipeline(
        module, module.get_inputs(), passes_with_exported_program=[RewriteConvPass]
    )
    # We can't run TOSA backend dialect operators in eager mode
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
