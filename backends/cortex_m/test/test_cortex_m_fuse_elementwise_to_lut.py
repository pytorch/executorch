# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from executorch.backends.arm._passes.quant_args import QuantArgs
from executorch.backends.cortex_m.library import cmsis_nn
from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
from executorch.backends.cortex_m.passes.fuse_elementwise_to_lut_pass import (
    FuseQuantizedElementwiseToLUTPass,
)
from executorch.backends.cortex_m.target_config import CortexM, CortexMTargetConfig
from executorch.backends.cortex_m.test.tester import CortexMTester
from executorch.backends.cortex_m.utils import _ACTIVATION_FNS, build_activation_lut
from executorch.backends.transforms.utils import get_param_tensor
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export


class _SigmoidTanh(torch.nn.Module):
    def forward(self, x):
        return torch.tanh(torch.sigmoid(x))


class _ScalarArithmetic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("constant", torch.tensor([0.25]))

    def forward(self, x):
        value = torch.sigmoid(x) + self.constant
        value = (value - self.constant) * self.constant
        return torch.tanh(value / self.constant)


@pytest.mark.parametrize("explicit_layout", [False, True])
def test_unary_after_two_input_add_lowers_to_table(explicit_layout):
    class AddSqrt(torch.nn.Module):
        def forward(self, x, y):
            return torch.sqrt(x + y)

    inputs = (torch.linspace(0, 2, 256), torch.linspace(0, 3, 256))
    tester = CortexMTester(
        AddSqrt(),
        inputs,
        target_config=CortexMTargetConfig(cpu=CortexM.M55, isa=cmsis_nn.Backend.MVE),
    )
    tester.test_dialect(
        {
            "executorch_exir_dialects_edge__ops_aten_add_Tensor": 1,
            "executorch_exir_dialects_edge__ops_aten_sqrt_default": 1,
        },
        {
            "executorch_exir_dialects_edge__ops_aten_sqrt_default": 0,
            "executorch_exir_dialects_edge__ops_cortex_m_quantized_add_default": 1,
            "executorch_exir_dialects_edge__ops_cortex_m_quantized_activation_default": 1,
        },
        qtol=2,
        use_explicit_layout=explicit_layout,
    )


@pytest.mark.parametrize("explicit_layout", [False, True])
@pytest.mark.parametrize("model", [_SigmoidTanh, _ScalarArithmetic])
def test_cortex_m_quantized_chain_lowers_to_one_table(explicit_layout, model):
    from executorch.backends.cortex_m.edge_compile_config import (
        cortex_m_edge_compile_config,
    )
    from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    x = torch.randn(2, 5, generator=torch.Generator().manual_seed(42))
    prepared = prepare_pt2e(
        export(model(), (x,)).module(),
        CortexMQuantizer(use_explicit_layout=explicit_layout),
    )
    prepared(x)
    quantized = convert_pt2e(prepared)
    expected = quantized(x)
    output_scale = [
        node.args[1]
        for node in quantized.graph.nodes
        if node.target == torch.ops.quantized_decomposed.dequantize_per_tensor.default
    ][-1]
    edge = to_edge(
        export(quantized, (x,)),
        compile_config=cortex_m_edge_compile_config(),
    )
    manager = CortexMPassManager(
        use_explicit_layout=explicit_layout,
        target_config=CortexMTargetConfig(cpu=CortexM.M55, isa=cmsis_nn.Backend.MVE),
    )
    passes = manager.passes[0].pass_classes
    assert FuseQuantizedElementwiseToLUTPass in passes

    program = edge.transform(manager).exported_program()

    program.validate()
    assert (
        sum(
            node.op == "call_function"
            and node.target == exir_ops.edge.cortex_m.quantized_activation.default
            for node in program.graph.nodes
        )
        == 1
    )
    assert len(program.state_dict) == 1
    torch.testing.assert_close(
        program.module()(x), expected, rtol=0, atol=2 * output_scale
    )


@pytest.mark.parametrize("folded_qdq", [False, True])
def test_chain_table_improves_accuracy_over_intermediate_qdq(folded_qdq):
    from executorch.backends.cortex_m.edge_compile_config import (
        cortex_m_edge_compile_config,
    )
    from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    x = torch.linspace(-6.5, 6.25, 256)
    model = _SigmoidTanh()
    prepared = prepare_pt2e(export(model, (x,)).module(), CortexMQuantizer())
    prepared(x)
    edge = to_edge(
        export(convert_pt2e(prepared), (x,)),
        compile_config=cortex_m_edge_compile_config(),
    )
    program = edge.exported_program()
    input_qparams, hidden_qparams, output_qparams = [
        QuantArgs.from_operator(node.target, node.args)
        for node in program.graph.nodes
        if node.target == FuseQuantizedElementwiseToLUTPass.quantize_op
    ]
    if folded_qdq:
        program = edge.transform(
            CortexMPassManager(
                target_config=CortexMTargetConfig(
                    cpu=CortexM.M55, isa=cmsis_nn.Backend.MVE
                )
            )
        ).exported_program()
    else:
        assert (
            FuseQuantizedElementwiseToLUTPass(program)
            .call(program.graph_module)
            .modified
        )
    program.validate()
    tables = [
        node
        for node in program.graph.nodes
        if node.target == exir_ops.edge.cortex_m.quantized_activation.default
    ]
    assert len(tables) == 1
    table = tables[0]
    assert table.meta["input_qparams"] == {0: input_qparams}
    assert table.meta["output_qparams"] == {0: output_qparams}

    def quantize(value, qparams):
        scaled = value / qparams.scale + qparams.zp
        return (
            (torch.sign(scaled) * torch.floor(torch.abs(scaled) + 0.5))
            .clamp(qparams.qmin, qparams.qmax)
            .to(qparams.dtype)
        )

    domain = torch.arange(-128, 128, dtype=torch.int8)
    inputs = input_qparams.dequantize_value(domain)
    reference = model(inputs)
    actual = get_param_tensor(program, table.args[1])
    torch.testing.assert_close(actual, quantize(reference, output_qparams))
    previous_hidden = quantize(torch.sigmoid(inputs), hidden_qparams)
    previous = quantize(
        torch.tanh(hidden_qparams.dequantize_value(previous_hidden)), output_qparams
    )
    error = (output_qparams.dequantize_value(actual) - reference).abs()
    previous_error = (output_qparams.dequantize_value(previous) - reference).abs()
    assert error.mean() < previous_error.mean()
    assert error.max() < previous_error.max()


@pytest.mark.parametrize("target", list(_ACTIVATION_FNS))
def test_standalone_activation_preserves_table_entries(target):
    program = to_edge(export(torch.nn.Sigmoid(), (torch.ones(256),))).exported_program()
    source = next(node for node in program.graph.nodes if node.op == "placeholder")
    sink = next(node for node in program.graph.nodes if node.op == "call_function")
    source.meta["val"] = torch.empty_like(source.meta["val"], dtype=torch.int8)
    sink.meta["val"] = torch.empty_like(sink.meta["val"], dtype=torch.int8)
    sink.target = target
    sink.meta["input_qparams"] = {0: QuantArgs(0.02, 0, -128, 127, torch.int8)}
    sink.meta["output_qparams"] = {0: QuantArgs(0.05, 7, -128, 127, torch.int8)}

    assert (
        FuseQuantizedElementwiseToLUTPass(program).call(program.graph_module).modified
    )

    program.validate()
    torch.testing.assert_close(
        get_param_tensor(program, sink.args[1]),
        build_activation_lut(target, 0.02, 0, 0.05, 7),
    )
