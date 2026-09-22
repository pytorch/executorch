# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from executorch.backends.cortex_m.edge_compile_config import (
    cortex_m_edge_compile_config,
)
from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
from executorch.backends.cortex_m.target_config import CortexM, CortexMTargetConfig
from executorch.exir import to_edge, to_edge_transform_and_lower
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


def _quantize(model, inputs, use_explicit_layout=False):
    prepared = prepare_pt2e(
        torch.export.export(model.eval(), inputs).module(),
        CortexMQuantizer(use_explicit_layout=use_explicit_layout),
    )
    with torch.no_grad():
        prepared(*inputs)
    quantized = convert_pt2e(prepared)
    expected = quantized(*inputs)
    output_scale = [
        node.args[1]
        for node in quantized.graph.nodes
        if node.target == torch.ops.quantized_decomposed.dequantize_per_tensor.default
    ][-1]
    return torch.export.export(quantized, inputs), expected, output_scale


@pytest.mark.parametrize("use_explicit_layout", [False, True])
@pytest.mark.parametrize("entry_point", ["transform", "combined", "legacy"])
def test_lowers_and_serializes(entry_point, use_explicit_layout):
    model = torch.nn.Conv2d(3, 4, 3, padding=1)
    inputs = (torch.randn(1, 3, 8, 8),)
    if not use_explicit_layout:
        model = model.to(memory_format=torch.channels_last)
        inputs = (inputs[0].to(memory_format=torch.channels_last),)
    exported, expected, output_scale = _quantize(model, inputs, use_explicit_layout)
    config = cortex_m_edge_compile_config()
    manager = CortexMPassManager(use_explicit_layout=use_explicit_layout)

    if entry_point == "combined":
        lowered = to_edge_transform_and_lower(
            exported, compile_config=config, transform_passes=manager
        )
    else:
        lowered = to_edge(exported, compile_config=config)
        if entry_point == "transform":
            lowered = lowered.transform(manager)
        else:
            manager = CortexMPassManager(
                lowered.exported_program(), use_explicit_layout=use_explicit_layout
            )
            lowered._edge_programs["forward"] = manager.transform()

    program = lowered.exported_program()
    program.validate()
    conv = (
        exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default
        if use_explicit_layout
        else exir_ops.edge.cortex_m.quantized_conv2d.default
    )
    assert sum(node.target == conv for node in program.graph.nodes) == 1
    assert all(node.op != "get_attr" for node in program.graph.nodes)
    torch.testing.assert_close(
        program.module()(*inputs), expected, rtol=0, atol=2 * output_scale
    )
    serialized = deserialize_pte_binary(lowered.to_executorch().buffer).program
    assert serialized.execution_plan[0].name == "forward"
    assert not serialized.execution_plan[0].delegates


def test_transform_uses_each_method_program():
    inputs = {
        "forward": (torch.randn(2, 4),),
        "other": (torch.randn(1, 8),),
    }
    models = {
        "forward": torch.nn.Linear(4, 3),
        "other": torch.nn.Linear(8, 2),
    }
    programs = {}
    expected = {}
    output_scales = {}
    for name, model in models.items():
        programs[name], expected[name], output_scales[name] = _quantize(
            model, inputs[name]
        )
    edge = to_edge(
        programs,
        compile_config=cortex_m_edge_compile_config(),
        constant_methods={"version": 17},
    )
    # A legacy bound program must not override the method supplied by transform().
    manager = CortexMPassManager(
        edge.exported_program(), target_config=CortexMTargetConfig(cpu=CortexM.M33)
    )
    lowered = edge.transform(manager)
    assert lowered.methods == {"forward", "other"}
    assert lowered.config_methods == {"version"}
    for name in models:
        program = lowered.exported_program(name)
        program.validate()
        [linear] = [
            node
            for node in program.graph.nodes
            if node.target == exir_ops.edge.cortex_m.quantized_linear.default
        ]
        assert linear.args[3] is None  # The M33 does not use the MVE kernel sum.
        torch.testing.assert_close(
            program.module()(*inputs[name]),
            expected[name],
            rtol=0,
            atol=2 * output_scales[name],
        )
    serialized = deserialize_pte_binary(lowered.to_executorch().buffer).program
    assert {plan.name for plan in serialized.execution_plan} == {
        "forward",
        "other",
        "version",
    }


def test_empty_pass_list_reports_no_change():
    inputs = (torch.tensor([-1.0, 2.0]),)
    edge = to_edge(torch.export.export(torch.nn.ReLU(), inputs))
    program = edge.exported_program()
    manager = CortexMPassManager(passes=[])
    assert not manager(program).modified
    assert edge.transform(manager).exported_program() is program
    assert CortexMPassManager(program, passes=[]).transform() is program
    with pytest.raises(ValueError, match="needs a real ExportedProgram"):
        manager.transform()


def test_empty_pass_list_preserves_unlifted_constants():
    class AddConstant(ExportPass):
        def call_operator(self, op, args, kwargs, meta):
            result = super().call_operator(op, args, kwargs, meta)
            return super().call_operator(
                exir_ops.edge.aten.add.Tensor,
                (result, torch.tensor([2.0])),
                {},
                meta,
            )

    inputs = (torch.tensor([-1.0, 2.0]),)
    edge = to_edge(torch.export.export(torch.nn.ReLU(), inputs))
    program = edge.transform([AddConstant()]).exported_program()
    assert any(node.op == "get_attr" for node in program.graph.nodes)

    result = CortexMPassManager(passes=[])(program)
    assert not result.modified
    assert result.exported_program.graph is program.graph
    assert result.exported_program.graph_signature is program.graph_signature
    program.validate()
    assert any(node.op == "get_attr" for node in program.graph.nodes)
    assert not program.graph_signature.buffers
    torch.testing.assert_close(program.module()(*inputs), torch.tensor([2.0, 4.0]))
