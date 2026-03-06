# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verify that clamp uses independent input/output quantization.

Clamp modifies the value range by enforcing min/max bounds, so its output
observer must be independent from its input observer. When observers are
shared, the pre-clamp (wider) values dominate the observed range and the
post-clamp tensor gets incorrect quantization parameters.

This test feeds a wide-range input through a narrow clamp and checks that
the quantization scale for the clamp output differs from the input scale.
"""

import torch
from executorch.backends.arm.quantizer import (
    get_symmetric_quantization_config,
    TOSAQuantizer,
)
from executorch.backends.arm.tosa import TosaSpecification
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

Q_PER_TENSOR = torch.ops.quantized_decomposed.quantize_per_tensor.default
DQ_PER_TENSOR = torch.ops.quantized_decomposed.dequantize_per_tensor.default


class ClampModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=0.0, max=1.0)


def test_clamp_has_different_input_output_qparams():
    """Input and output scales must differ when clamp narrows the range.

    A wide-range input ([-50, 50]) clamped to [0, 1] should produce a much
    smaller output scale than input scale, because the output observer only
    sees values in [0, 1] while the input observer sees the full [-50, 50].

    Before the fix (clamp in _one_to_one_shared_input_qspec), both observers
    were shared and would produce identical scales — the wider input range
    dominated, wasting output precision.
    """
    model = ClampModel()
    model.eval()

    # Use deterministic wide-range calibration data so the input observer
    # sees [-50, 50] while the output observer sees only [0, 1].
    calibration_input = torch.linspace(-50, 50, 200).reshape(1, 200)

    tosa_spec = TosaSpecification.create_from_string("TOSA-1.0+INT")
    quantizer = TOSAQuantizer(tosa_spec)
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=False))

    exported = torch.export.export(model, (calibration_input,))
    prepared = prepare_pt2e(exported.module(), quantizer)
    prepared(calibration_input)
    converted = convert_pt2e(prepared)

    # After conversion the graph has explicit quantize/dequantize nodes.
    # For clamp with independent qspecs the pattern is:
    #   dequantize_per_tensor(input_scale) -> clamp -> quantize_per_tensor(output_scale)
    # With shared qspecs both scales would be identical.
    clamp_nodes = [
        n
        for n in converted.graph.nodes
        if n.target in (torch.ops.aten.clamp.default, torch.ops.aten.clamp.Tensor)
    ]
    assert (
        len(clamp_nodes) == 1
    ), f"Expected exactly 1 clamp node, found {len(clamp_nodes)}"
    clamp_node = clamp_nodes[0]

    # Get the dequant feeding clamp's input — its scale is arg[1].
    input_dq = clamp_node.args[0]
    assert (
        input_dq.target == DQ_PER_TENSOR
    ), f"Expected dequantize_per_tensor before clamp, got {input_dq.target}"
    input_scale = float(input_dq.args[1])

    # Get the quant consuming clamp's output — its scale is arg[1].
    clamp_users = list(clamp_node.users)
    assert (
        len(clamp_users) == 1
    ), f"Expected exactly 1 user of clamp, found {len(clamp_users)}"
    output_q = clamp_users[0]
    assert (
        output_q.target == Q_PER_TENSOR
    ), f"Expected quantize_per_tensor after clamp, got {output_q.target}"
    output_scale = float(output_q.args[1])

    # With independent quantization the output scale (tracking [0, 1]) must
    # be much smaller than the input scale (tracking [-50, 50]).
    assert output_scale < input_scale, (
        f"Clamp output scale ({output_scale}) should be smaller than input "
        f"scale ({input_scale}) because clamp narrows [−50, 50] → [0, 1]. "
        "If they are equal, clamp is using shared observers (bug)."
    )
