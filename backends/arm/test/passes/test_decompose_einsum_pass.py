# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm._passes import DecomposeEinsumPass
from executorch.backends.arm.quantizer import (
    get_symmetric_quantization_config,
    TOSAQuantizer,
)
from executorch.backends.arm.test.tester.test_pipeline import QuantizationPipeline
from executorch.backends.arm.tosa import TosaSpecification
from torch.export import export


input_t = Tuple[torch.Tensor]


class EinsumPermuteModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nhwpqc->nchpwq", x)

    @staticmethod
    def get_inputs() -> input_t:
        return (torch.randn(2, 4, 16, 1, 16, 1),)


def _get_int8_quantizer() -> TOSAQuantizer:
    quantizer = TOSAQuantizer(TosaSpecification.create_from_string("TOSA-1.0+INT"))
    quantizer.set_global(get_symmetric_quantization_config())
    return quantizer


def test_decompose_einsum_no_target_rewrites_export_graph() -> None:
    module = EinsumPermuteModule().eval()
    exported_program = export(module, module.get_inputs())

    before_targets = [
        str(node.target)
        for node in exported_program.graph_module.graph.nodes
        if node.op == "call_function"
    ]
    assert before_targets == ["aten.einsum.default"]

    pass_result = DecomposeEinsumPass().call(exported_program.graph_module)

    after_targets = [
        str(node.target)
        for node in pass_result.graph_module.graph.nodes
        if node.op == "call_function"
    ]
    assert "aten.einsum.default" not in after_targets
    assert after_targets == ["aten.permute.default"]


def test_decompose_einsum_tosa_INT_quantizes_after_transform_for_annotation() -> None:
    module = EinsumPermuteModule().eval()
    quantization_annotations = {
        "aten.permute.default": {
            get_symmetric_quantization_config().output_activation: 1
        }
    }

    pipeline = QuantizationPipeline[input_t](
        module,
        module.get_inputs(),
        quantizer=_get_int8_quantizer(),
        qspecs=quantization_annotations,  # type: ignore[arg-type]
    )
    pipeline.run()
