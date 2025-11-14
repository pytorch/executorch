# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
from executorch.backends.arm.common.annotation_meta import ArmAnnotationInfo
from executorch.backends.arm.quantizer import (
    get_symmetric_quantization_config,
    TOSAQuantizer,
)
from executorch.backends.arm.tosa import TosaSpecification
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


class SimpleModule(torch.nn.Module):
    example_inputs = (torch.randn(1, 10),)

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def test_save_load_exported_int_model():
    module = SimpleModule().eval()
    example_inputs = module.example_inputs
    exported_module = torch.export.export(module, example_inputs)

    # Set up quantizer
    quantizer = TOSAQuantizer(TosaSpecification.create_from_string("TOSA-1.0+INT"))
    quantizer.set_global(get_symmetric_quantization_config())
    # Quantize model
    prepared_module = prepare_pt2e(exported_module.module(), quantizer)
    prepared_module(*example_inputs)
    quantized_module = convert_pt2e(prepared_module)
    quantized_exported_module = torch.export.export(quantized_module, example_inputs)

    base_path = "arm_test/misc/"
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    file_path = base_path + "exported_module.pt2"
    # Verify that we can save the model
    torch.export.save(quantized_exported_module, file_path)

    # Verify that we can load the model back
    loaded_model = torch.export.load(file_path)
    for original_node, loaded_node in zip(
        quantized_exported_module.graph.nodes, loaded_model.graph.nodes
    ):
        # Verify that the custom metadata is preserved after save/load
        assert original_node.meta.get("custom", {}) == loaded_node.meta.get(
            "custom", {}
        )
        if original_node.target == torch.ops.aten.linear.default:
            assert ArmAnnotationInfo.CUSTOM_META_KEY in original_node.meta.get(
                "custom", {}
            )
