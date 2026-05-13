# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file shows one way to export a small model to a PTE during the Zephyr
# build by pointing CONFIG_EXECUTORCH_EXPORT_PYTHON_SCRIPT at this script.
#
# This variant lowers the model for the Cortex-M backend. See the sibling files
# in this directory for the Ethos-U variants.

OUTPUT_PTE = "hello_executorch_cortex-m.pte"

import torch

from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager

from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.extension.export_util.utils import save_pte_program
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


# Model definition. This is intentionally a tiny add model to keep the export
# example focused on the Zephyr integration.
class myModelAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + x


# Define the model and example inputs used by the exporter code below.

ModelUnderTest = myModelAdd()
ModelInputs = (torch.ones(5),)

# Export the model to a `.pte` file for Cortex-M.


def _to_channels_last(value):
    if isinstance(value, torch.Tensor) and value.dim() == 4:
        return value.to(memory_format=torch.channels_last)
    if isinstance(value, tuple):
        return tuple(_to_channels_last(item) for item in value)
    return value


def _export_cortex_m(pte_file):
    model = ModelUnderTest.eval()
    example_inputs = tuple(_to_channels_last(value) for value in ModelInputs)
    if any(
        isinstance(value, torch.Tensor) and value.dim() == 4 for value in example_inputs
    ):
        model = model.to(memory_format=torch.channels_last)
    exported_model = torch.export.export(model, example_inputs, strict=True).module()

    quantizer = CortexMQuantizer()
    prepared = prepare_pt2e(exported_model, quantizer)
    prepared(*example_inputs)
    quantized_model = convert_pt2e(prepared)
    exported_program = torch.export.export(quantized_model, example_inputs, strict=True)

    edge_program = to_edge_transform_and_lower(
        exported_program,
        compile_config=EdgeCompileConfig(
            preserve_ops=[
                torch.ops.aten.linear.default,
                torch.ops.aten.hardsigmoid.default,
                torch.ops.aten.hardsigmoid_.default,
                torch.ops.aten.hardswish.default,
                torch.ops.aten.hardswish_.default,
            ],
            _check_ir_validity=False,
        ),
    )
    pass_manager = CortexMPassManager(edge_program.exported_program())
    edge_program._edge_programs["forward"] = pass_manager.transform()
    executorch_program = edge_program.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False)
    )
    save_pte_program(executorch_program, pte_file)


def main():
    _export_cortex_m(OUTPUT_PTE)


if __name__ == "__main__":
    main()
