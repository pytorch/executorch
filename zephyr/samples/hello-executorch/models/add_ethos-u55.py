# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file shows one way to export a small model to a PTE during the Zephyr
# build by pointing CONFIG_EXECUTORCH_EXPORT_PYTHON_SCRIPT at this script.
#
# This variant lowers the model for Ethos-U55. See the sibling files in this
# directory for the Cortex-M and Ethos-U85 variants.

OUTPUT_PTE = "hello_executorch_ethos-u55.pte"

import torch

from executorch.backends.arm.ethosu import EthosUCompileSpec, EthosUPartitioner
from executorch.backends.arm.quantizer import (
    EthosUQuantizer,
    get_symmetric_quantization_config,
)
from executorch.backends.cortex_m.passes.replace_quant_nodes_pass import (
    ReplaceQuantNodesPass,
)
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

# Export the model to a `.pte` file for Ethos-U55.


def _export_ethosu(target, pte_file):
    compile_spec = EthosUCompileSpec(target=target)

    model = ModelUnderTest
    example_inputs = ModelInputs
    exported_model = torch.export.export(model, example_inputs, strict=True).module()

    quantizer = EthosUQuantizer(compile_spec)
    quantizer.set_global(get_symmetric_quantization_config())
    prepared = prepare_pt2e(exported_model, quantizer)
    prepared(*example_inputs)
    quantized_model = convert_pt2e(prepared)
    exported_program = torch.export.export(quantized_model, example_inputs, strict=True)

    edge_program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[EthosUPartitioner(compile_spec)],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    edge_program = edge_program.transform([ReplaceQuantNodesPass()])
    executorch_program = edge_program.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False)
    )
    save_pte_program(executorch_program, pte_file)


def main():
    _export_ethosu("ethos-u55-128", OUTPUT_PTE)


if __name__ == "__main__":
    main()
