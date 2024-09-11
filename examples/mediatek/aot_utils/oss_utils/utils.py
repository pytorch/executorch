# Copyright (c) MediaTek Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional

import torch
from executorch import exir
from executorch.backends.mediatek import (
    NeuropilotPartitioner,
    NeuropilotQuantizer,
    Precision,
)
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e


def build_executorch_binary(
    model,
    inputs,
    file_name,
    dataset,
    quant_dtype: Optional[Precision] = None,
):
    if quant_dtype is not None:
        quantizer = NeuropilotQuantizer()
        quantizer.setup_precision(quant_dtype)
        if quant_dtype not in Precision:
            raise AssertionError(f"No support for Precision {quant_dtype}.")

        captured_model = torch._export.capture_pre_autograd_graph(model, inputs)
        annotated_model = prepare_pt2e(captured_model, quantizer)
        print("Quantizing the model...")
        # calibration
        for data in dataset:
            annotated_model(*data)
        quantized_model = convert_pt2e(annotated_model, fold_quantize=False)
        aten_dialect = torch.export.export(quantized_model, inputs)
    else:
        aten_dialect = torch.export.export(model, inputs)

    from executorch.exir.program._program import to_edge_transform_and_lower

    edge_compile_config = exir.EdgeCompileConfig(_check_ir_validity=False)
    # skipped op names are used for deeplabV3 model
    neuro_partitioner = NeuropilotPartitioner(
        [],
        op_names_to_skip={
            "aten_convolution_default_106",
            "aten_convolution_default_107",
        },
    )
    edge_prog = to_edge_transform_and_lower(
        aten_dialect,
        compile_config=edge_compile_config,
        partitioner=[neuro_partitioner],
    )

    exec_prog = edge_prog.to_executorch(
        config=exir.ExecutorchBackendConfig(extract_constant_segment=False)
    )
    with open(f"{file_name}.pte", "wb") as file:
        file.write(exec_prog.buffer)


def make_output_dir(path: str):
    if os.path.exists(path):
        for f in os.listdir(path):
            os.remove(os.path.join(path, f))
        os.removedirs(path)
    os.makedirs(path)
