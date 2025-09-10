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
from executorch.exir.backend.backend_details import CompileSpec
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


def build_executorch_binary(
    model,
    inputs,
    file_name,
    dataset,
    quant_dtype: Optional[Precision] = None,
    skip_op_name: Optional[set] = None,
    skip_op_type: Optional[set] = None,
):
    if quant_dtype is not None:
        quantizer = NeuropilotQuantizer()
        quantizer.setup_precision(quant_dtype)
        if quant_dtype not in Precision:
            raise AssertionError(f"No support for Precision {quant_dtype}.")

        captured_model = torch.export.export(model, inputs, strict=True).module()
        annotated_model = prepare_pt2e(captured_model, quantizer)
        print("Quantizing the model...")
        # calibration
        for data in dataset:
            annotated_model(*data)
        quantized_model = convert_pt2e(annotated_model, fold_quantize=False)
        aten_dialect = torch.export.export(quantized_model, inputs, strict=True)
    else:
        aten_dialect = torch.export.export(model, inputs, strict=True)

    from executorch.exir.program._program import to_edge_transform_and_lower

    edge_compile_config = exir.EdgeCompileConfig(_check_ir_validity=False)
    neuro_partitioner = NeuropilotPartitioner(
        [CompileSpec("platform-config", b"mt6989")],
        op_types_to_skip=skip_op_type,
        op_names_to_skip=skip_op_name,
    )

    edge_prog = to_edge_transform_and_lower(
        aten_dialect,
        compile_config=edge_compile_config,
        partitioner=[neuro_partitioner],
    )

    exec_prog = edge_prog.to_executorch(config=exir.ExecutorchBackendConfig())

    with open(f"{file_name}.pte", "wb") as file:
        file.write(exec_prog.buffer)


def make_output_dir(path: str):
    if os.path.exists(path):
        for f in os.listdir(path):
            os.remove(os.path.join(path, f))
        os.removedirs(path)
    os.makedirs(path)
