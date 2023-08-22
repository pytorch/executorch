# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting simple models to flatbuffer

import argparse
import copy

import executorch.exir as exir
import torch._export as export
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackFloatingPointPartitioner,
    XnnpackQuantizedPartitioner2,
)
from executorch.exir.backend.backend_api import to_backend, validation_disabled

from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)

from ..models import MODEL_NAME_TO_MODEL

# Note: for mv3, the mul op is not supported in XNNPACKQuantizer, that could be supported soon
XNNPACK_MODEL_NAME_TO_MODEL = {
    name: MODEL_NAME_TO_MODEL[name] for name in ["linear", "add", "add_mul", "mv2"]
}


def quantize(model, example_inputs):
    """This is the official recommended flow for quantization in pytorch 2.0 export"""
    m = model.eval()
    m = export.capture_pre_autograd_graph(m, copy.deepcopy(example_inputs))
    quantizer = XNNPACKQuantizer()
    # if we set is_per_channel to True, we also need to add out_variant of quantize_per_channel/dequantize_per_channel
    operator_config = get_symmetric_quantization_config(is_per_channel=False)
    quantizer.set_global(operator_config)
    m = prepare_pt2e(m, quantizer)
    # calibration
    m(*example_inputs)
    m = convert_pt2e(m)
    return m


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        help=f"Provide model name. Valid ones: {list(XNNPACK_MODEL_NAME_TO_MODEL.keys())}",
    )
    parser.add_argument(
        "-q",
        "--quantize",
        action="store_true",
        required=False,
        default=False,
        help="Flag for producing quantized or floating-point model",
    )
    args = parser.parse_args()

    if args.model_name not in XNNPACK_MODEL_NAME_TO_MODEL:
        raise RuntimeError(
            f"Model {args.model_name} is not a valid name. or not quantizable right now, "
            "please contact executorch team if you want to learn why or how to support "
            "quantization for the requested model"
            f"Available models are {list(XNNPACK_MODEL_NAME_TO_MODEL.keys())}."
        )

    model, example_inputs = MODEL_NAME_TO_MODEL[args.model_name]()
    model = model.eval()

    partitioner = XnnpackFloatingPointPartitioner
    if args.quantize:
        print("Quantizing Model...")
        model = quantize(model, example_inputs)
        # Partitioner will eventually be a single partitioner for both fp32 and quantized models
        partitioner = XnnpackQuantizedPartitioner2

    edge = exir.capture(
        model, example_inputs, exir.CaptureConfig(enable_aot=True, _unlift=True)
    ).to_edge(exir.EdgeCompileConfig(_check_ir_validity=False))
    print("Exported graph:\n", edge.exported_program.graph)

    with validation_disabled():
        edge.exported_program = to_backend(edge.exported_program, partitioner)
    print("Lowered graph:\n", edge.exported_program.graph)

    exec_prog = edge.to_executorch()
    buffer = exec_prog.buffer
    quant_tag = "_quantize" if args.quantize else ""
    filename = f"xnnpack_{args.model_name}{quant_tag}.pte"
    print(f"Saving exported program to {filename}.")
    with open(filename, "wb") as f:
        f.write(buffer)
