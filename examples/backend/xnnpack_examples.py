# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting simple models to flatbuffer

import argparse
import logging

from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackFloatingPointPartitioner,
    XnnpackQuantizedPartitioner,
)

from executorch.exir import CaptureConfig, EdgeCompileConfig
from executorch.exir.backend.backend_api import to_backend
from executorch.exir.backend.canonical_partitioners.duplicate_dequant_node_pass import (
    DuplicateDequantNodePass,
)

from ..export.utils import export_to_edge, save_pte_program

from ..models import MODEL_NAME_TO_MODEL
from ..models.model_factory import EagerModelFactory
from ..quantization.utils import quantize
from ..recipes.xnnpack_optimization import MODEL_NAME_TO_OPTIONS


FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        help=f"Provide model name. Valid ones: {list(MODEL_NAME_TO_OPTIONS.keys())}",
    )
    parser.add_argument(
        "-q",
        "--quantize",
        action="store_true",
        required=False,
        default=False,
        help="Flag for producing quantized or floating-point model",
    )
    parser.add_argument(
        "-d",
        "--delegate",
        action="store_true",
        required=False,
        default=True,
        help="Flag for producing XNNPACK delegated model",
    )

    args = parser.parse_args()

    if not args.delegate:
        raise NotImplementedError(
            "T161880157: Quantization-only without delegation is not supported yet"
        )

    if args.model_name not in MODEL_NAME_TO_OPTIONS and args.quantize:
        raise RuntimeError(
            f"Model {args.model_name} is not a valid name. or not quantizable right now, "
            "please contact executorch team if you want to learn why or how to support "
            "quantization for the requested model"
            f"Available models are {list(MODEL_NAME_TO_OPTIONS.keys())}."
        )

    model, example_inputs = EagerModelFactory.create_model(
        *MODEL_NAME_TO_MODEL[args.model_name]
    )

    model = model.eval()

    partitioner = XnnpackFloatingPointPartitioner
    if args.quantize:
        logging.info("Quantizing Model...")
        model = quantize(model, example_inputs)
        # TODO(T161849167): Partitioner will eventually be a single partitioner for both fp32 and quantized models
        partitioner = XnnpackQuantizedPartitioner

    # TODO(T161852812): Delegate implementation is currently on an unlifted graph.
    # It will eventually be changed to a lifted graph, in which _unlift=False,
    edge = export_to_edge(
        model,
        example_inputs,
        capture_config=CaptureConfig(enable_aot=True, _unlift=True),
        edge_compile_config=EdgeCompileConfig(
            # TODO(T162080278): Duplicated Dequant nodes will be in quantizer spec
            _check_ir_validity=False if args.quantize else True,
            passes=[DuplicateDequantNodePass()],
        ),
    )
    logging.info(f"Exported graph:\n{edge.exported_program.graph}")

    edge.exported_program = to_backend(edge.exported_program, partitioner)
    logging.info(f"Lowered graph:\n{edge.exported_program.graph}")

    exec_prog = edge.to_executorch()

    quant_tag = "q8" if args.quantize else "fp32"
    model_name = f"{args.model_name}_xnnpack_{quant_tag}"
    save_pte_program(exec_prog.buffer, model_name)
