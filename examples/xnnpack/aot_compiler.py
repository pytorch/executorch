# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

# Example script for exporting simple models to flatbuffer

import argparse
import copy
import logging

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.devtools import generate_etrecord
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig
from executorch.extension.export_util.utils import export_to_edge, save_pte_program

from ..models import MODEL_NAME_TO_MODEL
from ..models.model_factory import EagerModelFactory
from . import MODEL_NAME_TO_OPTIONS
from .quantization.utils import quantize


FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        help=f"Model name. Valid ones: {list(MODEL_NAME_TO_OPTIONS.keys())}",
    )
    parser.add_argument(
        "-q",
        "--quantize",
        action="store_true",
        required=False,
        default=False,
        help="Produce an 8-bit quantized model",
    )
    parser.add_argument(
        "-d",
        "--delegate",
        action="store_true",
        required=False,
        default=True,
        help="Produce an XNNPACK delegated model",
    )
    parser.add_argument(
        "-r",
        "--etrecord",
        required=False,
        help="Generate and save an ETRecord to the given file location",
    )
    parser.add_argument("-o", "--output_dir", default=".", help="output directory")

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

    model, example_inputs, _ = EagerModelFactory.create_model(
        *MODEL_NAME_TO_MODEL[args.model_name]
    )

    model = model.eval()
    # pre-autograd export. eventually this will become torch.export
    model = torch.export.export_for_training(model, example_inputs).module()

    if args.quantize:
        logging.info("Quantizing Model...")
        # TODO(T165162973): This pass shall eventually be folded into quantizer
        model = quantize(model, example_inputs)

    edge = export_to_edge(
        model,
        example_inputs,
        edge_compile_config=EdgeCompileConfig(
            _check_ir_validity=False if args.quantize else True,
            _skip_dim_order=True,  # TODO(T182187531): enable dim order in xnnpack
        ),
    )
    logging.info(f"Exported graph:\n{edge.exported_program().graph}")

    # this is needed for the ETRecord as lowering modifies the graph in-place
    edge_copy = copy.deepcopy(edge)

    edge = edge.to_backend(XnnpackPartitioner())
    logging.info(f"Lowered graph:\n{edge.exported_program().graph}")

    exec_prog = edge.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False)
    )

    if args.etrecord is not None:
        generate_etrecord(args.etrecord, edge_copy, exec_prog)
        logging.info(f"Saved ETRecord to {args.etrecord}")

    quant_tag = "q8" if args.quantize else "fp32"
    model_name = f"{args.model_name}_xnnpack_{quant_tag}"
    save_pte_program(exec_prog, model_name, args.output_dir)
