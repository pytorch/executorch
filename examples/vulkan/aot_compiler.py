# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for compiling models with Vulkan delegation

# pyre-unsafe

import argparse
import logging

import torch
from executorch.backends.transforms.convert_dtype_pass import I64toI32
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.extension.export_util.utils import save_pte_program

from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchao.quantization.pt2e.quantizer import Quantizer

from ..models import MODEL_NAME_TO_MODEL
from ..models.model_factory import EagerModelFactory


FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def quantize_and_lower_module(
    model: torch.nn.Module,
    sample_inputs,
    quantizer: Quantizer,
    dynamic_shapes=None,
) -> torch.nn.Module:
    """Quantize a model and lower it with Vulkan delegation"""
    compile_options = {}
    if dynamic_shapes is not None:
        compile_options["require_dynamic_shapes"] = True

    edge_compile_config = EdgeCompileConfig(
        _skip_dim_order=False,  # Proper handling for Vulkan memory format
    )

    program = torch.export.export_for_training(
        model, sample_inputs, dynamic_shapes=dynamic_shapes, strict=True
    ).module()

    program = prepare_pt2e(program, quantizer)
    # Calibrate
    program(*sample_inputs)

    program = convert_pt2e(program)

    program = torch.export.export(program, sample_inputs, dynamic_shapes=dynamic_shapes)

    edge_program = to_edge_transform_and_lower(
        program,
        compile_config=edge_compile_config,
        transform_passes=[
            I64toI32(edge_compile_config._skip_dim_order),
        ],
        partitioner=[VulkanPartitioner(compile_options)],
    )

    return edge_program


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        help=f"Model name. Valid ones: {list(MODEL_NAME_TO_MODEL.keys())}",
    )
    parser.add_argument(
        "-q",
        "--quantize",
        action="store_true",
        required=False,
        default=False,
        help="Produce a quantized model. Note: Quantization support may vary by model.",
    )
    parser.add_argument(
        "-d",
        "--delegate",
        action="store_true",
        required=False,
        default=True,
        help="Produce a Vulkan delegated model",
    )
    parser.add_argument(
        "-y",
        "--dynamic",
        action="store_true",
        required=False,
        default=False,
        help="Enable dynamic shape support",
    )
    parser.add_argument(
        "-r",
        "--etrecord",
        required=False,
        default="",
        help="Generate and save an ETRecord to the given file location",
    )
    parser.add_argument("-o", "--output_dir", default=".", help="output directory")

    args = parser.parse_args()

    model, example_inputs, _, dynamic_shapes = EagerModelFactory.create_model(
        *MODEL_NAME_TO_MODEL[args.model_name]
    )

    model = model.eval()

    if args.dynamic and dynamic_shapes is None:
        logging.warning("Dynamic shapes requested but not available for this model.")

    dynamic_shapes_to_use = dynamic_shapes if args.dynamic else None

    # Configure Edge compilation
    edge_compile_config = EdgeCompileConfig(
        _skip_dim_order=False,  # Proper handling for Vulkan memory format
        _check_ir_validity=True,
    )

    # Setup compile options
    compile_options = {}
    if dynamic_shapes_to_use is not None:
        compile_options["require_dynamic_shapes"] = True

    if args.quantize:
        logging.info("Quantization for Vulkan not fully supported yet. Using experimental path.")
        try:
            # Try to import quantization utilities if available
            try:
                from ..quantization.utils import get_quantizer_for_model
                quantizer = get_quantizer_for_model(args.model_name)
            except ImportError:
                # If the specific utility isn't available, create a basic quantizer
                logging.warning("Quantization utils not found. Using default quantizer.")
                from torchao.quantization.pt2e.quantizer import get_default_quantizer
                quantizer = get_default_quantizer()

            edge = quantize_and_lower_module(
                model, example_inputs, quantizer, dynamic_shapes=dynamic_shapes_to_use
            )
        except (ImportError, NotImplementedError) as e:
            logging.error(f"Quantization failed: {e}")
            logging.info("Falling back to non-quantized path")
            # Export the model using torch.export
            program = torch.export.export(
                model, example_inputs, dynamic_shapes=dynamic_shapes_to_use, strict=True
            )

            # Transform and lower with Vulkan partitioner
            edge = to_edge_transform_and_lower(
                program,
                compile_config=edge_compile_config,
                transform_passes=[
                    I64toI32(edge_compile_config._skip_dim_order),
                ],
                partitioner=[VulkanPartitioner(compile_options)],
                generate_etrecord=args.etrecord,
            )
    else:
        # Standard non-quantized path
        # Export the model using torch.export
        program = torch.export.export(
            model, example_inputs, dynamic_shapes=dynamic_shapes_to_use, strict=True
        )

        # Transform and lower with Vulkan partitioner
        edge = to_edge_transform_and_lower(
            program,
            compile_config=edge_compile_config,
            transform_passes=[
                I64toI32(edge_compile_config._skip_dim_order),
            ],
            partitioner=[VulkanPartitioner(compile_options)],
            generate_etrecord=args.etrecord,
        )

    logging.info(f"Exported and lowered graph:\n{edge.exported_program().graph}")

    exec_prog = edge.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False)
    )

    if args.etrecord:
        exec_prog.get_etrecord().save(args.etrecord)
        logging.info(f"Saved ETRecord to {args.etrecord}")

    quant_tag = "q8" if args.quantize else "fp32"
    model_name = f"{args.model_name}_vulkan_{quant_tag}"
    save_pte_program(exec_prog, model_name, args.output_dir)
    logging.info(f"Model exported and saved as {model_name}.pte in {args.output_dir}")
