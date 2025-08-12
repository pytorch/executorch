# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting models to flatbuffer with the Vulkan delegate

# pyre-unsafe

import argparse
import logging
import torch

from executorch.backends.transforms.convert_dtype_pass import I64toI32
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
)
from executorch.exir import to_edge_transform_and_lower
from executorch.extension.export_util.utils import save_pte_program

from ..models import MODEL_NAME_TO_MODEL
from ..models.model_factory import EagerModelFactory
from torch.export import export


FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        help=f"provide a model name. Valid ones: {list(MODEL_NAME_TO_MODEL.keys())}",
    )

    parser.add_argument(
        "-s",
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to export with strict mode. Default is True",
    )

    parser.add_argument(
        "-a",
        "--segment_alignment",
        required=False,
        help="specify segment alignment in hex. Default is 0x1000. Use 0x4000 for iOS",
    )

    parser.add_argument(
        "-e",
        "--external_constants",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save constants in external .ptd file. Default is False",
    )

    parser.add_argument(
        "-d",
        "--dynamic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable dynamic shape support. Default is False",
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

    if args.model_name not in MODEL_NAME_TO_MODEL:
        raise RuntimeError(
            f"Model {args.model_name} is not a valid name. "
            f"Available models are {list(MODEL_NAME_TO_MODEL.keys())}."
        )

    model, example_inputs, _, dynamic_shapes = EagerModelFactory.create_model(
        *MODEL_NAME_TO_MODEL[args.model_name]
    )

    # Prepare model
    model.eval()

    # Setup compile options
    compile_options = {}
    if args.dynamic or dynamic_shapes is not None:
        compile_options["require_dynamic_shapes"] = True

    # Configure Edge compilation
    edge_compile_config = EdgeCompileConfig(
        _skip_dim_order=False,  # Proper handling for Vulkan memory format
    )

    logging.info(f"Exporting model {args.model_name} with Vulkan delegate")

    # Export the model using torch.export
    if dynamic_shapes is not None:
        program = export(model, example_inputs, dynamic_shapes=dynamic_shapes, strict=args.strict)
    else:
        program = export(model, example_inputs, strict=args.strict)

    # Transform and lower with Vulkan partitioner
    edge_program = to_edge_transform_and_lower(
        program,
        compile_config=edge_compile_config,
        transform_passes=[
            I64toI32(edge_compile_config._skip_dim_order),
        ],
        partitioner=[VulkanPartitioner(compile_options)],
        generate_etrecord=args.etrecord,
    )

    logging.info(f"Exported and lowered graph:\n{edge_program.exported_program().graph}")

    # Configure backend options
    backend_config = ExecutorchBackendConfig(external_constants=args.external_constants)
    if args.segment_alignment is not None:
        backend_config.segment_alignment = int(args.segment_alignment, 16)

    # Create executorch program
    exec_prog = edge_program.to_executorch(config=backend_config)

    # Save ETRecord if requested
    if args.etrecord:
        exec_prog.get_etrecord().save(args.etrecord)
        logging.info(f"Saved ETRecord to {args.etrecord}")

    # Save the program
    model_name = f"{args.model_name}_vulkan"
    save_pte_program(exec_prog, model_name, args.output_dir)
    logging.info(f"Model exported and saved as {model_name}.pte in {args.output_dir}")


if __name__ == "__main__":
    with torch.no_grad():
        main()  # pragma: no cover
