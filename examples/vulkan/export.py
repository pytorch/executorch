# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting models to flatbuffer with the Vulkan delegate

# pyre-unsafe

import argparse
import logging

import backends.vulkan.test.utils as test_utils

import torch

from executorch.backends.transforms.convert_dtype_pass import I64toI32
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.devtools import BundledProgram
from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.devtools.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.extension.export_util.utils import save_pte_program
from executorch.extension.pytree import tree_flatten
from torch.export import export

from ..models import MODEL_NAME_TO_MODEL
from ..models.model_factory import EagerModelFactory

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def main() -> None:
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        help=f"provide a model name. Valid ones: {list(MODEL_NAME_TO_MODEL.keys())}",
    )

    parser.add_argument(
        "-fp16",
        "--force_fp16",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force fp32 tensors to be converted to fp16 internally. Input/s outputs "
        "will be converted to/from fp32 when entering/exiting the delegate. Default is "
        "False",
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

    parser.add_argument(
        "-b",
        "--bundled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Export as bundled program (.bpte) instead of regular program (.pte). Default is False",
    )

    parser.add_argument(
        "-t",
        "--test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Execute lower_module_and_test_output to validate the model. Default is False",
    )

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
    if args.force_fp16:
        compile_options["force_fp16"] = True

    # Configure Edge compilation
    edge_compile_config = EdgeCompileConfig(
        _skip_dim_order=False,  # Proper handling for Vulkan memory format
    )

    logging.info(f"Exporting model {args.model_name} with Vulkan delegate")

    # Export the model using torch.export
    if dynamic_shapes is not None:
        program = export(
            model, example_inputs, dynamic_shapes=dynamic_shapes, strict=args.strict
        )
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

    logging.info(
        f"Exported and lowered graph:\n{edge_program.exported_program().graph}"
    )

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
    output_filename = f"{args.model_name}_vulkan"

    atol = 1e-4
    rtol = 1e-4

    # If forcing fp16, then numerical divergence is expected
    if args.force_fp16:
        atol = 2e-2
        rtol = 1e-1

    # Test the model if --test flag is provided
    if args.test:
        test_result = test_utils.run_and_check_output(
            reference_model=model,
            executorch_program=exec_prog,
            sample_inputs=example_inputs,
            atol=atol,
            rtol=rtol,
        )

        if test_result:
            logging.info(
                "✓ Model test PASSED - outputs match reference within tolerance"
            )
        else:
            logging.error("✗ Model test FAILED - outputs do not match reference")
            raise RuntimeError(
                "Model validation failed: ExecutorTorch outputs do not match reference model outputs"
            )

    if args.bundled:
        # Create bundled program
        logging.info("Creating bundled program with test cases")

        # Generate expected outputs by running the model
        expected_outputs = [model(*example_inputs)]

        # Flatten sample inputs to match expected format
        inputs_flattened, _ = tree_flatten(example_inputs)

        # Create test suite with the sample inputs and expected outputs
        test_suites = [
            MethodTestSuite(
                method_name="forward",
                test_cases=[
                    MethodTestCase(
                        inputs=inputs_flattened,
                        expected_outputs=expected_outputs,
                    )
                ],
            )
        ]

        # Create bundled program
        bp = BundledProgram(exec_prog, test_suites)

        # Serialize to flatbuffer
        bp_buffer = serialize_from_bundled_program_to_flatbuffer(bp)

        # Save bundled program
        bundled_output_path = f"{args.output_dir}/{output_filename}.bpte"
        with open(bundled_output_path, "wb") as file:
            file.write(bp_buffer)

        logging.info(
            f"Bundled program exported and saved as {output_filename}.bpte in {args.output_dir}"
        )
    else:
        # Save regular program
        save_pte_program(exec_prog, output_filename, args.output_dir)
        logging.info(
            f"Model exported and saved as {output_filename}.pte in {args.output_dir}"
        )


if __name__ == "__main__":
    with torch.no_grad():
        main()  # pragma: no cover
