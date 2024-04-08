#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

# Example script for exporting simple models to flatbuffer

import argparse
import copy
import logging

import torch
from executorch import exir
from executorch.backends.apple.mps.mps_preprocess import MPSBackend
from executorch.backends.apple.mps.partition.mps_partitioner import MPSPartitioner

from executorch.exir import (
    EdgeCompileConfig,
    EdgeProgramManager,
    ExecutorchProgramManager,
)
from executorch.exir.backend.backend_api import to_backend
from executorch.exir.backend.backend_details import CompileSpec
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.sdk import BundledProgram, generate_etrecord
from executorch.sdk.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.sdk.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)

from ....models import MODEL_NAME_TO_MODEL
from ....models.model_factory import EagerModelFactory

from ....portable.utils import export_to_edge, save_pte_program

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        help=f"Provide model name. Valid ones: {list(MODEL_NAME_TO_MODEL.keys())}",
    )

    parser.add_argument(
        "--use_fp16",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to automatically convert float32 operations to float16 operations.",
    )

    parser.add_argument(
        "--use_partitioner",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use MPS partitioner to run the model instead of using whole graph lowering.",
    )

    parser.add_argument(
        "-b",
        "--bundled",
        action="store_true",
        required=False,
        default=False,
        help="Flag for bundling inputs and outputs in the final flatbuffer program",
    )

    parser.add_argument(
        "--generate_etrecord",
        action="store_true",
        required=False,
        default=False,
        help="Generate ETRecord metadata to link with runtime results (used for profiling)",
    )

    args = parser.parse_args()

    if args.model_name not in MODEL_NAME_TO_MODEL:
        raise RuntimeError(f"Available models are {list(MODEL_NAME_TO_MODEL.keys())}.")

    model, example_inputs, _ = EagerModelFactory.create_model(
        *MODEL_NAME_TO_MODEL[args.model_name]
    )

    model = model.eval()

    # pre-autograd export. eventually this will become torch.export
    model = torch._export.capture_pre_autograd_graph(model, example_inputs)

    edge: EdgeProgramManager = export_to_edge(
        model,
        example_inputs,
        edge_compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    edge_program_manager_copy = copy.deepcopy(edge)

    compile_specs = [CompileSpec("use_fp16", bytes([args.use_fp16]))]

    logging.info(f"Edge IR graph:\n{edge.exported_program().graph}")
    if args.use_partitioner:
        edge = edge.to_backend(MPSPartitioner(compile_specs=compile_specs))
        logging.info(f"Lowered graph:\n{edge.exported_program().graph}")

        executorch_program = edge.to_executorch(
            config=ExecutorchBackendConfig(extract_constant_segment=False)
        )
    else:
        lowered_module = to_backend(
            MPSBackend.__name__, edge.exported_program(), compile_specs
        )
        executorch_program: ExecutorchProgramManager = export_to_edge(
            lowered_module,
            example_inputs,
            edge_compile_config=exir.EdgeCompileConfig(_check_ir_validity=False),
        ).to_executorch(config=ExecutorchBackendConfig(extract_constant_segment=False))

    model_name = f"{args.model_name}_mps"

    if args.bundled:
        method_test_suites = [
            MethodTestSuite(
                method_name="forward",
                test_cases=[
                    MethodTestCase(
                        inputs=example_inputs, expected_outputs=[model(*example_inputs)]
                    )
                ],
            )
        ]
        logging.info(f"Expected output: {model(*example_inputs)}")

        bundled_program = BundledProgram(executorch_program, method_test_suites)
        bundled_program_buffer = serialize_from_bundled_program_to_flatbuffer(
            bundled_program
        )
        model_name = f"{model_name}_bundled"
        extension = "fp16"
        if not args.use_fp16:
            extension = "fp32"
        model_name = f"{model_name}_{extension}"

    if args.generate_etrecord:
        etrecord_path = "etrecord.bin"
        logging.info("generating etrecord.bin")
        generate_etrecord(etrecord_path, edge_program_manager_copy, executorch_program)

    save_pte_program(executorch_program, model_name)
