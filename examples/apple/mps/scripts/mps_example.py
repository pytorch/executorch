#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

# Example script for exporting simple models to flatbuffer

import argparse
import copy
import logging

import torch
from examples.apple.mps.scripts.bench_utils import bench_torch, compare_outputs
from executorch import exir
from executorch.backends.apple.mps import MPSBackend
from executorch.backends.apple.mps.partition import MPSPartitioner
from executorch.devtools import BundledProgram, generate_etrecord
from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.devtools.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)

from executorch.exir import (
    EdgeCompileConfig,
    EdgeProgramManager,
    ExecutorchProgramManager,
)
from executorch.exir.backend.backend_api import to_backend
from executorch.exir.backend.backend_details import CompileSpec
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.extension.export_util.utils import export_to_edge, save_pte_program

from ....models import MODEL_NAME_TO_MODEL
from ....models.model_factory import EagerModelFactory

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def get_bundled_program(executorch_program, example_inputs, expected_output):
    method_test_suites = [
        MethodTestSuite(
            method_name="forward",
            test_cases=[
                MethodTestCase(
                    inputs=example_inputs, expected_outputs=[expected_output]
                )
            ],
        )
    ]
    logging.info(f"Expected output: {expected_output}")

    bundled_program = BundledProgram(executorch_program, method_test_suites)
    bundled_program_buffer = serialize_from_bundled_program_to_flatbuffer(
        bundled_program
    )
    return bundled_program_buffer


def parse_args():
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
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use MPS partitioner to run the model instead of using whole graph lowering.",
    )

    parser.add_argument(
        "--bench_pytorch",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Bench ExecuTorch MPS foward pass with PyTorch MPS forward pass.",
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
        "-c",
        "--check_correctness",
        action="store_true",
        required=False,
        default=False,
        help="Whether to compare the ExecuTorch MPS results with the PyTorch forward pass",
    )

    parser.add_argument(
        "--generate_etrecord",
        action="store_true",
        required=False,
        default=False,
        help="Generate ETRecord metadata to link with runtime results (used for profiling)",
    )

    parser.add_argument(
        "--checkpoint",
        required=False,
        default=None,
        help="checkpoing for llama model",
    )

    parser.add_argument(
        "--params",
        required=False,
        default=None,
        help="params for llama model",
    )

    args = parser.parse_args()
    return args


def get_model_config(args):
    model_config = {}
    model_config["module_name"] = MODEL_NAME_TO_MODEL[args.model_name][0]
    model_config["model_class_name"] = MODEL_NAME_TO_MODEL[args.model_name][1]

    if args.model_name == "llama2":
        if args.checkpoint:
            model_config["checkpoint"] = args.checkpoint
        if args.params:
            model_config["params"] = args.params
        model_config["use_kv_cache"] = True
    return model_config


if __name__ == "__main__":
    args = parse_args()

    if args.model_name not in MODEL_NAME_TO_MODEL:
        raise RuntimeError(f"Available models are {list(MODEL_NAME_TO_MODEL.keys())}.")

    model_config = get_model_config(args)
    model, example_inputs, _ = EagerModelFactory.create_model(**model_config)

    model = model.eval()

    # Deep copy the model inputs to check against PyTorch forward pass
    if args.check_correctness or args.bench_pytorch:
        model_copy = copy.deepcopy(model)
        inputs_copy = []
        for t in example_inputs:
            inputs_copy.append(t.detach().clone())
        inputs_copy = tuple(inputs_copy)

    # pre-autograd export. eventually this will become torch.export
    with torch.no_grad():
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
            config=ExecutorchBackendConfig(extract_delegate_segments=False)
        )
    else:
        lowered_module = to_backend(
            MPSBackend.__name__, edge.exported_program(), compile_specs
        )
        executorch_program: ExecutorchProgramManager = export_to_edge(
            lowered_module,
            example_inputs,
            edge_compile_config=exir.EdgeCompileConfig(_check_ir_validity=False),
        ).to_executorch(config=ExecutorchBackendConfig(extract_delegate_segments=False))

    model_name = f"{args.model_name}_mps"

    if args.bundled:
        expected_output = model(*example_inputs)
        bundled_program_buffer = get_bundled_program(
            executorch_program, example_inputs, expected_output
        )
        model_name = f"{model_name}_bundled"
        extension = "fp16"
        if not args.use_fp16:
            extension = "fp32"
        model_name = f"{model_name}_{extension}.pte"

    if args.generate_etrecord:
        etrecord_path = "etrecord.bin"
        logging.info("generating etrecord.bin")
        generate_etrecord(etrecord_path, edge_program_manager_copy, executorch_program)

    if args.bundled:
        with open(model_name, "wb") as file:
            file.write(bundled_program_buffer)
        logging.info(f"Saved bundled program to {model_name}")
    else:
        save_pte_program(executorch_program, model_name)

    if args.bench_pytorch:
        bench_torch(executorch_program, model_copy, example_inputs, model_name)

    if args.check_correctness:
        compare_outputs(
            executorch_program, model_copy, inputs_copy, model_name, args.use_fp16
        )
