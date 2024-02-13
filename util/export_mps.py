#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

# Example script for exporting simple models to flatbuffer

import copy
import logging

import torch
from executorch import exir
from executorch.backends.apple.mps.mps_preprocess import MPSBackend
from executorch.backends.apple.mps.partition.mps_partitioner import MPSPartitioner

from executorch.exir import EdgeCompileConfig, EdgeProgramManager
from executorch.exir.backend.backend_api import to_backend
from executorch.exir.backend.backend_details import CompileSpec
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.sdk import BundledProgram, generate_etrecord
from executorch.sdk.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.sdk.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)
from executorch.util.export_edge_ir import export_to_edge

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

def export_to_mps(model, example_inputs, use_partitioner=False, bundled=False,use_fp16=True, generate_etrecord=False):

    model = model.eval()

    # pre-autograd export. eventually this will become torch.export
    model = torch._export.capture_pre_autograd_graph(model, example_inputs)

    edge: EdgeProgramManager = export_to_edge(
        model,
        example_inputs,
        edge_compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    edge_program_manager_copy = copy.deepcopy(edge)

    compile_specs = [CompileSpec("use_fp16", bytes([1]))]

    logging.info(f"Edge IR graph:\n{edge.exported_program().graph}")
    if use_partitioner:
        edge = edge.to_backend(MPSPartitioner(compile_specs=compile_specs))
        logging.info(f"Lowered graph:\n{edge.exported_program().graph}")

        executorch_program = edge.to_executorch(
            config=ExecutorchBackendConfig(extract_constant_segment=False)
        )
    else:
        lowered_module = to_backend(
            MPSBackend.__name__, edge.exported_program(), compile_specs
        )
        executorch_program = (
            exir.capture(
                lowered_module,
                example_inputs,
                exir.CaptureConfig(enable_aot=True, _unlift=False),
            )
            .to_edge(exir.EdgeCompileConfig(_check_ir_validity=False))
            .to_executorch(
                config=ExecutorchBackendConfig(extract_constant_segment=False)
            )
        )

    model_name = f"model_mps"

    if bundled:
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
        if not use_fp16:
            extension = "fp32"
        model_name = f"{model_name}_{extension}"
        program_buffer = bundled_program_buffer
    else:
        program_buffer = executorch_program.buffer

    if generate_etrecord:
        etrecord_path = "etrecord.bin"
        logging.info("generating etrecord.bin")
        generate_etrecord(etrecord_path, edge_program_manager_copy, executorch_program)

    # save_pte_program(program_buffer, model_name)

