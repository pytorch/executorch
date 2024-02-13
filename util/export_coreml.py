#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

# Example script for exporting simple models to flatbuffer

import copy
import logging
import os

import torch
from executorch import exir
from executorch.backends.apple.coreml.compiler import CoreMLBackend

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

compute_units = ["cpu_only", "cpu_and_gpu", "cpu_and_ane", "all"]

def export_to_coreml(model, example_inputs, compute_units="all"):

    model = model.eval()

    # pre-autograd export. eventually this will become torch.export
    model = torch._export.capture_pre_autograd_graph(model, example_inputs)

    edge: EdgeProgramManager = export_to_edge(
        model,
        example_inputs,
        edge_compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    edge_program_manager_copy = copy.deepcopy(edge)

    lowered_module = to_backend(
        CoreMLBackend.__name__,
        edge.exported_program,
        [CompileSpec("compute_units", bytes(compute_units, "utf-8"))],
    )

    logging.info(f"Edge IR graph:\n{edge.exported_program().graph}")

    lowered_module(*example_inputs)

    _CAPTURE_CONFIG = exir.CaptureConfig(enable_aot=True, _unlift=False)
    _EDGE_COMPILE_CONFIG = exir.EdgeCompileConfig(
        _check_ir_validity=False,
    )

    exec_prog = (
        exir.capture(lowered_module, example_inputs, _CAPTURE_CONFIG)
        .to_edge(_EDGE_COMPILE_CONFIG)
        .to_executorch(
            config=exir.ExecutorchBackendConfig(extract_constant_segment=False)
        )
    )

    filename = os.path.join("", f"model_coreml.pte")
    try:
        with open(filename, "wb") as file:
            file.write(exec_program.buffer)
            logging.info(f"Saved exported program to {filename}")
    except Exception as e:
        logging.error(f"Error while saving to {filename}: {e}")

