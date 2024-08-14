# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting simple models to flatbuffer

import logging
import tempfile

from executorch.backends.cadence.aot.ops_registrations import *  # noqa
import os
from typing import Any, Tuple

from executorch.backends.cadence.aot.compiler import (
    convert_pt2,
    export_to_cadence,
    export_to_edge,
    quantize_pt2,
)
from executorch.backends.cadence.aot.quantizer.quantizer import CadenceQuantizer
from executorch.backends.cadence.runtime import runtime
from executorch.backends.cadence.runtime.executor import BundledProgramManager
from executorch.exir import ExecutorchProgramManager
from torch import nn

from .utils import print_ops_info


FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def _save_pte_program(
    prog: ExecutorchProgramManager, model_name: str, output_dir: str = ""
) -> None:
    if model_name.endswith(".pte"):
        filename = model_name
    else:
        filename = os.path.join(output_dir, f"{model_name}.pte")

    try:
        with open(filename, "wb") as file:
            prog.write_to_file(file)
            logging.info(f"Saved exported program to {filename}")
    except Exception as e:
        logging.error(f"Error while saving to {filename}: {e}")


def _save_bpte_program(
    buffer: bytes,
    model_name: str,
    output_dir: str = "",
) -> None:
    if model_name.endswith(".bpte"):
        filename = model_name
    else:
        filename = os.path.join(output_dir, f"{model_name}.bpte")
    try:
        with open(filename, "wb") as f:
            f.write(buffer)
        logging.info(f"Saved exported program to {filename}")
    except Exception as e:
        logging.error(f"Error while saving to {output_dir}: {e}")


def export_model(
    model: nn.Module,
    example_inputs: Tuple[Any, ...],
    file_name: str = "CadenceDemoModel",
):
    # create work directory for outputs and model binary
    working_dir = tempfile.mkdtemp(dir="/tmp")
    logging.debug(f"Created work directory {working_dir}")

    # convert the model (also called in quantize_pt2)
    converted_model = convert_pt2(model, example_inputs, CadenceQuantizer())

    # Get reference outputs from quantized_model
    ref_outputs = converted_model(*example_inputs)

    # Quantize the model
    quantized_model = quantize_pt2(model, example_inputs)

    # Get edge program (also called in export_to_cadence)
    edge_prog_manager = export_to_edge(quantized_model, example_inputs)

    # Get edge program after Cadence specific passes
    cadence_prog_manager = export_to_cadence(quantized_model, example_inputs)

    exec_prog: ExecutorchProgramManager = cadence_prog_manager.to_executorch()

    logging.info("Final exported graph:\n")
    exec_prog.exported_program().graph_module.graph.print_tabular()

    # Print some information to terminal
    print_ops_info(
        edge_prog_manager.exported_program().graph_module,
        cadence_prog_manager.exported_program().graph_module,
    )

    forward_test_data = BundledProgramManager.bundled_program_test_data_gen(
        method="forward", inputs=example_inputs, expected_outputs=ref_outputs
    )
    bundled_program_manager = BundledProgramManager([forward_test_data])
    buffer = bundled_program_manager._serialize(
        exec_prog,
        bundled_program_manager.get_method_test_suites(),
        forward_test_data,
    )
    # Save the program as pte (default name is CadenceDemoModel.pte)
    _save_pte_program(exec_prog, file_name, working_dir)
    # Save the program as btpe (default name is CadenceDemoModel.bpte)
    _save_bpte_program(buffer, file_name, working_dir)

    logging.debug(
        f"Executorch bundled program buffer saved to {file_name} is {len(buffer)} total bytes"
    )

    # TODO: move to test infra
    runtime.run_and_compare(
        executorch_prog=exec_prog,
        inputs=example_inputs,
        ref_outputs=ref_outputs,
        working_dir=working_dir,
    )
