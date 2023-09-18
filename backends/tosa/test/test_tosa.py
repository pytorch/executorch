#
# SPDX-FileCopyrightText: Copyright 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: BSD-3-Clause
#

import json
import os
import subprocess

import executorch.exir as exir
import numpy as np
from executorch.backends.tosa.test.test_tosa_models import TestList, TosaProfile
from executorch.backends.tosa.tosa_backend import TosaPartitioner

from executorch.exir.backend.backend_api import to_backend

# Assumes you have these two tools on your path
TOSA_REF_MODEL_PATH = "tosa_reference_model"
VELA_COMPILER_PATH = "vela"

# Config for Capturting the weights, will be moved in the future
_CAPTURE_CONFIG = exir.CaptureConfig(enable_aot=True, enable_dynamic_shape=False)
_EDGE_COMPILE_CONFIG = exir.EdgeCompileConfig(
    _check_ir_validity=False,
)


def tosa_run_test(op, profile=TosaProfile.MI):  # noqa: C901
    #
    # Minimal sequence to take model through TosaPartitioner and emit
    # tosaout/ debug directory containing the flatbuffer - assumes one and will only save last output
    # tosaout is generated even for partial/broken subgraph capture to aid in debg
    # delegated.pte containing the flatbuffer within the executorch flatbuffer binary
    #
    print("\n\033[96m" + "Processing:::{} ".format(op) + "\033[0m")

    # TODO: Don't know how to pass data into tosa backend, setting as an env var as a workaround
    os.environ["TOSA_TESTING_OP"] = str(op)

    model = TestList[op]
    if model.inputs.get(profile) is None:
        print(
            "\033[96m" + "Skipping test as no inputs for this TOSA profile" + "\033[0m"
        )
        return

    torch_output = model.forward(*model.inputs[profile])

    TORCH_OUT_PATH = os.path.join("torchout", op)
    torch_dir_exists = os.path.exists(TORCH_OUT_PATH)
    if not torch_dir_exists:
        os.makedirs(TORCH_OUT_PATH)

    # Save ground truth results to file
    with open(TORCH_OUT_PATH + "/torch_output.npy", "wb") as f:
        np.save(f, torch_output.detach().numpy())

    # capture ExirExportedProgram
    captured_model = exir.capture(
        model, model.inputs[profile], _CAPTURE_CONFIG
    ).to_edge(_EDGE_COMPILE_CONFIG)
    # convert ExportedProgram using TosaPartitioner and assign back to captured model
    captured_model.exported_program = to_backend(
        captured_model.exported_program, TosaPartitioner
    )
    # Output ExecutorchProgram from ExportedProgram
    exec_prog = captured_model.to_executorch()

    # Emit TOSA test data from the model inputs - assumes whole graph lowered so we just have
    # placeholders for the TOSA delegate.
    # - Skips placeholders which are encoded as constants (i.e. are already captured weights)
    # - Assumes argument order is fixed
    argument_names = []
    for node in captured_model.exported_program.graph.nodes:
        if node.op == "placeholder":
            if (
                node.name
                in captured_model.exported_program.graph_signature.inputs_to_parameters
            ):
                pass
            elif (
                node.name
                in captured_model.exported_program.graph_signature.inputs_to_buffers
            ):
                pass
            else:
                argument_names.append(node.name)
        else:
            break

    TOSA_OUT_PATH = os.path.join("tosaout", op)
    tosa_dir_exists = os.path.exists(TOSA_OUT_PATH)
    if not tosa_dir_exists:
        os.makedirs(TOSA_OUT_PATH)

    for arg in zip(argument_names, model.inputs[profile]):
        name = arg[0]
        data = arg[1].detach().numpy()
        path = TOSA_OUT_PATH + "/" + name + ".npy"
        np.save(path, data, allow_pickle=False)

    # this is the .pte binary file
    with open(TORCH_OUT_PATH + "/delegated.pte", "wb") as fh:
        fh.write(exec_prog.buffer)

    # Convert TOSA Flatbuffer into JSON format for human debugging
    cmd_flatc = (
        "flatc"
        + " -o "
        + TOSA_OUT_PATH
        + " --raw-binary -t ./backends/tosa/serialization_lib/schema/tosa.fbs -- ./"
        + TOSA_OUT_PATH
        + "/output.tosa"
    )
    subprocess.run([cmd_flatc], shell=True, check=True)

    ### Run the TOSA flatbuffer through TOSA Ref_Model and print the results
    DESC_FILE_NAME = "/desc.json"
    DESC_FILE_PATH = TOSA_OUT_PATH + DESC_FILE_NAME
    cmd_ref_model = TOSA_REF_MODEL_PATH + " --test_desc " + DESC_FILE_PATH
    subprocess.run([cmd_ref_model], shell=True, check=True)

    ## Load in the JSON File, Read the tosa output
    desc_file = open(DESC_FILE_PATH)
    desc_json = json.load(desc_file)
    tosa_out_filenames = desc_json["ofm_file"]
    for tosa_out_fm_file_name in tosa_out_filenames:
        f = open(TOSA_OUT_PATH + "/" + tosa_out_fm_file_name, "rb")
        tosa_output = np.load(f)

    ## Read the Torch Output
    torch_file = open(TORCH_OUT_PATH + "/torch_output.npy", "rb")
    torch_output = np.load(torch_file)

    ## Compare Tosa and Torch Results
    if np.allclose(tosa_output, torch_output, 1e-1, equal_nan=True):
        print(
            "\033[92m"
            + "Torch and Tosa Reference results are matching for operator: "
            + op
            + "\033[0m"
        )
    else:
        print("\033[91m" + "Sorry, Torch and Tosa Reference Results Do not Match!")
        print("============================")
        print("TOSA Output Shape is: " + str(tosa_output.shape))
        print("TOSA Output is: ")
        print(tosa_output)
        print("\033[93m")
        print("============================")
        print("Torch Output Shape is: " + str(torch_output.shape))
        print("Torch Output is: ")
        print(torch_output)
        print("\033[0m")

    if profile == TosaProfile.BI:
        cmd_vela = "cd " + TOSA_OUT_PATH + "; " + VELA_COMPILER_PATH + " ./output.tosa"
        try:
            subprocess.run([cmd_vela], shell=True, check=True)
            print("\033[92m" + "Vela compile worked for: " + op + "\033[0m")
        except:
            print("\033[91m" + "Vela compile failed for: " + op + "\033[0m")
    else:
        print("\033[96m" + "Skipping Vela test on non-BI profile." + "\033[0m")


for op in TestList:
    tosa_run_test(op, profile=TosaProfile.MI)
