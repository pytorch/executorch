# Copyright 2023 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import subprocess
import tempfile

import numpy as np
from executorch.backends.arm.test.test_models import TestList, TosaProfile
from executorch.backends.arm.test.test_tosa import export_model, prepare_model_and_ref

from executorch.exir.backend.compile_spec_schema import CompileSpec

# Assumes you have these two tools on your path
TOSA_REF_MODEL_PATH = "tosa_reference_model"
VELA_COMPILER_PATH = "vela"

# Temp directory that any debug output is written to
DEBUG_OUTPUT_PATH = tempfile.mkdtemp(prefix="arm_tosa_")


def tosa_ref_dump_inputs(model_edge, inputs, path):
    # Emit TOSA test data from the model inputs - assumes whole graph lowered so we just have
    # placeholders for the TOSA delegate. Emits data in tosa_ref_model expected layout.
    # - Skips placeholders which are encoded as constants (i.e. are already captured weights)
    # - Assumes argument order is fixed
    argument_names = []
    for node in model_edge.exported_program.graph.nodes:
        gs = model_edge.exported_program.graph_signature
        if node.op == "placeholder":
            print("got placholder", node.target)
            if node.name in gs.inputs_to_parameters:
                pass
            elif node.name in gs.inputs_to_buffers:
                pass
            else:
                argument_names.append(node.name)
        else:
            break

    for arg in zip(argument_names, inputs):
        name = arg[0]
        data = arg[1].detach().numpy()
        file_path = path + "/" + name + ".npy"
        np.save(file_path, data, allow_pickle=False)


def tosa_run_test(op, profile=TosaProfile.MI):  # noqa: C901
    #
    # Minimal sequence to take model through TosaPartitioner and emit
    # tosaout/ debug directory containing the flatbuffer - assumes one and will only save last output
    # tosaout is generated even for partial/broken subgraph capture to aid in debg
    # delegated.pte containing the flatbuffer within the executorch flatbuffer binary
    #
    print(f"\n\033[96mProcessing:::{op}\033[0m")
    print(f"\033[96m Debug output path for intermediates: {DEBUG_OUTPUT_PATH}\033[0m")

    # Debug output for TORCH
    TORCH_OUT_PATH = os.path.join(DEBUG_OUTPUT_PATH, op, "torch", "")
    os.makedirs(TORCH_OUT_PATH, exist_ok=True)

    # Debug output for TOSA
    TOSA_OUT_PATH = os.path.join(DEBUG_OUTPUT_PATH, op, "tosa", "")
    os.makedirs(TOSA_OUT_PATH, exist_ok=True)

    # Debug flag for compilers
    compile_spec = [CompileSpec("debug_tosa_path", bytes(TOSA_OUT_PATH, "utf8"))]

    model, inputs, torch_output = prepare_model_and_ref(op, profile)

    if inputs is None:
        print("\033[96m Skipping, no inputs for TOSA profile \033[0m")
        return

    captured_model, exec_prog = export_model(model, inputs, compile_spec)

    # Save ground truth results to file
    with open(TORCH_OUT_PATH + "/torch_output.npy", "wb") as f:
        np.save(f, torch_output.detach().numpy())

    tosa_ref_dump_inputs(captured_model, inputs, TOSA_OUT_PATH)

    print(TORCH_OUT_PATH, TOSA_OUT_PATH)

    # this is the .pte binary file
    with open(TORCH_OUT_PATH + "/delegated.pte", "wb") as fh:
        fh.write(exec_prog.buffer)

    # Convert TOSA Flatbuffer into JSON format for human debugging
    cmd_flatc = (
        "flatc"
        + " -o "
        + TOSA_OUT_PATH
        + " --raw-binary -t ./backends/arm/third-party/serialization_lib/schema/tosa.fbs -- "
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


# Temp systest mode for running all models against both inference profiles
if __name__ == "__main__":
    for op in TestList:
        tosa_run_test(op, profile=TosaProfile.MI)

    # TODO: haven't added the quantized lowerings for BI, comment out for now
    # for op in TestList:
    #     tosa_run_test(op, profile=TosaProfile.BI)
