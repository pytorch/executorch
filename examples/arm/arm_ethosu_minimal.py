# Copyright 2023 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import subprocess
import tempfile

import executorch.exir as exir

import numpy as np
from executorch.backends.arm.arm_backend import ArmPartitioner
from executorch.backends.arm.test.test_models import TestList, TosaProfile
from executorch.backends.arm.test.test_tosa import prepare_model_and_ref

from executorch.exir.backend.backend_api import to_backend
from executorch.exir.backend.canonical_partitioners.duplicate_dequant_node_pass import (
    DuplicateDequantNodePass,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec

from executorch.exir.dialects._ops import ops as exir_ops

# Assumes you have these two tools on your path
TOSA_REF_MODEL_PATH = "tosa_reference_model"
VELA_COMPILER_PATH = "vela"

# Basic config for graph capture
_CAPTURE_CONFIG = exir.CaptureConfig(enable_aot=True)
_EDGE_COMPILE_CONFIG = exir.EdgeCompileConfig(
    _check_ir_validity=False,
)

EXAMPLE_TEST_LIST = ["simple_add", "simple_add_2"]

#
#
#
#
def tosa_ref_capture_inputs(
    model_edge,
    inputs,
    path,
    input_quantization_scales,
    input_quantization_zps,
    profile=TosaProfile.MI,
):
    # Emit TOSA test data from the model inputs - assumes whole graph lowered so we just have
    # placeholders for the TOSA delegate. Emits data in tosa_ref_model expected layout.
    # - Skips placeholders which are encoded as constants (i.e. are already captured weights)
    # - Assumes argument order is fixed
    argument_names = []
    for node in model_edge.exported_program.graph.nodes:
        gs = model_edge.exported_program.graph_signature
        if node.op == "placeholder":
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

        # Torch is doing Input[FP32]->Q[INT8]->DQ[FP32]->Operator[FP32]->Q[INT]->DQ[FP32]->[Output]FP32
        # Need to quantize the input to INT8 for TOSA comsumption
        if profile is TosaProfile.BI:
            data_quantized = (
                (data / input_quantization_scales[name]) - input_quantization_zps[name]
            ).astype(np.int8)
            np.save(file_path, data_quantized, allow_pickle=False)
        else:
            np.save(file_path, data, allow_pickle=False)

#
# Minimal sequence to take a model through the ArmPartitioner and produce
# both TOSA intermediate output, and an Ethos-U55 command stream within
# the ExecuTorch .pte binary
#
def run_test(op, profile=TosaProfile.MI, output_path="./ethosout/"):
    #
    # Minimal sequence to take model through TosaPartitioner and emit
    # tosaout/ debug directory containing the flatbuffer - assumes one and will only save last output
    # tosaout is generated even for partial/broken subgraph capture to aid in debg
    # delegated.pte containing the flatbuffer within the executorch flatbuffer binary
    #
    print(f"\n\033[96mProcessing:::{op}\033[0m")
    print(f"\033[96mDebug output path for intermediates: {output_path}\033[0m")
    
    os.makedirs(output_path, exist_ok=True)
    
    # Debug output for TORCH
    TORCH_OUT_PATH = os.path.join(output_path, op, "torch", "")
    os.makedirs(TORCH_OUT_PATH, exist_ok=True)

    # Debug output for TOSA
    TOSA_OUT_PATH = os.path.join(output_path, op, "tosa", "")
    os.makedirs(TOSA_OUT_PATH, exist_ok=True)

    model, inputs, torch_output = prepare_model_and_ref(op, profile)

    if inputs is None:
        print("\033[96m Skipping, model has no inputs for TOSA profile \033[0m")
        return

    print(f"  Model: {op}\n  Inputs: {inputs}\n  Outputs: {torch_output}")
    
    # Export model
    model_capture = exir.capture(model, inputs, _CAPTURE_CONFIG)
    model_edge = model_capture.to_edge(_EDGE_COMPILE_CONFIG)

    # Partition with ArmBackend
    ArmPartitioner.compile_spec = [CompileSpec("debug_tosa_path", bytes(TOSA_OUT_PATH, "utf8"))]
    model_edge.exported_program = to_backend(
        model_edge.transform(DuplicateDequantNodePass()).exported_program,
        ArmPartitioner,
    )
    exec_prog = model_edge.to_executorch()

    # Save .pte including delegated Vela section
    with open(TORCH_OUT_PATH + "/delegated.pte", "wb") as fh:
        fh.write(exec_prog.buffer)

    # NOTE:
    #   Additional steps from here are optional but can be helpful with
    # debug as they will capture the inputs and outputs as well as running
    # the intermediate output on the tosa_reference_model.
    #   This can ensure the compilation flow is working correctly as part of
    # a development loop, ahead of running the example on hardware.
        
    # Save inputs for TOSA reference run
    tosa_ref_capture_inputs(model_edge, inputs, TOSA_OUT_PATH, {}, {}, profile)

    # Save ground truth results to file
    with open(TORCH_OUT_PATH + "/torch_output.npy", "wb") as f:
        np.save(f, torch_output.detach().numpy())

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
    if np.allclose(tosa_output, torch_output, rtol=1e-1, atol=1e-1, equal_nan=True):
        print(
            "\033[92m"
            + "Torch and Tosa Reference results are matching for operator: "
            + op
            + " from "
            + str(str(profile))
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

    if profile in ( TosaProfile.BI,  TosaProfile.BI_INT ):
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
    for op in EXAMPLE_TEST_LIST:
        run_test(op, profile=TosaProfile.BI_INT)
