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
from executorch.backends.arm.arm_partitioner import ArmPartitioner
from executorch.backends.arm.test.test_models import TestList, TosaProfile
from executorch.backends.arm.test.test_tosa import prepare_model_and_ref
from executorch.exir import to_edge

from executorch.exir.backend.compile_spec_schema import CompileSpec

from executorch.exir.dialects._ops import ops as exir_ops

from torch.export import export

# Assumes you have these two tools on your path
TOSA_REF_MODEL_PATH = "tosa_reference_model"
VELA_COMPILER_PATH = "vela"

# Temp directory that any debug output is written to
DEBUG_OUTPUT_PATH = tempfile.mkdtemp(prefix="arm_tosa_")

# Config for Capturing the weights, will be moved in the future
_EDGE_COMPILE_CONFIG = exir.EdgeCompileConfig(
    _check_ir_validity=False,
)

SUPPORTED_BI_TEST_LIST = [
    "simple_add",
    "simple_add_broadcast",
    "simple_linear",
    "simple_linear_rank4",
    "simple_conv2d_3x3_1x3x256x256_stride1",
    "simple_conv2d_1x1_1x2x128x128_stride1",
    "simple_conv2d_2x2_1x1x14x14_stride2",
    "simple_conv2d_5x5_3x2x128x128_stride1",
    "simple_conv2d_2x2_3x1x40x40_non_bias",
    "block_two_conv2d",
    "block_two_conv2d_non_bias",
]


def get_input_quantization_params(captured_model):
    input_scales = {}
    input_zeropoints = {}
    input_names = []
    for node in captured_model.exported_program().graph.nodes:
        if node.op == "placeholder":
            input_names.append(node.name)
            continue

    for node in captured_model.exported_program().graph.nodes:
        if (
            node.target
            == exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
            and node.args[0].name in input_names
        ):
            # input_scales.append(float(node.args[1]))
            # input_zeropoints.append(int(node.args[2]))
            input_scales[node.args[0].name] = float(node.args[1])
            input_zeropoints[node.args[0].name] = int(node.args[2])

    return input_scales, input_zeropoints


def get_output_quantization_param(captured_model):
    output_scale = 0.0
    output_zeropoint = 0
    output_name = ""
    for node in captured_model.exported_program().graph.nodes:
        if node.op == "output":
            output_name = node.args[0][0]

    for node in captured_model.exported_program().graph.nodes:
        if (
            node.target
            == exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
            and node == output_name
        ):
            output_scale = float(node.args[1])
            output_zeropoint = int(node.args[2])

    return output_scale, output_zeropoint


def tosa_ref_dump_inputs(
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
    for node in model_edge.exported_program().graph.nodes:
        gs = model_edge.exported_program().graph_signature
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


def tosa_run_test(op, profile=TosaProfile.MI):  # noqa: C901
    #
    # Minimal sequence to take model through TosaPartitioner and emit
    # tosaout/ debug directory containing the flatbuffer - assumes one and will only save last output
    # tosaout is generated even for partial/broken subgraph capture to aid in debg
    # delegated.pte containing the flatbuffer within the executorch flatbuffer binary
    #
    print(f"\n\033[96mProcessing:::{op}\033[0m")
    print(f"\033[96mDebug output path for intermediates: {DEBUG_OUTPUT_PATH}\033[0m")
    if profile == TosaProfile.BI and op not in SUPPORTED_BI_TEST_LIST:
        print(f"\033[33m{op} hasn't been supported for BI profile. Skip...\033[0m")
        return

    # Debug output for TORCH
    TORCH_OUT_PATH = os.path.join(DEBUG_OUTPUT_PATH, op, "torch", "")
    os.makedirs(TORCH_OUT_PATH, exist_ok=True)

    # Debug output for TOSA
    TOSA_OUT_PATH = os.path.join(DEBUG_OUTPUT_PATH, op, "tosa", "")
    os.makedirs(TOSA_OUT_PATH, exist_ok=True)

    # Debug flags for compilers
    # - Emit some debug files into /tmp
    # - output_format TOSA for this test (and pure tosa flows)
    compile_spec = [
        CompileSpec("debug_tosa_path", bytes(TOSA_OUT_PATH, "utf8")),
        CompileSpec("output_format", bytes("tosa", "utf8")),
    ]

    model, inputs, torch_output = prepare_model_and_ref(op, profile)

    if inputs is None:
        print("\033[96m Skipping, no inputs for TOSA profile \033[0m")
        return

    # Export model
    model_capture = export(model, inputs)
    model_edge = to_edge(model_capture, compile_config=_EDGE_COMPILE_CONFIG)

    ArmPartitioner.compile_spec = compile_spec

    if profile == TosaProfile.BI:
        (
            input_quantization_scales,
            input_quantization_zps,
        ) = get_input_quantization_params(model_edge)
        (
            output_quantization_scale,
            output_quantization_zp,
        ) = get_output_quantization_param(model_edge)

    model_edge = model_edge.to_backend(ArmPartitioner())
    exec_prog = model_edge.to_executorch()

    # Save ground truth results to file
    with open(TORCH_OUT_PATH + "/torch_output.npy", "wb") as f:
        np.save(f, torch_output.detach().numpy())

    if profile is TosaProfile.BI:
        tosa_ref_dump_inputs(
            model_edge,
            inputs,
            TOSA_OUT_PATH,
            input_quantization_scales,
            input_quantization_zps,
            profile,
        )
    else:
        tosa_ref_dump_inputs(model_edge, inputs, TOSA_OUT_PATH, {}, {})

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

        # Torch is doing Input[FP32]->Q[INT8]->DQ[FP32]->Operator[FP32]->Q[INT]->DQ[FP32]->[Output]FP32
        # Need to dequant back to FP32 for running comparison with Torch output
        if profile is TosaProfile.BI:
            tosa_output = (
                np.round(tosa_output - output_quantization_zp)
                * output_quantization_scale
            )

    ## Read the Torch Output
    torch_file = open(TORCH_OUT_PATH + "/torch_output.npy", "rb")
    torch_output = np.load(torch_file)

    ## Compare Tosa and Torch Results
    ## TODO: Torch is doing [Q, DQ, Operation (FP32), Q, DQ] for quantization
    ## While TOSA is doing everything in INT8 which is causing a large diff
    ## Between two final results. Need to fix this to have a smaller error margin.
    ## Set tolerance values to 1.5e-1 for conv2d testing as that operation can
    ## generate larger difference with ground-truth floating point output on random
    ## input data.
    if np.allclose(tosa_output, torch_output, rtol=1.5e-1, atol=1.5e-1, equal_nan=True):
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

    # if profile == TosaProfile.BI:
    #     cmd_vela = "cd " + TOSA_OUT_PATH + "; " + VELA_COMPILER_PATH + " ./output.tosa"
    #     try:
    #         subprocess.run([cmd_vela], shell=True, check=True)
    #         print("\033[92m" + "Vela compile worked for: " + op + "\033[0m")
    #     except:
    #         print("\033[91m" + "Vela compile failed for: " + op + "\033[0m")
    # else:
    #     print("\033[96m" + "Skipping Vela test on non-BI profile." + "\033[0m")


# Temp systest mode for running all models against both inference profiles
if __name__ == "__main__":
    for op in TestList:
        tosa_run_test(op, profile=TosaProfile.MI)

    for op in TestList:
        tosa_run_test(op, profile=TosaProfile.BI)
