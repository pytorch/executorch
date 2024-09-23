# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import shutil
import subprocess
import tempfile

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from torch.export import ExportedProgram
from torch.fx.node import Node

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class QuantizationParams:
    __slots__ = ["node_name", "zp", "scale", "qmin", "qmax", "dtype"]

    # todo: zps and scales can be per tensors or per channel => a list??
    def __init__(
        self,
        node_name: str,
        zp: int,
        scale: float,
        qmin: int,
        qmax: int,
        dtype: torch.dtype,
    ):
        self.node_name = node_name  # not need I think, but good for error check
        self.zp = zp
        self.scale = scale
        self.qmin = qmin
        self.qmax = qmax
        self.dtype = dtype


def _get_input_names(program: ExportedProgram) -> list[str]:
    """
    Get a list[str] with the names of the inputs to this model.

    Args:
        program (ExportedProgram): The program to get input names from.
    Returns:
        A list of strings with the names of the model input.
    """
    input_names = []

    # E.g. bias and weights are 'placeholders' as well. This is used to
    # get only the use inputs.
    usr_inputs = program.graph_signature.user_inputs
    for node in program.graph.nodes:
        if node.op == "placeholder" and node.name in usr_inputs:
            input_names.append(node.name)

    return input_names


def _get_input_quantization_params(
    program: ExportedProgram, input_names: list[str]
) -> list[QuantizationParams]:
    """
    Get input QuantizationParams in a program, maximum one per input to the program.
    Args:
        program (ExportedProgram): The program to get input quantization parameters from.
    Returns:
        list[QuantizationParams]: The found quantization parameters.
    Raises:
        RuntimeError if no quantization parameters are found.
    """

    quant_params = []
    num_inputs = len(input_names)
    for node in program.graph.nodes:
        if (
            node.target == torch.ops.quantized_decomposed.quantize_per_tensor.default
            and node.args[0].name in input_names
        ):
            qp = QuantizationParams(
                node_name=node.args[0].name,
                scale=node.args[1],
                zp=node.args[2],
                qmin=node.args[3],
                qmax=node.args[4],
                dtype=node.args[5],
            )
            quant_params.append(qp)
            if (
                len(quant_params) == num_inputs
            ):  # break early if we have all the inputs quantized parameters
                break
    if len(quant_params) == 0:
        raise RuntimeError("No Quantization parameters not found in exported model.")
    return quant_params


def _get_output_node(program: ExportedProgram) -> Node:
    """
    Get output node to this model.

    Args:
        program (ExportedProgram): The program to get output node from.
    Returns:
        The node that is the output of 'program'.
    """

    for node in program.graph.nodes:
        if node.op == "output":
            return node
    raise RuntimeError("No output node found.")


def _get_output_quantization_params(
    program: ExportedProgram, output_node: Node
) -> QuantizationParams:
    """
    Get output QuantizationParams from a program.
    Args:
        program (ExportedProgram): The program to get output quantization parameters from.
    Returns:
        QuantizationParams: The found quantization parameters.
    Raises:
        RuntimeError if no output quantization parameters are found.
    """

    quant_params = None
    for node in program.graph.nodes:
        if (
            node.target == torch.ops.quantized_decomposed.dequantize_per_tensor.default
            and node == output_node.args[0][0]
        ):
            quant_params = QuantizationParams(
                node_name=node.args[0].name,
                scale=node.args[1],
                zp=node.args[2],
                qmin=node.args[3],
                qmax=node.args[4],
                dtype=node.args[5],
            )
            break  # break early, there's only one output node
    if quant_params is None:
        raise RuntimeError("No Quantization parameters not found in exported model.")
    return quant_params


"""
A class to store parameters needed for running programs, either in tosa or .pte format.
"""


class RunnerUtil:
    def __init__(
        self,
        intermediate_path: str,
        tosa_ref_model_path: Optional[str] = None,
    ):
        self.intermediate_path = intermediate_path
        self.tosa_ref_model_path = tosa_ref_model_path or "tosa_reference_model"
        assert os.path.exists(
            self.intermediate_path
        ), f"TOSA artifact path don't exist! Path: {self.intermediate_path}"

        self.is_quantized: bool = False
        self.input_names: list[str] = None
        self.output_name: str = None
        self.qp_input: list[QuantizationParams] = None
        self.qp_output: QuantizationParams = None
        self.timeout = 120

        self._has_init_run = False

    def init_run(self, exported_program: ExportedProgram, is_quantized: bool):
        self.input_names = _get_input_names(exported_program)
        self.output_node = _get_output_node(exported_program)
        self.output_name = self.output_node.name
        self.is_quantized = is_quantized

        if is_quantized:
            self.qp_input = _get_input_quantization_params(
                exported_program, self.input_names
            )
            self.qp_output = _get_output_quantization_params(
                exported_program, self.output_node
            )
        else:
            self.qp_input = [None] * len(self.input_names)
            self.qp_output = None

        self._has_init_run = True

    def set_timeout(self, timeout: int):
        self.timeout = timeout

    def run_corstone300(
        self,
        inputs: Tuple[torch.Tensor],
    ) -> list[torch.Tensor]:

        assert (
            self._has_init_run
        ), "RunnerUtil needs to be initialized using init_run() before running Corstone300."

        pte_path = os.path.join(self.intermediate_path, "program.pte")
        assert os.path.exists(pte_path), f"Pte path '{pte_path}' not found."

        for input_name, quant_param, data in zip(
            self.input_names, self.qp_input, inputs
        ):
            save_bytes(self.intermediate_path, data, False, input_name, quant_param)

        out_path = os.path.join(self.intermediate_path, "out")
        out_path_with_suffix = out_path + "-0.bin"
        input_paths = []
        for name in self.input_names:
            input_paths.append(
                os.path.join(self.intermediate_path, f"{name}.bin"),
            )
        elf_path = os.path.join(
            "cmake-out", "arm_semihosting_executor_runner", "arm_executor_runner"
        )
        assert os.path.exists(
            elf_path
        ), f"Did not find build arm_executor_runner in path {elf_path}, run setup_testing.sh?"

        cmd_line = f"executor_runner -m {pte_path} -o {out_path}"
        for input_path in input_paths:
            cmd_line += f" -i {input_path}"

        command_args = [
            "FVP_Corstone_SSE-300_Ethos-U55",
            "-C",
            "ethosu.num_macs=128",
            "-C",
            "mps3_board.visualisation.disable-visualisation=1",
            "-C",
            "mps3_board.telnetterminal0.start_telnet=0",
            "-C",
            "mps3_board.uart0.out_file='-'",
            "-C",
            "cpu0.CFGITCMSZ=11",
            "-C",
            "cpu0.semihosting-enable=1",
            "-C",
            "cpu0.semihosting-stack_base=0",
            "-C",
            "cpu0.semihosting-heap_limit=0",
            "-C",
            f"cpu0.semihosting-cmd_line='{cmd_line}'",
            "-a",
            elf_path,
            "--timelimit",
            f"{self.timeout}",
        ]
        result = _run_cmd(command_args, check=False)
        result_stdout = result.stdout.decode()
        if "Hard fault" in result_stdout or len(result.stderr) > 0:
            raise RuntimeError(
                f"Corstone simulation failed, log: \n {result_stdout}\n{result.stderr.decode()}"
            )
        elif "E [" in result_stdout:
            logger.error(result_stdout)

        tosa_ref_output = np.fromfile(out_path_with_suffix, dtype=np.float32)
        output_shape = self.output_node.args[0][0].meta["val"].shape
        tosa_ref_output = torch.from_numpy(tosa_ref_output).reshape(output_shape)
        return [tosa_ref_output]

    def run_tosa_ref_model(
        self,
        inputs: Tuple[torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        Run TOSA reference model using the tosa_reference_model program.

        In order to do that we need:
        1. desc.json, which points to files needed by tosa_reference_model.
        2. output.tosa, which is the TOSA buffer that describes the model we're
           trying to run.

        These two files are created by arm_backend.py as part of partition stage

        All these files are saved on disk in self.intermediate_path.

        Args:
            inputs (Tuple[torch.Tensor]): The input data to run the TOSA

        Returns:
            torch.Tensor: The output of the TOSA reference model, as a torch
                tensor.

        Here's a sample desc.json file:
        {
            "tosa_file": "output.tosa",
            "ifm_name": [
                "arg0_1"
            ],
            "ifm_file": [
                "arg0_1.npy"
            ],
            "ofm_name": [
                "quantized_decomposed_dequantize_per_tensor_default_1"
            ],
            "ofm_file": [
                "ref-quantized_decomposed_dequantize_per_tensor_default_1.npy"
            ],
            "expected_return_code": 0,
            "expected_failure": false
        }

        Todo:
            * It would be nice to not rely on files on disk. Should be possible
              as a next step. See:
              https://review.mlplatform.org/plugins/gitiles/tosa/reference_model/#executable-usage
        """

        assert (
            self._has_init_run
        ), "RunnerUtil needs to be initialized using init_run() before running tosa reference."

        all_desc_file_paths = [
            str(path) for path in Path(self.intermediate_path).glob("desc*.json")
        ]
        assert (
            all_desc_file_paths
        ), f"No TOSA description file found in '{self.intermediate_path}'."
        if len(all_desc_file_paths) != 1:
            raise NotImplementedError(
                "Graphs with more than one partition are currently not supported."
            )

        desc_file_path = all_desc_file_paths[0]
        assert os.path.exists(
            desc_file_path
        ), f"desc_file_path: {desc_file_path} does not exist"

        # Save the input data to disk as a .npy file, since that's what the TOSA
        # reference model expects. Name of the file must match the name in
        # desc.json, which is the tensor name from the graph + .npy
        for input_name, quant_param, data in zip(
            self.input_names, self.qp_input, inputs, strict=True
        ):
            save_npy(
                self.intermediate_path, data, self.is_quantized, input_name, quant_param
            )

        # Run the TOSA reference model via command line, this will produce a
        # .npy file with the result (aka OFM).
        assert (
            shutil.which(self.tosa_ref_model_path) is not None
        ), f"tosa_reference_model tool not found, did you run examples/arm/setup.sh? Path: {self.tosa_ref_model_path}"
        loglevel_map = {
            logging.INFO: "INFO",
            logging.CRITICAL: "LOW",
            logging.ERROR: "LOW",
            logging.WARNING: "MED",
            logging.DEBUG: "HIGH",
            logging.NOTSET: "MED",
        }
        clamped_logging_level = max(min(logger.level // 10 * 10, 50), 0)
        cmd_ref_model = [
            self.tosa_ref_model_path,
            "--test_desc",
            desc_file_path,
            "-l",
            loglevel_map[clamped_logging_level],
        ]
        _run_cmd(cmd_ref_model)

        # Load desc.json, just to get the name of the output file above
        with open(desc_file_path) as f:
            desc_json = json.load(f)

        tosa_ref_outputs = []
        for ofm_file in desc_json["ofm_file"]:
            ofm_file_npy = os.path.join(self.intermediate_path, ofm_file)

            # Load the output file (OFM) and return it as a numpy array
            tosa_ref_output = np.load(ofm_file_npy)

            if self.is_quantized:
                # Need to dequant back to FP32 for comparison with torch output
                quant_param = self.qp_output
                assert (
                    quant_param is not None
                ), "There are no quantization parameters, check output parameters"
                tosa_ref_output = (tosa_ref_output - quant_param.zp) * quant_param.scale

            # tosa_output is a numpy array, convert to torch tensor for comparison
            tosa_ref_outputs.append(torch.from_numpy(tosa_ref_output.astype("float32")))

        return tosa_ref_outputs


def prep_data_for_save(
    data, is_quantized: bool, input_name: str, quant_param: QuantizationParams
):
    data_np = data.detach().numpy().astype(np.float32)

    if is_quantized:
        assert (
            quant_param.node_name == input_name
        ), "These quantization params do not match the input tensor name"
        data_np = (
            ((data_np / np.float32(quant_param.scale)) + quant_param.zp)
            .round()
            .clip(quant_param.qmin, quant_param.qmax)
            .astype(
                f"{quant_param.dtype}".replace("torch.", "")
            )  # Use string format of dtype to convert to numpy dtype
        )
    return data_np


def save_npy(
    path: str,
    data,
    is_quantized: bool,
    input_name: str,
    quant_param: QuantizationParams,
) -> str:
    """Serializes and saves 'data' as a .npy file, possibly quantizing it before.

    Parameters:
        path: the directory where to save the data.
        data: the data to save.
        is_quantized: whether to quantize the data before saving it.
        input_name: the name of the file, without file-ending.
        quant_param: the parameters to use for quantization.
    Returns:
        the full file path of the output.
    """
    data_np = prep_data_for_save(data, is_quantized, input_name, quant_param)
    file_path = os.path.join(path, input_name + ".npy")
    np.save(file_path, data_np, allow_pickle=False)

    return file_path


def save_bytes(
    path: str,
    data,
    is_quantized: bool,
    input_name: str,
    quant_param: QuantizationParams,
) -> str:
    """Serializes and saves 'data' in byte format, possibly quantizing it before.

    Parameters:
        path: the directory where to save the data.
        data: the data to save.
        is_quantized: whether to quantize the data before saving it.
        input_name: the name of the file, without file-ending.
        quant_param: the parameters to use for quantization.
    Returns:
        the full file path of the output.
    """
    data_np = prep_data_for_save(data, is_quantized, input_name, quant_param)
    file_path = os.path.join(path, input_name + ".bin")
    with open(file_path, "w+b") as f:
        data_np_bytes = data_np.tobytes()
        f.write(data_np_bytes)

    return file_path


def _run_cmd(cmd: List[str], check=True) -> subprocess.CompletedProcess[bytes]:
    """
    Run a command and check for errors.

    Args:
    cmd (List[str]): The command to run as a list.
    """
    try:
        result = subprocess.run(cmd, check=check, capture_output=True)
        return result
    except subprocess.CalledProcessError as e:
        arg_string = " ".join(cmd)
        raise RuntimeError(
            f"Failed running command {arg_string}\nStderr: {e.stderr.decode()}\nStdout: {e.stdout.decode()}"
        )


def dbg_tosa_fb_to_json(tosa_fb: bytes) -> Dict:
    """
    This function is used to dump the TOSA flatbuffer to a human readable
    format, using flatc. It is used for debugging purposes.
    """

    tmp = tempfile.mkdtemp()
    tosa_input_file = os.path.join(tmp, "output.tosa")
    with open(tosa_input_file, "wb") as f:
        f.write(tosa_fb)

    tosa_schema_file = "./backends/arm/third-party/serialization_lib/schema/tosa.fbs"
    assert os.path.exists(
        tosa_schema_file
    ), f"tosa_schema_file: {tosa_schema_file} does not exist"

    assert shutil.which("flatc") is not None
    cmd_flatc = [
        "flatc",
        "--json",
        "--strict-json",
        "-o",
        tmp,
        "--raw-binary",
        "-t",
        tosa_schema_file,
        "--",
        tosa_input_file,
    ]
    _run_cmd(cmd_flatc)
    with open(os.path.join(tmp, "output.json"), "r") as f:
        json_out = json.load(f)

    # Cast float tensors to proper dtype.
    try:
        for region in json_out["regions"]:
            for block in region["blocks"]:
                for tensor in block["tensors"]:
                    if "data" in tensor:
                        if tensor["type"] == "FP32":
                            data = np.array(tensor["data"])
                            data = data.astype(np.int8)
                            data = np.frombuffer(data, dtype=np.float32)
                        data = data.reshape(tensor["shape"])
                        tensor["data"] = data
    except Exception:
        # This is just nice-to-have if it works, don't care if it fails.
        pass

    return json_out
