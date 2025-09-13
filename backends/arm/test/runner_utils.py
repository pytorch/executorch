# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile

from pathlib import Path

from typing import Any, cast, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from executorch.backends.arm.common.arm_compile_spec import ArmCompileSpec

from executorch.backends.arm.ethosu import EthosUCompileSpec
from executorch.backends.arm.test.conftest import is_option_enabled
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.tosa.specification import Tosa_1_00, TosaSpecification
from executorch.backends.arm.vgf import VgfCompileSpec
from executorch.exir import ExecutorchProgramManager, ExportedProgram
from executorch.exir.lowered_backend_module import LoweredBackendModule
from torch.fx.node import Node

from torch.overrides import TorchFunctionMode
from tosa.TosaGraph import TosaGraph

logger = logging.getLogger(__name__)

# Copied from PyTorch.
# From torch/testing/_internal/common_utils.py:torch_to_numpy_dtype_dict
# To avoid a dependency on _internal stuff.
_torch_to_numpy_dtype_dict = {
    torch.bool: np.bool_,
    torch.uint8: np.uint8,
    torch.uint16: np.uint16,
    torch.uint32: np.uint32,
    torch.uint64: np.uint64,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.bfloat16: np.float32,
    torch.complex32: np.complex64,
    torch.complex64: np.complex64,
    torch.complex128: np.complex128,
}

VALID_TARGET = {"corstone-300", "corstone-320", "vkml_emulation_layer"}


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


def get_input_names(program: ExportedProgram) -> list[str]:
    """
    Get a list[str] with the names of the inputs to this model.

    Args:
        program (ExportedProgram): The program to get input names from.
    Returns:
        A list of strings with the names of the model input.
    """
    return [spec.arg.name for spec in program.graph_signature.input_specs]


def get_input_quantization_params(
    program: ExportedProgram,
) -> list[QuantizationParams]:
    """
    Get input QuantizationParams in a program, maximum one per input to the program.
    Args:
        program (ExportedProgram): The program to get input quantization parameters from.
    Returns:
        list[QuantizationParams]: The found quantization parameters.
    """

    quant_params = []
    input_names = get_input_names(program)
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
        logger.warning("No input quantization parameters found in exported model.")
    return quant_params


def get_output_quantization_params(
    output_node: Node,
) -> dict[Node, QuantizationParams | None]:
    """
    Get output QuantizationParams from a program.
    Args:
        output_nodes (list(Node)): A list of output nodes to get output quantization parameters from.
    Returns:
        dictionary mapping the output nodes to the found quantization parameters.
        If no quantization parameters were found, the entry is None.
    Raises:
        RuntimeError if no output quantization parameters are found.
    """
    quant_params = {}
    for node in output_node.args[0]:
        if node.target == torch.ops.quantized_decomposed.dequantize_per_tensor.default:
            quant_params[node] = QuantizationParams(
                node_name=node.args[0].name,
                scale=node.args[1],
                zp=node.args[2],
                qmin=node.args[3],
                qmax=node.args[4],
                dtype=node.args[5],
            )
        else:
            quant_params[node] = None
    return quant_params


class TosaReferenceModelDispatch(TorchFunctionMode):
    """A context manager for executing call_delegate nodes using the reference model"""

    def __init__(self):
        self.ran_tosa_dispatch = False
        super().__init__()

    def _tosa_dispatch(self, lowered_backend_module: LoweredBackendModule, inputs):
        tosa_buffer = lowered_backend_module.processed_bytes
        compile_spec = TosaCompileSpec.from_list(lowered_backend_module.compile_specs)

        return run_tosa_graph(tosa_buffer, compile_spec.tosa_spec, inputs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        # Only raise this error if we ran the model without errors.
        if not self.ran_tosa_dispatch and exc_type is None:
            raise RuntimeError(
                "Ran model with TosaReferenceModelDispatch but never ran TOSABackend delegate."
            )

    def __torch_function__(self, func, types, args=..., kwargs=None):
        if func is torch._higher_order_ops.executorch_call_delegate:
            lowered_backend_module = cast(LoweredBackendModule, args[0])
            if lowered_backend_module.backend_id == "TOSABackend":
                self.ran_tosa_dispatch = True
                return self._tosa_dispatch(lowered_backend_module, args[1:])
            else:
                raise RuntimeError(
                    f"Ran model with TosaReferenceModelDispatch but call_delegate with {lowered_backend_module.backend_id=} != 'TOSABackend'."
                )

        kwargs = kwargs or {}
        return func(*args, **kwargs)


def run_target(
    executorch_program_manager: ExecutorchProgramManager,
    inputs: Tuple[torch.Tensor],
    intermediate_path: str | Path,
    target_board: Literal["corestone-300", "corestone-320", "vkml_emulation_layer"],
    elf_path: str | Path,
    timeout: int = 120,  # s
):
    if target_board not in VALID_TARGET:
        raise ValueError(f"Unsupported target: {target_board}")

    if target_board in ("corstone-300", "corstone-320"):
        return run_corstone(
            executorch_program_manager,
            inputs,
            intermediate_path,
            target_board,
            elf_path,
            timeout,
        )
    elif target_board == "vkml_emulation_layer":
        return run_vkml_emulation_layer(
            executorch_program_manager,
            inputs,
            intermediate_path,
            elf_path,
        )


def save_inputs_to_file(
    exported_program: ExportedProgram,
    inputs: Tuple[torch.Tensor],
    intermediate_path: str | Path,
):
    input_file_paths = []
    input_names = get_input_names(exported_program)
    for input_name, input_ in zip(input_names, inputs):
        input_path = save_bytes(intermediate_path, input_, input_name)
        input_file_paths.append(input_path)

    return input_file_paths


def get_output_from_file(
    exported_program: ExportedProgram,
    intermediate_path: str | Path,
    output_base_name: str,
):
    output_np = []
    output_node = exported_program.graph_module.graph.output_node()
    for i, node in enumerate(output_node.args[0]):
        output_shape = node.meta["val"].shape
        output_dtype = node.meta["val"].dtype
        tosa_ref_output = np.fromfile(
            os.path.join(intermediate_path, f"{output_base_name}-{i}.bin"),
            _torch_to_numpy_dtype_dict[output_dtype],
        )

        output_np.append(torch.from_numpy(tosa_ref_output).reshape(output_shape))
    return tuple(output_np)


def run_vkml_emulation_layer(
    executorch_program_manager: ExecutorchProgramManager,
    inputs: Tuple[torch.Tensor],
    intermediate_path: str | Path,
    elf_path: str | Path,
):
    """Executes an inference of the exported_program on ML Emulation Layer for Vulkan
    Args:
        `executorch_program_manager`: The executorch program to run.
        `intermediate_path`: Directory to save the .pte and capture outputs.
        `elf_path`: Path to the Vulkan-capable executor_runner binary.
    """
    exported_program = executorch_program_manager.exported_program()
    intermediate_path = Path(intermediate_path)
    intermediate_path.mkdir(exist_ok=True)
    elf_path = Path(elf_path)
    if not elf_path.exists():
        raise FileNotFoundError(f"Did not find elf file {elf_path}")

    # Save pte to file
    pte_path = os.path.join(intermediate_path, "program.pte")
    with open(pte_path, "wb") as f:
        f.write(executorch_program_manager.buffer)

    output_base_name = "out"
    out_path = os.path.join(intermediate_path, output_base_name)

    cmd_line = f"{elf_path} -model_path {pte_path} -output_file {out_path}"

    input_string = None
    input_paths = save_inputs_to_file(exported_program, inputs, intermediate_path)
    for input_path in input_paths:
        if input_string is None:
            input_string = f" -inputs={input_path}"
        else:
            input_string += f",{input_path}"
    if input_string is not None:
        cmd_line += input_string
    cmd_line = cmd_line.split()

    result = _run_cmd(cmd_line)

    # TODO: MLETORCH-1234: Support VGF e2e tests in VgfPipeline
    # TODO: Add regex to check for error or fault messages in stdout from Emulation Layer
    result_stdout = result.stdout.decode()  # noqa: F841

    return get_output_from_file(exported_program, intermediate_path, output_base_name)


def run_corstone(
    executorch_program_manager: ExecutorchProgramManager,
    inputs: Tuple[torch.Tensor],
    intermediate_path: str | Path,
    target_board: Literal["corestone-300", "corestone-320"],
    elf_path: str | Path,
    timeout: int = 120,  # s
) -> list[torch.Tensor]:
    """Executes an inference of the exported_program on FVP.
    Returns a list of tensors with the output.
    Args:
        `executorch_program_manager`: The executorch program to run.
        The output of a EdgeProgramManager.to_executorch() call.
        `inputs`: A list of tensors with the inputs of the inference.
        `dump_path`: A directory where the .pte and inputs are saved to file.
                     The output tensors are saved in `dump_path`/out.
        `target_board`: Whether to run the corstone-300 FVP or the corstone-320 FVP
        `elf_path`: The path to the runtime elf. Needs to have semihosting enabled
        and match the target_board.
        `timeout`: The timeout until the FVP terminates the elf, in seconds.
    A runtime with semihosting needs
    Limitations:
        Relies on the output tensors from the exported program
        to figure out the shape and dtype of the buffer that was
        output from the FVP.
    """

    exported_program = executorch_program_manager.exported_program()
    intermediate_path = Path(intermediate_path)
    intermediate_path.mkdir(exist_ok=True)
    elf_path = Path(elf_path)
    if not elf_path.exists():
        raise FileNotFoundError(f"Did not find elf file {elf_path}")

    # Save pte to file
    pte_path = os.path.join(intermediate_path, "program.pte")
    with open(pte_path, "wb") as f:
        f.write(executorch_program_manager.buffer)

    input_paths = save_inputs_to_file(exported_program, inputs, intermediate_path)

    output_base_name = "out"
    out_path = os.path.join(intermediate_path, output_base_name)

    cmd_line = f"executor_runner -m {pte_path} -o {out_path}"
    for input_path in input_paths:
        cmd_line += f" -i {input_path}"

    ethos_u_extra_args = ""
    if is_option_enabled("fast_fvp"):
        ethos_u_extra_args = ethos_u_extra_args + "--fast"

    match target_board:
        case "corstone-300":
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
                "cpu0.semihosting-enable=1",
                "-C",
                "cpu0.semihosting-stack_base=0",
                "-C",
                f"ethosu.extra_args='{ethos_u_extra_args}'",
                "-C",
                "cpu0.semihosting-heap_limit=0",
                "-C",
                f"cpu0.semihosting-cmd_line='{cmd_line}'",
                "-a",
                str(elf_path),
                "--timelimit",
                f"{timeout}",
            ]
        case "corstone-320":
            command_args = [
                "FVP_Corstone_SSE-320",
                "-C",
                "mps4_board.subsystem.ethosu.num_macs=128",
                "-C",
                "mps4_board.visualisation.disable-visualisation=1",
                "-C",
                "vis_hdlcd.disable_visualisation=1",
                "-C",
                "mps4_board.telnetterminal0.start_telnet=0",
                "-C",
                "mps4_board.uart0.out_file='-'",
                "-C",
                "mps4_board.uart0.unbuffered_output=1",
                "-C",
                "mps4_board.uart0.shutdown_on_eot=1",
                "-C",
                "mps4_board.subsystem.cpu0.semihosting-enable=1",
                "-C",
                "mps4_board.subsystem.cpu0.semihosting-stack_base=0",
                "-C",
                "mps4_board.subsystem.cpu0.semihosting-heap_limit=0",
                "-C",
                f"mps4_board.subsystem.ethosu.extra_args='{ethos_u_extra_args}'",
                "-C",
                f"mps4_board.subsystem.cpu0.semihosting-cmd_line='{cmd_line}'",
                "-a",
                str(elf_path),
                "--timelimit",
                f"{timeout}",
            ]
        case _:
            raise ValueError(f"Unknown target board {target_board}")

    result = _run_cmd(command_args)

    # Regex to check for error or fault messages in stdout from FVP
    result_stdout = result.stdout.decode()
    error_regex = r"(^[EF][: ].*$)|(^.*Hard fault.*$)|(^.*Assertion.*$)"
    if re.compile(error_regex, re.MULTILINE).search(result_stdout):
        raise RuntimeError(
            f"Corstone simulation failed:\ncmd: {' '.join(command_args)}\nlog: \n {result_stdout}\n{result.stderr.decode()}"
        )

    return get_output_from_file(exported_program, intermediate_path, output_base_name)


def prep_data_for_save(
    data,
    input_name: str,
    quant_param: Optional[QuantizationParams] = None,
):
    if isinstance(data, torch.Tensor):
        data_np = np.array(data.detach(), order="C").astype(
            _torch_to_numpy_dtype_dict[data.dtype]
        )
    else:
        data_np = np.array(data)
    if quant_param is not None:
        assert quant_param.node_name in input_name, (
            f"The quantization params name '{quant_param.node_name}' does not "
            f"match the input tensor name '{input_name}'."
        )
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
    input_name: str,
    quant_param: Optional[QuantizationParams] = None,
) -> str:
    """Serializes and saves 'data' as a .npy file, possibly quantizing it before.

    Parameters:
        path: the directory where to save the data.
        data: the data to save.
        input_name: the name of the file, without file-ending.
        quant_param: the parameters to use for quantization.
    Returns:
        the full file path of the output.
    """
    data_np = prep_data_for_save(data, input_name, quant_param)
    file_path = os.path.join(path, input_name + ".npy")
    np.save(file_path, data_np, allow_pickle=False)

    return file_path


def save_bytes(
    path: str,
    data,
    input_name: str,
    quant_param: Optional[QuantizationParams] = None,
) -> str:
    """Serializes and saves 'data' in byte format, possibly quantizing it before.

    Parameters:
        path: the directory where to save the data.
        data: the data to save.
        input_name: the name of the file, without file-ending.
        quant_param: the parameters to use for quantization.
    Returns:
        the full file path of the output.
    """
    data_np = prep_data_for_save(data, input_name, quant_param)
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
    tosa_graph = TosaGraph.GetRootAsTosaGraph(tosa_fb)
    version = tosa_graph.Version()
    major = version._Major()
    minor = version._Minor()
    patch = version._Patch()
    if not ((major == 1 and minor == 0)):
        raise RuntimeError(
            f"Unsupported version in TOSA flatbuffer: version={major}.{minor}.{patch}"
        )

    arm_backend_path = os.path.realpath(os.path.dirname(__file__) + "/..")
    tosa_schema_file = os.path.join(
        arm_backend_path, f"tosa/schemas/tosa_{major}.{minor}.fbs"
    )
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


def _tosa_refmodel_loglevel(loglevel: int) -> str:
    """Converts a logging loglevel to tosa_reference_model logginglevel,
    returned as string.
    """
    loglevel_map = {
        logging.INFO: "INFO",
        logging.CRITICAL: "LOW",
        logging.ERROR: "LOW",
        logging.WARNING: "MED",
        logging.DEBUG: "HIGH",
        logging.NOTSET: "MED",
    }
    clamped_logging_level = max(min(loglevel // 10 * 10, 50), 0)
    return loglevel_map[clamped_logging_level]


def corstone300_installed() -> bool:
    cmd = ["FVP_Corstone_SSE-300_Ethos-U55", "--version"]
    try:
        _run_cmd(cmd, check=True)
    except:
        return False
    return True


def corstone320_installed() -> bool:
    cmd = ["FVP_Corstone_SSE-320", "--version"]
    try:
        _run_cmd(cmd, check=True)
    except:
        return False
    return True


def model_converter_installed() -> bool:
    cmd = ["model-converter", "--version"]
    try:
        _run_cmd(cmd, check=True)
    except:
        return False
    return True


def vkml_emulation_layer_installed() -> bool:
    # Check VK_INSTANCE_LAYERS
    vk_instance_layers = os.environ.get("VK_INSTANCE_LAYERS", "")
    required_layers = {
        "VK_LAYER_ML_Graph_Emulation",
        "VK_LAYER_ML_Tensor_Emulation",
    }
    existing_layers = set(vk_instance_layers.split(":"))
    layers_exists = required_layers.issubset(existing_layers)

    # Check LD_LIBRARY_PATH for "emulation-layer/deploy"
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    deploy_exists = False
    for path in ld_library_path.split(os.path.pathsep):
        if "emulation-layer/deploy" in path and os.path.isdir(path):
            deploy_exists = True

    return layers_exists and deploy_exists


def assert_elf_path_exists(elf_path):
    if not os.path.exists(elf_path):
        raise FileNotFoundError(
            f"Did not find build arm_executor_runner or executor_runner in path {elf_path}, \
            run setup_testing.sh or setup_testing_vkml.sh?"
        )


def get_elf_path(target_board):
    if target_board not in VALID_TARGET:
        raise ValueError(f"Unsupported target: {target_board}")

    if target_board in ("corstone-300", "corstone-320"):
        elf_path = os.path.join(
            "arm_test",
            f"arm_semihosting_executor_runner_{target_board}",
            "arm_executor_runner",
        )
        assert_elf_path_exists(elf_path)
    elif target_board == "vkml_emulation_layer":
        elf_path = os.path.join(
            "arm_test/arm_executor_runner_vkml",
            "executor_runner",
        )
        assert_elf_path_exists(elf_path)

    return elf_path


def arm_executor_runner_exists(target_board):
    try:
        get_elf_path(target_board)
    except:
        return False
    else:
        return True


def run_tosa_graph(
    graph: Any,
    tosa_version: TosaSpecification,
    inputs: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Runs the TOSA reference model with inputs and returns the result."""
    inputs_np = [input.numpy() for input in inputs]

    if isinstance(tosa_version, Tosa_1_00):
        import tosa_reference_model as reference_model

        debug_mode = "ALL" if logger.level <= logging.DEBUG else None
        outputs_np, status = reference_model.run(
            graph,
            inputs_np,
            verbosity=_tosa_refmodel_loglevel(logger.level),
            initialize_variable_tensor_from_numpy=True,
            debug_mode=debug_mode,
        )
    else:
        raise ValueError(
            f"Unknown TOSA specification: {tosa_version}. No refererence model available to run for this specification version"
        )

    assert (
        status == reference_model.GraphStatus.TOSA_VALID
    ), "Non-valid TOSA given to reference model."

    return [torch.from_numpy(output) for output in outputs_np]


def get_target_board(compile_spec: ArmCompileSpec) -> str | None:
    if isinstance(compile_spec, VgfCompileSpec):
        return "vkml_emulation_layer"
    if isinstance(compile_spec, EthosUCompileSpec):
        if "u55" in compile_spec.target:
            return "corstone-300"
        if "u85" in compile_spec.target:
            return "corstone-320"
    return None
