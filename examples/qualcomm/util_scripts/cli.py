# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This tool supports the QC internal QA pipeline by quantizing, compiling,
# and executing models under various configuration flags.

import argparse
import importlib
import logging
import os
import re
import shutil
from pathlib import Path

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManagerAdaptor
import numpy as np

import torch

from executorch.backends.qualcomm._passes.qnn_pass_manager import (
    get_capture_program_passes,
)
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
from executorch.backends.qualcomm.utils.constants import QCOM_PASS_ACTIVATE_KEY
from executorch.backends.qualcomm.utils.utils import (
    draw_graph,
    dump_context_from_pte,
    from_context_binary,
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    generate_qnn_executorch_option,
    QNN_QUANT_TYPE_MAP,
    QNN_TENSOR_TYPE_MAP,
    to_edge_transform_and_lower_to_qnn,
)
from executorch.examples.qualcomm.qaihub_scripts.utils.utils import preprocess_binary
from executorch.examples.qualcomm.utils import make_quantizer, SimpleADB
from executorch.exir import ExecutorchBackendConfig
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from torchao.quantization import pt2e
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

INPUT_ORDER = "input_order"


def get_logger():
    logger = logging.getLogger("examples.qualcomm.util_scripts.cli")
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="[%(asctime)s %(prefix)s] %(levelname)-8s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logging.LoggerAdapter(logger, extra={"prefix": "QNN_BACKEND"})


def get_io_info(pte_path, compiler_specs):
    dtype_map = {}
    for type_map in (QNN_QUANT_TYPE_MAP, QNN_TENSOR_TYPE_MAP):
        for k, v in type_map.items():
            dtype_map.setdefault(v, k)

    def fill_tensor_info(info, qnn_tensors, category):
        for tensor in qnn_tensors:
            encoding = tensor.GetEncodings()
            quantization_info = {
                "scale": encoding.data["scale"].tolist(),
                "offset": encoding.data["offset"].tolist(),
                "axis": encoding.axis,
            }

            info[category].append(
                {
                    "name": tensor.GetName(),
                    "shape": tensor.GetDims().tolist(),
                    "dtype": dtype_map[tensor.GetDataType()],
                    "encoding": quantization_info,
                }
            )

    in_key, out_key = "inputs", "outputs"
    tensor_info = {in_key: [], out_key: []}

    path_of_pte = Path(pte_path)
    dump_context_from_pte(path_of_pte.absolute())
    ctx_bin = [f for f in os.listdir(path_of_pte.parent) if Path(f).suffix == ".bin"][0]
    # assume graph is fully delegated or it will be too hard to handle
    with open(f"{path_of_pte.parent}/{ctx_bin}", "rb") as f:
        ctx_bin = preprocess_binary(f.read(), compiler_specs)
        # leverage QNN pybind interface to retrieve tensor encodings
        qnn_mgr = PyQnnManagerAdaptor.QnnManager(
            generate_qnn_executorch_option(compiler_specs), ctx_bin
        )
        assert qnn_mgr.Init().value == 0, "failed to load context binary"
        graph_name = qnn_mgr.GetGraphNames()[0]
        qnn_mgr.AllocateTensor(graph_name)
        fill_tensor_info(tensor_info, qnn_mgr.GetGraphInputs(graph_name), in_key)
        fill_tensor_info(tensor_info, qnn_mgr.GetGraphOutputs(graph_name), out_key)
        qnn_mgr.Destroy()

    return tensor_info


class InputListParser:
    def __init__(self, input_list):
        self.input_list = input_list

    def __iter__(self):
        with open(self.input_list, "r") as f:
            for line in re.split(r"\r?\n", f.read()):
                if not line:
                    continue
                split_line = line.strip().split(" ")
                inputs = {}
                if ":=" in line:
                    for input_assignment in split_line:
                        name, path = input_assignment.split(":=")
                        inputs[name] = torch.load(path, weights_only=True)
                else:
                    inputs = [torch.load(t, weights_only=True) for t in split_line]
                yield inputs


def quantize(args):
    logger = get_logger()

    # get corresponding QnnQuantizer
    try:
        quant_dtype = getattr(QuantDtype, args.config)
        act_observer = getattr(pt2e, args.activation_observer)
        quantizer = make_quantizer(
            quant_dtype=quant_dtype,
            per_channel_conv=args.per_channel,
            per_channel_linear=args.per_row,
            act_observer=act_observer,
        )
    except Exception:
        logger.error(
            f"Failed to retrieve expected config {args.config} / {args.activation_observer}."
        )
        exit(1)

    # step 0: load saved model
    ep = torch.export.load(args.artifact)
    # step 1: use prepare_pt2e to annotate QDQ pairs
    ep_prepared = prepare_pt2e(ep.module(), quantizer)
    logger.info(f"perform calibration on {args.artifact}")
    # step 2: perform calibration
    input_list_parser = InputListParser(args.input_list)
    graph_input_names = [
        spec.arg.name
        for spec in ep.graph_signature.input_specs
        if spec.kind.name == "USER_INPUT"
    ]
    for inputs in input_list_parser:
        if isinstance(inputs, dict):
            inputs = [inputs[name] for name in graph_input_names]
        ep_prepared(*inputs)
    # step 3: use convert_pt2e to fix encodings of QDQ pairs
    logger.info(f"saving calibrated model for {args.artifact}")
    ep_converted = convert_pt2e(ep_prepared)
    ep_quantized = torch.export.export(ep_converted, tuple(inputs))
    os.makedirs(args.output_folder, exist_ok=True)
    torch.export.save(
        ep_quantized, f"{args.output_folder}/{Path(args.artifact).stem}_quantized.pt2"
    )


def compile(args):
    logger = get_logger()

    # setup memory planning
    memory_planning_pass = MemoryPlanningPass(
        alloc_graph_input=args.shared_buffer is None,
        alloc_graph_output=args.shared_buffer is None,
    )

    file_name, extension = Path(args.artifact).stem, Path(args.artifact).suffix
    os.makedirs(args.output_folder, exist_ok=True)
    # setup compiler spec dedicated to QNN HTP backend
    backend_options = generate_htp_compiler_spec(use_fp16=True)
    # setup general compiler spec for QNN
    compiler_specs = generate_qnn_executorch_compiler_spec(
        soc_model=getattr(QcomChipset, args.model),
        backend_options=backend_options,
        is_from_context_binary=extension == "bin",
    )
    if extension == ".bin":
        custom_op_name = f"ctx_loader_{file_name}"
        # step 1: generate ExportedProgram with custom op as a binary loader & lower it w/QnnBackend
        logger.info(f"exporting program for {args.artifact}")
        prog_info = from_context_binary(
            args.artifact, custom_op_name, getattr(QcomChipset, args.model)
        )
        # step 2: write pte files and store final graph
        logger.info(f"exporting {file_name}.pte")
        with open(f"{args.output_folder}/{file_name}.pte", "wb") as f:
            prog_info["edge_program_manager"].to_executorch(
                config=ExecutorchBackendConfig(
                    memory_planning_pass=memory_planning_pass
                )
            ).write_to_file(f)
        logger.info(f"exporting network graph with {file_name}.svg")
        draw_graph(file_name, args.output_folder, prog_info["exported_program"])
    elif extension == ".pt2":
        # step 0: prepare exported_program
        ep = torch.export.load(args.artifact)
        sample_inputs = ep.example_inputs[0]
        # step 1: start lowering to QnnBackend
        logger.info(f"start lowering program for {args.artifact}")
        passes, user_passes = get_capture_program_passes(), []
        if args.pass_job is not None:
            for job in args.pass_job:
                try:
                    user_passes.append(
                        importlib.import_module(
                            "executorch.backends.qualcomm._passes", job
                        )
                    )
                except Exception:
                    logger.error(f"failed to extract designated pass '{args.artifact}'")

        for user_pass in user_passes:
            passes[user_pass][QCOM_PASS_ACTIVATE_KEY] = True
        input_order = {INPUT_ORDER: ep.graph_signature.user_inputs}
        edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
            module=ep.module(),
            inputs=sample_inputs,
            compiler_specs=compiler_specs,
            passes_job=passes,
            constant_methods=input_order,
        )
        # step 2: write pte files and store final graph
        logger.info(f"exporting {file_name}.pte")
        with open(f"{args.output_folder}/{file_name}.pte", "wb") as f:
            edge_prog_mgr.to_executorch(
                config=ExecutorchBackendConfig(
                    memory_planning_pass=memory_planning_pass
                )
            ).write_to_file(f)
        logger.info(f"exporting network graph with {file_name}.svg")
        draw_graph(file_name, args.output_folder, edge_prog_mgr.exported_program())
    else:
        logger.error(f"unsupported file extension for '{args.artifact}'")


def execute(args):
    logger = get_logger()

    pte_name = Path(args.artifact).stem

    # get input order
    from executorch.runtime import Runtime, Verification

    et_runtime = Runtime.get()
    program = et_runtime.load_program(
        args.artifact,
        verification=Verification.Minimal,
    )
    input_order_func = program.load_method(INPUT_ORDER)
    input_order = input_order_func.execute([])

    # load input files
    logger.info("loading user inputs")
    input_list_parser = InputListParser(args.input_list)
    user_inputs = []
    for inputs in input_list_parser:
        if isinstance(inputs, dict):
            ordered_inputs = []
            # since io_info is dict and it is ordered in python
            # we use it to reorder input assignments here
            for name in input_order:
                ordered_inputs.append(inputs[name])
            user_inputs.append(ordered_inputs)
        else:
            user_inputs.append(inputs)

    logger.info("retrieving graph I/O")
    # setup compiler spec dedicated to QNN HTP backend
    backend_options = generate_htp_compiler_spec(use_fp16=True)
    # setup general compiler spec for QNN
    compiler_specs = generate_qnn_executorch_compiler_spec(
        soc_model=getattr(QcomChipset, args.model),
        backend_options=backend_options,
    )
    io_info = get_io_info(args.artifact, compiler_specs)
    logger.info("preparing ADB connection")
    # leverage SimpleADB for e2e inference
    adb = SimpleADB(
        qnn_sdk=os.getenv("QNN_SDK_ROOT"),
        build_path=args.build_folder,
        pte_path=args.artifact,
        workspace=f"/data/local/tmp/executorch/{pte_name}",
        device_id=args.device,
        soc_model=args.model,
        host_id=args.host,
        shared_buffer=args.shared_buffer,
        target=args.target,
    )

    logger.info("pushing QNN libraries & other artifacts")

    adb.push(inputs=user_inputs)

    logger.info("starting inference")
    adb.execute()

    tmp_dir = f"{args.output_folder}/tmp_outputs"
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(args.output_folder, exist_ok=True)

    def post_process():
        torch_to_numpy_dtype_dict = {
            torch.bool: np.dtype("bool"),
            torch.uint8: np.dtype("uint8"),
            torch.int8: np.dtype("int8"),
            torch.int16: np.dtype("int16"),
            torch.int32: np.dtype("int32"),
            torch.int64: np.dtype("int64"),
            torch.float16: np.dtype("float16"),
            torch.float32: np.dtype("float32"),
            torch.float64: np.dtype("float64"),
            torch.complex64: np.dtype("complex64"),
            torch.complex128: np.dtype("complex128"),
        }
        output_info = io_info["outputs"]
        tmp_output_folder = f"{tmp_dir}/outputs"
        for _, f in enumerate(os.listdir(tmp_output_folder)):
            filename = os.path.join(tmp_output_folder, f)
            match_res = re.match(r".*output_([0-9]+)_([0-9]+)\.raw$", filename)
            data_index, output_index = int(match_res.group(1)), int(match_res.group(2))

            output_result_folder = f"{args.output_folder}/Result_{data_index}"
            os.makedirs(output_result_folder, exist_ok=True)
            output = np.fromfile(
                filename,
                dtype=eval(
                    f"np.{torch_to_numpy_dtype_dict[output_info[output_index]['dtype']]}"
                ),
            )
            output = torch.from_numpy(
                output.reshape(output_info[output_index]["shape"])
            )
            torch.save(output, f"{output_result_folder}/output_{output_index}.pt")

    logger.info("collecting output data")
    adb.pull(tmp_dir, post_process)
    shutil.rmtree(tmp_dir)
    logger.info(f"execution finished, please check {args.output_folder} for results")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Utility to quantize / compile / execute models via Qualcomm backend"
        ),
    )
    subparsers = parser.add_subparsers(
        title="subcommands",
        description=(
            "[quantize]: Perform PTQ with QnnQuantizer for models in .pt2 extension. "
            "[compile]: Compile model in .pt2 extenstion / context binary into .pte file. "
            "[execute]: Perform on-device inference with given .pte."
        ),
    )

    sub_quantize = subparsers.add_parser(
        name="quantize",
        help=(
            "e.g. python -m executorch.example.qualcomm.util_scripts.cli quantize "
            "-a model.pt2 -c use_8a8w -i calibration_data"
        ),
    )
    sub_quantize.add_argument(
        "-a",
        "--artifact",
        type=str,
        required=True,
        help="Path to saved .pt2 model in floating point precision.",
    )
    sub_quantize.add_argument(
        "-o",
        "--output_folder",
        type=str,
        default="./output_quantized",
        help="Path to output artifact, store in 'output_quantized' if not given.",
    )
    sub_quantize.add_argument(
        "-c",
        "--config",
        type=str,
        default="use_8a8w",
        help=(f"Configuration to be applied: {list(QuantDtype.__members__.keys())}."),
    )
    sub_quantize.add_argument(
        "-i",
        "--input_list",
        type=str,
        required=True,
        help=(
            "List of input files specified for calibration. "
            'e.g. File content with: "input_0_0.pt2 input_0_1.pt2\\ninput_1_0.pt2 input_1_1.pt2" '
            "means there are 2 sets of data for calibration on a graph with 2 inputs."
        ),
    )
    sub_quantize.add_argument(
        "--per_channel",
        action="store_true",
        help="Use per_channel encoding for operator convolution and its' families.",
    )
    sub_quantize.add_argument(
        "--per_row",
        action="store_true",
        help="Use per_row encoding for operator linear.",
    )
    sub_quantize.add_argument(
        "--activation_observer",
        type=str,
        default="MovingAverageMinMaxObserver",
        help=(
            "Activation observer for PTQ "
            "(MinMaxObserver / MovingAverageMinMaxObserver / HistogramObserver)."
        ),
    )
    sub_quantize.set_defaults(callback=quantize)

    sub_compile = subparsers.add_parser(
        name="compile",
        help=(
            "e.g. python -m executorch.example.qualcomm.util_scripts.cli compile "
            "-a model.(pt2 / bin) -m SM8750"
        ),
    )
    sub_compile.add_argument(
        "-a",
        "--artifact",
        type=str,
        required=True,
        help="Path to saved .pt2 model or pre-generated context binary.",
    )
    sub_compile.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="SoC model. e.g. SM8750",
    )
    sub_compile.add_argument(
        "-o",
        "--output_folder",
        type=str,
        default="./output_pte",
        help="Path to output artifacts, store in 'output_pte' if not given.",
    )
    sub_compile.add_argument(
        "-p",
        "--pass_job",
        nargs="+",
        type=str,
        help=('Add extra passes for model lowering. e.g. "TagQuantIO".'),
    )
    sub_compile.add_argument(
        "--shared_buffer",
        help=(
            "Enable usage of shared buffer between application and backend for graph I/O."
        ),
        action="store_true",
    )
    sub_compile.set_defaults(callback=compile)

    sub_execute = subparsers.add_parser(
        name="execute",
        help=(
            "e.g. python -m executorch.example.qualcomm.util_scripts.cli "
            "execute -p model.pte -i execution_data -s device_serial"
        ),
    )
    sub_execute.add_argument(
        "-a",
        "--artifact",
        type=str,
        required=True,
        help="Path to .pte file generated from 'compile' subcommand.",
    )
    sub_execute.add_argument(
        "-i",
        "--input_list",
        type=str,
        help=(
            "List of input files specified for execution. "
            'e.g. File content with: "input_0_0.pt2 input_0_1.pt2\\ninput_1_0.pt2 input_1_1.pt2" '
            "means there are 2 sets of data for execution on a graph with 2 inputs.\n"
        ),
    )
    sub_execute.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="SoC model. e.g. SM8750",
    )
    sub_execute.add_argument(
        "-s",
        "--device",
        type=str,
        required=True,
        help="Serial no of device which could be obtained by 'adb devices'.",
    )
    sub_execute.add_argument(
        "-o",
        "--output_folder",
        type=str,
        default="./output_data",
        help="Path to output data, store in 'output_data' if not given.",
    )
    sub_execute.add_argument(
        "-b",
        "--build_folder",
        help="Path to cmake binary directory for android, e.g., /path/to/build-android",
        type=str,
        required=True,
    )
    sub_execute.add_argument(
        "-H",
        "--host",
        type=str,
        help="Gateway hostname.",
    )
    sub_execute.add_argument(
        "-t",
        "--target",
        help="Target platform for deployment",
        choices=[
            "aarch64-android",
            "aarch64-oe-linux-gcc9.3",
            "aarch64-oe-linux-gcc11.2",
        ],
        default="aarch64-android",
        type=str,
    )
    sub_execute.add_argument(
        "--shared_buffer",
        help=(
            "Enable usage of shared buffer between application and backend for graph I/O."
            " Please use with `--shared_buffer` in compile command."
        ),
        action="store_true",
    )
    sub_execute.set_defaults(callback=execute)

    args = parser.parse_args()
    args.callback(args)


if __name__ == "__main__":
    main()
