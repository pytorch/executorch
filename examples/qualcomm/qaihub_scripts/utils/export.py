# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import io
import json
import logging
import os
from pathlib import Path

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManagerAdaptor
import numpy as np

import torch
from executorch.backends.qualcomm.serialization.qnn_compile_spec_schema import (
    QcomChipset,
)
from executorch.backends.qualcomm.utils.utils import (
    draw_graph,
    from_context_binary,
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    generate_qnn_executorch_option,
)
from executorch.examples.qualcomm.utils import make_output_dir, SimpleADB
from executorch.exir.backend.backend_api import to_backend
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass


def get_logger():
    logger = logging.getLogger("aihub.utils.export")
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
    return logging.LoggerAdapter(logger, extra={"prefix": "UTILS.EXPORT"})


def get_io_info(prog_info, ctx_bin_path, compiler_spec):
    def fill_tensor_info(info, qnn_tensors, category):
        # fetch related IO info stored in prog_info
        for i, (name, tensor) in enumerate(prog_info[category].items()):
            assert qnn_tensors[i].GetName() == name, "tensor name unmatch"
            encoding = qnn_tensors[i].GetEncodings()
            quantization_info = {
                "scale": encoding.data["scale"].tolist(),
                "offset": encoding.data["offset"].tolist(),
                "axis": encoding.axis,
            }
            info[category].append(
                {
                    "name": name,
                    "shape": tuple(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "encoding": quantization_info,
                }
            )

    # dictionary to be serialized into json format
    in_key, out_key = "inputs", "outputs"
    tensor_info = {in_key: [], out_key: []}

    with open(ctx_bin_path, "rb") as f:
        ctx_bin = f.read()
        # leverage QNN pybind interface to retrieve tensor encodings
        qnn_mgr = PyQnnManagerAdaptor.QnnManager(
            generate_qnn_executorch_option(compiler_spec), ctx_bin
        )
        assert qnn_mgr.Init().value == 0, "failed to load context binary"
        qnn_mgr.AllocateTensor()
        fill_tensor_info(tensor_info, qnn_mgr.GetGraphInputs(), in_key)
        fill_tensor_info(tensor_info, qnn_mgr.GetGraphOutputs(), out_key)
        qnn_mgr.Destroy()

    return tensor_info


def get_ones_tensor(tensor_info, logger):
    logger.warning(
        f"tensor '{tensor_info['name']}' use ones tensor, "
        "unexpected outputs might generate"
    )
    return torch.ones(tensor_info["shape"], dtype=eval(tensor_info["dtype"]))


def get_tensor_with_encoding(tensor, tensor_info, logger):
    scale = tensor_info["encoding"]["scale"]
    offset = tensor_info["encoding"]["offset"]

    # user gave wrong tensor for no encoding appears
    if len(scale) == 0:
        logger.error(f"tensor '{tensor_info['name']}' has no encoding")
        return get_ones_tensor(tensor_info, logger)

    # quant if tensor is float with encoding
    return (
        tensor.div(scale).add(offset).round().to(eval(tensor_info["dtype"]))
        if tensor.dtype == torch.float
        else tensor.sub(offset).mul(scale).to(torch.float32)
    )


def get_tensor(io_info, tensors, logger, checking_output=False):
    # check if enough tensors have been given
    if len(tensors) != len(io_info):
        logger.error(
            "given tensor numbers mismatch, "
            f"expected {len(io_info)} but got {len(tensors)}"
        )
        if checking_output:
            logger.error(
                "output tensors failed to generate, "
                "please check executor_runner logs."
            )
            exit(-1)

        return [get_ones_tensor(t, logger) for t in io_info]

    # list of tensors to be returned
    ret_tensors, ret_list = [], []
    for i, info in enumerate(io_info):
        ret_list.append(f"input_0_{i}.raw")
        if list(tensors[i].shape) != info["shape"]:
            logger.error(
                f"tensor '{info['name']}' shape mismatch: "
                f"users > {tensors[i].shape} - "
                f"required > {info['shape']}"
            )
            ret_tensors.append(get_ones_tensor(info, logger))
            continue

        ret_tensors.append(
            tensors[i]
            if tensors[i].dtype == eval(info["dtype"])
            else
            # try quant / dequant for given tensor if possible
            ret_tensors.append(get_tensor_with_encoding(tensors[i], info, logger))
        )
    return [ret_tensors], " ".join(ret_list)


def to_context_binary(
    model_lib, soc_model, device, host, build_folder, output_folder, logger
):
    ext = Path(model_lib).suffix
    if ext == ".bin":
        return model_lib

    assert (
        device is not None
    ), "Please assign device serial for model library conversion."
    logger.info(f"Generating context binary for {model_lib}")
    # leverage SimpleADB for model library conversion
    lib_name = Path(model_lib).stem
    sdk_root = os.getenv("QNN_SDK_ROOT")
    adb = SimpleADB(
        qnn_sdk=sdk_root,
        build_path=build_folder,
        pte_path=model_lib,
        workspace=f"/data/local/tmp/executorch/{lib_name}",
        device_id=device,
        soc_model=soc_model,
        host_id=host,
    )

    logger.info("pushing QNN libraries & tool")
    arch = adb.arch_table[soc_model]
    files = [
        f"{sdk_root}/bin/aarch64-android/qnn-context-binary-generator",
        f"{sdk_root}/lib/aarch64-android/libQnnHtp.so",
        f"{sdk_root}/lib/aarch64-android/libQnnHtpV{arch}Stub.so",
        f"{sdk_root}/lib/aarch64-android/libQnnHtpPrepare.so",
        f"{sdk_root}/lib/hexagon-v{arch}/unsigned/libQnnHtpV{arch}Skel.so",
    ]
    adb.push(files=files)

    logger.info("starting conversion")
    commands = " ".join(
        [
            f"cd {adb.workspace} &&",
            "export LD_LIBRARY_PATH=. &&",
            "./qnn-context-binary-generator",
            f"--model {Path(model_lib).name}",
            "--backend libQnnHtp.so",
            f"--binary_file {lib_name}",
        ]
    )
    adb.execute(custom_runner_cmd=commands)

    logger.info(f"collecting converted context binary - {lib_name}.bin")
    adb._adb(["pull", f"{adb.workspace}/output/{lib_name}.bin", output_folder])

    bin_path = f"{output_folder}/{lib_name}.bin"
    assert os.path.exists(bin_path), (
        "Failed to convert context binary, " "please check logcat for more details."
    )
    return bin_path


def compile(args):
    logger = get_logger()
    logger.info("prepare compiler spec for qualcomm backend")

    # setup compiler spec dedicated to QNN HTP backend
    backend_options = generate_htp_compiler_spec(use_fp16=False)
    # setup general compiler spec for QNN
    compiler_specs = generate_qnn_executorch_compiler_spec(
        soc_model=getattr(QcomChipset, args.model),
        backend_options=backend_options,
        is_from_context_binary=True,
    )
    # setup memory planning
    memory_planning_pass = MemoryPlanningPass(
        alloc_graph_input=args.allocate_graph_io,
        alloc_graph_output=args.allocate_graph_io,
    )

    # dictionary for avoiding name collision when creating custom ops
    name_map = {}
    num_bins = len(args.artifacts)
    for i, ctx_bin in enumerate(args.artifacts):
        index = i + 1
        binary_name = Path(ctx_bin).stem
        output_dir = f"{args.output_pte_folder}/{binary_name}"
        make_output_dir(output_dir)
        # conversion model library into context binary if required
        ctx_bin = to_context_binary(
            model_lib=ctx_bin,
            soc_model=args.model,
            device=args.device,
            host=args.host,
            build_folder=args.build_folder,
            output_folder=output_dir,
            logger=logger,
        )
        # step 0: check if name collision happens for context binaries
        logger.info(f"({index}/{num_bins}) checking custom op name of {ctx_bin}")
        custom_op_name = f"ctx_loader_{binary_name}"
        postfix = name_map.get(custom_op_name, 0)
        if postfix > 0:
            postfix += 1
            custom_op_name = f"{custom_op_name}_{postfix}"
        name_map[custom_op_name] = postfix
        # step 1: generate ExportedProgram with custom op as binary loader
        logger.info(f"({index}/{num_bins}) exporting program for {ctx_bin}")
        prog_info = from_context_binary(
            ctx_bin, custom_op_name, getattr(QcomChipset, args.model)
        )
        # step 2: lower to QnnBackend
        logger.info(f"({index}/{num_bins}) start lowering {ctx_bin} to QnnBackend")
        lowered_module = to_backend(
            "QnnBackend", prog_info["edge_program"], compiler_specs
        )
        # step 3: write pte files and IO information
        logger.info(f"({index}/{num_bins}) exporting {binary_name}.pte")
        with open(f"{output_dir}/{binary_name}.pte", "wb") as f:
            f.write(
                lowered_module.buffer(
                    extract_delegate_segments=True, memory_planning=memory_planning_pass
                )
            )
        logger.info(
            f"({index}/{num_bins}) exporting network graph with {binary_name}.svg"
        )
        draw_graph(binary_name, output_dir, prog_info["edge_program"].graph_module)
        logger.info(
            f"({index}/{num_bins}) exporting graph description with {binary_name}.json"
        )
        with open(f"{output_dir}/{binary_name}.json", "w") as f:
            graph_info = get_io_info(prog_info, ctx_bin, compiler_specs)
            graph_info["soc_model"] = args.model
            json.dump(graph_info, f, indent=2)


def execute(args):
    logger = get_logger()

    # load graph description file
    pte_name = Path(args.pte_directory).stem
    graph_desc = f"{args.pte_directory}/{pte_name}.json"
    logger.info(f"loading graph description: {graph_desc}")
    with open(graph_desc, "r") as f:
        graph_info = json.load(f)

    # load input files
    logger.info("loading user inputs")
    user_inputs = []
    for input_file in args.input_files:
        with open(input_file, "rb") as f:
            buffer = io.BytesIO(f.read())
            user_inputs.append(torch.load(buffer, weights_only=False))

    # check if inputs are valid, fallback to ones tensor if any
    logger.info("generating input data")
    inputs, input_list = get_tensor(graph_info["inputs"], user_inputs, logger)

    logger.info("preparing ADB connection")
    # leverage SimpleADB for e2e inference
    adb = SimpleADB(
        qnn_sdk=os.getenv("QNN_SDK_ROOT"),
        build_path=args.build_folder,
        pte_path=f"{args.pte_directory}/{pte_name}.pte",
        workspace=f"/data/local/tmp/executorch/{pte_name}",
        device_id=args.device,
        soc_model=graph_info["soc_model"],
        host_id=args.host,
        shared_buffer=args.shared_buffer,
    )

    logger.info("pushing QNN libraries & other artifacts")
    adb.push(inputs=inputs, input_list=input_list)

    logger.info("starting inference")
    adb.execute()

    logger.info("collecting output data")

    def post_process():
        output_info, outputs = graph_info["outputs"], []
        output_folder = f"{args.output_data_folder}/outputs"
        for i, f in enumerate(sorted(os.listdir(output_folder))):
            filename = os.path.join(output_folder, f)
            output = np.fromfile(
                filename, dtype=eval(f"np.{output_info[i]['dtype'].split('.')[-1]}")
            )
            outputs.append(torch.from_numpy(output.reshape(output_info[i]["shape"])))
            os.remove(filename)

        os.rmdir(output_folder)
        outputs, _ = get_tensor(output_info, outputs, logger, checking_output=True)
        # dataset length equals to 1
        for i, output in enumerate(outputs[0]):
            torch.save(output, f"{args.output_data_folder}/{output_info[i]['name']}.pt")

    make_output_dir(args.output_data_folder)
    adb.pull(args.output_data_folder, post_process)
    logger.info(
        f"execution finished, please check {args.output_data_folder} for results"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Utility to lower precompiled model libraries / "
            "context binaries from Qualcomm AI Engine Direct to executorch"
            " .pte program. Please visit https://aihub.qualcomm.com/ to "
            "download your favorite models."
        ),
    )
    subparsers = parser.add_subparsers(
        title="subcommands",
        description=(
            "[compile]: Compile designated model libraries / "
            "context binaries into .pte files. "
            "[execute]: Perform on-device inference with given .pte."
        ),
    )

    sub_compile = subparsers.add_parser(
        name="compile",
        help=(
            "e.g. python export.py compile -a model.bin -m SM8650 "
            "-b /path/to/build-android"
        ),
    )
    sub_compile.add_argument(
        "-a",
        "--artifacts",
        nargs="+",
        type=str,
        required=True,
        help=(
            "Path to AI HUB or QNN tool generated artifacts, "
            "batch process is supported. "
            "e.g. python export.py compile -a a.bin b.so c.bin "
            "-m SM8650 -s $SERIAL_NO -b /path/to/build-android"
        ),
    )
    sub_compile.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="SoC model. e.g. SM8650",
    )
    sub_compile.add_argument(
        "-s",
        "--device",
        type=str,
        help="Serial no of device which could be obtained by 'adb devices'.",
    )
    sub_compile.add_argument(
        "-o",
        "--output_pte_folder",
        type=str,
        default="./output_pte",
        help=(
            "Path to output artifacts, store in 'output_pte' if not given. "
            "graph descriptions & diagram will also be exported."
        ),
    )
    sub_compile.add_argument(
        "-b",
        "--build_folder",
        help="Path to cmake binary directory for android, e.g., /path/to/build-android",
        type=str,
        required=True,
    )
    sub_compile.add_argument(
        "-l",
        "--allocate_graph_io",
        type=bool,
        default=True,
        help=(
            "True if IO tensors are pre-allocated by framework. "
            "False for users who want to manage resources in runtime."
        ),
    )
    sub_compile.add_argument(
        "-H",
        "--host",
        type=str,
        help="Gateway hostname.",
    )
    sub_compile.set_defaults(callback=compile)

    sub_execute = subparsers.add_parser(
        name="execute",
        help=(
            "e.g. python export.py execute -p model_dir -i inp.raw " "-s device_serial"
        ),
    )
    sub_execute.add_argument(
        "-p",
        "--pte_directory",
        type=str,
        required=True,
        help="Path to .pte file folder generated from 'compile' subcommand.",
    )
    sub_execute.add_argument(
        "-i",
        "--input_files",
        nargs="*",
        type=str,
        help=(
            "Path to input files stored via torch.save. "
            "If the number / spec of input files doesn't match given .pte file, "
            "tensors filled with value 1 will be taken as inputs."
        ),
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
        "--output_data_folder",
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
        "-z",
        "--shared_buffer",
        help=(
            "Enables usage of shared buffer between application and backend for graph I/O."
            " Please use with `--allocate_graph_io False` in compile command."
        ),
        action="store_true",
    )
    sub_execute.add_argument(
        "-H",
        "--host",
        type=str,
        help="Gateway hostname.",
    )
    sub_execute.set_defaults(callback=execute)

    args = parser.parse_args()
    args.callback(args)


if __name__ == "__main__":
    main()
