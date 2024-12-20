# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# TODO: reenable pyre after fixing the issues
# pyre-ignore-all-errors

import argparse
import os
import subprocess
import sys
from pathlib import Path

from typing import Callable, List, Optional

import numpy as np

import torch
from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer, QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
from executorch.backends.qualcomm.utils.utils import (
    capture_program,
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    get_soc_to_arch_map,
)
from executorch.exir import EdgeCompileConfig, EdgeProgramManager, to_edge
from executorch.exir.backend.backend_api import to_backend
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from torch.ao.quantization.observer import MovingAverageMinMaxObserver
from torch.ao.quantization.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)


class SimpleADB:
    """
    A wrapper class for communicating with Android device

    Attributes:
        qnn_sdk (str): QNN SDK path setup in environment variable
        build_path (str): Path where artifacts were built
        pte_path (str): Path where executorch binary was stored
        workspace (str): Folder for storing artifacts on android device
        device_id (str): Serial number of android device
        soc_model (str): Chipset of device
        host_id (str): Hostname of machine where device connects
        error_only (bool): Redirect stdio and leave error messages only
        shared_buffer (bool): Apply zero-copy mechanism in runtime
        runner (str): Runtime executor binary
    """

    def __init__(
        self,
        qnn_sdk,
        build_path,
        pte_path,
        workspace,
        device_id,
        soc_model,
        host_id=None,
        error_only=False,
        shared_buffer=False,
        dump_intermediate_outputs=False,
        runner="examples/qualcomm/executor_runner/qnn_executor_runner",
    ):
        self.qnn_sdk = qnn_sdk
        self.build_path = build_path
        self.pte_path = pte_path if isinstance(pte_path, list) else [pte_path]
        self.workspace = workspace
        self.device_id = device_id
        self.host_id = host_id
        self.working_dir = Path(self.pte_path[0]).parent.absolute()
        self.input_list_filename = "input_list.txt"
        self.etdump_path = f"{self.workspace}/etdump.etdp"
        self.dump_intermediate_outputs = dump_intermediate_outputs
        self.debug_output_path = f"{self.workspace}/debug_output.bin"
        self.output_folder = f"{self.workspace}/outputs"
        self.htp_arch = get_soc_to_arch_map()[soc_model]
        self.error_only = error_only
        self.shared_buffer = shared_buffer
        self.runner = runner

    def _adb(self, cmd):
        if not self.host_id:
            cmds = ["adb", "-s", self.device_id]
        else:
            cmds = ["adb", "-H", self.host_id, "-s", self.device_id]
        cmds.extend(cmd)

        subprocess.run(
            cmds, stdout=subprocess.DEVNULL if self.error_only else sys.stdout
        )

    def push(self, inputs=None, input_list=None, files=None):
        self._adb(["shell", f"rm -rf {self.workspace}"])
        self._adb(["shell", f"mkdir -p {self.workspace}"])

        # necessary artifacts
        artifacts = [
            *self.pte_path,
            f"{self.qnn_sdk}/lib/aarch64-android/libQnnHtp.so",
            (
                f"{self.qnn_sdk}/lib/hexagon-v{self.htp_arch}/"
                f"unsigned/libQnnHtpV{self.htp_arch}Skel.so"
            ),
            (
                f"{self.qnn_sdk}/lib/aarch64-android/"
                f"libQnnHtpV{self.htp_arch}Stub.so"
            ),
            f"{self.qnn_sdk}/lib/aarch64-android/libQnnHtpPrepare.so",
            f"{self.qnn_sdk}/lib/aarch64-android/libQnnSystem.so",
            f"{self.build_path}/{self.runner}",
            f"{self.build_path}/backends/qualcomm/libqnn_executorch_backend.so",
        ]
        input_list_file, input_files = generate_inputs(
            self.working_dir, self.input_list_filename, inputs, input_list
        )

        if input_list_file is not None:
            # prepare input list
            artifacts.append(input_list_file)

        for artifact in artifacts:
            self._adb(["push", artifact, self.workspace])

        # input data
        for file_name in input_files:
            self._adb(["push", file_name, self.workspace])

        # custom files
        if files is not None:
            for file_name in files:
                self._adb(["push", file_name, self.workspace])

    def execute(self, custom_runner_cmd=None, method_index=0):
        self._adb(["shell", f"mkdir -p {self.output_folder}"])
        # run the delegation
        if custom_runner_cmd is None:
            qnn_executor_runner_args = " ".join(
                [
                    f"--model_path {os.path.basename(self.pte_path[0])}",
                    f"--output_folder_path {self.output_folder}",
                    f"--input_list_path {self.input_list_filename}",
                    f"--etdump_path {self.etdump_path}",
                    "--shared_buffer" if self.shared_buffer else "",
                    f"--debug_output_path {self.debug_output_path}",
                    (
                        "--dump_intermediate_outputs"
                        if self.dump_intermediate_outputs
                        else ""
                    ),
                    f"--method_index {method_index}",
                ]
            )
            qnn_executor_runner_cmds = " ".join(
                [
                    f"cd {self.workspace} &&",
                    f"./qnn_executor_runner {qnn_executor_runner_args}",
                ]
            )
        else:
            qnn_executor_runner_cmds = custom_runner_cmd

        self._adb(["shell", f"{qnn_executor_runner_cmds}"])

    def pull(self, output_path, callback=None):
        self._adb(["pull", "-a", self.output_folder, output_path])
        if callback:
            callback()

    def pull_etdump(self, output_path, callback=None):
        self._adb(["pull", self.etdump_path, output_path])
        if callback:
            callback()

    def pull_debug_output(self, etdump_path, debug_ouput_path, callback=None):
        self._adb(["pull", self.etdump_path, etdump_path])
        self._adb(["pull", self.debug_output_path, debug_ouput_path])
        if callback:
            callback()


def ptq_calibrate(captured_model, quantizer, dataset):
    annotated_model = prepare_pt2e(captured_model, quantizer)
    print("Quantizing(PTQ) the model...")
    # calibration
    if callable(dataset):
        dataset(annotated_model)
    else:
        for data in dataset:
            annotated_model(*data)
    return annotated_model


def qat_train(ori_model, captured_model, quantizer, dataset):
    data, targets = dataset
    annotated_model = torch.ao.quantization.move_exported_model_to_train(
        prepare_qat_pt2e(captured_model, quantizer)
    )
    optimizer = torch.optim.SGD(annotated_model.parameters(), lr=0.00001)
    criterion = torch.nn.CrossEntropyLoss()
    for i, d in enumerate(data):
        print(f"Epoch {i}")
        if i > 3:
            # Freeze quantizer parameters
            annotated_model.apply(torch.ao.quantization.disable_observer)
        if i > 2:
            # Freeze batch norm mean and variance estimates
            annotated_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

        output = annotated_model(*d)
        loss = criterion(output, targets[i])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return torch.ao.quantization.quantize_pt2e.convert_pt2e(
        torch.ao.quantization.move_exported_model_to_eval(annotated_model)
    )


def make_quantizer(
    quant_dtype: Optional[QuantDtype] = QuantDtype.use_8a8w,
    custom_annotations=(),
    per_channel_conv=True,
    per_channel_linear=False,
    act_observer=MovingAverageMinMaxObserver,
    is_qat=False,
):
    quantizer = QnnQuantizer()
    quantizer.add_custom_quant_annotations(custom_annotations)
    quantizer.set_per_channel_conv_quant(per_channel_conv)
    quantizer.set_per_channel_linear_quant(per_channel_linear)
    quantizer.set_quant_config(quant_dtype, is_qat, act_observer)
    return quantizer


# TODO: refactor to support different backends
def build_executorch_binary(
    model,  # noqa: B006
    inputs,  # noqa: B006
    soc_model,
    file_name,
    dataset: List[torch.Tensor] | Callable[[torch.fx.GraphModule], None],
    skip_node_id_set=None,
    skip_node_op_set=None,
    quant_dtype: Optional[QuantDtype] = None,
    custom_quantizer=None,
    shared_buffer=False,
    metadata=None,
    dump_intermediate_outputs=False,
    custom_pass_config=frozenset(),
    qat_training_data=None,
):
    """
    A function to generate an ExecuTorch binary for Qualcomm platforms.

    Attributes:
        model (torch.nn.Module): The model to be converted into an ExecuTorch binary.
        inputs (torch.Tensor): Sample input tensors required for model export.
        soc_model (QcomChipset): The target Qualcomm System on Chip (SoC) model.
        file_name (str): Name for the output binary file (.pte).
        dataset (List[torch.Tensor] | Callable): A dataset for quantization calibration.
        skip_node_id_set (set, optional): Set of node IDs to be skipped during partition.
        skip_node_op_set (set, optional): Set of operation node  to be skipped during partition.
        quant_dtype (QuantDtype, optional): Data type for quantization.
        custom_quantizer (Callable, optional): Custom quantizer.
        shared_buffer (bool, optional): Applies zero-copy mechanism to optimize runtime memory allocation.
        metadata (dict, optional): An optional dictionary that maps each method name to a constant value in eager mode.
        dump_intermediate_outputs (bool, optional): Enables dumping model intermediate outputs.
        custom_pass_config (frozenset, optional): Set of custom passes for model processing.

    Returns:
        None: The function writes the output to a specified .pte file.
    """
    if quant_dtype is not None:
        captured_model = torch.export.export(model, inputs).module()
        if qat_training_data:
            quantizer = custom_quantizer or make_quantizer(
                quant_dtype=quant_dtype, is_qat=True
            )
            # qat training
            annotated_model = qat_train(
                model, captured_model, quantizer, qat_training_data
            )
        else:
            quantizer = custom_quantizer or make_quantizer(quant_dtype=quant_dtype)
            # ptq calibration
            annotated_model = ptq_calibrate(captured_model, quantizer, dataset)

        quantized_model = convert_pt2e(annotated_model)
        edge_prog = capture_program(quantized_model, inputs, custom_pass_config)
    else:
        edge_prog = capture_program(model, inputs, custom_pass_config)

    backend_options = generate_htp_compiler_spec(
        use_fp16=False if quant_dtype else True
    )
    qnn_partitioner = QnnPartitioner(
        generate_qnn_executorch_compiler_spec(
            soc_model=getattr(QcomChipset, soc_model),
            backend_options=backend_options,
            shared_buffer=shared_buffer,
            dump_intermediate_outputs=dump_intermediate_outputs,
        ),
        skip_node_id_set,
        skip_node_op_set,
    )

    executorch_config = ExecutorchBackendConfig(
        # For shared buffer, user must pass the memory address
        # which is allocated by RPC memory to executor runner.
        # Therefore, won't want to pre-allocate
        # by memory manager in runtime.
        memory_planning_pass=MemoryPlanningPass(
            alloc_graph_input=not shared_buffer,
            alloc_graph_output=not shared_buffer,
        ),
    )

    if metadata is None:
        exported_program = to_backend(edge_prog.exported_program, qnn_partitioner)
        exported_program.graph_module.graph.print_tabular()
        exec_prog = to_edge(exported_program).to_executorch(config=executorch_config)
        with open(f"{file_name}.pte", "wb") as file:
            file.write(exec_prog.buffer)
    else:
        edge_prog_mgr = EdgeProgramManager(
            edge_programs={"forward": edge_prog.exported_program},
            constant_methods=metadata,
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )

        edge_prog_mgr = edge_prog_mgr.to_backend(qnn_partitioner)
        exec_prog_mgr = edge_prog_mgr.to_executorch(config=executorch_config)
        with open(f"{file_name}.pte", "wb") as file:
            file.write(exec_prog_mgr.buffer)


def make_output_dir(path: str):
    if os.path.exists(path):
        for f in os.listdir(path):
            os.remove(os.path.join(path, f))
        os.removedirs(path)
    os.makedirs(path)


def topk_accuracy(predictions, targets, k):
    def solve(prob, target, k):
        _, indices = torch.topk(prob, k=k, sorted=True)
        golden = torch.reshape(target, [-1, 1])
        correct = (golden == indices) * 1.0
        top_k_accuracy = torch.mean(correct) * k
        return top_k_accuracy

    cnt = 0
    for index, pred in enumerate(predictions):
        cnt += solve(torch.from_numpy(pred), targets[index], k)

    return cnt * 100.0 / len(predictions)


def segmentation_metrics(predictions, targets, classes):
    def make_confusion(goldens, predictions, num_classes):
        def histogram(golden, predict):
            mask = golden < num_classes
            hist = np.bincount(
                num_classes * golden[mask].astype(int) + predict[mask],
                minlength=num_classes**2,
            ).reshape(num_classes, num_classes)
            return hist

        confusion = np.zeros((num_classes, num_classes))
        for g, p in zip(goldens, predictions):
            confusion += histogram(g.flatten(), p.flatten())

        return confusion

    eps = 1e-6
    confusion = make_confusion(targets, predictions, len(classes))
    pa = np.diag(confusion).sum() / (confusion.sum() + eps)
    mpa = np.mean(np.diag(confusion) / (confusion.sum(axis=1) + eps))
    iou = np.diag(confusion) / (
        confusion.sum(axis=1) + confusion.sum(axis=0) - np.diag(confusion) + eps
    )
    miou = np.mean(iou)
    cls_iou = dict(zip(classes, iou))
    return (pa, mpa, miou, cls_iou)


def get_imagenet_dataset(
    dataset_path, data_size, image_shape, crop_size=None, shuffle=True
):
    from torchvision import datasets, transforms

    def get_data_loader():
        preprocess = transforms.Compose(
            [
                transforms.Resize(image_shape),
                transforms.CenterCrop(crop_size or image_shape[0]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        imagenet_data = datasets.ImageFolder(dataset_path, transform=preprocess)
        return torch.utils.data.DataLoader(
            imagenet_data,
            shuffle=shuffle,
        )

    # prepare input data
    inputs, targets, input_list = [], [], ""
    data_loader = get_data_loader()
    for index, data in enumerate(data_loader):
        if index >= data_size:
            break
        feature, target = data
        inputs.append((feature,))
        targets.append(target)
        input_list += f"input_{index}_0.raw\n"

    return inputs, targets, input_list


def setup_common_args_and_variables():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        help="SoC model of current device. e.g. 'SM8550' for Snapdragon 8 Gen 2.",
        type=str,
        required=True,
    )

    parser.add_argument(
        "-b",
        "--build_folder",
        help="path to cmake binary directory for android, e.g., /path/to/build-android",
        type=str,
        required=True,
    )

    parser.add_argument(
        "-H",
        "--host",
        help="hostname where android device is connected.",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--ip",
        help="IPC address for delivering execution result",
        default="",
        type=str,
    )

    parser.add_argument(
        "--port",
        help="IPC port for delivering execution result",
        default=-1,
        type=int,
    )

    parser.add_argument(
        "-S",
        "--skip_delegate_node_ids",
        help="If specified, skip delegation for the specified node based on node ids. Node ids should be seperated by comma. e.g., aten_relu_default_10,aten_relu_default_2",
        default=None,
        type=str,
    )

    parser.add_argument(
        "-f",
        "--skip_delegate_node_ops",
        help="If specified, skip delegation for the specified op. Node ops should be seperated by comma. e.g., aten.add.Tensor,aten.relu.default",
        default=None,
        type=str,
    )

    parser.add_argument(
        "-c",
        "--compile_only",
        help="If specified, only compile the model.",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "-s",
        "--device",
        help="serial number for android device communicated via ADB.",
        type=str,
    )

    parser.add_argument(
        "-z",
        "--shared_buffer",
        help="Enables usage of shared buffer between application and backend for graph I/O.",
        action="store_true",
    )

    parser.add_argument(
        "--skip_push",
        help="If specified, skip pushing files to device.",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--dump_intermediate_outputs",
        help="If specified, enable dump intermediate outputs",
        action="store_true",
        default=False,
    )

    # QNN_SDK_ROOT might also be an argument, but it is used in various places.
    # So maybe it's fine to just use the environment.
    if "QNN_SDK_ROOT" not in os.environ:
        raise RuntimeError("Environment variable QNN_SDK_ROOT must be set")
    print(f"QNN_SDK_ROOT={os.getenv('QNN_SDK_ROOT')}")

    return parser


def parse_skip_delegation_node(args):
    skip_node_id_set = set()
    skip_node_op_set = set()

    if args.skip_delegate_node_ids is not None:
        skip_node_id_set = set(map(str, args.skip_delegate_node_ids.split(",")))
        print("Skipping following node ids: ", skip_node_id_set)

    if args.skip_delegate_node_ops is not None:
        skip_node_op_set = set(map(str, args.skip_delegate_node_ops.split(",")))
        print("Skipping following node ops: ", skip_node_op_set)

    return skip_node_id_set, skip_node_op_set


def generate_inputs(dest_path: str, file_name: str, inputs=None, input_list=None):
    input_list_file = None
    input_files = []

    # Prepare input list
    if input_list is not None:
        input_list_file = f"{dest_path}/{file_name}"
        with open(input_list_file, "w") as f:
            f.write(input_list)
            f.flush()

    # Prepare input data
    if inputs is not None:
        for idx, data in enumerate(inputs):
            for i, d in enumerate(data):
                file_name = f"{dest_path}/input_{idx}_{i}.raw"
                d.detach().numpy().tofile(file_name)
                input_files.append(file_name)

    return input_list_file, input_files
