# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
from pathlib import Path

import numpy as np

import torch
from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
from executorch.backends.qualcomm.qnn_quantizer import (
    get_default_16bit_qnn_ptq_config,
    get_default_8bit_qnn_ptq_config,
    QnnQuantizer,
)
from executorch.backends.qualcomm.utils.utils import (
    capture_program,
    generate_qnn_executorch_compiler_spec,
    SoCModel,
)
from executorch.exir.backend.backend_api import to_backend
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e


class SimpleADB:
    def __init__(
        self,
        qnn_sdk,
        artifact_path,
        pte_path,
        workspace,
        device_id,
        soc_model,
        host_id=None,
    ):
        self.qnn_sdk = qnn_sdk
        self.artifact_path = artifact_path
        self.pte_path = pte_path
        self.workspace = workspace
        self.device_id = device_id
        self.host_id = host_id
        self.working_dir = Path(self.pte_path).parent.absolute()
        self.input_list_filename = "input_list.txt"
        self.output_folder = f"{self.workspace}/outputs"
        arch_table = {
            "SM8550": "73",
            "SM8475": "69",
            "SM8450": "69",
        }
        self.soc_model = arch_table[soc_model]

    def _adb(self, cmd):
        if not self.host_id:
            cmds = ["adb", "-s", self.device_id]
        else:
            cmds = ["adb", "-H", self.host_id, "-s", self.device_id]
        cmds.extend(cmd)
        subprocess.run(cmds)

    def push(self, inputs, input_list):
        self._adb(["shell", f"rm -rf {self.workspace}"])
        self._adb(["shell", f"mkdir -p {self.workspace}"])

        # prepare input list
        input_list_file = f"{self.working_dir}/{self.input_list_filename}"
        with open(input_list_file, "w") as f:
            f.write(input_list)
            f.flush()

        # necessary artifacts
        for artifact in [
            f"{self.pte_path}",
            f"{self.qnn_sdk}/lib/aarch64-android/libQnnHtp.so",
            (
                f"{self.qnn_sdk}/lib/hexagon-v{self.soc_model}/"
                f"unsigned/libQnnHtpV{self.soc_model}Skel.so"
            ),
            (
                f"{self.qnn_sdk}/lib/aarch64-android/"
                f"libQnnHtpV{self.soc_model}Stub.so"
            ),
            f"{self.qnn_sdk}/lib/aarch64-android/libQnnSystem.so",
            f"{self.artifact_path}/examples/qualcomm/qnn_executor_runner",
            f"{self.artifact_path}/backends/qualcomm/libqnn_executorch_backend.so",
            input_list_file,
        ]:
            self._adb(["push", artifact, self.workspace])

        # input data
        for idx, data in enumerate(inputs):
            for i, d in enumerate(data):
                file_name = f"{self.working_dir}/input_{idx}_{i}.raw"
                d.detach().numpy().tofile(file_name)
                self._adb(["push", file_name, self.workspace])

    def execute(self):
        self._adb(["shell", f"mkdir -p {self.output_folder}"])
        # run the delegation
        qnn_executor_runner_args = " ".join(
            [
                f"--model_path {os.path.basename(self.pte_path)}",
                f"--output_folder_path {self.output_folder}",
                f"--input_list_path {self.input_list_filename}",
            ]
        )
        qnn_executor_runner_cmds = " ".join(
            [
                f"cd {self.workspace} &&",
                "export ADSP_LIBRARY_PATH=. &&",
                "export LD_LIBRARY_PATH=. &&",
                f"./qnn_executor_runner {qnn_executor_runner_args}",
            ]
        )
        self._adb(["shell", f"{qnn_executor_runner_cmds}"])

    def pull(self, output_path, callback=None):
        self._adb(["pull", "-a", f"{self.output_folder}", output_path])
        if callback:
            callback()


def build_executorch_binary(
    model,  # noqa: B006
    inputs,  # noqa: B006
    soc_model,
    file_name,
    dataset,
    use_fp16=False,
    use_16bit_quant=False,
    custom_annotations=(),
):
    if not use_fp16:
        quantizer = QnnQuantizer()
        quantizer.add_custom_quant_annotations(custom_annotations)
        if use_16bit_quant:
            quantizer.add_16bit_quant_ops(quantizer.SUPPORTED_OPS)
            quantizer.set_bit16_op_quant_config(get_default_16bit_qnn_ptq_config())
        else:
            quantizer.set_bit8_op_quant_config(get_default_8bit_qnn_ptq_config())

        captured_model = torch._export.capture_pre_autograd_graph(model, inputs)
        annotated_model = prepare_pt2e(captured_model, quantizer)
        print("Quantizing the model...")
        # calibration
        for data in dataset:
            annotated_model(*data)
        quantized_model = convert_pt2e(annotated_model)

        edge_prog = capture_program(quantized_model, inputs)
    else:
        edge_prog = capture_program(model, inputs)

    arch_table = {
        "SM8550": SoCModel.SM8550,
        "SM8475": SoCModel.SM8475,
        "SM8450": SoCModel.SM8450,
    }

    QnnPartitioner.set_compiler_spec(
        generate_qnn_executorch_compiler_spec(
            is_fp16=use_fp16,
            soc_model=arch_table[soc_model],
            debug=False,
            saver=False,
        )
    )
    edge_prog.exported_program = to_backend(
        edge_prog.exported_program, QnnPartitioner()
    )
    edge_prog.exported_program.graph_module.graph.print_tabular()
    exec_prog = edge_prog.to_executorch()
    with open(f"{file_name}.pte", "wb") as file:
        file.write(exec_prog.buffer)


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
