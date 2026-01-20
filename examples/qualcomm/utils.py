# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# TODO: reenable pyre after fixing the issues
# pyre-ignore-all-errors
import argparse
import csv
import inspect
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchao
import transformers
from executorch.backends.qualcomm.debugger.qnn_intermediate_debugger import (
    QNNIntermediateDebugger,
)
from executorch.backends.qualcomm.quantizer.quantizer import (
    ModuleQConfig,
    QnnQuantizer,
    QuantDtype,
)
from executorch.backends.qualcomm.serialization.qc_schema import (
    QcomChipset,
    QnnExecuTorchBackendType,
    QnnExecuTorchOpPackageOptions,
)
from executorch.backends.qualcomm.utils.utils import (
    generate_gpu_compiler_spec,
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    get_soc_to_arch_map,
    to_edge_transform_and_lower_to_qnn,
)
from executorch.exir.backend.utils import get_delegates
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from torchao.quantization.pt2e import MovingAverageMinMaxObserver
from torchao.quantization.pt2e.quantize_pt2e import (
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
        target (str): Target toolchain name
        expected_input_shape (Tuple[torch.Size]): Input shape of dynamic graph
        expected_output_shape (Tuple[torch.Size]): Output shape of dynamic graph
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
        target="aarch64-android",
        backend=QnnExecuTorchBackendType.kHtpBackend,
        expected_input_shape=None,
        expected_output_shape=None,
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
        self.target = target
        self.backend = backend
        self.expected_input_shape = expected_input_shape
        self.expected_output_shape = expected_output_shape
        self.extra_cmds = ""

    def _adb(self, cmd, output_callback: Optional[Callable[[str], None]] = None):
        if not self.host_id:
            cmds = ["adb", "-s", self.device_id]
        else:
            cmds = ["adb", "-H", self.host_id, "-s", self.device_id]
        cmds.extend(cmd)

        if output_callback:
            result = subprocess.run(
                cmds, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            output_callback(result)
        else:
            subprocess.run(
                cmds, stdout=subprocess.DEVNULL if self.error_only else sys.stdout
            )

    def push(self, inputs=None, input_list=None, files=None, init_env=True):
        artifacts = []
        if init_env:
            self._adb(["shell", f"rm -rf {self.workspace}"])
            self._adb(["shell", f"mkdir -p {self.workspace}"])

        # necessary artifacts
        artifacts = {
            QnnExecuTorchBackendType.kHtpBackend: [
                f"{self.qnn_sdk}/lib/{self.target}/libQnnHtp.so",
                (
                    f"{self.qnn_sdk}/lib/hexagon-v{self.htp_arch}/"
                    f"unsigned/libQnnHtpV{self.htp_arch}Skel.so"
                ),
                (
                    f"{self.qnn_sdk}/lib/{self.target}/"
                    f"libQnnHtpV{self.htp_arch}Stub.so"
                ),
                f"{self.qnn_sdk}/lib/{self.target}/libQnnHtpPrepare.so",
            ],
            QnnExecuTorchBackendType.kGpuBackend: [
                f"{self.qnn_sdk}/lib/{self.target}/libQnnGpu.so",
            ],
        }[self.backend]

        artifacts.extend(
            [
                *self.pte_path,
                f"{self.qnn_sdk}/lib/{self.target}/libQnnSystem.so",
                f"{self.build_path}/{self.runner}",
                f"{self.build_path}/backends/qualcomm/libqnn_executorch_backend.so",
                f"{self.qnn_sdk}/lib/{self.target}/libQnnModelDlc.so",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_list_file, input_files = generate_inputs(
                tmp_dir, self.input_list_filename, inputs
            )

            if input_list_file is not None:
                # prepare input list
                artifacts.append(input_list_file)

            for artifact in artifacts:
                self._adb(["push", artifact, self.workspace])

            # input data
            for file_name in input_files:
                self._adb(["push", file_name, self.workspace])

            # dynamic shape related
            if self.expected_input_shape and self.expected_output_shape:
                shape_info = {
                    "input_shape": self.expected_input_shape,
                    "output_shape": self.expected_output_shape,
                }
                for name, shapes in shape_info.items():
                    with open(f"{tmp_dir}/{name}.txt", "w") as f:
                        for s in shapes:
                            f.write(str(tuple(s)).strip("()") + "\n")
                    self._adb(["push", f"{tmp_dir}/{name}.txt", self.workspace])
                    self.extra_cmds += f" --{name}_path {name}.txt"

        # custom files
        if files is not None:
            for file_name in files:
                self._adb(["push", file_name, self.workspace])

    def execute(
        self,
        custom_runner_cmd=None,
        method_index=0,
        output_callback: Optional[Callable[[str], None]] = None,
    ):
        self._adb(["shell", f"mkdir -p {self.output_folder}"])
        # run the delegation
        if custom_runner_cmd is None:
            qnn_executor_runner_args = (
                " ".join(
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
                + self.extra_cmds
            )
            qnn_executor_runner_cmds = " ".join(
                [
                    f"cd {self.workspace} &&",
                    "chmod +x ./qnn_executor_runner &&",
                    f"./qnn_executor_runner {qnn_executor_runner_args}",
                ]
            )
        else:
            qnn_executor_runner_cmds = custom_runner_cmd
        self._adb(
            ["shell", f"{qnn_executor_runner_cmds}"], output_callback=output_callback
        )

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
    annotated_model = torchao.quantization.pt2e.move_exported_model_to_train(
        prepare_qat_pt2e(captured_model, quantizer)
    )
    optimizer = torch.optim.SGD(annotated_model.parameters(), lr=0.00001)
    criterion = torch.nn.CrossEntropyLoss()
    for i, d in enumerate(data):
        print(f"Epoch {i}")
        if i > 3:
            # Freeze quantizer parameters
            annotated_model.apply(
                torchao.quantization.pt2e.fake_quantize.disable_observer
            )
        if i > 2:
            # Freeze batch norm mean and variance estimates
            annotated_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

        output = annotated_model(*d)
        loss = criterion(output, targets[i])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return convert_pt2e(
        torchao.quantization.pt2e.move_exported_model_to_eval(annotated_model),
    )


def make_quantizer(
    quant_dtype: Optional[QuantDtype] = QuantDtype.use_8a8w,
    custom_annotations=(),
    per_channel_conv=True,
    per_channel_linear=False,
    act_observer=MovingAverageMinMaxObserver,
    is_qat=False,
    submodule_qconfig_list: Optional[List[Tuple[Callable, ModuleQConfig]]] = None,
    eps=None,
):
    quantizer = QnnQuantizer()
    quantizer.add_custom_quant_annotations(custom_annotations)
    quantizer.set_default_quant_config(
        quant_dtype,
        is_qat=is_qat,
        is_conv_per_channel=per_channel_conv,
        is_linear_per_channel=per_channel_linear,
        act_observer=act_observer,
        eps=eps,
    )
    submodule_qconfig_list = submodule_qconfig_list or []
    quantizer.set_submodule_qconfig_list(submodule_qconfig_list)
    return quantizer


def replace_module_with_custom_class(
    model: torch.nn.Module,
    target_class: torch.nn.Module,
    custom_class: torch.nn.Module,
    strict: bool = False,
    extra_custom_kwargs: Optional[Dict] = None,
):
    """
    Recursively replaces all instances of `target_class` in `model` with `custom_class`.

    Args:
        model (torch.nn.Module): The root module to search within.
        target_class (type): The class to be replaced.
        custom_class (type): The class to replace with.
        strict (bool): Whether to strictly enforce that the keys in `state_dict` match the model.
        extra_custom_kwargs: Extra keyword arguments to override or extend the constructor args.

    Example:
        >>> class MyDecoder(Decoder):
        ...     def __init__(self, ...)
        ...         super().__init__()
        ...         freqs_cos, freqs_sin = precompute_freqs_cis(...)
        ...         self.register_buffer("freqs_cos", freqs_cos)
        ...         self.register_buffer("freqs_sin", freqs_sin)
        ...
        ...     def forward(self, x):
        ...         ....
        >>> model = Decoder()
        >>> replace_module_with_custom_class(model, Decoder, MyDecoder)
    """

    def extract_init_args_from_instance(instance):
        init_signature = inspect.signature(instance.__init__)
        init_params = [
            param
            for param in init_signature.parameters.values()
            if param.name != "self"
        ]

        extracted_args = {}
        for param in init_params:
            name = param.name
            if hasattr(instance, name):
                extracted_args[name] = getattr(instance, name)
            elif param.default is not inspect.Parameter.empty:
                extracted_args[name] = param.default

        return extracted_args

    if extra_custom_kwargs is None:
        extra_custom_kwargs = {}

    for name, child in model.named_children():
        if isinstance(child, target_class):
            state_dict = child.state_dict()

            original_args = extract_init_args_from_instance(child)
            new_module = custom_class(**{**original_args, **extra_custom_kwargs})
            new_module.load_state_dict(state_dict, strict=strict)
            new_module.eval()

            setattr(model, name, new_module)
        else:
            replace_module_with_custom_class(
                child, target_class, custom_class, strict, extra_custom_kwargs
            )


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
    custom_quantizer: Optional[QnnQuantizer] = None,
    shared_buffer=False,
    metadata=None,
    dump_intermediate_outputs=False,
    qnn_intermediate_debugger: QNNIntermediateDebugger = None,
    backend=QnnExecuTorchBackendType.kHtpBackend,
    passes_job=None,
    passes_dependency=None,
    qat_training_data=None,
    online_prepare=False,
    optrace=False,
    op_package_options: QnnExecuTorchOpPackageOptions = None,
):
    """
    A function to generate an ExecuTorch binary for Qualcomm platforms.

    Attributes:
        model (torch.nn.Module): The model to be converted into an ExecuTorch binary.
        inputs (torch.Tensor): Sample input tensors required for model export.
        soc_model (QcomChipset): The target Qualcomm System on Chip (SoC) model.
        backend (QnnExecuTorchBackendType): The target backend.
        file_name (str): Name for the output binary file (.pte).
        dataset (List[torch.Tensor] | Callable): A dataset for quantization calibration.
        skip_node_id_set (set, optional): Set of node IDs to be skipped during partition.
        skip_node_op_set (set, optional): Set of operation node  to be skipped during partition.
        quant_dtype (QuantDtype, optional): Data type for quantization.
        custom_quantizer (Callable, optional): Custom quantizer.
        shared_buffer (bool, optional): Applies zero-copy mechanism to optimize runtime memory allocation.
        metadata (dict, optional): An optional dictionary that maps each method name to a constant value in eager mode.
        dump_intermediate_outputs (bool, optional): Enables dumping model intermediate outputs.
        passes_job (OrderedDict, optional): Custom passes job in capture_program, users can enable/disable specific passes or modify their attributes.
        passes_dependency (Dict, optional): A dictionary mapping each pass to its corresponding list of dependencies.
        qat_training_data (List[torch.Tensor], optional): A dataset for quantization aware training(QAT). Typically is a pair of tensors, such as [features, ground truth].
        online_prepare (bool, optional): Compose QNN graph on device if set to True.
        optrace (bool, optional): Enable optrace mode for performance analysis if set to True.
        op_package_options: Optional structure to specify op packages
            loaded and used by the backend.

    Returns:
        None: The function writes the output to a specified .pte file.
    """
    if backend == QnnExecuTorchBackendType.kGpuBackend and not online_prepare:
        raise RuntimeError("Currently GPU backend only supports online_prepare.")
    backend_options = {
        QnnExecuTorchBackendType.kGpuBackend: generate_gpu_compiler_spec(),
        QnnExecuTorchBackendType.kHtpBackend: generate_htp_compiler_spec(
            use_fp16=False if quant_dtype is not None else True
        ),
    }[backend]
    compile_spec = generate_qnn_executorch_compiler_spec(
        soc_model=getattr(QcomChipset, soc_model),
        backend_options=backend_options,
        online_prepare=online_prepare,
        optrace=optrace,
        shared_buffer=shared_buffer,
        dump_intermediate_outputs=dump_intermediate_outputs,
        op_package_options=op_package_options,
    )
    if quant_dtype is not None or custom_quantizer is not None:
        captured_model = torch.export.export(model, inputs, strict=False).module()
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
        edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
            quantized_model,
            inputs,
            compile_spec,
            constant_methods=metadata,
            passes_job=passes_job,
            dep_table=passes_dependency,
            skip_node_id_set=skip_node_id_set,
            skip_node_op_set=skip_node_op_set,
        )
    else:
        edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
            model,
            inputs,
            compile_spec,
            constant_methods=metadata,
            passes_job=passes_job,
            skip_node_id_set=skip_node_id_set,
            skip_node_op_set=skip_node_op_set,
        )

    if qnn_intermediate_debugger:
        lowered_module_nodes = get_delegates(edge_prog_mgr.exported_program().graph)
        assert (
            len(lowered_module_nodes) == 1
        ), "Graph with partitions are currently unsupported."

        lowered_module_node = lowered_module_nodes[0]
        lowered_module = getattr(
            edge_prog_mgr.exported_program().graph_module, lowered_module_node.name
        )
        edge_module = lowered_module.original_module.module()
        qnn_intermediate_debugger._set_edge_module(
            edge_module=edge_module,
            debug_handle_map=lowered_module.meta["debug_handle_map"],
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
    pte_name = f"{file_name}.pte"
    exec_prog_mgr = edge_prog_mgr.to_executorch(config=executorch_config)
    with open(pte_name, "wb") as file:
        exec_prog_mgr.write_to_file(file)


def make_output_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
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


def class_agnostic_mIoU(predictions, targets):
    total_iou = 0
    for pred, tar in zip(predictions, targets):
        inter = np.count_nonzero(pred & tar)
        union = np.count_nonzero(pred | tar)
        total_iou += inter / (union + 1e-10)
    return total_iou / len(predictions)


def evaluate_squad(predicted_texts: List[str], target_texts: List[str]):
    import evaluate

    squad_metric = evaluate.load("squad")

    predictions = []
    references = []

    for i, (pred, target) in enumerate(zip(predicted_texts, target_texts)):
        predictions.append({"id": str(i), "prediction_text": pred.strip()})
        references.append(
            {
                "id": str(i),
                "answers": {
                    "text": [target.strip()],
                    "answer_start": [0],  # answer_start could be dummy
                },
            }
        )

    results = squad_metric.compute(predictions=predictions, references=references)
    results["f1"] /= 100
    results["exact_match"] /= 100
    return results


def get_backend_type(backend: str):
    return getattr(QnnExecuTorchBackendType, f"k{backend.title()}Backend")


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
    inputs, targets = [], []
    data_loader = get_data_loader()
    for index, data in enumerate(data_loader):
        if index >= data_size:
            break
        feature, target = data
        inputs.append((feature,))
        targets.append(target)

    return inputs, targets


def get_masked_language_model_dataset(dataset_path, tokenizer, data_size, shuffle=True):

    def get_data_loader():
        class MaskedSentencesDataset(torch.utils.data.Dataset):
            def __init__(self, dataset_path, tokenizer, data_size) -> None:
                self.data_size = data_size
                self.dataset = self._get_val_dataset(dataset_path, data_size, tokenizer)

            def _get_val_dataset(self, dataset_path, data_size, tokenizer):
                data_collator = transformers.DataCollatorForLanguageModeling(
                    tokenizer=tokenizer
                )
                with open(dataset_path, "r") as f:
                    texts = f.read().split("\n")
                    texts = [
                        text for text in random.choices(texts, k=2000) if len(text) > 1
                    ]
                    dataset = data_collator([tokenizer(text) for text in texts])
                return dataset

            def __getitem__(self, idx):
                return (
                    self.dataset["input_ids"][idx].to(torch.int32),
                    self.dataset["attention_mask"][idx].to(torch.float32),
                    self.dataset["labels"][idx],
                )

            def __len__(self):
                return self.data_size

        dataset = MaskedSentencesDataset(dataset_path, tokenizer, data_size)
        return torch.utils.data.DataLoader(
            dataset,
            shuffle=shuffle,
        )

    # prepare input data
    inputs, targets = [], []
    data_loader = get_data_loader()
    for data in data_loader:
        if len(inputs) >= data_size:
            break
        input_ids = data[0]
        attention_mask = data[1]
        target = data[2][0]
        indice = [i for i, x in enumerate(target) if x != -100]
        # continue if no mask annotated
        if len(indice) == 0:
            continue
        inputs.append((input_ids, attention_mask))
        targets.append(target)

    return inputs, targets


def get_seq2seq_dataset_from_squad_csv(  # noqa: C901
    dataset_path,
    tokenizer,
    data_size,
    max_hidden_seq_length=384,
    shuffle=True,
):

    def get_data_loader(max_hidden_seq_length):
        class SquadSeq2SeqDataset(torch.utils.data.Dataset):
            def __init__(
                self,
                dataset_path,
                tokenizer,
                data_size,
                max_hidden_seq_length,
            ):
                self.max_hidden_seq_length = max_hidden_seq_length
                self.tokenizer = tokenizer
                self.samples = self._load_and_process(dataset_path, data_size)

            def _load_and_process(self, path, max_samples):
                with open(path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                if shuffle:
                    random.shuffle(rows)
                samples = []
                for row in rows:
                    question = row["question"].strip()
                    context = row["context"].strip()
                    answer = row["answer"].strip()
                    if not question or not context or not answer:
                        continue
                    input_text = f"question: {question} context: {context}"
                    target_text = answer
                    samples.append((input_text, target_text))
                    if len(samples) >= max_samples:
                        break
                return samples

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                input_text, target_text = self.samples[idx]
                model_input = tokenizer(
                    input_text,
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_hidden_seq_length,
                    return_tensors="pt",
                )

                label = tokenizer(
                    target_text,
                    truncation=True,
                    padding="max_length",
                    max_length=64,
                    return_tensors="pt",
                )
                return {
                    "input_ids": model_input["input_ids"].squeeze(0),
                    "attention_mask": model_input["attention_mask"].squeeze(0),
                    "decoder_input_ids": torch.tensor([0], dtype=torch.long),
                    "labels": label["input_ids"].squeeze(0),
                }

        dataset = SquadSeq2SeqDataset(
            dataset_path, tokenizer, data_size, max_hidden_seq_length
        )
        collator = transformers.DataCollatorForSeq2Seq(tokenizer)
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=shuffle, collate_fn=collator
        )

    inputs, targets = [], []
    data_loader = get_data_loader(max_hidden_seq_length)
    for batch in data_loader:
        if len(inputs) >= data_size:
            break
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        decoder_input_ids = batch["decoder_input_ids"]
        labels = batch["labels"][0]

        if (labels != -100).sum().item() == 0:
            continue

        inputs.append(
            (
                input_ids.to(torch.long),
                attention_mask.to(torch.long),
                decoder_input_ids,
            )
        )
        targets.append(labels)

    return inputs, targets


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
        "--online_prepare",
        help="If specified, compose QNN graph on device.",
        action="store_true",
        default=False,
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
        help="If specified, skip delegation for the specified node based on node ids. Node ids should be separated by comma. e.g., aten_relu_default_10,aten_relu_default_2",
        default=None,
        type=str,
    )

    parser.add_argument(
        "-f",
        "--skip_delegate_node_ops",
        help="If specified, skip delegation for the specified op. Node ops should be separated by comma. e.g., aten.add.Tensor,aten.relu.default",
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
        "--backend",
        help="Backend to be deployed ('htp'/'gpu' are currently supported).",
        choices=["htp", "gpu"],
        default="htp",
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
        "-D",
        "--dump_intermediate_outputs",
        help="If specified, enable dump intermediate outputs",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "-x",
        "--enable_x86_64",
        help="Enable unittest to be executed on x86_64 platform",
        action="store_true",
    )

    parser.add_argument(
        "--ci",
        help="This flag is for Continuous Integration(CI) purpose and is NOT recommended to turn on for typical use cases. It will use random inputs instead of real inputs.",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--seed",
        help="Set the seed for generating random numbers in both torch and random.",
        type=int,
    )

    parser.add_argument(
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

    parser.add_argument(
        "--pre_gen_pte",
        help="Run the pre-generated pte in the given directory.",
        type=str,
    )

    # QNN_SDK_ROOT might also be an argument, but it is used in various places.
    # So maybe it's fine to just use the environment.
    if "QNN_SDK_ROOT" not in os.environ:
        raise RuntimeError("Environment variable QNN_SDK_ROOT must be set")
    print(f"QNN_SDK_ROOT={os.getenv('QNN_SDK_ROOT')}")

    def validate(args):
        if not args.compile_only and args.device is None:
            raise RuntimeError(
                "device serial is required if not compile only. "
                "Please specify a device serial by -s/--device argument."
            )
        if args.seed:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)

    parser.set_defaults(validate=validate)

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


def generate_inputs(dest_path: str, file_name: str, inputs=None):
    input_list_file = None
    input_files = []

    def prepare_input_file(tensor, fd, index, sub_index):
        # transform torch.Tensor to raw file
        input_file_name = f"input_{index}_{sub_index}.raw"
        input_file_path = f"{dest_path}/{input_file_name}"
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)
        tensor.detach().numpy().tofile(input_file_path)
        input_files.append(input_file_path)
        # prepare input_list
        if sub_index > 0:
            fd.write(" ")
        fd.write(input_file_name)

    # Prepare input data
    if inputs is not None:
        input_list_file = f"{dest_path}/{file_name}"

        with open(input_list_file, "w") as f:
            for idx, data in enumerate(inputs):
                sub_index = 0
                for d in data:
                    if isinstance(d, (list, tuple)):
                        for sub_d in d:
                            prepare_input_file(sub_d, f, idx, sub_index)
                            sub_index += 1
                    else:
                        prepare_input_file(d, f, idx, sub_index)
                        sub_index += 1

                f.write("\n")

    return input_list_file, input_files
