# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from pathlib import Path
from typing import Optional

import evaluate
import numpy as np
import requests

import torch
import torch.nn as nn
import torchao

from datasets import ClassLabel, DatasetDict, load_dataset

from executorch.backends.samsung.quantizer import EnnQuantizer, Precision
from executorch.backends.samsung.serialization.compile_options import (
    gen_samsung_backend_compile_spec,
)
from executorch.backends.samsung.utils.export_utils import (
    to_edge_transform_and_lower_to_enn,
)
from executorch.examples.samsung.utils import save_tensors
from executorch.exir import ExecutorchBackendConfig
from executorch.extension.export_util.utils import save_pte_program
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    MobileBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# For removing the tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "true"


class MobileBertFinetune:
    def __init__(self, metric, args):
        self.tokenizer = self.load_tokenizer()
        self.artifact = args.artifact
        self.max_length = args.max_length
        self.csv_dataset = args.csv_dataset
        self.metric = metric if metric is not None else evaluate.load("accuracy")
        self.batch_size_training = args.batch_size
        self.num_epochs = args.num_epochs_for_finetune

    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained("google/mobilebert-uncased")

    def load_CSV_dataset(self):
        # grab dataset
        if self.csv_dataset is None:
            url = "https://raw.githubusercontent.com/susanli2016/NLP-with-Python/master/data/title_conference.csv"
            print(
                "Because a CSV file is not assigned, a CSV file is downloaded from ",
                str(url),
            )
            response = requests.get(url, allow_redirects=True)
            cvs_file_path = os.path.join(self.artifact, "title_conference.csv")
            if response.status_code == 200:
                with open(cvs_file_path, "wb") as f:
                    f.write(response.content)
                print("CSV file downloaded successfully!\n\n")
            else:
                print(
                    f"Failed to download the file. Status code: {response.status_code}\n\n"
                )
        else:
            cvs_file_path = self.csv_dataset

        # load dataset
        try:
            loaded_datasets = load_dataset("csv", data_files=cvs_file_path)
            raw_labels = loaded_datasets["train"].unique("Conference")
        except:
            print(f"Error: the file '{cvs_file_path}' was not avaiable.")

        # Creating ClassLabel
        class_labels = ClassLabel(names=raw_labels)
        labels = {key: index for index, key in enumerate(raw_labels)}

        def encode_labels(example):
            example["label"] = class_labels.str2int(example["Conference"])
            return example

        loaded_datasets = loaded_datasets.map(encode_labels)

        split_dataset = loaded_datasets["train"].train_test_split(
            test_size=0.15, seed=51
        )
        raw_datasets = DatasetDict(
            {"train": split_dataset["train"], "validation": split_dataset["test"]}
        )

        if self.max_length is None:

            def preprocess_function(examples):
                return self.tokenizer(examples["Title"], truncation=True, padding=True)

        else:

            def preprocess_function(examples):
                return self.tokenizer(
                    examples["Title"],
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                )

        print("Preprocessing data...")
        tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
        tokenized_datasets.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )
        return tokenized_datasets, labels

    # Define compute metrics function
    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)

    def training(
        self,
        model,
        tokenized_datasets,
        tokenizer,
        compute_metrics,
        batch_size=8,
        num_epochs=3,
        device="cpu",
    ):
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            logging_dir="./logs",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            dataloader_pin_memory=False if device == torch.device(type="cpu") else True,
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
        )
        return trainer

    def get_finetune_mobilebert(self, artifacts_dir):
        # Pretrained bert's output ranges in a large scale. It is challenge for enn backend to support directly.
        # Please finetune mobilebert on specific tasks, make sure that bert's output and hidden states are friendly
        # to resource-constraint device.

        # Load data for classification
        tokenized_datasets, labels = self.load_CSV_dataset()

        artifacts_dir = artifacts_dir if artifacts_dir is not None else "./mobilebert"
        need_finetune = True
        os.makedirs(artifacts_dir, exist_ok=True)
        pretrained_required_files = ["config.json", "model.safetensors"]
        path = Path(artifacts_dir)
        if (path / pretrained_required_files[0]).exists() and (
            path / pretrained_required_files[1]
        ).exists():
            need_finetune = False

        # get pre-trained mobilebert
        model = MobileBertForSequenceClassification.from_pretrained(
            "google/mobilebert-uncased" if need_finetune else artifacts_dir,
            num_labels=len(labels),
            # return_dict=False,
        )

        if not need_finetune:
            return model.eval(), tokenized_datasets

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        trainer = self.training(
            model,
            tokenized_datasets,
            self.tokenizer,
            self.compute_metrics,
            self.batch_size_training,
            self.num_epochs,
            device,
        )

        # Train the model
        print(
            "\n==== Starting training for fine tuning for ",
            self.num_epochs,
            "epochs....",
        )
        trainer.train()

        # Evaluate on validation set
        print("\n==== Starting evaluating the fine tuned model ....")
        FP_eval_results = trainer.evaluate()
        print("The eval results of the trained model =", FP_eval_results)

        model.save_pretrained(artifacts_dir)

        return model, tokenized_datasets


def get_dataset(data_size, tokenized_datasets, batch_size, num_workers, device):
    # making dataset for calibrating the model...
    inputs, labels = [], []
    for i, (batch) in enumerate(
        tqdm(
            DataLoader(
                tokenized_datasets["validation"],
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=False if device == torch.device(type="cpu") else True,
            )
        )
    ):
        inputs.append((batch["input_ids"], batch["attention_mask"]))
        labels.append(batch["label"].tolist())
        if i >= int(data_size):
            break

    return inputs, labels


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def trainingQuantModel_QAT(
    model, tokenized_datasets, batch_size, workers, device, num_epochs
):
    avgloss = AverageMeter("Loss", "1.5f")

    model = torchao.quantization.pt2e.move_exported_model_to_train(model)
    optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

    criterion = nn.CrossEntropyLoss()
    model.to(device)
    print(f"\n=== Starting training on {device} for {num_epochs} epochs...")

    data_loader = DataLoader(
        tokenized_datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=False if device == torch.device(type="cpu") else True,
    )

    # --- Training and Evaluation Loop ---
    for nepoch in range(num_epochs):
        for batch in tqdm(data_loader, desc=f"Training Epoch {nepoch + 1}"):
            batch_input_ids = batch["input_ids"].to(device)
            batch_attention_mask = batch["attention_mask"].to(device)
            batch_label = batch["label"].to(device)
            logits = model(batch_input_ids, batch_attention_mask).logits
            loss = criterion(logits, batch_label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            avgloss.update(loss, batch["label"].size(0))
        print(f"Epoch {nepoch + 1} | Average Training Loss: {avgloss.avg:.4f} \n")

    return torchao.quantization.pt2e.move_exported_model_to_eval(model)


# Eval a mobileBert model
def evaluatingQuantModel_mobileBert(
    quantized_model,
    tokenized_datasets,
    device,
    batch_size_edge,
    workers,
    metric=None,
):
    if metric is None:
        metric = evaluate.load("glue", "mrpc")

    # Collect predictions
    predictions = []
    labels = []

    for batch in tqdm(
        DataLoader(
            tokenized_datasets["validation"],
            batch_size=batch_size_edge,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
        )
    ):
        batch_input_ids = batch["input_ids"].to(device)
        batch_attention_mask = batch["attention_mask"].to(device)
        outputs = quantized_model(batch_input_ids, batch_attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        predictions.extend(preds.tolist())
        labels.extend(batch["label"].tolist())

    # Compute accuracy and F1
    results = metric.compute(predictions=predictions, references=labels)
    print("Evaluation results:", results)

    return results


def build_aten_to_qat_mobilebert(
    model,
    inputs,
    quant_dtype: Optional[Precision] = None,
    is_per_channel=True,
    is_qat=True,
    tokenized_datasets="",
    batch_size_training=1,
    batch_size_edge=1,
    num_workers=8,
    num_epochs=100,
    qat_file_name="mobilebert_qat_model.pt2",
    qat_file_name_for_cpu="mobilebert_qat_model_for_cpu.pt2",
    metric=None,
    device="cpu",
):
    # Evaluating a FP32 model
    print("==================================================")
    print("\nEvaluation a FP model")
    FP_results = evaluatingQuantModel_mobileBert(
        model.eval().to(device),
        tokenized_datasets,
        device,
        batch_size_training,
        num_workers,
        metric,
    )

    # Training a quantized model with QAT
    print("\n\n==================================================")
    print("==== Starting QAT(Quantization Aware Training)....")
    quantizer = EnnQuantizer()
    quantizer.setup_quant_params(quant_dtype, is_per_channel, is_qat)
    batch_dim = torch.export.Dim("batch_size", min=1, max=batch_size_training)

    size_input_ids = (batch_size_training, inputs[0].size(1))
    size_attention_mask = (batch_size_training, inputs[1].size(1))
    vector_input_ids = torch.randint(0, 256, size_input_ids).to(device)
    vector_attention_mask = torch.randint(0, 1, size_attention_mask).to(device)
    example_inputs = (
        vector_input_ids,
        vector_attention_mask,
    )

    exported_model = torch.export.export(
        model.eval().to(device),
        example_inputs,
        dynamic_shapes={"input_ids": {0: batch_dim}, "attention_mask": {0: batch_dim}},
    ).module()
    prepared_model = prepare_pt2e(exported_model, quantizer)

    prepared_model = trainingQuantModel_QAT(
        prepared_model,
        tokenized_datasets,
        batch_size=batch_size_training,
        workers=num_workers,
        device=device,
        num_epochs=num_epochs,
    )

    quantized_model = convert_pt2e(prepared_model)

    # Evaluating a quantized model with QAT
    print("\nEvaluation a quantized model")
    results = evaluatingQuantModel_mobileBert(
        quantized_model.to(device),
        tokenized_datasets,
        device,
        batch_size_training,
        num_workers,
        metric,
    )

    print("\n------------------------------------")
    print("     FP32 Model, accuracy=", FP_results["accuracy"])
    print("Quantized Model, accuracy=", results["accuracy"])
    print(
        "  Accurarcy drop, accuracy=",
        (results["accuracy"] / FP_results["accuracy"]) * 100,
        "%",
    )
    print("------------------------------------")
    print("==== Model Evaluation complete! \n\n")

    # Saving a quantized model for GPU servers
    size_input_ids = (batch_size_edge, inputs[0].size(1))
    size_attention_mask = (batch_size_edge, inputs[1].size(1))
    vector_input_ids = torch.randint(0, 256, size_input_ids).to(device)
    vector_attention_mask = torch.randint(0, 1, size_attention_mask).to(device)
    example_inputs = (
        vector_input_ids,
        vector_attention_mask,
    )

    exported_model = torch.export.export(quantized_model, example_inputs)
    torch.export.save(exported_model, qat_file_name)
    print(f"QAT model for {device} is saved in ", qat_file_name)

    # Saving a quantized model for CPU servers
    device_cpu = torch.device(type="cpu")
    quantized_model = quantized_model.to(device_cpu)
    quantized_model = removing_gpu_node_in_graph(quantized_model)
    cpu_vector_input_ids = torch.randint(0, 256, size_input_ids).to(device_cpu)
    cpu_vector_attention_mask = torch.randint(0, 1, size_attention_mask).to(device_cpu)
    example_inputs_cpu = (
        cpu_vector_input_ids,
        cpu_vector_attention_mask,
    )

    exported_model = torch.export.export(quantized_model, example_inputs_cpu)
    torch.export.save(exported_model, qat_file_name_for_cpu)
    print(f"QAT model for {device_cpu} is saved in ", qat_file_name_for_cpu)

    # Reloading a quantized model for GPU servers
    exported_model = torch.export.load(qat_file_name)
    print("==== QAT Training complete! \n\n")

    return exported_model.module()


def removing_gpu_node_in_graph(model):
    graph = model.graph
    for node in list(graph.nodes):
        if node.target == torch.ops.aten._assert_tensor_metadata.default:
            # remove torch.ops.aten._assert_tensor_metadata.default
            node.replace_all_uses_with(node.args[0])  # bypass
            graph.erase_node(node)
        if node.target == torch.ops.aten.zeros.default:
            # Change torch.ops.aten.zeros.default
            node.kwargs = {
                "dtype": torch.int64,
                "device": torch.device("cpu"),
                "pin_memory": False,
            }
    model.graph.eliminate_dead_code()
    model.recompile()
    # complete converting GPU target ops to CPU ones.

    return model


def main(args):
    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    # define the metric for the model evaluation
    metric = evaluate.load("accuracy")

    # Fine tuning model with a csv dataset
    mobilebert_finetune = MobileBertFinetune(metric, args)
    model, tokenized_datasets = mobilebert_finetune.get_finetune_mobilebert(
        args.artifact
    )

    # Setting for QAT training
    batch_size_edge = 1  # The batch of the final graph for a target edge device is 1
    batch_size_training = args.batch_size
    num_workers = args.num_workers  # Num of dataset loaders
    num_epochs = args.num_epochs_for_QAT  # Num of epochs in QAT training
    data_num = args.calibration_number  # Num of dataset for quantization calibration

    # searching an avaiable device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # making dataset for calibrating the model...
    print("\n==== Loading calibration dataset for PTQ quantization....")
    inputs, labels = get_dataset(
        data_num, tokenized_datasets, batch_size_edge, num_workers, device
    )

    # running an example
    example_ref_input_ids = inputs[0][0].to(device)
    example_ref_attention_mask = inputs[0][1].to(device)
    example_inputs = (example_ref_input_ids, example_ref_attention_mask)
    float_out = model(*example_inputs)

    # QAT Training with a csv dataset
    qat_file_path = os.path.join(args.artifact, "mobilebert_qat_model_csv.pt2")
    qat_file_path_for_cpu = os.path.join(
        args.artifact, "mobilebert_qat_model_csv_for_cpu.pt2"
    )
    if args.qat and args.precision is not None:
        model = build_aten_to_qat_mobilebert(
            model.train(),
            example_inputs,
            quant_dtype=getattr(Precision, args.precision),
            is_qat=True,
            tokenized_datasets=tokenized_datasets,
            batch_size_training=batch_size_training,
            batch_size_edge=batch_size_edge,
            num_epochs=num_epochs,
            qat_file_name=qat_file_path,
            qat_file_name_for_cpu=qat_file_path_for_cpu,
            metric=metric,
            device=device,
        )
        quant_out = model(*example_inputs)
    else:
        # trying to load a pretrained QAT model
        if device == torch.device(type="cpu"):
            model_path = qat_file_path_for_cpu
        else:
            model_path = qat_file_path

        print(f"\n==== Loading a pretrained QAT model from '{model_path}'....")
        try:
            loaded_model = torch.export.load(model_path)
            model = loaded_model.module().to(device)
        except:
            print(f"Error: the file '{model_path}' was not avaiable.")

        quant_out = model(*example_inputs)

    compile_specs = [gen_samsung_backend_compile_spec(args.chipset)]
    edge = to_edge_transform_and_lower_to_enn(
        model, example_inputs, compile_specs=compile_specs
    )
    model_name = "mobilebert_exynos"
    exec_prog = edge.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=True)
    )
    save_pte_program(exec_prog, model_name, args.artifact)

    if args.dump:
        # Expect example inputs are tuple, including input ids and attn mask
        save_tensors(example_inputs, prefix="float_input", artifact_dir=args.artifact)
        save_tensors(float_out, prefix="float_output", artifact_dir=args.artifact)
        if args.precision:
            save_tensors(quant_out, "quant_out", artifact_dir=args.artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--chipset",
        required=True,
        help="Samsung chipset, i.e. E9955, etc",
        type=str,
    )
    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts by this example.",
        default="./mobilebert",
        type=str,
    )
    parser.add_argument(
        "--csv_dataset",
        default=None,
        help=(
            "path of a csv file  "
            "e.g. --csv_dataset ./mobilebert/title_conference.csv "
            "If you don't assign a cvs file, a csv file is loaded automatically "
            "from https://raw.githubusercontent.com/susanli2016/NLP-with-Python/master/data/title_conference.csv"
        ),
        type=str,
    )
    parser.add_argument(
        "-p",
        "--precision",
        default="A8W8",
        help=("Quantizaiton precision. If not set, the model will not be quantized."),
        choices=[None, "A8W8"],
        type=str,
    )
    parser.add_argument(
        "-cn",
        "--calibration_number",
        default=100,
        help=(
            "Assign the number of data you want "
            "to use for calibrating the quant params."
        ),
        type=int,
    )
    parser.add_argument(
        "--num-epochs-for-finetune",
        default=12,
        type=int,
        help="# of epochs for finetune training",
    )
    parser.add_argument(
        "-m",
        "--max-length",
        default=256,
        type=int,
        help="The max length of input tokens",
    )
    parser.add_argument(
        "--batch-size",
        default=32,
        type=int,
        help=(
            "Batch size for finetuning and QAT training"
            "The batch of the final graph for a target edge device is 1."
            "  It is independent on the setting of batch-size. "
        ),
    )
    parser.add_argument(
        "--num-workers",
        default=8,
        type=int,
        help="# of workers for DataLoader in QAT training",
    )
    parser.add_argument(
        "--qat",
        default=False,
        const=True,
        nargs="?",
        help=("Whether to train the model with QAT."),
        type=bool,
    )
    parser.add_argument(
        "--num-epochs-for-QAT",
        default=12,
        type=int,
        help=(
            "# of epochs for QAT training"
            ">1000 epochs is recommended to get proper accuracy"
            " with a GPU server."
        ),
    )
    parser.add_argument(
        "--dump",
        default=False,
        action="store_true",
        help=("Whether to dump all outputs. If not set, we only dump pte."),
    )
    args = parser.parse_args()
    main(args)
