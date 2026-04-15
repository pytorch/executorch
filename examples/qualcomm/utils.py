# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# TODO: reenable pyre after fixing the issues
# pyre-ignore-all-errors
import csv
import inspect
import os
import random
import shutil
from typing import Dict, List, Optional

import numpy as np
import torch
import transformers

from executorch.backends.qualcomm.export_utils import *  # noqa: F401,F403


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
                    "attention_mask": model_input["attention_mask"]
                    .reshape(1, 1, -1)
                    .to(torch.float32),
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
                torch.where(attention_mask == 0.0, -255.0, 0.0),
                decoder_input_ids,
            )
        )
        targets.append(labels)

    return inputs, targets
