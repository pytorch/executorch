# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from pathlib import Path

import torch

from executorch.backends.samsung.serialization.compile_options import (
    gen_samsung_backend_compile_spec,
)
from executorch.backends.samsung.utils.export_utils import (
    to_edge_transform_and_lower_to_enn,
)
from executorch.examples.samsung.utils import save_tensors
from executorch.exir import ExecutorchBackendConfig
from executorch.extension.export_util.utils import save_pte_program
from transformers import AutoTokenizer, MobileBertForSequenceClassification


# Output from pretrained model exceeds the representation scale of half-float.
# Finetune bert model on specific task and make output more reasonable for hardware.
# Here is an example.
class MobileBertFinetune:
    def __init__(self):
        self.tokenizer = self.load_tokenizer()

    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained("google/mobilebert-uncased")

    def get_example_inputs(self):
        encoding = self.tokenizer.encode_plus(
            "Hello, my dog is cute",
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            return_attention_mask=True,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )

        return (
            encoding["input_ids"],
            encoding["attention_mask"].to(torch.float32),
        )

    def build_loader_from_dataset(self, dataset, batch_size, usage="train"):
        """
        :param data: Provide dataset in pandas table type. The header names should be ['text', 'label'],
                    and label range from 0 (include) to total number of classification kinds (not include).
                    For example:
                    index          text                                                label
                        0     despite its title , punch drunk love is never heavy handed    1
                        1     at once half baked and overheated                             0
                        2     this is a shameless sham, ...                                 0
                        ...
        :param batch_size: Size of data fetch in one batch.
        :param usage: The type of dataset which is used to build dataloader, like train, val.
        :return: dataloader
        """
        from torch.utils.data import (
            DataLoader,
            RandomSampler,
            SequentialSampler,
            TensorDataset,
        )

        encoded_dataset = self.tokenizer.batch_encode_plus(
            dataset.text.values.tolist(),
            return_attention_mask=True,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )

        labels = torch.tensor(dataset.label.values.tolist())

        tensor_dataset = TensorDataset(
            encoded_dataset["input_ids"], encoded_dataset["attention_mask"], labels
        )
        data_loader = None
        if usage == "train":
            data_loader = DataLoader(
                tensor_dataset,
                sampler=RandomSampler(tensor_dataset),
                batch_size=batch_size,
            )
        elif usage == "val":
            data_loader = DataLoader(
                tensor_dataset,
                sampler=SequentialSampler(tensor_dataset),
                batch_size=batch_size,
                drop_last=True,
            )
        else:
            raise NotImplementedError(
                f"Unsupported `{usage}` dataset for building dataloader."
            )

        return data_loader

    def get_finetune_mobilebert(self, artifacts_dir):
        # Pretrained bert's output ranges in a large scale. It is challenge for enn backend to support directly.
        # Please finetune mobilebert on specific tasks, make sure that bert's output and hidden states are friendly
        # to resource-constraint device.
        from io import BytesIO

        import pandas as pd
        import requests

        from tqdm import tqdm
        from transformers import get_linear_schedule_with_warmup

        # sentiment classification
        train_url = "https://raw.githubusercontent.com/clairett/pytorch-sentiment-classification/refs/heads/master/data/SST2/train.tsv"
        content = requests.get(train_url, allow_redirects=True).content
        train_data = pd.read_csv(
            BytesIO(content), delimiter="\t", header=None, names=["text", "label"]
        )
        labels_set = train_data.label.unique()

        train_data_loader = self.build_loader_from_dataset(
            train_data, batch_size=64, usage="train"
        )

        val_url = "https://raw.githubusercontent.com/clairett/pytorch-sentiment-classification/refs/heads/master/data/SST2/test.tsv"
        content = requests.get(val_url, allow_redirects=True).content
        val_data = pd.read_csv(
            BytesIO(content), delimiter="\t", header=None, names=["text", "label"]
        )
        val_data_loader = self.build_loader_from_dataset(
            val_data, batch_size=64, usage="val"
        )

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
            num_labels=len(labels_set),
            return_dict=False,
        )

        if not need_finetune:
            return model.eval(), val_data_loader

        num_epochs = 5
        num_train_steps = len(train_data_loader) * num_epochs

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        for epoch in range(1, num_epochs + 1):
            print(f"Epoch {epoch}")
            model.train()
            for batch in tqdm(train_data_loader):
                texts, attention_mask, labels = batch
                texts = texts.to(device)
                labels = labels.to(device)

                loss = model(texts, attention_mask=attention_mask, labels=labels)[0]
                # backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                # update learning rate
                scheduler.step()
        model.to("cpu")

        model.save_pretrained(artifacts_dir)

        return model.eval(), val_data_loader

    def validate(self, model, val_data_loader):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_data_loader:
                texts, attention_mask, labels = batch

                loss, output = model(
                    texts, attention_mask=attention_mask, labels=labels
                )
                total_loss += loss.item()
                predictions = torch.argmax(output, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        val_loss, val_accuracy = total_loss / len(val_data_loader), correct / total
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")


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
        "--dump",
        default=False,
        action="store_true",
        help=("Whether to dump all outputs. If not set, we only dump pte."),
    )
    args = parser.parse_args()
    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    mobilebert_finetune = MobileBertFinetune()
    model, val_dataset = mobilebert_finetune.get_finetune_mobilebert(args.artifact)
    mobilebert_finetune.validate(model, val_dataset)

    example_inputs = mobilebert_finetune.get_example_inputs()
    output = model(*example_inputs)

    compile_specs = [gen_samsung_backend_compile_spec(args.chipset)]
    edge = to_edge_transform_and_lower_to_enn(
        model, example_inputs, compile_specs=compile_specs
    )
    model_name = "mobilebert_exynos_fp32"
    exec_prog = edge.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=True)
    )
    save_pte_program(exec_prog, model_name, args.artifact)

    if args.dump:
        # Expect example inputs are tuple, including input ids and attn mask
        save_tensors(example_inputs, prefix="float_input", artifact_dir=args.artifact)
        save_tensors(output, prefix="float_output", artifact_dir=args.artifact)
