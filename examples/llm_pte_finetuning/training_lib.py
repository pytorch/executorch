# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from functools import partial
from typing import Any, Dict, Mapping, Optional

import torch
from executorch.extension.pybindings.aten_lib import ExecuTorchModule  # @manual

from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchtune.data import InstructTemplate
from torchtune.data._collate import padded_collate_sft
from tqdm import tqdm


class TrainingModule(torch.nn.Module):
    """
    The model being trained should return the loss from forward(). This
    class wraps the actual model and computes the loss for an LLM
    fine-tuning task. The loss is computed as the cross entropy between
    the tokens and a shifted version of the labels so we learn to predict
    the next token.
    """

    def __init__(
        self, model: torch.nn.Module, loss: torch.nn.modules.loss._Loss
    ) -> None:
        super().__init__()
        self.model = model
        self.loss = loss

    def forward(self, input: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Output is of the shape (seq_len, vocab_size).
        logits = self.model(input)
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
        logits = logits.transpose(1, 2)
        return self.loss(logits, labels)


class DatabricksDolly(InstructTemplate):
    """
    Used for the Dolly dataset from Databricks.

    https://huggingface.co/datasets/databricks/databricks-dolly-15k
    """

    template = "Instruction:\n{instruction}\n\nContext:\n{input}\n\nResponse: "

    @classmethod
    def format(
        cls,
        sample: Mapping[str, Any],
        column_map: Optional[Dict[str, str]],
    ) -> str:
        assert column_map is not None
        instruction = sample[column_map["instruction"]]
        input = sample[column_map["input"]]
        return cls.template.format(instruction=instruction, input=input)


class PythonCodeInstructions(InstructTemplate):
    """
    https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca
    """

    template = (
        "{prompt}\n\n"
        "Instruction:\n{instruction}"
        "\n\nContext:\n{input}\n\nResponse: "
    )

    @classmethod
    def format(
        cls,
        sample: Mapping[str, Any],
        column_map: Optional[Dict[str, str]],
    ) -> str:
        assert column_map is not None
        instruction = sample[column_map["instruction"]]
        input = sample[column_map["input"]]
        prompt = sample[column_map["prompt"]]
        return cls.template.format(instruction=instruction, input=input, prompt=prompt)


def update_function(
    param: torch.Tensor,
    grad: torch.Tensor,
    learning_rate: float,
    weight_decay: float = 1.0,
) -> None:
    """SGD update function."""
    grad = grad + weight_decay * param
    param.sub_(learning_rate * grad)


def eval_model(
    model: ExecuTorchModule,
    dataloader: DataLoader,
    loss_fn: torch.nn.modules.loss._Loss,
    max_seq_len: int,
    num_eval_steps: int,
) -> float:
    total_loss = 0
    for i, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
        if i >= num_eval_steps:
            break
        tokens, labels = batch["tokens"], batch["labels"]
        token_size = tokens.shape[1]
        labels_size = labels.shape[1]

        tokens, labels = batch["tokens"], batch["labels"]
        token_size = tokens.shape[1]
        labels_size = labels.shape[1]

        # Fixed length for now. We need to resize as the input shapes
        # should be the same passed as examples to the export function.
        if token_size > max_seq_len:
            tokens = tokens[:, :max_seq_len]
        else:
            tokens = F.pad(tokens, (0, max_seq_len - token_size), value=0)

        if labels_size > max_seq_len:
            labels = labels[:, :max_seq_len]
        else:
            labels = F.pad(labels, (0, max_seq_len - labels_size), value=0)

        out = model.forward((tokens, labels))
        loss = out[0]
        total_loss += loss
    return total_loss / num_eval_steps


def get_dataloader(
    cfg: Any,  # pyre-ignore[2]
    ds: Dataset[Any],  # pyre-ignore[2]
    tokenizer: Any,  # pyre-ignore[2]
    loss_fn: torch.nn.modules.loss._Loss,
) -> DataLoader:
    """Given a dataset, tokenizer, and loss function, return a dataloader."""
    packed = cfg.dataset.get("packed", False)

    sampler = DistributedSampler(
        ds,
        num_replicas=1,
        rank=0,
        shuffle=cfg.shuffle,
        seed=0,
    )
    dataloader = DataLoader(
        dataset=ds,
        sampler=sampler,
        batch_size=cfg.batch_size,
        collate_fn=(
            partial(
                padded_collate_sft,
                padding_idx=tokenizer.pad_id,
                ignore_idx=loss_fn.ignore_index,
            )
            if not packed
            else None
        ),
    )
    return dataloader
