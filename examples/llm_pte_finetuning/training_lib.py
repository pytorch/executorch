# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from functools import partial
from typing import Any

import torch
from executorch.extension.pybindings.portable_lib import ExecuTorchModule  # @manual

from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchtune.data import AlpacaToMessages
from torchtune.data._collate import padded_collate_sft
from torchtune.datasets import PackedDataset, SFTDataset
from torchtune.modules.tokenizers import ModelTokenizer
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
        if loss.__class__.__name__ == "CEWithChunkedOutputLoss":
            # set num_output_chunks for model
            # pyre-ignore
            model.set_num_output_chunks(self.loss.num_output_chunks)

        # (batch_size, 1) tensor of ignore_index
        # pyre-ignore
        self.ignore_labels_cache = torch.full(
            (1, 1), self.loss.ignore_index, device="cpu"  # pyre-ignore
        )

    def forward(self, input: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Output is of the shape (seq_len, vocab_size).
        logits = self.model(input)
        labels = torch.hstack(
            (labels[..., 1:], self.ignore_labels_cache[: labels.shape[0]])
        )
        if not isinstance(logits, list):
            labels = labels.reshape(-1)
            logits = logits.reshape(-1, logits.size(-1))
        return self.loss(logits, labels)


def python_code_instructions_alpaca(tokenizer: ModelTokenizer) -> PackedDataset:
    """
    Python code instruction-input-output pairs from iamtarun/python_code_instructions_18k_alpaca templated with Alpaca.
    """
    ds = SFTDataset(
        # pyre-ignore[6]: Incompatible parameter type
        model_transform=tokenizer,
        source="iamtarun/python_code_instructions_18k_alpaca",
        message_transform=AlpacaToMessages(
            train_on_input=False,
        ),
        # pyre-ignore[6]: Incompatible parameter type
        split="train",
    )
    if tokenizer.max_seq_len is None:
        raise ValueError(
            "PackedDataset requires a max_seq_len to be set on the tokenizer."
        )
    return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len, split_across_pack=False)


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
