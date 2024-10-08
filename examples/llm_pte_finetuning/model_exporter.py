# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse

import torch
from executorch.examples.llm_pte_finetuning.model_loading_lib import (
    export_model_lora_training,
    load_checkpoint,
    setup_model,
)

from executorch.examples.llm_pte_finetuning.training_lib import (
    get_dataloader,
    TrainingModule,
)

from omegaconf import OmegaConf
from torch.nn import functional as F
from torchtune import config

from torchtune.training import MODEL_KEY

parser = argparse.ArgumentParser(
    prog="ModelExporter",
    description="Export a LoRA model to ExecuTorch.",
    epilog="Model exported to be used for fine-tuning.",
)

parser.add_argument("--cfg", type=str, help="Path to the config file.")
parser.add_argument("--output_file", type=str, help="Path to the output ET model.")


def main() -> None:
    args = parser.parse_args()
    config_file = args.cfg
    output_file = args.output_file
    cfg = OmegaConf.load(config_file)
    tokenizer = config.instantiate(
        cfg.tokenizer,
    )

    loss_fn = config.instantiate(cfg.loss)

    ds = config.instantiate(cfg.dataset, tokenizer)
    train_set, val_set = torch.utils.data.random_split(ds, [0.8, 0.2])
    train_dataloader = get_dataloader(cfg, train_set, tokenizer, loss_fn)

    max_seq_len = cfg.tokenizer.max_seq_len

    # Example inputs, needed for ET export.
    batch = next(iter(train_dataloader))
    tokens, labels = batch["tokens"], batch["labels"]
    token_size = tokens.shape[1]
    labels_size = labels.shape[1]

    if token_size > max_seq_len:
        tokens = tokens[:, :max_seq_len]
    else:
        tokens = F.pad(tokens, (0, max_seq_len - token_size), value=0)

    if labels_size > max_seq_len:
        labels = labels[:, :max_seq_len]
    else:
        labels = F.pad(labels, (0, max_seq_len - labels_size), value=0)

    # Load pre-trained checkpoint.
    checkpoint_dict = load_checkpoint(cfg=cfg)
    model = setup_model(
        # pyre-ignore
        cfg=cfg,
        base_model_state_dict=checkpoint_dict[MODEL_KEY],
    )

    training_module = TrainingModule(model, loss_fn)

    # Export the model to ExecuTorch for training.
    export_model_lora_training(training_module, (tokens, labels), output_file)


if __name__ == "__main__":
    main()
