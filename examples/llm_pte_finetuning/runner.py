# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse

import torch
from executorch.examples.llm_pte_finetuning.training_lib import (
    eval_model,
    get_dataloader,
    update_function,
)

from executorch.extension.pybindings.aten_lib import (  # @manual
    _load_for_executorch_from_buffer,
)
from omegaconf import OmegaConf
from torch.nn import functional as F
from torchtune import config
from tqdm import tqdm

parser = argparse.ArgumentParser(
    prog="Runner",
    description="Fine tunes LoRA model using ExecuTorch.",
    epilog="Model exported to be used for fine-tuning.",
)
parser.add_argument("--cfg", type=str, help="Path to the config file.")
parser.add_argument("--model_file", type=str, help="Path to the ET model file.")


def main() -> None:
    args = parser.parse_args()
    config_file = args.cfg
    file = args.model_file
    cfg = OmegaConf.load(config_file)
    tokenizer = config.instantiate(
        cfg.tokenizer,
    )

    loss_fn = config.instantiate(cfg.loss)

    ds = config.instantiate(cfg.dataset, tokenizer)
    train_set, val_set = torch.utils.data.random_split(ds, [0.8, 0.2])
    train_dataloader = get_dataloader(cfg, train_set, tokenizer, loss_fn)
    val_dataloader = get_dataloader(cfg, val_set, tokenizer, loss_fn)

    max_seq_len = cfg.tokenizer.max_seq_len
    # Num of steps to run training. Assume 1 epoch
    num_steps = 100
    with open(file, "rb") as f:
        model_bytes = f.read()
        et_mod = _load_for_executorch_from_buffer(model_bytes)

        # Evaluate the model before training.
        print("Evaluating the model before training")
        eval_loss = eval_model(
            model=et_mod,
            dataloader=val_dataloader,
            loss_fn=loss_fn,
            max_seq_len=max_seq_len,
            num_eval_steps=10,
        )
        print("Eval loss: ", eval_loss)

        # Based on executorch/extension/training/module/training_module.cpp
        # grads run from [grad_start, param_start]
        # params run from [param_start, outputs_end]
        grad_start = et_mod.run_method("__et_training_gradients_index_forward", [])[0]
        param_start = et_mod.run_method("__et_training_parameters_index_forward", [])[0]
        learning_rate = 5e-3
        f.seek(0)
        losses = []
        for i, batch in tqdm(enumerate(train_dataloader), total=num_steps):
            # Run for a limited number of steps.
            if i >= num_steps:
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

            # Do not clone outputs, since we want the original weights to be returned
            # for us to update with the gradients in-place.
            # See https://github.com/pytorch/executorch/blob/main/extension/pybindings/pybindings.cpp#L736
            # for more info.
            out = et_mod.forward((tokens, labels), clone_outputs=False)  # pyre-ignore

            loss = out[0]
            losses.append(loss.item())
            with torch.no_grad():
                for grad, param in zip(out[grad_start:param_start], out[param_start:]):
                    update_function(param, grad, learning_rate)

        print("Losses: ", losses)
        # Evaluate the model after training.
        eval_loss = eval_model(
            model=et_mod,
            dataloader=val_dataloader,
            loss_fn=loss_fn,
            max_seq_len=max_seq_len,
            num_eval_steps=10,
        )
    print("Eval loss: ", eval_loss)


if __name__ == "__main__":
    main()
