# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir import to_edge
from torch import int64, long, no_grad, randint, Tensor, zeros
from torch.export import export, ExportedProgram
from torch.export.experimental import _export_forward_backward
from torch.nn.attention import sdpa_kernel, SDPBackend
from torchtune.models.phi3._model_builders import lora_phi3_mini
from torchtune.modules.peft import get_adapter_params, set_trainable_params

vocab_size = 32064


class TrainingModule(torch.nn.Module):
    """
    The model being trained should return the loss from forward(). This
    class wraps the actual phi3-mini model and calculates an arbitrary
    loss with its forward() output.
    """

    def __init__(self, model, loss):
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


@no_grad()
def export_phi3_mini_lora(model) -> None:
    """
    Export the example phi3-mini with LoRA model to executorch.

    Note: need to use the SDPBackend's custom kernel for sdpa (scalable
    dot product attention) because the default sdpa kernel used in the
    model results in a internally mutating graph.
    """
    model.eval()
    # 1. torch.export: Defines the program with the ATen operator set.
    print("Exporting to aten dialect")
    batch_size = 1
    vocab_size = 100
    seq_len = 10
    tokens = randint(0, vocab_size, (batch_size, seq_len), dtype=long)
    example_args = (tokens,)
    with sdpa_kernel([SDPBackend.MATH]):
        aten_dialect: ExportedProgram = export(model, example_args)

        # 2. to_edge: Make optimizations for Edge devices.
        print("Lowering to edge dialect")
        edge_program = to_edge(aten_dialect)

    # 3. to_executorch: Convert the graph to an ExecuTorch program.
    print("Exporting to executorch")
    executorch_program = edge_program.to_executorch()

    # 4. Save the compiled .pte program.
    print("Saving to phi3_mini_lora.pte")
    with open("phi3_mini_lora.pte", "wb") as file:
        file.write(executorch_program.buffer)

    print("Done.")


def export_phi3_mini_lora_training(model) -> None:
    """
    Export the example phi3-mini with LoRA model to executorch for training, only.
    """

    # 0. Mark the LoRA layers as trainable (requires_grad = True) in order
    # to just export the backwards pass for these layers later in the
    # export process.
    set_trainable_params(model, get_adapter_params(model))

    print("Exporting phi3-mini with LoRA for training")
    # 1. torch.export: Defines the program with the ATen operator set.
    print("Exporting to aten dialect")
    batch_size = 1
    vocab_size = 100
    seq_len = 10
    tokens = randint(0, vocab_size, (batch_size, seq_len), dtype=long)
    labels = tokens
    example_args = (tokens, labels)
    with sdpa_kernel([SDPBackend.MATH]):
        exported_graph: ExportedProgram = export(model, example_args)
        print("Creating a joint forward-backwards graph for training")
        joint_graph = _export_forward_backward(exported_graph)

        # 2. to_edge: Make optimizations for Edge devices.
        print("Lowering to edge dialect")
        edge_program = to_edge(joint_graph)

        print(edge_program._edge_programs["forward"].graph_module)

    # 3. to_executorch: Convert the graph to an ExecuTorch program.
    print("Exporting to executorch")
    executorch_program = edge_program.to_executorch()

    # 4. Save the compiled .pte program.
    print("Saving to phi3_mini_lora_training.pte")
    with open("phi3_mini_lora_training.pte", "wb") as file:
        file.write(executorch_program.buffer)

    print("Done.")


def run_phi3_mini_lora(model) -> Tensor:
    """Run the model and return the result."""
    # Input shape: (batch_size, seq_len).
    args = zeros((1, 10), dtype=int64)
    model.eval()
    res = model(args)
    return res


def main() -> None:
    print("Main")
    lora_model = lora_phi3_mini(
        lora_attn_modules=[
            "q_proj",
        ]
    )

    # Export for inference.
    export_phi3_mini_lora(lora_model)

    # Export for training.
    lora_training_model = TrainingModule(lora_model, torch.nn.CrossEntropyLoss())
    export_phi3_mini_lora_training(lora_training_model)


if __name__ == "__main__":
    main()
