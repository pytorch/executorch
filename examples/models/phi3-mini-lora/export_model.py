# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.exir import to_edge
from torch import int64, long, no_grad, randint, Tensor, zeros
from torch.export import export, ExportedProgram
from torch.nn.attention import sdpa_kernel, SDPBackend
from torchtune.models.phi3._model_builders import lora_phi3_mini


@no_grad()
def export_mini_phi3_lora(model) -> None:
    """
    Export the example mini-phi3 with LoRA model to executorch.

    Note: need to use the SDPBackend's custom kernel for sdpa (scalable
    dot product attention) because the default sdpa kernel used in the
    model results in a internally mutating graph.
    """
    model.eval()
    # 1. torch.export: Defines the program with the ATen operator set.
    print("Exporting to aten dialect")
    example_args = (randint(0, 100, (1, 100), dtype=long),)
    with sdpa_kernel([SDPBackend.MATH]):
        aten_dialect: ExportedProgram = export(model, example_args)

        # 2. to_edge: Make optimizations for Edge devices.
        print("Lowering to edge dialect")
        edge_program = to_edge(aten_dialect)

    # 3. to_executorch: Convert the graph to an ExecuTorch program.
    print("Exporting to executorch")
    executorch_program = edge_program.to_executorch()

    # 4. Save the compiled .pte program.
    print("Saving to mini_phi3_lora.pte")
    with open("mini_phi3_lora.pte", "wb") as file:
        file.write(executorch_program.buffer)

    print("Done.")


def run_mini_phi3_lora(model) -> Tensor:
    """Run the model and return the result."""
    args = zeros([3072, 1], dtype=int64)
    model.eval()
    res = model(args)
    return res


def main() -> None:
    mini_lora_model = lora_phi3_mini(
        lora_attn_modules=[
            "q_proj",
        ]
    )
    export_mini_phi3_lora(mini_lora_model)


if __name__ == "__main__":
    main()
