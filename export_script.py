# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch
from executorch.exir import to_edge
from torch import int64, long, no_grad, randint, Tensor, zeros
from torch.export import export, ExportedProgram
from torch.export.experimental import _export_forward_backward
from torch.nn.attention import sdpa_kernel, SDPBackend

from torchtune.models.llama3_2._model_builders import lora_llama3_2_3b
from torchtune.models.phi3._model_builders import lora_phi3_mini
from torchtune.modules.peft import get_adapter_params, set_trainable_params


def export_llama3_lora(model, checkpoint_path, adapter_path) -> None:
    model.eval()
    # 1. torch.export: Defines the program with the ATen operator set.
    print("Exporting to aten dialect")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True)
    adapter = torch.load(adapter_path, map_location="cpu", mmap=True)

    missing, unexpected = model.load_state_dict(
        checkpoint,
        strict=False,
        assign=True,
    )

    # Should be missing all the LoRA adapter weights.
    # print("Missing: ", missing)
    print("Unexpected: ", unexpected)

    missing, unexpected = model.load_state_dict(
        adapter,
        strict=False,
        assign=True,
    )
    # Should not be missing aything now.
    print("Missing a: ", missing)
    print("Unexpected a: ", unexpected)

    example_args = (
        torch.tensor([[1, 2, 3]], dtype=torch.long),
    )  # tokens, with kv cache our input token length is always just 1 token.

    with sdpa_kernel([SDPBackend.MATH]):
        aten_dialect: ExportedProgram = export(model, example_args, strict=True)

        # 2. to_edge: Make optimizations for Edge devices.
        print("Lowering to edge dialect")
        edge_program = to_edge(aten_dialect)

    # 3. to_executorch: Convert the graph to an ExecuTorch program.
    print("Exporting to executorch")
    executorch_program = edge_program.to_executorch()

    # 4. Save the compiled .pte program.
    print("Saving to llama3_2_lora.pte")
    with open("llama3_2_lora.pte", "wb") as file:
        file.write(executorch_program.buffer)

    print("Done.")


def main() -> None:
    print("Main")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--checkpoint",
        help="checkpoint path",
    )

    parser.add_argument(
        "-a",
        "--adapter",
        help="adapter path",
    )

    args = parser.parse_args()

    llama3_2_model = lora_llama3_2_3b(
        lora_attn_modules=[
            "q_proj",
            "v_proj",
            "o_proj",
            # "gate_proj",
            # "down_proj",
            # "up_proj",
        ],
        apply_lora_to_mlp=True,
        lora_rank=64,
    )

    llama3_2_model.to(dtype=torch.bfloat16)

    # Export for inference.
    export_llama3_lora(llama3_2_model, args.checkpoint, args.adapter)


if __name__ == "__main__":
    main()
