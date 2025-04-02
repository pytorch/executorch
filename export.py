# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch
from executorch.exir import to_edge
from executorch.extension.llm.modules.attention import replace_mha_with_inference_mha
from torch import int64, long, no_grad, randint, Tensor, zeros
from torch.export import export, ExportedProgram
from torch.nn.attention import sdpa_kernel, SDPBackend
from torchtune.models import convert_weights

from torchtune.models.llama3_2._model_builders import lora_llama3_2_3b

# from torchtune.modules.peft import get_adapter_params, set_trainable_params


def export_llama3_lora(model, checkpoint_path, adapter_path) -> None:
    example_args = (
        torch.tensor(
            [[1]], dtype=torch.long
        ),  # tokens, with kv cache our input token length is always just 1 token.
    )
    example_kwargs = {
        "input_pos": torch.tensor(
            [0], dtype=torch.long
        )  # start_pos, what token of output are we on.
    }
    breakpoint()
    model.requires_grad_(False)
    model = replace_mha_with_inference_mha(model)

    # print("Loading checkpoint")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=False
    )
    state_dict = convert_weights.meta_to_tune(checkpoint)
    breakpoint()
    state_dict.pop("output.weight")
    adapter = torch.load(
        adapter_path, map_location="cpu", mmap=True, weights_only=False
    )

    state_dict.update(adapter)

    breakpoint()
    # missing, unexpected = model.load_state_dict(
    #     state_dict,
    #     # strict=False,
    #     # assign=True,
    # )

    # print("Missing: ", missing)
    # print("Unexpected: ", unexpected)

    eager_result = model.forward(
        example_args[0],
    )

    breakpoint()
    with sdpa_kernel([SDPBackend.MATH]):
        print("Exporting to aten dialect")
        aten_dialect: ExportedProgram = export(
            model, args=example_args, kwargs=example_kwargs, strict=True
        )

        exported_result = aten_dialect.module()(
            example_args[0], input_pos=example_kwargs["input_pos"]
        )

        print("Checking eager and exported results are close")
        print(torch.allclose(eager_result, exported_result))

        # print("EXPORTED_MODEL")
        # with open("/data/users/lfq/executorch/exported_model.txt", "w") as f:
        #     f.write(aten_dialect.graph_module.print_readable())

        # print("EXPORTED_MODEL No decomp")
        # with open("/data/users/lfq/executorch/exported_decomp.txt", "w") as f:
        #     run_decomp = aten_dialect.run_decompositions()
        #     f.write(run_decomp.graph_module.print_readable())

        # 2. to_edge: Make optimizations for Edge devices.
        print("Lowering to edge dialect")
        edge_program = to_edge(aten_dialect)

        edge_result = edge_program._edge_programs["forward"].module()(
            example_args[0], input_pos=example_kwargs["input_pos"]
        )

        print("Checking eager and edge results are close")
        print(torch.allclose(eager_result, edge_result))
        breakpoint()

    # 3. to_executorch: Convert the graph to an ExecuTorch program.
    print("Exporting to executorch")
    executorch_program = edge_program.to_executorch()

    # 4. Save the compiled .pte program.
    print("Saving to llama3_2_lora.pte")
    with open("llama3_2_lora.pte", "wb") as file:
        file.write(executorch_program.buffer)

    print("Done.")


def export_llama3(model, checkpoint_path) -> None:
    pass


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

    lora_model = lora_llama3_2_3b(
        lora_attn_modules=[
            "q_proj",
            # "v_proj",
            # "o_proj",
            # "gate_proj",
            # "down_proj",
            # "up_proj",
        ],
        # apply_lora_to_mlp=False,
        # lora_rank=8,
    )
    lora_model.eval()

    # Export for inference.
    export_llama3_lora(lora_model, args.checkpoint, args.adapter)

    # Is this something to do with the torchtune checkpoint?
    # export_llama3(llama3_model, args.checkpoint)


if __name__ == "__main__":
    main()
