# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.extension.gguf_util.convert_gguf_to_checkpoint import (
    convert_gguf_to_checkpoint,
)
from executorch.extension.gguf_util.load_gguf import GGUFModelArgs, GGUFWeights
from executorch.extension.gguf_util.quantize import quantize


def convert_to_pte(
    gguf_model_args: GGUFModelArgs,
    gguf_weights: GGUFWeights,
) -> None:
    """Convert a GGUF model into a PTE file, an ExecuTorch program.

    Args:
        gguf_model_args: The arguments for the GGUF model.
        gguf_weights: The weights of the GGUF model.
    """

    # Step 1: Create the corresponding checkpoint state dict
    state_dict = convert_gguf_to_checkpoint(gguf_model_args, gguf_weights)

    # Switch statement based on the architecture enum.
    # Each enum has its own converter function.
    if gguf_model_args.arch == "llama":

        # Step 2: Create the PyTorch nn.Module.
        #
        # NOTE: Currently it is using the Llama2 model in the executorch/examples/ directory
        # But in the future, should point to a generic/reference implementation instead.
        #
        # Currently, we are doing this so that it is exportable.
        from executorch.examples.models.llama2.llama_transformer import (
            ModelArgs as LlamaModelArgs,
            Transformer as LlamaTransformer,
        )

        llama_model_args = LlamaModelArgs(
            dim=gguf_model_args.embedding_length,
            n_layers=gguf_model_args.block_count,
            n_heads=gguf_model_args.attention.head_count,
            n_kv_heads=gguf_model_args.attention.head_count_kv,
            vocab_size=gguf_model_args.vocab_size,
            norm_eps=gguf_model_args.attention.layer_norm_rms_epsilon,
            hidden_dim=gguf_model_args.feed_forward_length,
            rope_freq_base=gguf_model_args.rope.freq_base,
        )
        pt_model = LlamaTransformer(llama_model_args)
        pt_model.eval()

        # Step 3: Transform the PyTorch nn.Module to another PyTorch nn.Module that
        # is compatible with the quantized weights.
        pt_model = quantize(pt_model, gguf_model_args)

        # Step 4: Convert the PyTorch nn.Module to a PTE program.
        from executorch.extension.gguf_util.pte_converters.llama_pte_converter import (
            convert_to_pte as llama_convert_to_pte,
        )
        return llama_convert_to_pte(pt_model, state_dict, gguf_model_args)

    else:
        raise NotImplementedError("Unsupported architecture.")
