# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.extension.llm.export.builder import LLMEdgeManager

from executorch.extension.llm.export.partitioner_lib import get_xnnpack_partitioner
from executorch.extension.llm.export.quantizer_lib import (
    DynamicQuantLinearOptions,
    get_pt2e_quantizers,
    PT2EQuantOptions,
)

from transformers import Phi3ForCausalLM, AutoTokenizer


def main() -> None:
    torch.manual_seed(42)

    pre_trained_model_name = "microsoft/Phi-3-mini-4k-instruct"
    # pyre-ignore: Undefined attribute [16]: Module `transformers` has no attribute `Phi3ForCausalLM`
    model = Phi3ForCausalLM.from_pretrained(pre_trained_model_name)
    tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_name)

    tokens = tokenizer.encode("Tell me a story", return_tensors="pt")
    result = model.forward(input_ids=tokens, use_cache=True, return_dict=True)

    model_name = "phi-3-mini"

    (
        LLMEdgeManager(
            model=model,
            modelname=model_name,
            max_seq_len=128,
            dtype=model.dtype,
            use_kv_cache=False,
            example_inputs=tokens,
            dynamic_shapes={
                "input_ids": {1: torch.export.Dim("sequence_length", max=128)}
            },
            enable_dynamic_shape=True,
            verbose=True,
            kwargs={
                "use_cache": True,
                "return_dict": True,
                "past_key_values": result.past_key_values
            }
        )
        .set_output_dir(".")
        .capture_pre_autograd_graph()
        .pt2e_quantize(
            get_pt2e_quantizers(PT2EQuantOptions(None, DynamicQuantLinearOptions()))
        )
        .export_to_edge()
        .to_backend([get_xnnpack_partitioner()])
        .to_executorch()
        .save_to_pte(f"{model_name}.pte")
    )


if __name__ == "__main__":
    main()
