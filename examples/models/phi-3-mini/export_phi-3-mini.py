# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.extension.llm.export.builder import DType, LLMEdgeManager

from executorch.extension.llm.export.partitioner_lib import get_xnnpack_partitioner
from executorch.extension.llm.export.quantizer_lib import (
    DynamicQuantLinearOptions,
    get_pt2e_quantizers,
    PT2EQuantOptions,
)

from transformers import Phi3ForCausalLM


def main() -> None:
    torch.manual_seed(42)

    # pyre-ignore: Undefined attribute [16]: Module `transformers` has no attribute `Phi3ForCausalLM`
    model = Phi3ForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

    modelname = "phi-3-mini"

    (
        LLMEdgeManager(
            model=model,
            modelname=modelname,
            max_seq_len=128,
            dtype=DType.fp32,
            use_kv_cache=False,
            example_inputs=(torch.randint(0, 100, (1, 100), dtype=torch.long),),
            enable_dynamic_shape=True,
            verbose=True,
        )
        .set_output_dir(".")
        .capture_pre_autograd_graph()
        .pt2e_quantize(
            get_pt2e_quantizers(PT2EQuantOptions(None, DynamicQuantLinearOptions()))
        )
        .export_to_edge()
        .to_backend([get_xnnpack_partitioner()])
        .to_executorch()
        .save_to_pte(f"{modelname}.pte")
    )


if __name__ == "__main__":
    main()
