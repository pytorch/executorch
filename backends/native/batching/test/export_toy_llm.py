# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export a tiny random Llama with an off-graph KV cache and selected logits,
for test_native_executor.cpp:

    python -m executorch.backends.native.batching.test.export_toy_llm toy.ptn
    ET_NATIVE_BATCHING_PTN=toy.ptn native_batching_test
"""

import sys
import tempfile
from unittest import mock

from executorch.backends.native.examples.hf_llm import export_hf_llm
from transformers import LlamaConfig


def main() -> None:
    output = sys.argv[1]
    with tempfile.TemporaryDirectory() as model:
        LlamaConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=256,
        ).save_pretrained(model)
        argv = [
            "export_hf_llm",
            "--model",
            model,
            "--random_weights",
            "--offgraph_cache",
            "--logits_to_keep",
            "selected",
            "--max_seq_length",
            "16",
            "--output",
            output,
        ]
        with mock.patch.object(sys, "argv", argv):
            export_hf_llm.main()


if __name__ == "__main__":
    main()
