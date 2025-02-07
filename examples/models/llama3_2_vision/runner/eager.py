# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import Optional

import torch

from executorch.examples.models.llama.export_llama_lib import _prepare_for_llama_export
from executorch.examples.models.llama.runner.eager import execute_runner
from executorch.examples.models.llama3_2_vision.runner.generation import (
    TorchTuneLlamaRunner,
)
from executorch.extension.llm.export import LLMEdgeManager


class EagerLlamaRunner(TorchTuneLlamaRunner):
    """
    Runs llama in eager mode with provided checkpoint file.
    """

    def __init__(self, args):
        with open(args.params, "r") as f:
            params = json.loads(f.read())
        super().__init__(
            tokenizer_path=args.tokenizer_path,
            max_seq_len=args.max_seq_length,
            max_batch_size=1,
            use_kv_cache=args.use_kv_cache,
            vocab_size=params["vocab_size"],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        manager: LLMEdgeManager = _prepare_for_llama_export(args)
        self.model = manager.model.eval().to(device=self.device)

    def forward(
        self,
        tokens: Optional[torch.LongTensor] = None,
        input_pos: Optional[torch.LongTensor] = None,
        mask: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        return self.model.forward(tokens=tokens, input_pos=input_pos, mask=mask)


def main() -> None:
    execute_runner(EagerLlamaRunner)


if __name__ == "__main__":
    main()  # pragma: no cover
