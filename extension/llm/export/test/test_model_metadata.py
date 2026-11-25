# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from executorch.exir import to_edge
from executorch.extension.llm.export.model_metadata import EOS_IDS_METHOD, write_eos_ids


class _Identity(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


class WriteEosIdsTest(unittest.TestCase):
    def test_sorted_and_deduplicated(self) -> None:
        self.assertEqual(write_eos_ids([106, 1, 50, 1]), {EOS_IDS_METHOD: [1, 50, 106]})

    def test_empty_is_omitted(self) -> None:
        # No method -> the runner falls back to the tokenizer's EOS.
        self.assertEqual(write_eos_ids([]), {})
        self.assertEqual(write_eos_ids(set()), {})

    def test_negative_rejected(self) -> None:
        with self.assertRaises(ValueError):
            write_eos_ids([1, -1])

    def test_runtime_returns_one_scalar_per_id(self) -> None:
        # extension/llm/runner get_eos_ids() executes the method and reads each
        # output with toScalar(), so every id must be its own int output.
        from executorch.runtime import Runtime

        exported = torch.export.export(_Identity(), (torch.ones(1),), strict=True)
        program = to_edge(
            exported, constant_methods=write_eos_ids([1, 106, 50])
        ).to_executorch()
        method = Runtime.get().load_program(program.buffer).load_method(EOS_IDS_METHOD)
        self.assertEqual(list(method.execute([])), [1, 50, 106])


if __name__ == "__main__":
    unittest.main()
