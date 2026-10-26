# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

import torch
from executorch.examples.models.lfm2.short_conv import ShortConv
from executorch.examples.models.llama.llama_transformer import construct_transformer
from executorch.examples.models.llama.model_args import ModelArgs


def _pos(start: int, length: int) -> torch.Tensor:
    return torch.arange(start, start + length, dtype=torch.long)


class ShortConvStateTest(unittest.TestCase):
    def test_short_conv_resets_state_on_new_sequence(self):
        torch.manual_seed(0)
        conv = ShortConv(dim=8).eval()
        first = torch.randn(1, 5, 8)
        second = torch.randn(1, 4, 8)

        with torch.no_grad():
            fresh = conv(second, _pos(0, 4))
            conv(first, _pos(0, 5))
            self.assertFalse(torch.allclose(conv.conv_state, torch.zeros(1, 8, 2)))
            after_first = conv(second, _pos(0, 4))

        torch.testing.assert_close(after_first, fresh)

    def test_short_conv_keeps_state_within_sequence(self):
        torch.manual_seed(0)
        conv = ShortConv(dim=8).eval()
        x = torch.randn(1, 6, 8)

        with torch.no_grad():
            full = conv(x, _pos(0, 6))
            prefill = conv(x[:, :3], _pos(0, 3))
            decode = [conv(x[:, i : i + 1], _pos(i, 1)) for i in range(3, 6)]

        torch.testing.assert_close(torch.cat([prefill, *decode], dim=1), full)

    def test_short_conv_no_input_pos_does_not_leak_state(self):
        torch.manual_seed(0)
        conv = ShortConv(dim=8).eval()
        x = torch.randn(1, 4, 8)

        with torch.no_grad():
            first = conv(x)
            second = conv(x)

        torch.testing.assert_close(first, second)


class Lfm2HybridModelStateTest(unittest.TestCase):
    def _make_model(self) -> torch.nn.Module:
        args = ModelArgs(
            dim=32,
            hidden_dim=64,
            n_layers=3,
            n_heads=4,
            n_kv_heads=2,
            vocab_size=64,
            max_seq_len=32,
            max_context_len=32,
            max_batch_size=1,
            use_kv_cache=True,
            use_hf_rope=True,
            use_qk_norm=True,
            qk_norm_before_rope=True,
            generate_full_logits=True,
            layer_types=["conv", "full_attention", "conv"],
        )
        return construct_transformer(args).eval()

    def test_new_sequence_matches_fresh_model(self):
        torch.manual_seed(0)
        model = self._make_model()
        reused = copy.deepcopy(model)
        first = torch.randint(0, 64, (1, 7))
        second = torch.randint(0, 64, (1, 5))

        with torch.no_grad():
            fresh = model(second, {"input_pos": _pos(0, 5)})
            reused(first, {"input_pos": _pos(0, 7)})
            after_first = reused(second, {"input_pos": _pos(0, 5)})

        torch.testing.assert_close(after_first, fresh)


if __name__ == "__main__":
    unittest.main()
