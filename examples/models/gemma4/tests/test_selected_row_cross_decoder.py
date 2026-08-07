# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from executorch.examples.models.gemma4.text_decoder.gemma4_transformer import (
    Gemma4TextModel,
)


class _SelfDecoder(torch.nn.Module):
    def forward(self, input_ids, input_pos=None, inputs_embeds=None):
        del input_pos, inputs_embeds
        seq_len = input_ids.shape[1]
        hidden = torch.arange(seq_len * 3, dtype=torch.float32).reshape(1, seq_len, 3)
        per_layer = hidden.unsqueeze(0).repeat(2, 1, 1, 1)
        return hidden, per_layer, {}


class _CrossDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_hidden_shape = None
        self.last_per_layer_shape = None
        self.last_query_start_pos = None

    def forward(
        self,
        hidden_states,
        per_layer_inputs,
        shared_kv,
        input_pos=None,
        query_start_pos=None,
    ):
        del shared_kv, input_pos
        self.last_hidden_shape = tuple(hidden_states.shape)
        self.last_per_layer_shape = tuple(per_layer_inputs.shape)
        self.last_query_start_pos = query_start_pos
        return hidden_states


def _model() -> Gemma4TextModel:
    model = Gemma4TextModel.__new__(Gemma4TextModel)
    torch.nn.Module.__init__(model)
    model.self_decoder = _SelfDecoder()
    model.cross_decoder = _CrossDecoder()
    model.norm = torch.nn.Identity()
    model.lm_head = torch.nn.Identity()
    model.final_logit_softcapping = 0.0
    return model


class SelectedRowCrossDecoderTest(unittest.TestCase):
    def test_generation_narrows_before_cross_decoder(self) -> None:
        model = _model()
        input_ids = torch.ones((1, 4), dtype=torch.long)
        logits = model(input_ids, input_pos=torch.arange(4))

        self.assertEqual(tuple(logits.shape), (1, 1, 3))
        self.assertEqual(model.cross_decoder.last_hidden_shape, (1, 1, 3))
        self.assertEqual(model.cross_decoder.last_per_layer_shape, (2, 1, 1, 3))
        self.assertEqual(model.cross_decoder.last_query_start_pos, 3)
        torch.testing.assert_close(logits, torch.tensor([[[9.0, 10.0, 11.0]]]))

    def test_non_generation_keeps_full_cross_decoder_input(self) -> None:
        model = _model()
        input_ids = torch.ones((1, 4), dtype=torch.long)
        logits = model(input_ids)

        self.assertEqual(tuple(logits.shape), (1, 1, 3))
        self.assertEqual(model.cross_decoder.last_hidden_shape, (1, 4, 3))
        self.assertEqual(model.cross_decoder.last_per_layer_shape, (2, 1, 4, 3))
        self.assertIsNone(model.cross_decoder.last_query_start_pos)
