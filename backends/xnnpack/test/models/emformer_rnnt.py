# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

import torchaudio
from executorch.backends.xnnpack.test.tester import Tester


class TestEmformerModel(unittest.TestCase):
    class EmformerRnnt(torch.nn.Module):
        def __init__(self):
            super().__init__()
            bundle = torchaudio.pipelines.EMFORMER_RNNT_BASE_LIBRISPEECH
            decoder = bundle.get_decoder()
            self.rnnt = decoder.model

    class Joiner(EmformerRnnt):
        def forward(self, a, b, c, d):
            return self.rnnt.join(a, b, c, d)

        def get_example_inputs(self):
            join_inputs = (
                torch.rand([1, 128, 1024]),
                torch.tensor([128]),
                torch.rand([1, 128, 1024]),
                torch.tensor([128]),
            )
            return join_inputs

    def test_fp32_emformer_joiner(self):
        joiner = self.Joiner()
        (
            Tester(joiner, joiner.get_example_inputs())
            .export()
            .to_edge_transform_and_lower()
            .check(["torch.ops.higher_order.executorch_call_delegate"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    class Predictor(EmformerRnnt):
        def forward(self, a, b):
            return self.rnnt.predict(a, b, None)

        def get_example_inputs(self):
            predict_inputs = (
                torch.zeros([1, 128], dtype=int),
                torch.tensor([128], dtype=int),
            )
            return predict_inputs

    @unittest.skip(
        "T183426271: Emformer Predictor Takes too long to export + partition"
    )
    def _test_fp32_emformer_predictor(self):
        predictor = self.Predictor()
        (
            Tester(predictor, predictor.get_example_inputs())
            .export()
            .to_edge_transform_and_lower()
            .check(["torch.ops.higher_order.executorch_call_delegate"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    class Transcriber(EmformerRnnt):
        def forward(self, a, b):
            return self.rnnt.transcribe(a, b)

        def get_example_inputs(self):
            transcribe_inputs = (
                torch.randn(1, 128, 80),
                torch.tensor([128]),
            )
            return transcribe_inputs

    def test_fp32_emformer_transcriber(self):
        transcriber = self.Transcriber()
        (
            Tester(transcriber, transcriber.get_example_inputs())
            .export()
            .to_edge_transform_and_lower()
            .check(["torch.ops.higher_order.executorch_call_delegate"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )
