# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging

import torch
import torchaudio

from ..model_base import EagerModelBase


FORMAT = "[%(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(format=FORMAT)


__all__ = [
    "EmformerRnntTranscriberModel",
    "EmformerRnntPredictorModel",
    "EmformerRnntJoinerModel",
]


class EmformerRnntTranscriberExample(torch.nn.Module):
    """
    This is a wrapper for validating transcriber for the Emformer RNN-T architecture.
    It does not reflect the actual usage such as beam search, but rather an example for the export workflow.
    """

    def __init__(self) -> None:
        super().__init__()
        bundle = torchaudio.pipelines.EMFORMER_RNNT_BASE_LIBRISPEECH
        decoder = bundle.get_decoder()
        m = decoder.model
        self.rnnt = m

    def forward(self, transcribe_inputs):
        return self.rnnt.transcribe(*transcribe_inputs)


class EmformerRnntTranscriberModel(EagerModelBase):
    def __init__(self):
        pass

    def get_eager_model(self) -> torch.nn.Module:
        logging.info("Loading emformer rnnt transcriber")
        m = EmformerRnntTranscriberExample()
        logging.info("Loaded emformer rnnt transcriber")
        return m

    def get_example_inputs(self):
        transcribe_inputs = (
            torch.randn(1, 128, 80),
            torch.tensor([128]),
        )
        return (transcribe_inputs,)


class EmformerRnntPredictorExample(torch.nn.Module):
    """
    This is a wrapper for validating predictor for the Emformer RNN-T architecture.
    It does not reflect the actual usage such as beam search, but rather an example for the export workflow.
    """

    def __init__(self) -> None:
        super().__init__()
        bundle = torchaudio.pipelines.EMFORMER_RNNT_BASE_LIBRISPEECH
        decoder = bundle.get_decoder()
        m = decoder.model
        self.rnnt = m

    def forward(self, predict_inputs):
        return self.rnnt.predict(*predict_inputs)


class EmformerRnntPredictorModel(EagerModelBase):
    def __init__(self):
        pass

    def get_eager_model(self) -> torch.nn.Module:
        logging.info("Loading emformer rnnt predictor")
        m = EmformerRnntPredictorExample()
        logging.info("Loaded emformer rnnt predictor")
        return m

    def get_example_inputs(self):
        predict_inputs = (
            torch.zeros([1, 128], dtype=int),
            torch.tensor([128], dtype=int),
            None,
        )
        return (predict_inputs,)


class EmformerRnntJoinerExample(torch.nn.Module):
    """
    This is a wrapper for validating joiner for the Emformer RNN-T architecture.
    It does not reflect the actual usage such as beam search, but rather an example for the export workflow.
    """

    def __init__(self) -> None:
        super().__init__()
        bundle = torchaudio.pipelines.EMFORMER_RNNT_BASE_LIBRISPEECH
        decoder = bundle.get_decoder()
        m = decoder.model
        self.rnnt = m

    def forward(self, predict_inputs):
        return self.rnnt.join(*predict_inputs)


class EmformerRnntJoinerModel(EagerModelBase):
    def __init__(self):
        pass

    def get_eager_model(self) -> torch.nn.Module:
        logging.info("Loading emformer rnnt joiner")
        m = EmformerRnntJoinerExample()
        logging.info("Loaded emformer rnnt joiner")
        return m

    def get_example_inputs(self):
        join_inputs = (
            torch.rand([1, 128, 1024]),
            torch.tensor([128]),
            torch.rand([1, 128, 1024]),
            torch.tensor([128]),
        )
        return (join_inputs,)
