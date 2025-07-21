# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch

from transformers import AutoFeatureExtractor, WhisperModel # @manual
from datasets import load_dataset

from ..model_base import EagerModelBase


class WhisperTinyModel(EagerModelBase):
    def __init__(self):
        pass

    def get_eager_model(self) -> torch.nn.Module:
        logging.info("Loading whipser-tiny model")
        # pyre-ignore
        model = WhisperModel.from_pretrained("openai/whisper-tiny", return_dict=False)
        model.eval()
        logging.info("Loaded whisper-tiny model")
        return model

    def get_example_inputs(self):
        feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-tiny")
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
        print(inputs)
        print(inputs.input_features)
        return (inputs.input_features,)
        # Raw audio input: 1 second of 16kHz audio
        #input_values = torch.randn(1, 16000)
        #print(input_values)
        #return (input_values,)
