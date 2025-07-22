# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch

from transformers import AutoFeatureExtractor, WhisperModel # @manual
from transformers import AutoProcessor, WhisperForConditionalGeneration # @manual
from datasets import load_dataset

from ..model_base import EagerModelBase


class WhisperTinyModel(EagerModelBase):
    def __init__(self):
        #self.max_cache_length=1024
        #self.batch_size=1
        pass

    def get_eager_model(self) -> torch.nn.Module:
        logging.info("Loading whipser-tiny model")
        # pyre-ignore
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en", return_dict=False)
        model.eval()
        logging.info("Loaded whisper-tiny model")
        return model

    def get_example_inputs(self):
        #input_ids = torch.tensor([[0]], dtype=torch.long)
        #encoder_hidden_states = torch.rand(1, 1500, 384)
        #cache_position = torch.tensor([0], dtype=torch.long)
        #atten_mask = torch.full((1, self.max_cache_length), torch.tensor(-255.0))
        #atten_mask *= torch.arange(self.max_cache_length) > cache_position.reshape(
        #    -1, 1
        #)
        #atten_mask = atten_mask[None, None, :, :].expand(self.batch_size, 1, -1, -1)
        #return (input_ids, atten_mask, encoder_hidden_states, cache_position)

        processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en", return_dict=False)
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
        input_features = inputs.input_features
        #expected_shape = (1, processor.feature_extractor.feature_size, processor.feature_extractor.nb_max_frames)
        #print("Expected shape: " + str(expected_shape))
        print("Input features has shape: " + str(input_features.shape))
        #generated_ids = model.generate(inputs=input_features)
        #return (torch.rand(expected_shape),) #(input_features,) #(generated_ids,)
        return (input_features,) #(generated_ids,)

        #feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-tiny")
        #ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        #inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
        #print(inputs)
        #print(inputs.input_features)
        #print(inputs.input_features.shape)
        #decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id

        #return (inputs.input_features,decoder_input_ids)
        # Raw audio input: 1 second of 16kHz audio
        #input_values = torch.randn(1, 16000)
        #print(input_values)
        #return (input_values,)
