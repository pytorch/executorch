# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hugging Face export specification for FLUX.2-dev."""

from executorch.extension.diffusers.pipeline_behaviors import Flux2PipelineBehavior

from ._flux2 import _Flux2PipelineExportSpec


class Flux2PipelineExportSpec(_Flux2PipelineExportSpec):
    pipeline_behavior = Flux2PipelineBehavior()
    text_encoder_layers = (10, 20, 30)

    def _load_text_encoder(self, model_id, revision, dtype):
        from transformers import Mistral3ForConditionalGeneration

        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_id,
            subfolder="text_encoder",
            revision=revision,
            torch_dtype=dtype,
        )
        return model.model.language_model.eval()
