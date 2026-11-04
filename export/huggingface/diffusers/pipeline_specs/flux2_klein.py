# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hugging Face export specification for FLUX.2 Klein variants."""

from dataclasses import replace

from executorch.extension.diffusers.pipeline_behaviors import Flux2KleinPipelineBehavior

from ._flux2 import _Flux2PipelineExportSpec


class Flux2KleinPipelineExportSpec(_Flux2PipelineExportSpec):
    pipeline_behavior = Flux2KleinPipelineBehavior()
    text_encoder_layers = (9, 18, 27)

    def resolve_model_info(self, components, model_info):
        model_info = super().resolve_model_info(components, model_info)
        if model_info.accepts_guidance_input:
            raise ValueError("FLUX.2 Klein does not use embedded guidance")
        return replace(
            model_info,
            supports_cfg=not model_info.is_distilled,
            default_inference_steps=4 if model_info.is_distilled else 50,
        )

    def _load_text_encoder(self, model_id, revision, dtype):
        from transformers import Qwen3ForCausalLM

        model = Qwen3ForCausalLM.from_pretrained(
            model_id,
            subfolder="text_encoder",
            revision=revision,
            torch_dtype=dtype,
        )
        return model.model.eval()
