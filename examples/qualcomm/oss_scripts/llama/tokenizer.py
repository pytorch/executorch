# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
from typing import Callable

from executorch.examples.qualcomm.oss_scripts.llama import LLMModelConfig
from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
    VISION_ENCODER,
)
from pytorch_tokenizers import get_tokenizer, TiktokenTokenizer
from pytorch_tokenizers.llama2c import Llama2cTokenizer as SentencePieceTokenizer

from transformers import AutoTokenizer

# Special tokens for Vision-Language Model
VLM_SPECIAL_TOKENS = {
    "smolvlm_500m_instruct": {
        "image_token": "<image>",
        "global_img": "<global-img>",
        "fake_wrap_start": "<fake_token_around_image>",
        "fake_wrap_end": "<fake_token_around_image>",
    },
    "internvl3_1b": {
        "image_token": "<IMG_CONTEXT>",
        "fake_wrap_start": "<img>",
        "fake_wrap_end": "</img>",
    },
}
# TODO: add special tokens Audio-Language Model
ALM_SPECIAL_TOKENS = {}


class TokenizerWrapper:
    """
    A unified tokenization wrapper for multimodal models and LLM including:
    - Vision-Language Models (VLM) with image token handling
    - Audio-Language Models (ALM) with audio token handling (under development)
    - Text-only Language Models with standard tokenization

    This converting tokenizers from multiple sources into runtime tokenizers.
    The supported sources include: HuggingFace tokenizers, tokenizer_model, and tokenizer_bin,
    which are processed and transformed into runtime format based on the specific model requirements.
    """

    def __init__(self, control_args: argparse.Namespace, config: LLMModelConfig):
        self.artifact = control_args.artifact
        self.decoder_model = control_args.decoder_model
        self.verbose = control_args.verbose

        self.config = config
        self.repo_id = config.repo_id
        self.apply_chat_template = config.instruct_model

    def _from_tokenizer_model_and_bin(self, tokenizer_model, tokenizer_bin):
        tokenizer = get_tokenizer(tokenizer_model)
        assert isinstance(
            tokenizer, SentencePieceTokenizer
        ), "Wrong tokenizer provided for stories."
        assert tokenizer_bin is not None, "Please provide tokenizer_bin for stories."
        runtime_tokenizer_path = tokenizer_bin
        return runtime_tokenizer_path, tokenizer

    def _from_tokenizer_model(self, tokenizer_model):
        tokenizer = get_tokenizer(tokenizer_model)
        assert isinstance(
            tokenizer, TiktokenTokenizer
        ), "Wrong tokenizer provided for llama3_2."
        runtime_tokenizer_path = tokenizer_model
        return runtime_tokenizer_path, tokenizer

    def _from_hf(self):
        tokenizer = AutoTokenizer.from_pretrained(self.repo_id)
        chat_template = (
            tokenizer.apply_chat_template
            if hasattr(tokenizer, "apply_chat_template") and self.apply_chat_template
            else None
        )
        tokenizer_artifacts = tokenizer.save_pretrained(self.artifact)
        tokenizer_config = tokenizer_artifacts[0]
        if self.decoder_model == "gemma-2b":
            # For Gemma, use tokenizer.model as it doesn't provide pre_tokenizer in tokenizer.json.
            runtime_tokenizer_path = tokenizer_artifacts[-3]
        else:
            if self.decoder_model == "glm-1_5b":
                with open(tokenizer_config, "r+") as file:
                    data = json.load(file)
                    # Verified with HF flow and it uses <|user|> as eos condition
                    data["bos_token"] = "<|user|>"
                    data["eos_token"] = "<|user|>"
                    file.seek(0)
                    json.dump(data, file, indent=4)
                    file.truncate()
            runtime_tokenizer_path = tokenizer_artifacts[-1]

        tokenizer = get_tokenizer(runtime_tokenizer_path, tokenizer_config)

        if self.decoder_model == "codegen2_1b":
            # Override the default BOS and EOS token IDs for codegen2_1b
            tokenizer.bos_id = 1
            tokenizer.eos_id = 2
        elif self.decoder_model == "phi_4_mini":
            with open(runtime_tokenizer_path, "r+") as file:
                data = json.load(file)
                # TODO: Encountered the following error during runtime, so switched behavior for now.
                # Error: libc++abi: terminating due to uncaught exception of type std::runtime_error: invert=true is not supported for Split PreTokenizer. Only invert=false is supported.
                data["pre_tokenizer"]["pretokenizers"][-2]["invert"] = False
                file.seek(0)
                json.dump(data, file, indent=4)
                file.truncate()

        return runtime_tokenizer_path, tokenizer, chat_template

    def get_runtime_tokenizer(self, tokenizer_model, tokenizer_bin):
        tokenizer = None
        runtime_tokenizer_path = ""
        chat_template = None
        if self.decoder_model in {"stories110m", "stories260k"}:
            runtime_tokenizer_path, tokenizer = self._from_tokenizer_model_and_bin(
                tokenizer_model, tokenizer_bin
            )
        elif "llama3_2" in self.decoder_model:
            runtime_tokenizer_path, tokenizer = self._from_tokenizer_model(
                tokenizer_model
            )
        else:
            runtime_tokenizer_path, tokenizer, chat_template = self._from_hf()

        return runtime_tokenizer_path, tokenizer, chat_template

    def prepare_multimodal_prompt(
        self,
        prompt: str,
    ) -> str:
        """
        Prepare multimodal prompt by expanding special tokens.

        This method processes prompts containing multimodal tokens (e.g., <image>, <audio>)
        and expands them into the format expected by the multimodal model. For vision-language
        models, image tokens are expanded to include wrapper tokens and repeated based on
        the image sequence length.

        Args:
            prompt (str): Input prompt containing multimodal tokens such as <image> or <audio>

        Returns:
            str: Processed prompt with expanded multimodal tokens ready for model inference
        """
        if (
            self.decoder_model not in VLM_SPECIAL_TOKENS
            and self.decoder_model not in ALM_SPECIAL_TOKENS
        ):
            raise ValueError(
                f"No special tokens defined for model {self.decoder_model}"
            )

        if self.decoder_model in VLM_SPECIAL_TOKENS:
            specials = VLM_SPECIAL_TOKENS[self.decoder_model]

            image_seq_len = getattr(self.config, VISION_ENCODER, None).img_seq_len

            # Build the expanded image prompt
            image_prompt = (
                f"{specials['fake_wrap_start']}"
                f"{specials.get('global_img', '')}"
                f"{specials['image_token'] * image_seq_len}"
                f"{specials['fake_wrap_end']}"
            )
            # Replace image token with expanded version
            expanded = prompt.replace(specials["image_token"], image_prompt)

            if self.verbose:
                logging.info(f"Prompt after expanding image token: {expanded}")

            return expanded

        elif self.decoder_model in ALM_SPECIAL_TOKENS:
            raise NotImplementedError(
                "Audio-language model expanded tokens still under development"
            )

    def apply_prompt_template(
        self,
        chat_template: Callable,
        prompt: str,
        system_prompt: str = None,
    ) -> str:
        """
        Apply chat template to format the prompt for different modalities.

        Args:
            chat_template: The chat template function from tokenizer
            prompt: Input text prompt
            system_prompt: Optional system prompt

        Returns:
            Formatted prompt string
        """
        if self.decoder_model in VLM_SPECIAL_TOKENS:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        elif self.decoder_model in ALM_SPECIAL_TOKENS:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        template_prompt = chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        logging.info(f"Prompt after applying template: {template_prompt}")

        # edge cases handling:
        # Gemma may produce unexpected output if the prompt contains an extra <bos> token.
        # This can happen after applying a prompt template, which might inject <bos> unintentionally.
        # To prevent decoding issues, we explicitly remove <bos> token
        if chat_template and self.decoder_model in {
            "gemma-2b",
            "gemma3-1b",
        }:
            template_prompt = template_prompt.replace("<bos>", "")

        return template_prompt
