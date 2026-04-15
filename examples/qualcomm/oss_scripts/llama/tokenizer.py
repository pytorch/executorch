# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import re
import warnings
from typing import Callable, List

from executorch.examples.qualcomm.oss_scripts.llama import LLMModelConfig
from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
    VISION_ENCODER,
)
from pytorch_tokenizers import get_tokenizer, TiktokenTokenizer
from pytorch_tokenizers.llama2c import Llama2cTokenizer as SentencePieceTokenizer

from transformers import AutoTokenizer

IMG_TOKEN = "<image>"
AUDIO_TOKEN = "<audio>"

# Special tokens for Vision-Language Model
VLM_SPECIAL_TOKENS = {
    "smolvlm_500m_instruct": {
        IMG_TOKEN: "<image>",
        "global_img": "<global-img>",
        "fake_wrap_start": "<fake_token_around_image>",
        "fake_wrap_end": "<fake_token_around_image>",
    },
    "internvl3_1b": {
        IMG_TOKEN: "<IMG_CONTEXT>",
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

        self.control_args = control_args
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

    def prepare_messages(self, prompts: List[str]):
        """
        Validate and normalize a multi-turn prompt sequence, then prepare it into
        a message list.

        This function checks image-token usage against provided image paths, auto-injects
        image tokens when none of them were present, and constructs a per-turn message structure.

        Args:
            prompts (List[str]):
                A list of user prompts representing a multi-turn conversation.
                If `VISION_ENCODER` is present in `self.config`, image usage is validated:
                - The total count of image tokens (IMG_TOKEN) across all prompts must
                    match the number of image paths, unless no image token is present at all
                    (in which case tokens will be auto-prepended to the first prompt).

        Returns:
            List[Dict[str, Any]]:
                A list of message dictionaries, one per prompt/turn, in the same order as `prompts`.
                Each message has the following schema:

                - `id` (int): 0-based turn index (i.e., position in `prompts`).
                - `text` (str): The raw prompt text for this turn. If no image tokens were
                present anywhere and images were provided/assumed, the first prompt's text
                is auto-prefixed with `IMG_TOKEN * num_images`.
                - `files_path` (List[str]): Image paths (local or URLs) associated with this
                turn, assigned left-to-right based on the number of `IMG_TOKEN` occurrences
                in `text`. Empty when the turn contains no image tokens.

                Example return value:
                [
                    {"id": 0, "text": "<image><image> Compare these images", "files_path": ["a.png", "b.png"]},
                    {"id": 1, "text": "Answer the question: What's the main object in first image?", "files_path": []},
                ]

        Raises:
            ValueError:
                Raised only if the user has already included one or more image tokens (IMG_TOKEN)
                across `prompts` and the total number of those tokens does not equal the number of
                provided `image_paths`.

        Examples:
            >>> self.control_args.image_path = ["img1.jpg", "img2.jpg"]
            >>> prompts = ["<image><image>Compare these images above and list the differences.", "Answer the question: What's the main object in first image?"]
            >>> prepare_messages(prompts)
            [
                {"id": 0, "text": "<image><image>Compare these images above and list the differences.", "files_path": ["img1.jpg", "img2.jpg"]},
                {"id": 1, "text": "Answer the question: What's the main object in first image?", "files_path": []},
            ]
        """

        messages = []

        image_paths = self.control_args.image_path
        if hasattr(self.config, VISION_ENCODER):
            # Load image from user-specified path (URL or local file)
            # fall back to the default image URL if no image is provided.
            if not image_paths:
                image_paths = [getattr(self.config, VISION_ENCODER).img_url]
                warnings.warn(
                    f"No image path/URL provided, using default image URL: {image_paths}",
                    UserWarning,
                    stacklevel=1,
                )

            num_images = len(image_paths)

            total_image_tokens = sum(prompt.count(IMG_TOKEN) for prompt in prompts)

            if total_image_tokens == 0:
                prompts[0] = (IMG_TOKEN * num_images) + prompts[0]
            elif total_image_tokens != num_images:
                raise ValueError(
                    f"Number of <image> tokens ({total_image_tokens}) does not match "
                    f"number of images ({num_images}). Please check your prompts and image paths."
                    "Please check your prompts and image paths.\n\n"
                    f"=== Prompt ===\n{prompts}\n"
                    f"=== Image paths ===\n{image_paths}"
                )

        img_idx = 0
        for i, prompt in enumerate(prompts):
            message = {"id": i, "text": prompt, "files_path": []}
            if IMG_TOKEN in prompt:
                num_img = prompt.count(IMG_TOKEN)
                message["files_path"] = image_paths[img_idx : img_idx + num_img]
                img_idx += num_img
            messages.append(message)

        if self.control_args.verbose:
            logging.info("Simulation multi-turn:")
            logging.info(messages)
        return messages

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
                f"{specials[IMG_TOKEN] * image_seq_len}"
                f"{specials['fake_wrap_end']}"
            )
            # Replace image token with expanded version
            expanded = prompt.replace(specials[IMG_TOKEN], image_prompt)
            if self.verbose:
                logging.info(f"Prompt after expanding image token: {expanded}")

            return expanded

        elif self.decoder_model in ALM_SPECIAL_TOKENS:
            raise NotImplementedError(
                "Audio-language model expanded tokens still under development"
            )

    def _split_prompt(self, prompt: str):
        """
        Split user prompt by special tokens.

        Args:
            prompt (str): Input prompt containing special tokens

        Returns:
            List[str]: List of prompt segments split by special tokens
        """
        split_tokens = set()
        if self.decoder_model in VLM_SPECIAL_TOKENS:
            split_tokens.add(IMG_TOKEN)
        if self.decoder_model in ALM_SPECIAL_TOKENS:
            split_tokens.add(AUDIO_TOKEN)

        if not split_tokens:
            return [prompt]
        pattern = f"({'|'.join(map(re.escape, split_tokens))})"
        return [part for part in re.split(pattern, prompt) if part]

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

        messages = []
        message = {"role": "user", "content": prompt}
        if self.decoder_model in VLM_SPECIAL_TOKENS:
            contents = self._split_prompt(prompt)
            message["content"] = []
            for content in contents:
                if content == IMG_TOKEN:
                    message["content"].append(
                        {"type": "image"},
                    )
                else:
                    message["content"].append(
                        {"type": "text", "text": content},
                    )
        elif self.decoder_model in ALM_SPECIAL_TOKENS:
            message["content"] = prompt

        messages.append(message)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        template_prompt = chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

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
