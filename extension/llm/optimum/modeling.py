import logging
from typing import Optional

import torch
from executorch.extension.pybindings.portable_lib import ExecuTorchModule
from transformers import AutoModelForCausalLM, PretrainedConfig, PretrainedTokenizer

logger = logging.getLogger(__name__)


class ExecuTorchModelForImageTextToTextCausalLM:
    """
    ExecuTorch model with an image-text-to-text causal language modeling head for inference using the ExecuTorch Runtime.

    Although the auto_model_class is `AutoModelForCausalLM` same as `ExecuTorchModelForCausalLM`, this model is specifically designed for
    image-text-to-text tasks. This class provides an interface for loading, running, and generating outputs from a vision-language model
    optimized for ExecuTorch Runtime. It includes utilities for exporting and loading pre-trained models
    compatible with ExecuTorch runtime.

    Attributes:
        auto_model_class (`Type`):
            Associated Transformers class, `AutoModelForCausalLM`.
        model (`ExecuTorchModule`):
            The loaded ExecuTorch model.
    """

    auto_model_class = AutoModelForCausalLM

    task = "image-text-to-text"

    def __init__(self, model: "ExecuTorchModule", config: "PretrainedConfig"):
        if self.__class__.auto_model_class is None:
            raise ValueError(
                f"Class {self.__class__.__name__} must set auto_model_class. "
                f"This attribute is used to identify the corresponding AutoModel class."
            )

        self.model = model
        self.config = config

        # Make sure config contains vision_config and text_config, otherwise raise an error
        if not hasattr(config, "vision_config") or not hasattr(config, "text_config"):
            raise ValueError(
                "The configuration must contain 'vision_config' and 'text_config' attributes for image-text-to-text task."
            )
        metadata = self.model.method_names()
        logging.debug(f"Load all static methods: {metadata}")
        if "use_kv_cache" in metadata:
            self.use_kv_cache = self.model.run_method("use_kv_cache")[0]
        if "get_max_seq_len" in metadata:
            self.max_cache_size = self.model.run_method("get_max_seq_len")[0]
        if "get_max_batch_size" in metadata:
            self.max_batch_size = self.model.run_method("get_max_batch_size")[0]
        if "get_dtype" in metadata:
            self.dtype = self.model.run_method("get_dtype")[0]
        if "get_bos_id" in metadata:
            self.bos_token_id = self.model.run_method("get_bos_id")[0]
        for key in ("get_eos_id", "get_eos_ids"):
            if key in metadata:
                self.eos_token_ids = self.model.run_method(key)
                break
        if "get_vocab_size" in metadata:
            self.vocab_size = self.model.run_method("get_vocab_size")[0]
        if "use_sdpa_with_kv_cache" in metadata:
            self.use_sdpa_with_kv_cache = self.model.run_method(
                "use_sdpa_with_kv_cache"
            )[0]

    def forward(
        self,
        cache_position: torch.LongTensor,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the model, which is compatible with the ExecuTorch runtime for LLM. Here we are assuming pixel_values only represent 1 image.

        Args:
            input_ids (`torch.Tensor`): Tensor representing current input token id to the model.
            pixel_values (`torch.Tensor`): Tensor representing image input to the model.
            cache_position (`torch.Tensor`): Tensor representing current input position in the cache.

        Returns:
            torch.Tensor: Logits output from the model.
        """
        if (input_ids is None) and (pixel_values is None):
            raise ValueError(
                "You must specify at least one of input_ids or pixel_values"
            )

        inputs_embeds = self.model.run_method("token_embeddings", (input_ids,))[0]

        if pixel_values is not None:
            image_features = self.model.run_method(
                "vision_embeddings", (pixel_values,)
            )[0]

            if input_ids is None:
                special_image_mask = (
                    inputs_embeds
                    == self.model.run_method(
                        "token_embeddings",
                        (
                            torch.tensor(
                                self.config.image_token_id,
                                dtype=torch.long,
                                device=inputs_embeds.device,
                            ),
                        ),
                    )[0]
                )
            else:
                special_image_mask = (
                    input_ids == self.config.image_token_id
                ).unsqueeze(-1)
                special_image_mask = special_image_mask.expand_as(inputs_embeds).to(
                    inputs_embeds.device
                )
            image_features = image_features.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_image_mask, image_features
            )

        logits = self.model.run_method("decoder", (cache_position, inputs_embeds))[0]
        return logits

    def generate(
        self,
        tokenizer: "PretrainedTokenizer",
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.FloatTensor] = None,
        max_new_tokens: int = 100,
    ):
        # Sanity check

        if max_new_tokens <= 0:
            raise ValueError(
                f"max_new_tokens must be greater than 0, got {max_new_tokens}."
            )
        elif max_new_tokens > self.max_cache_size:
            logging.warning(
                f"max_new_tokens={max_new_tokens} is larger than max_cache_size={self.max_cache_size}. Generating tokens will be truncated to max_cache_size."
            )
            max_new_tokens = self.max_cache_size

        # Prefill
        logits = self.forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            cache_position=torch.arange(
                input_ids.size(1), dtype=torch.long, device=input_ids.device
            ),
        )

        tokens = []

        token = torch.argmax(logits[:, -1, :], dim=-1).item()
        tokens.append(token)
        i = 1
        while i < max_new_tokens:
            # Generate next token
            logits = self.forward(
                input_ids=torch.tensor(
                    [token], dtype=torch.long, device=input_ids.device
                ).unsqueeze(0),
                cache_position=torch.tensor(
                    [input_ids.size(1) + i - 1],
                    dtype=torch.long,
                    device=input_ids.device,
                ),
            )
            token = torch.argmax(logits[:, -1, :], dim=-1).item()
            tokens.append(token)

            if token in self.eos_token_ids:
                break
            i += 1

        return tokenizer.decode(tokens, skip_special_tokens=True)
