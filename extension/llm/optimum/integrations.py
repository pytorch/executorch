# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import Dict, Optional

import torch

from executorch.extension.llm.optimum.custom_sdpa import (
    get_custom_sdpa_for_ring_kv_cache,
)

from executorch.extension.llm.optimum.utils import save_config_to_constant_methods
from packaging.version import parse
from torch.export import ExportedProgram
from transformers import PreTrainedModel
from transformers.cache_utils import HybridCache
from transformers.integrations.executorch import sdpa_mask_without_vmap
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


class TorchExportableModuleWithHybridCache(torch.nn.Module):
    """
    A recipe module designed to make a `PreTrainedModel` exportable with `torch.export`,
    specifically for decoder-only LM to `HybridCache`. This module ensures that the
    exported model is compatible with further lowering and execution in `ExecuTorch`.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        max_batch_size: int = 1,
        max_cache_len: int = 4096,
    ):
        """
        Initializes the exportable module with `HybridCache`.

        Args:
            model (`PreTrainedModel`): The pretrained model to wrap.
            max_batch_size (int): Maximum batch size for the cache.
            max_cache_len (int): Maximum sequence length for the cache.

        Raises:
            AssertionError: If the model doesn't have the expected configuration for HybridCache.
        """
        super().__init__()
        self.model = model

        # Verify the model is configured for HybridCache
        if not self.model.config.text_config.use_cache:
            raise AssertionError("Model must have caching enabled")

        # Initialize the HybridCache
        self.cache = HybridCache(
            config=self.model.config.text_config,
            max_batch_size=max_batch_size,
            max_cache_len=max_cache_len,
            device=self.model.device,
            dtype=self.model.dtype,
        )

        # Register all key and value cache tensors as buffers
        for i in range(len(self.cache.key_cache)):
            self.register_buffer(
                f"key_cache_{i}", self.cache.key_cache[i], persistent=False
            )
            self.register_buffer(
                f"value_cache_{i}", self.cache.value_cache[i], persistent=False
            )

    def forward(
        self,
        *,
        cache_position: torch.Tensor,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the module, which is compatible with the ExecuTorch llm runner.

        Args:
            cache_position (`torch.Tensor`): Tensor representing current input position in the cache.
            input_ids (`torch.Tensor`): Tensor representing current input token id to the module.
            inputs_embeds (`torch.Tensor`): Optional tensor representing input embeddings.

        Returns:
            torch.Tensor: Logits output from the model.
        """
        batch_size = (
            input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        )

        # Generate position_ids from cache_position
        position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)

        # Forward pass with the model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=self.cache,
            use_cache=True,
            cache_position=cache_position,
            inputs_embeds=inputs_embeds,
        )

        # Return only the logits to simplify the export
        return outputs.logits


class TorchExportableModuleForImageTextLM(torch.nn.Module):
    """
    A recipe module designed to make a `PreTrainedModel` exportable with `torch.export`,
    specifically for image-text LM with cache. This module ensures that the
    exported model is compatible with further lowering and execution in `ExecuTorch`.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        max_batch_size: int = 1,
        max_cache_len: int = 4096,
    ):
        """
        Initializes the exportable module with `HybridCache`.

        Args:
            model (`PreTrainedModel`): The pretrained model to wrap.
            max_batch_size (int): Maximum batch size for the cache.
            max_cache_len (int): Maximum sequence length for the cache.

        Raises:
            ValueError: If the model is configured with a unsupported cache implementation.
        """
        super().__init__()

        if (
            not hasattr(model.config.text_config, "use_cache")
            or model.config.text_config.use_cache is False
        ):
            raise ValueError("The model must have caching enabled to be performant.")

        if (
            hasattr(model.config.text_config, "layer_types")
            and getattr(model.config.text_config, "sliding_window", None) is not None
        ):
            self.model = TorchExportableModuleWithHybridCache(
                model, max_batch_size, max_cache_len
            )
        else:
            # If `layer_types` is not specified explicitly in the config or `sliding_window` is null,
            # there is only 1 type of layers, so export will use `StaticCache` by default.
            raise NotImplementedError(
                "Using `StaticCache` for exporting image-text LM is not implemented yet."
            )
        # This is the same as sdpa, but mask creation does not use `vmap` which is not exportable
        ALL_MASK_ATTENTION_FUNCTIONS.register(
            "sdpa_without_vmap", sdpa_mask_without_vmap
        )
        ALL_ATTENTION_FUNCTIONS.register(
            "sdpa_without_vmap", ALL_ATTENTION_FUNCTIONS["sdpa"]
        )
        self.model.model.config._attn_implementation = "sdpa_without_vmap"

    def forward(
        self,
        input_embeds: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the module, which is compatible with the ExecuTorch llm runner.

        Args:
            input_embeds (`torch.Tensor`): Tensor representing current input embeddings to the module.
            cache_position (`torch.Tensor`): Tensor representing current input position in the cache.

        Returns:
            torch.Tensor: Logits output from the model.
        """
        return self.model.forward(input_embeds, cache_position)

    def export(
        self,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        dynamic_shapes: Optional[dict] = None,
        strict: Optional[bool] = None,
    ) -> torch.export.ExportedProgram:
        """
        Export the wrapped module using `torch.export`.

        Args:
            input_embeds (`Optional[torch.Tensor]`):
                Tensor representing current input embeddings to the module. If not provided, a default tensor will be used.
            cache_position (`Optional[torch.Tensor]`):
                Tensor representing current input position in the cache. If not provided, a default tensor will be used.
            dynamic_shapes (`Optional[dict]`):
                Dynamic shapes to use for export if specified.
            strict(`Optional[bool]`):
                Flag to instruct `torch.export` to use `torchdynamo`.
        """
        seq_length = 3

        if dynamic_shapes is None:
            seq_len_dim = torch.export.Dim("seq_length_dim", max=seq_length)
            dynamic_shapes = {
                "inputs_embeds": {1: seq_len_dim},
                "cache_position": {0: seq_len_dim},
            }

        exported_program = torch.export.export(
            self.model,
            args=(),
            kwargs={"cache_position": cache_position, "inputs_embeds": inputs_embeds},
            dynamic_shapes=dynamic_shapes,
            strict=strict if strict is not None else True,
        )
        return exported_program


class ImageEncoderExportableModule(torch.nn.Module):
    """
    A wrapper module designed to make a vision encoder-only model exportable with `torch.export`.
    This module ensures that the exported model is compatible with ExecuTorch.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        """
        Projects the last hidden state from the vision model into language model space.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
               The tensors corresponding to the input images.
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        vision_outputs = self.model.vision_tower(
            pixel_values=pixel_values
        ).last_hidden_state
        image_features = self.model.multi_modal_projector(vision_outputs)
        return image_features


class ImageTextToTextExportableModule(torch.nn.Module):
    """
    A wrapper module designed to make an image-text-to-text model exportable with `torch.export`.
    This module ensures that the exported model is compatible with ExecuTorch.
    """

    def __init__(self, model, use_custom_kv_cache=False, use_custom_sdpa=False):
        super().__init__()
        self.model = model
        self.config = model.config
        self.use_custom_kv_cache = use_custom_kv_cache
        self.use_custom_sdpa = use_custom_sdpa
        self.metadata = save_config_to_constant_methods(
            model.config.text_config, model.generation_config
        )
        logging.info(f"Metadata to be recorded in PTE: {self.metadata}")

    def _prepare_vision_embedding_export_inputs(self):
        """
        Prepare example inputs and configurations for export.

        Returns:
            pixel_values (torch.Tensor): Example pixel values tensor.
            dynamic_shapes (dict or None): Dynamic shape specifications for export.
            strict (bool): Whether to use strict export mode.
        """
        image_size = self.config.vision_config.image_size
        pixel_values = torch.rand((1, 3, image_size, image_size))
        dynamic_shapes = None
        strict = False

        return pixel_values, dynamic_shapes, strict

    def _prepare_text_embedding_export_inputs(self):
        """
        Prepare example inputs and configurations for export.

        Returns:
            input_ids (torch.Tensor): Example input IDs tensor.
            dynamic_shapes (dict or None): Dynamic shape specifications for export.
            strict (bool): Whether to use strict export mode.
        """
        # Prepare inputs with dynamic shapes
        seq_length = 3  # Sequence length > 1 to avoid specialization issues
        example_input_ids = torch.zeros((1, seq_length), dtype=torch.long)
        max_seq_len = self.metadata.get("get_max_seq_len")
        sliding_window = self.metadata.get("sliding_window", float("inf"))
        max_dim = min(max_seq_len, sliding_window) - 1
        seq_len_dim = torch.export.Dim("seq_length_dim", max=max_dim)
        dynamic_shapes = {
            "input_ids": {1: seq_len_dim},
        }
        strict = parse(torch.__version__) != parse(
            "2.7.0"
        )  # Workaround for PyTorch bug #150994
        return example_input_ids, dynamic_shapes, strict

    def _prepare_decoder_only_export_inputs(self):
        """
        Prepare example inputs and configurations for export.

        Returns:
            inputs_embeds (torch.Tensor): Example input embeddings tensor.
            cache_position (torch.Tensor): Example cache position tensor.
            dynamic_shapes (dict or None): Dynamic shape specifications for export.
            strict (bool): Whether to use strict export mode.
        """

        # Prepare inputs with dynamic shapes
        seq_length = 3
        example_inputs_embeds = torch.zeros(
            (1, seq_length, self.config.text_config.hidden_size), dtype=torch.float
        )
        example_cache_position = torch.arange(seq_length, dtype=torch.long)
        max_seq_len = self.metadata.get("get_max_seq_len")
        sliding_window = self.metadata.get("sliding_window", float("inf"))
        max_dim = min(max_seq_len, sliding_window) - 1
        seq_len_dim = torch.export.Dim("seq_length_dim", max=max_dim)
        dynamic_shapes = {
            "inputs_embeds": {1: seq_len_dim},
            "cache_position": {0: seq_len_dim},
        }
        strict = parse(torch.__version__) != parse(
            "2.7.0"
        )  # Workaround for PyTorch bug #150994
        return example_inputs_embeds, example_cache_position, dynamic_shapes, strict

    def _register_attention_mask_for_4_53(self, exportable_module: torch.nn.Module):
        from transformers.integrations.executorch import sdpa_mask_without_vmap
        from transformers.masking_utils import AttentionMaskInterface
        from transformers.modeling_utils import AttentionInterface

        _custom_sdpa_for_ring_kv_cache = get_custom_sdpa_for_ring_kv_cache(
            exportable_module
        )
        if self.use_custom_sdpa:
            if self.use_custom_kv_cache:
                AttentionInterface.register(
                    "custom_sdpa_ring_kv_cache", _custom_sdpa_for_ring_kv_cache
                )
                AttentionMaskInterface.register(
                    "custom_sdpa_ring_kv_cache", sdpa_mask_without_vmap
                )
                # Manually set the attention implementation to custom_sdpa_ring_kv_cache
                # This handles both regular sdpa and one for sliding window/local attention
                exportable_module.model.model.config._attn_implementation = (
                    "custom_sdpa_ring_kv_cache"
                )
            else:
                # Manually set the attention implementation to custom_sdpa_ring_kv_cache
                # This handles both regular sdpa and one for sliding window/local attention
                exportable_module.model.model.config._attn_implementation = (
                    "custom_sdpa"
                )

    def export(
        self,
    ) -> Dict[str, ExportedProgram]:

        exportable_module = TorchExportableModuleForImageTextLM(
            self.model,
            max_batch_size=1,
            max_cache_len=self.metadata.get("get_max_seq_len"),
        )
        self._register_attention_mask_for_4_53(exportable_module)

        if self.use_custom_kv_cache:
            from executorch.extension.llm.optimum.custom_kv_cache import (
                replace_with_et_custom_kv_cache,
            )

            replace_with_et_custom_kv_cache(
                exportable_module.model,
                self.model.config.text_config,
                self.model.generation_config,
                self.model.dtype,
            )

        with torch.no_grad():
            inputs_embeds, cache_position, dynamic_shapes, strict = (
                self._prepare_decoder_only_export_inputs()
            )
            logging.info(
                f"Exporting decoder using inputs_embeds({inputs_embeds.shape}), cache_position({cache_position.shape})={cache_position}, dynamic_shapes={dynamic_shapes}, strict={strict}"
            )
            exported_program = exportable_module.export(
                inputs_embeds, cache_position, dynamic_shapes, strict
            )
            # Apply RemoveTransposes pass to remove
            # any back-to-back transpose ops that are not needed
            # e.g. output of update_cache is transposed and
            # input to custom_sdpa is transposed.
            from executorch.extension.llm.export.export_passes import (
                RemoveRedundantTransposes,
            )

            mutated_gm = RemoveRedundantTransposes()(exported_program.module())[0]
            exported_program = torch.export.export(
                mutated_gm,
                args=(),
                kwargs={
                    "cache_position": cache_position,
                    "inputs_embeds": inputs_embeds,
                },
                dynamic_shapes=dynamic_shapes,
                strict=strict,
            )

            # Export token embeddings
            input_ids, dynamic_shapes, strict = (
                self._prepare_text_embedding_export_inputs()
            )
            logging.info(
                f"Exporting token embeddings using input_ids({input_ids.shape}), dynamic_shapes={dynamic_shapes}, strict={strict}"
            )

            token_embeddings_exported_program = torch.export.export(
                exportable_module.model.model.language_model.get_input_embeddings(),
                args=(input_ids,),
                kwargs={},
                dynamic_shapes=dynamic_shapes,
                strict=strict,
            )

            # Export vision embeddings
            pixel_values, dynamic_shapes, strict = (
                self._prepare_vision_embedding_export_inputs()
            )
            logging.info(
                f"Exporting vision embeddings using pixel_values({pixel_values.shape}), dynamic_shapes={dynamic_shapes}, strict={strict}"
            )
            # Setting the _attn_implementation to "sdpa_without_vmap" for vision encoder
            exportable_module.model.model.vision_tower.config._attn_implementation = (
                "sdpa_without_vmap"
            )
            vision_encoder = ImageEncoderExportableModule(exportable_module.model.model)
            vision_embeddings_exported_program = torch.export.export(
                vision_encoder,
                args=(pixel_values,),
                kwargs={},
                dynamic_shapes=dynamic_shapes,
                strict=strict,
            )
        return {
            "text_model": exported_program,
            "token_embedding": token_embeddings_exported_program,
            "image_encoder": vision_embeddings_exported_program,
        }
