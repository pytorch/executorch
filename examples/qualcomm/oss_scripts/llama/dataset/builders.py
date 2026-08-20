# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
from executorch.examples.qualcomm.oss_scripts.llama import LLMModelConfig
from executorch.examples.qualcomm.oss_scripts.llama.dataset.collators import (
    LLMCalibCollator,
    ModalityEncoderCollator,
)
from executorch.examples.qualcomm.oss_scripts.llama.dataset.config import DataConfig
from executorch.examples.qualcomm.oss_scripts.llama.dataset.datasets import (
    LLMDataset,
    ModalityEncoderDataset,
)
from executorch.examples.qualcomm.oss_scripts.llama.dataset.loaders import (
    collect_lm_eval_tokens,
    load_audio_file,
    load_conversation_samples,
)
from executorch.examples.qualcomm.oss_scripts.llama.dataset.preprocessors import (
    ModalityPreprocessor,
)
from executorch.examples.qualcomm.oss_scripts.llama.dataset.schema import MessageSample
from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
    AUDIO_ENCODER,
    TEXT_DECODER,
    TEXT_ENCODER,
    TOK_EMBEDDING,
    VISION_ENCODER,
)
from executorch.examples.qualcomm.oss_scripts.llama.encoder.encoder_config import (
    AudioModalityConfig,
    MultiModalityConfig,
    VisionModalityConfig,
)
from executorch.examples.qualcomm.oss_scripts.llama.masking_utils import AttentionMask
from executorch.examples.qualcomm.oss_scripts.llama.tokenizer import TokenizerWrapper
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from transformers.image_utils import load_image

logger = logging.getLogger(__name__)

_ALL_MODALITY_KEYS = (
    AUDIO_ENCODER,
    TEXT_ENCODER,
    VISION_ENCODER,
    TOK_EMBEDDING,
    TEXT_DECODER,
)


class DecoderDatasetBuilder:
    """Builds LLMDataset instances. Each from_* method corresponds to one data source.

    Datasets contain raw (unpadded) token sequences. Padding, masking, and label
    generation are handled by the collators owned by DatasetBuilder.
    """

    def __init__(
        self,
        tokenizer_wrapper: TokenizerWrapper,
        max_context_len: int,
        is_multimodal: bool = False,
    ):
        self._tokenizer_wrapper = tokenizer_wrapper
        self._max_context_len = max_context_len
        self._is_multimodal = is_multimodal

    def from_conversation(self, samples: List[MessageSample]) -> LLMDataset:
        max_context_len = self._max_context_len
        sequences = []
        for sample in samples:
            prompt = self._tokenizer_wrapper.apply_chat_template(sample.messages)
            if self._is_multimodal:
                prompt = self._tokenizer_wrapper.prepare_multimodal_prompt(prompt)
            tokens = self._tokenizer_wrapper.tokenizer.encode(
                prompt, bos=True, eos=False
            )
            if self._is_multimodal and len(tokens) > max_context_len:
                raise ValueError(
                    f"Sample: {prompt}\n"
                    f"has {len(tokens)} tokens which exceeds "
                    f"max_context_len={max_context_len}."
                )
            sequences.extend(
                tokens[i : i + max_context_len]
                for i in range(0, len(tokens), max_context_len)
            )
        return LLMDataset(sequences)

    def from_lm_eval(
        self,
        tasks: Union[str, List[str]],
        limit: int,
        num_fewshot: Optional[int] = None,
    ) -> LLMDataset:
        sequences = collect_lm_eval_tokens(
            tokenizer=self._tokenizer_wrapper.tokenizer,
            max_context_length=self._max_context_len,
            vocab_size=self._tokenizer_wrapper.vocab_size,
            tasks=tasks,
            tasks_limit=limit,
            num_fewshot=num_fewshot,
        )
        return LLMDataset(sequences)

    def from_tokens(self, token_ids: List[int]) -> LLMDataset:
        """Wrap a single pre-tokenized sequence in an LLMDataset."""
        return LLMDataset([token_ids])


class EncoderDatasetBuilder:
    """Builds modality-encoder Datasets."""

    def __init__(
        self,
        llm_config: LLMModelConfig,
        tokenizer_wrapper: TokenizerWrapper,
    ):
        self.llm_config = llm_config
        self._preprocessor = ModalityPreprocessor(llm_config.repo_id)
        self._tokenizer_wrapper = tokenizer_wrapper

    def load_multimodal_inputs(
        self,
        config: Type[MultiModalityConfig],
        prompt: str,
        paths: List[str],
    ) -> List[Tuple[torch.Tensor, ...]]:
        if issubclass(config, AudioModalityConfig):
            return [
                self._preprocessor.preprocess_audio(
                    config, prompt, load_audio_file(p, self.llm_config.repo_id)
                )
                for p in paths
            ]
        elif issubclass(config, VisionModalityConfig):
            return self._preprocessor.preprocess_images(
                config, prompt, [load_image(p) for p in paths]
            )
        raise NotImplementedError(f"Unsupported config type: {config}")

    def from_message_samples(
        self, samples: List[MessageSample], modality: str
    ) -> Optional[ModalityEncoderDataset]:
        """Build calibration datasets for each encoder modality."""
        if not samples:
            return None

        missing = [i for i, s in enumerate(samples) if not s.files]
        if missing:
            raise ValueError(
                f"Multimodal model requires files in each sample, "
                f"but samples at indices {missing} provide none."
            )

        # [N samples, N files]
        modality_data: List[List] = []
        for sample in samples:
            prompt = self._tokenizer_wrapper.apply_chat_template(sample.messages)
            if not hasattr(self.llm_config, modality):
                continue
            inputs = self.load_multimodal_inputs(
                getattr(self.llm_config, modality), prompt, sample.files
            )
            modality_data.extend(inputs)
        return ModalityEncoderDataset(modality_data)


class DatasetBuilder:
    """Orchestrates DecoderDatasetBuilder and EncoderDatasetBuilder for calibration."""

    def __init__(
        self,
        data_config: DataConfig,
        llm_config: LLMModelConfig,
        tokenizer_wrapper: TokenizerWrapper,
        attn_mask: AttentionMask,
    ):
        self._data_config = data_config
        self._llm_config = llm_config
        self._tokenizer_wrapper = tokenizer_wrapper
        self._attn_mask = attn_mask
        self._is_multimodal = any(
            hasattr(llm_config, m) for m in [AUDIO_ENCODER, VISION_ENCODER]
        )
        self._text_decoder_dataset_builder = DecoderDatasetBuilder(
            tokenizer_wrapper=tokenizer_wrapper,
            max_context_len=data_config.max_context_len,
            is_multimodal=self._is_multimodal,
        )
        self._modality_encoder_dataset_builder: Optional[EncoderDatasetBuilder] = (
            EncoderDatasetBuilder(
                llm_config=llm_config, tokenizer_wrapper=tokenizer_wrapper
            )
            if self._is_multimodal
            else None
        )

        # Collator for multimodal encoders
        self._encoder_collator = ModalityEncoderCollator()
        # Collator for PTQ calibration: pads tokens and builds attention masks.
        self._calib_collator = LLMCalibCollator(
            self._attn_mask,
            self._data_config.max_context_len,
            self._data_config.token_dtype,
        )

    def build_calib_dataloaders(self) -> Dict[str, Optional[DataLoader]]:
        """Calibration DataLoaders for all modalities; all modality keys always present."""
        cfg = self._data_config
        datasets: Dict[str, Dataset] = {}

        samples: List[MessageSample] = load_conversation_samples(cfg.calib_samples)

        # build encoder datasets
        if self._is_multimodal:
            datasets[AUDIO_ENCODER] = (
                self._modality_encoder_dataset_builder.from_message_samples(
                    samples, AUDIO_ENCODER
                )
            )
            datasets[VISION_ENCODER] = (
                self._modality_encoder_dataset_builder.from_message_samples(
                    samples, VISION_ENCODER
                )
            )

        # build decoder datasets
        decoder_datasets: List[Dataset] = list(
            filter(
                None,
                [
                    # build from samples
                    (
                        self._text_decoder_dataset_builder.from_conversation(samples)
                        if samples
                        else None
                    ),
                    # build from lm tasks
                    (
                        self._text_decoder_dataset_builder.from_lm_eval(
                            cfg.calib_tasks, cfg.calib_limit, cfg.calib_num_fewshot
                        )
                        if cfg.calib_tasks is not None
                        else None
                    ),
                ],
            )
        )
        if not decoder_datasets:
            raise ValueError(
                "No calibration data specified. Provide at least one of: "
                "--calib_tasks, --calib_samples"
            )

        datasets[TEXT_DECODER] = (
            ConcatDataset(decoder_datasets)
            if len(decoder_datasets) > 1
            else decoder_datasets[0]
        )
        self._log_dataset_stats(datasets, cfg.batch_size, phase="calibration")

        return dict.fromkeys(_ALL_MODALITY_KEYS) | {
            modality: DataLoader(
                dataset,
                # Encoders are forced to batch_size=1: multi-batch quantization is not yet supported.
                batch_size=cfg.batch_size if modality == TEXT_DECODER else 1,
                shuffle=False,
                drop_last=cfg.batch_size > 1 if modality == TEXT_DECODER else False,
                collate_fn=(
                    self._calib_collator
                    if modality == TEXT_DECODER
                    else self._encoder_collator
                ),
            )
            for modality, dataset in datasets.items()
        }

    def build_runtime_dataloader(
        self,
        message: MessageSample,
    ) -> Dict[str, Optional[DataLoader]]:
        datasets = {}
        # build encoder datasets
        if self._is_multimodal:
            datasets[AUDIO_ENCODER] = (
                self._modality_encoder_dataset_builder.from_message_samples(
                    [message], AUDIO_ENCODER
                )
            )
            datasets[VISION_ENCODER] = (
                self._modality_encoder_dataset_builder.from_message_samples(
                    [message], VISION_ENCODER
                )
            )
        # build decoder datasets
        datasets[TEXT_DECODER] = self._text_decoder_dataset_builder.from_conversation(
            [message]
        )

        return dict.fromkeys(_ALL_MODALITY_KEYS) | {
            modality: DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=(
                    self._calib_collator
                    if modality == TEXT_DECODER
                    else self._encoder_collator
                ),
            )
            for modality, dataset in datasets.items()
        }

    @staticmethod
    def _log_dataset_stats(
        datasets: Dict[str, Dataset],
        batch_size: int,
        phase: str = "calibration",
    ) -> None:
        """Log sample/batch counts per modality; raises if any dataset < batch_size."""
        for modality, ds in datasets.items():
            n = len(ds)
            n_batches = n // batch_size
            dropped = n - n_batches * batch_size
            drop_str = f" ({dropped} dropped)" if batch_size > 1 and dropped else ""
            logging.info(
                "%s '%s': %d samples, batch_size=%d, %d batches%s",
                phase,
                modality,
                n,
                batch_size,
                n_batches,
                drop_str,
            )
            if batch_size > 1 and n < batch_size:
                raise ValueError(
                    f"{phase} '{modality}' has {n} samples but "
                    f"batch_size={batch_size}. "
                    "Increase the data limit or reduce the batch size."
                )
