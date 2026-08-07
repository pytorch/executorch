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
    LLMTrainingCollator,
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
    load_hf_chat_dataset,
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
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
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

    def from_hf_source(
        self,
        dataset_name: str,
        num_samples: int,
    ) -> LLMDataset:
        """Build an LLMDataset from a HuggingFace dataset.

        Populates assistant_masks so training computes loss only on assistant-response
        tokens (assistant-only loss), ignoring user and system turns. Plain-text corpora
        with no assistant turns yield all-zero masks, which fall back to full causal
        next-token labels at collation time.
        """
        sequences, assistant_masks = load_hf_chat_dataset(
            dataset_name,
            self._tokenizer_wrapper,
            self._max_context_len,
            num_samples=num_samples,
        )
        return LLMDataset(sequences, assistant_masks=assistant_masks)

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
    """Orchestrates DecoderDatasetBuilder and EncoderDatasetBuilder for calibration or training."""

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
        # Collator for QAT training: pads tokens, builds masks, and labels.
        self._train_collator = LLMTrainingCollator(
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
                    # build from huggingface source
                    (
                        self._text_decoder_dataset_builder.from_hf_source(
                            cfg.calib_hf_dataset, cfg.calib_hf_limit
                        )
                        if cfg.calib_hf_dataset is not None
                        else None
                    ),
                ],
            )
        )
        if not decoder_datasets:
            raise ValueError(
                "No calibration data specified. Provide at least one of: "
                "--calib_tasks, --calib_samples, --calib_hf_dataset."
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

    def build_qat_dataloaders(self) -> Tuple[
        Dict[str, Optional[DataLoader]],
        Dict[str, Optional[DataLoader]],
        Dict[str, Optional[DataLoader]],
    ]:
        """Build calib/train/val DataLoaders for QAT.

        Full split: --qat_full_tasks / --qat_full_hf_dataset provide a single
        pool that is automatically split into calib/train/val by --calib_train_ratio / --train_val_ratio.

        Explicit split: --calib_tasks provides the calibration set and
        --train_tasks / --train_hf_dataset provide the training set. An optional
        val split is cut from the training pool via --train_val_ratio.

        Returns three modality-keyed dicts: (calib_loaders, train_loaders, val_loaders).
        Only TEXT_DECODER carries non-None loaders.
        """
        if self._is_multimodal:
            raise NotImplementedError(
                "QAT is not supported for multimodal models yet. "
                "build_qat_dataloaders operates on the text decoder only."
            )

        cfg = self._data_config

        if cfg.qat_mode == "full":
            full_datasets: List[Dataset] = list(
                filter(
                    None,
                    [
                        # build from lm tasks
                        (
                            self._text_decoder_dataset_builder.from_lm_eval(
                                cfg.qat_full_tasks, cfg.qat_full_limit
                            )
                            if cfg.qat_full_tasks is not None
                            else None
                        ),
                        # build from huggingface source
                        (
                            self._text_decoder_dataset_builder.from_hf_source(
                                cfg.qat_full_hf_dataset,
                                cfg.qat_full_hf_limit,
                            )
                            if cfg.qat_full_hf_dataset is not None
                            else None
                        ),
                    ],
                )
            )
            full_dataset = (
                ConcatDataset(full_datasets)
                if len(full_datasets) > 1
                else full_datasets[0]
            )
            self._log_dataset_stats(
                {TEXT_DECODER: full_dataset}, cfg.batch_size, phase="qat-full"
            )
            calib_loader, train_loader, val_loader = self._split_qat_dataset(
                full_dataset,
                calib_train_ratio=cfg.calib_train_ratio,
                train_val_ratio=cfg.train_val_ratio,
                batch_size=cfg.batch_size,
                seed=cfg.seed,
                calib_collator=self._calib_collator,
                train_collator=self._train_collator,
            )
        else:
            # Explicit split calibration dataset and training dataset
            calib_datasets: List[Dataset] = list(
                filter(
                    None,
                    [
                        # build from samples
                        (
                            self._text_decoder_dataset_builder.from_conversation(
                                load_conversation_samples(cfg.calib_samples)
                            )
                            if cfg.calib_samples
                            else None
                        ),
                        # build from lm tasks
                        (
                            self._text_decoder_dataset_builder.from_lm_eval(
                                cfg.calib_tasks, cfg.calib_limit
                            )
                            if cfg.calib_tasks is not None
                            else None
                        ),
                        # build from huggingface source
                        (
                            self._text_decoder_dataset_builder.from_hf_source(
                                cfg.calib_hf_dataset, cfg.calib_hf_limit
                            )
                            if cfg.calib_hf_dataset is not None
                            else None
                        ),
                    ],
                )
            )
            train_datasets: List[Dataset] = list(
                filter(
                    None,
                    [
                        # build from lm tasks
                        (
                            self._text_decoder_dataset_builder.from_lm_eval(
                                cfg.train_tasks, cfg.train_limit
                            )
                            if cfg.train_tasks is not None
                            else None
                        ),
                        # build from huggingface source
                        (
                            self._text_decoder_dataset_builder.from_hf_source(
                                cfg.train_hf_dataset,
                                cfg.train_hf_limit,
                            )
                            if cfg.train_hf_dataset is not None
                            else None
                        ),
                    ],
                )
            )
            calib_full = (
                ConcatDataset(calib_datasets)
                if len(calib_datasets) > 1
                else calib_datasets[0]
            )
            train_full = (
                ConcatDataset(train_datasets)
                if len(train_datasets) > 1
                else train_datasets[0]
            )
            self._log_dataset_stats(
                {TEXT_DECODER: calib_full}, cfg.batch_size, phase="qat-calibration"
            )
            self._log_dataset_stats(
                {TEXT_DECODER: train_full}, cfg.batch_size, phase="qat-training"
            )
            calib_loader = DataLoader(
                calib_full,
                batch_size=cfg.batch_size,
                shuffle=False,
                drop_last=cfg.batch_size > 1,
                collate_fn=self._calib_collator,
            )
            train_loader, val_loader = self._split_train_val(
                train_full,
                train_val_ratio=cfg.train_val_ratio,
                batch_size=cfg.batch_size,
                seed=cfg.seed,
                train_collator=self._train_collator,
            )

        return (
            dict.fromkeys(_ALL_MODALITY_KEYS) | {TEXT_DECODER: calib_loader},
            dict.fromkeys(_ALL_MODALITY_KEYS) | {TEXT_DECODER: train_loader},
            dict.fromkeys(_ALL_MODALITY_KEYS) | {TEXT_DECODER: val_loader},
        )

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
    def _split_train_val(
        dataset: Dataset,
        train_val_ratio: float,
        batch_size: int,
        seed: int,
        train_collator,
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        n_total = len(dataset)
        n_train = max(1, int(n_total * train_val_ratio))
        n_val = n_total - n_train

        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(n_total, generator=generator).tolist()

        train_loader = DataLoader(
            Subset(dataset, indices[:n_train]),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=train_collator,
        )
        val_loader: Optional[DataLoader] = (
            DataLoader(
                Subset(dataset, indices[n_train:]),
                batch_size=batch_size,
                shuffle=False,
                drop_last=True,
                collate_fn=train_collator,
            )
            if n_val > 0
            else None
        )
        logging.info(
            "QAT train/val split (seed=%d): total=%d  train=%d  val=%d",
            seed,
            n_total,
            n_train,
            n_val,
        )
        return train_loader, val_loader

    @staticmethod
    def _split_qat_dataset(
        dataset: Dataset,
        calib_train_ratio: float,
        train_val_ratio: float,
        batch_size: int,
        seed: int,
        calib_collator,
        train_collator,
    ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        n_total = len(dataset)
        n_calib = max(1, int(n_total * calib_train_ratio))
        n_remaining = n_total - n_calib
        n_train = max(1, int(n_remaining * train_val_ratio))
        n_val = n_remaining - n_train

        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(n_total, generator=generator).tolist()

        calib_idx = indices[:n_calib]
        train_idx = indices[n_calib : n_calib + n_train]
        val_idx = indices[n_calib + n_train : n_calib + n_train + n_val]

        calib_loader = DataLoader(
            Subset(dataset, calib_idx),
            batch_size=batch_size,
            shuffle=False,
            drop_last=batch_size > 1,
            collate_fn=calib_collator,
        )
        # drop_last=True: the exported graph module has static input shapes, so every batch
        # must have exactly batch_size samples. Incomplete trailing batches are dropped.
        train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=train_collator,
        )
        val_loader: Optional[DataLoader] = (
            DataLoader(
                Subset(dataset, val_idx),
                batch_size=batch_size,
                shuffle=False,
                drop_last=True,
                collate_fn=train_collator,
            )
            if n_val > 0
            else None
        )
        logging.info(
            "QAT splits (seed=%d): total=%d  calib=%d (%.0f%%)  train=%d  val=%d",
            seed,
            n_total,
            n_calib,
            calib_train_ratio * 100,
            n_train,
            n_val,
        )
        return calib_loader, train_loader, val_loader

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
