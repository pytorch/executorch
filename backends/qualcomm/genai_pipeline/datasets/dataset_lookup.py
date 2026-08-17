# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dataset loader, collector, and purpose-adapter lookup."""

from __future__ import annotations


def get_dataset_adapter(dataset_options, is_multimodal=False):
    """Build purpose-agnostic dataset loaders selected by ``dataset_options``.

    Each loader exposes ``load_dataset(tokenizer, extra_options)`` returning a
    component-keyed dataset dict. The purpose (calibration / training / eval)
    lives in the purpose adapters, not here. Multimodal message samples also
    need the model config to build encoder datasets, so it is bound into that
    loader here instead of being routed through the quantization strategy.
    """
    if dataset_options is None:
        return []

    loaders = []
    if is_multimodal:
        from executorch.backends.qualcomm.genai_pipeline.datasets.loaders.mllm.message_sample_adapter import (
            MessageSampleAdapter as MLLMMessageSampleAdapter,
        )

        if dataset_options.calib_samples:
            loaders.append(
                MLLMMessageSampleAdapter(
                    dataset_options.calib_samples,
                    llm_config=dataset_options.llm_config,
                )
            )
        return loaders

    from executorch.backends.qualcomm.genai_pipeline.datasets.loaders.llm.hf_dataset_adapter import (
        HFDatasetAdapter,
    )
    from executorch.backends.qualcomm.genai_pipeline.datasets.loaders.llm.lm_eval_adapter import (
        LMEvalAdapter,
    )
    from executorch.backends.qualcomm.genai_pipeline.datasets.loaders.llm.message_sample_adapter import (
        MessageSampleAdapter,
    )

    if dataset_options.calib_tasks:
        loaders.append(
            LMEvalAdapter(
                dataset_options.calib_tasks,
                dataset_options.calib_limit,
                dataset_options.calib_num_fewshot,
            )
        )
    if dataset_options.calib_samples:
        loaders.append(MessageSampleAdapter(dataset_options.calib_samples))
    if dataset_options.calib_hf_dataset:
        loaders.append(
            HFDatasetAdapter(
                dataset_options.calib_hf_dataset,
                dataset_options.calib_hf_limit,
            )
        )
    return loaders


def get_collector(is_multimodal=False):
    """Return the component-aware collator provider for the model modality."""
    if is_multimodal:
        from executorch.backends.qualcomm.genai_pipeline.datasets.collators.mllm_collator import (
            MLLMDatasetCollector,
        )

        return MLLMDatasetCollector()

    from executorch.backends.qualcomm.genai_pipeline.datasets.collators.llm_collator import (
        LLMDatasetCollector,
    )

    return LLMDatasetCollector()


def get_calibration_dataset_adapter(dataset_options, is_multimodal=False):
    """Assemble the calibration purpose adapter.

    Selects source loaders from ``dataset_options``. With no source, returns
    the modality-agnostic ``DefaultCalibrationDataAdapter`` (random pipeline
    sanity data, sized from the options). With sources, wires them and the
    modality's collector through :class:`CalibrationDatasetBuilder`.
    """
    loaders = get_dataset_adapter(dataset_options, is_multimodal)

    if not loaders:
        from executorch.backends.qualcomm.genai_pipeline.datasets.calibration.default_calibration_data_adapter import (
            DefaultCalibrationDataAdapter,
        )

        kwargs = {}
        if dataset_options is not None:
            if dataset_options.num_samples is not None:
                kwargs["num_samples"] = dataset_options.num_samples
            kwargs["batch_size"] = dataset_options.batch_size
            if dataset_options.seed is not None:
                kwargs["seed"] = dataset_options.seed
        return DefaultCalibrationDataAdapter(**kwargs)

    builder = CalibrationDatasetBuilder(is_multimodal)
    for loader in loaders:
        builder.with_dataset(loader)
    builder.with_collector(get_collector(is_multimodal))
    builder.with_context_len(dataset_options.max_context_len)
    builder.with_batch_size(dataset_options.batch_size)
    return builder.build()


class CalibrationDatasetBuilder:
    """Assembler for a corpus-backed calibration adapter.

    The axes are modality (fixed at construction), data sources
    (:meth:`with_dataset`), collator (:meth:`with_collector`), and shaping
    (:meth:`with_context_len` / :meth:`with_batch_size`). Modality is decided
    once and reused for adapter selection. :meth:`build` wires the modality's
    corpus adapter with the accumulated sources, the supplied collector, and
    the shaping; the random default is not this builder's concern (see
    ``get_calibration_dataset_adapter``).

    Example usage:

        adapter = (
            CalibrationDatasetBuilder(is_multimodal)
            .with_dataset(lm_eval_loader)
            .with_dataset(message_sample_loader)
            .with_collector(collector)
            .with_context_len(1024)
            .with_batch_size(1)
            .build()
        )

    Args:
        is_multimodal: Whether the model is multimodal (VLM/ALM). Selects the
            MLLM adapter instead of the LLM one.
    """

    def __init__(self, is_multimodal: bool = False) -> None:
        self._is_multimodal = is_multimodal
        self._dataset_adapters: list = []
        self._collector = None
        self._max_context_len = None
        self._batch_size = None

    def with_dataset(self, dataset_adapter) -> "CalibrationDatasetBuilder":
        """Accumulate one data-source loader. May be called more than once."""
        self._dataset_adapters.append(dataset_adapter)
        return self

    def with_collector(self, collector) -> "CalibrationDatasetBuilder":
        """Set the collator provider."""
        self._collector = collector
        return self

    def with_context_len(self, max_context_len) -> "CalibrationDatasetBuilder":
        """Set the sequence length for encoding and padding."""
        self._max_context_len = max_context_len
        return self

    def with_batch_size(self, batch_size) -> "CalibrationDatasetBuilder":
        """Set the text-decoder DataLoader batch size."""
        self._batch_size = batch_size
        return self

    def build(self):
        """Build and validate the calibration adapter.

        Returns:
            An ``LLMCalibrationDataAdapter`` or ``MLLMCalibrationDataAdapter``.

        Raises:
            ValueError: If required fields are missing.
        """
        missing = []
        if not self._dataset_adapters:
            missing.append("dataset sources (use .with_dataset())")
        if self._collector is None:
            missing.append("collector (use .with_collector())")
        if self._max_context_len is None:
            missing.append("max_context_len (use .with_context_len())")

        if missing:
            raise ValueError(
                f"Cannot build calibration adapter, missing required fields: "
                f"{', '.join(missing)}"
            )

        kwargs = {
            "dataset_loaders": self._dataset_adapters,
            "collector": self._collector,
            "max_context_len": self._max_context_len,
        }
        if self._batch_size is not None:
            kwargs["batch_size"] = self._batch_size

        if self._is_multimodal:
            from executorch.backends.qualcomm.genai_pipeline.datasets.calibration.mllm_calibration_data_adapter import (
                MLLMCalibrationDataAdapter,
            )

            return MLLMCalibrationDataAdapter(**kwargs)

        from executorch.backends.qualcomm.genai_pipeline.datasets.calibration.llm_calibration_data_adapter import (
            LLMCalibrationDataAdapter,
        )

        return LLMCalibrationDataAdapter(**kwargs)


def get_training_dataset_adapter(dataset_options, is_multimodal=False):
    """Assemble the training purpose adapter."""
    from executorch.backends.qualcomm.genai_pipeline.datasets.training.default_training_data_adapter import (
        DefaultTrainingDataAdapter,
    )

    return DefaultTrainingDataAdapter(
        dataset_loaders=get_dataset_adapter(dataset_options, is_multimodal),
        collector=get_collector(is_multimodal),
    )


def get_eval_dataset_adapter(dataset_options, is_multimodal=False):
    """Assemble the evaluation purpose adapter."""
    from executorch.backends.qualcomm.genai_pipeline.datasets.evaluation.default_evaluation_data_adapter import (
        DefaultEvaluationDataAdapter,
    )

    return DefaultEvaluationDataAdapter(
        dataset_loaders=get_dataset_adapter(dataset_options, is_multimodal),
        collector=get_collector(is_multimodal),
    )
