# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.examples.qualcomm.oss_scripts.llama.dataset.builders import (
    DatasetBuilder,
    DecoderDatasetBuilder,
    EncoderDatasetBuilder,
)
from executorch.examples.qualcomm.oss_scripts.llama.dataset.collators import (
    LLMCalibCollator,
    LLMTrainingCollator,
)
from executorch.examples.qualcomm.oss_scripts.llama.dataset.config import DataConfig
from executorch.examples.qualcomm.oss_scripts.llama.dataset.constants import (
    LABEL_IGNORE_INDEX,
)
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
    preprocess_encoder_inputs,
)
from executorch.examples.qualcomm.oss_scripts.llama.dataset.schema import MessageSample
from executorch.examples.qualcomm.oss_scripts.llama.dataset.targets import (
    make_causal_labels,
    make_conversation_labels,
)

__all__ = [
    # config
    "DataConfig",
    # schema
    "MessageSample",
    # constants
    "LABEL_IGNORE_INDEX",
    # loaders
    "collect_lm_eval_tokens",
    "load_audio_file",
    "load_conversation_samples",
    "load_hf_chat_dataset",
    # targets (label generation — low-level functions)
    "make_causal_labels",
    "make_conversation_labels",
    # datasets
    "LLMDataset",
    "ModalityEncoderDataset",
    # collators
    "ModalityEncoderCollator",
    "LLMCalibCollator",
    "LLMTrainingCollator",
    # builders
    "DatasetBuilder",
    "DecoderDatasetBuilder",
    "EncoderDatasetBuilder",
    # preprocessors
    "ModalityPreprocessor",
    "preprocess_encoder_inputs",
]
