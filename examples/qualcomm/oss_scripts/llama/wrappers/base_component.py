# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import logging

import math
import time

from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Tuple

import torch
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchBackendType,
)
from executorch.backends.qualcomm.utils.utils import (
    get_sdk_build_id,
    is_qnn_sdk_version_less_than,
)
from executorch.examples.qualcomm.oss_scripts.llama import LLMModelConfig
from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
    AUDIO_ENCODER,
    VISION_ENCODER,
)
from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import ModelArgs
from executorch.examples.qualcomm.oss_scripts.llama.static_llm_quant_recipe import (
    StaticLLMQuantRecipe,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from transformers import AutoConfig


class Mode(Enum):
    PREFILL = 1
    DECODE = 2


def log_info(func):
    class TimeIt:
        def __init__(self, event):
            self.event = event

        def __enter__(self):
            self.start = time.time()
            return self

        def __exit__(self, type, value, traceback):
            self.time = time.time() - self.start
            logging.info(f"{self.event}{self.time}s")

    @wraps(func)
    def wrapper(cls, *args, **kwargs):
        func_name = f"{cls.__class__.__name__}::{func.__name__}"
        logging.info(f"calling {func_name}")
        with TimeIt(f"{func_name} completed in "):
            func(cls, *args, **kwargs)

    return wrapper


def next_power_of_two(n):
    return 1 if n == 0 else 2 ** math.ceil(math.log2(n))


def process_model_args(
    control_args: argparse.Namespace,
    model_args: ModelArgs,
    quant_recipe: StaticLLMQuantRecipe,
    config: LLMModelConfig,
    mode: Mode,
):
    """
    Based on the mode and arguments, set the appropriate model args for compilation.
    Args:
        control_args: Arguments from command line.
        model_args: ModelArgs object to be modified.
        quant_recipe: Quantization recipe to be used.
        config: LLMModelConfig object to be used.
        mode: Mode of operation (PREFILL or DECODE).
    """
    # TODO: support batch inputs if necessary
    if mode == Mode.DECODE:
        ar_len = (
            # To get better performance, we round up to the nearest power of 2.
            next_power_of_two(
                (control_args.window + control_args.gcap) * (control_args.ngram - 1)
            )
            if control_args.model_mode == "lookahead"
            else 1
        )
    else:
        ar_len = control_args.prefill_ar_len

    model_args.max_batch_size = 1
    model_args.max_seq_len = control_args.max_seq_len
    model_args.max_context_len = control_args.max_context_len
    model_args.use_kv_cache = control_args.max_context_len != ar_len
    model_args.enable_r3 = config.r3
    model_args.ar_len = ar_len
    model_args.kv_io_bit_width = quant_recipe.get_kv_io_bit_width()

    if config.masked_softmax:
        if is_qnn_sdk_version_less_than("2.35"):
            logging.warning(
                f"Masked softmax is supported after QNN SDK 2.35. Given sdk version {get_sdk_build_id()}"
                " is lower the target version. Disabling the feature."
            )
            model_args.enable_masked_softmax = False
        else:
            model_args.enable_masked_softmax = True

    return model_args


def get_model_specific_kwargs(control_args: argparse.Namespace, config: LLMModelConfig):
    """
    Retrieve model-specific config required for Static LLaMA.
    This method handles architecture-specific requirements for both Vision-Language Models (VLMs)
    and Language-only Models (LLMs), extracting necessary config from HuggingFace configs.

    """
    kwargs = {}
    # For multimodal models, we need the special token ID that represents modality placeholders
    # in the input sequence. This token is used to mark positions where modality embeddings
    # should be inserted during inference.
    if hasattr(config, AUDIO_ENCODER):
        hf_config = AutoConfig.from_pretrained(config.repo_id)
        kwargs["audio_token_id"] = hf_config.audio_token_index
    if hasattr(config, VISION_ENCODER):
        hf_config = AutoConfig.from_pretrained(config.repo_id)
        kwargs["image_token_id"] = hf_config.image_token_id
    return kwargs


class Processor:
    _next_handler = None

    def set_next(self, processor) -> Processor:
        self._next_handler = processor
        return processor

    def process(self, request: Any):
        if self._next_handler:
            return self._next_handler.process(request)


@dataclass
class Request:
    @dataclass
    class CalibrationData:
        datasets: List[Tuple[torch.Tensor]] = None
        intermediate_outputs: List[Tuple[torch.Tensor]] = None
        qdq_intermediate_outputs: List[Tuple[torch.Tensor]] = None

    @dataclass
    class Data:
        compile_spec: List[CompileSpec] = None
        pte_filename: str = None
        custom_annotation: Any = ()
        calibration_data: Request.CalibrationData = None
        tokenizer: callable = None
        skip_quantize: bool = False
        backend: QnnExecuTorchBackendType = QnnExecuTorchBackendType.kHtpBackend
        soc_model: str = "SM8750"

    method_name: str
    method_data: Dict[str, Data]


class Component(Processor):
    def process(self, request: Request) -> Request:
        getattr(self, request.method_name)(request)
        super().process(request)

    def compile(self, request: Request):
        return

    def quantize(self, request: Request):
        return
