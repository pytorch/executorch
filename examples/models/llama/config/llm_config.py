# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Configurations for exporting Llama.

Uses dataclases, which integrate with OmegaConf and Hydra.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


################################################################################
################################## BaseConfig ##################################
################################################################################


class ModelType(str, Enum):
    STORIES110M = "stories110m"
    LLAMA2 = "llama2"
    LLAMA3 = "llama3"
    LLAMA3_1 = "llama3_1"
    LLAMA3_2 = "llama3_2"
    LLAMA3_2_VISION = "llama3_2_vision"
    STATIC_LLAMA = "static_llama"
    QWEN2_5 = "qwen2_5"
    QWEN3_0_6B = "qwen3-0_6b"
    QWEN3_1_7B = "qwen3-1_7b"
    QWEN3_4B = "qwen3-4b"
    PHI_4_MINI = "phi_4_mini"
    SMOLLM2 = "smollm2"


class PreqMode(str, Enum):
    PREQ_8DA4W = "8da4w"
    PREQ_8DA4W_OUT_8DA8W = "8da4w_output_8da8w"


@dataclass
class BaseConfig:
    """
    These are specific to the specific model, e.g. whether it’s Qwen3 0.6B or Phi-4-mini.
    For each of these different models, you can expect each of these fields to change.
    """

    model_class: ModelType = ModelType.LLAMA3
    params: Optional[str] = None
    checkpoint: Optional[str] = None
    checkpoint_dir: Optional[str] = None  # For sharded checkpoint.
    tokenizer_path: Optional[str] = None
    metadata: Optional[str] = None
    use_lora: bool = False
    fairseq2: bool = False  # For legacy internal use cases.

    # Legacy pre-quantization options that happen during model weight loading.
    preq_mode: Optional[PreqMode] = None
    preq_group_size: int = 32
    preq_embedding_quantize: str = "8,0"


################################################################################
################################# ModelConfig ##################################
################################################################################


class DtypeOverride(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


@dataclass
class ModelConfig:
    """
    These are not necessarily specific to the model, but are needed to finish off
    the rest of the model configuration in eager. You can think of these like
    optimizations / actual configurations. The same ModelConfig can be applied
    to different models.
    """

    dtype_override: DtypeOverride = DtypeOverride.FP32
    enable_dynamic_shape: bool = True
    use_shared_embedding: bool = False
    use_sdpa_with_kv_cache: bool = False
    expand_rope_table: bool = False
    use_attention_sink: Optional[str] = None
    output_prune_map: Optional[str] = None
    input_prune_map: Optional[str] = None

    # Below are config options relating to kv cache.
    use_kv_cache: bool = False
    quantize_kv_cache: bool = False
    local_global_attention: Optional[List[int]] = None


################################################################################
################################ ExportConfig ##################################
################################################################################


@dataclass
class ExportConfig:
    max_seq_length: int = 128
    max_context_length: int = 128
    output_dir: Optional[str] = None
    output_name: Optional[str] = None
    so_library: Optional[str] = None
    export_only: bool = False


################################################################################
################################# DebugConfig ##################################
################################################################################


@dataclass
class DebugConfig:
    profile_memory: bool = False
    profile_path: Optional[str] = None
    generate_etrecord: bool = False
    generate_full_logits: bool = False
    verbose: bool = False


################################################################################
############################# QuantizationConfig ###############################
################################################################################


class Pt2eQuantize(str, Enum):
    XNNPACK_DYNAMIC = "xnnpack_dynamic"
    XNNPACK_DYNAMIC_QC4 = "xnnpack_dynamic_qc4"
    QNN_8A8W = "qnn_8a8w"
    QNN_16A16W = "qnn_16a16w"
    QNN_16A4W = "qnn_16a4w"
    COREML_C4W = "coreml_c4w"
    COREML_8A_C8W = "coreml_8a_c8w"
    COREML_8A_C4W = "coreml_8a_c4w"
    COREML_BASELINE_8A_C8W = "coreml_baseline_8a_c8w"
    COREML_BASELINE_8A_C4W = "coreml_baseline_8a_c4w"
    VULKAN_8W = "vulkan_8w"


class SpinQuant(str, Enum):
    CUDA = "cuda"
    NATIVE = "native"


@dataclass
class QuantizationConfig:
    qmode: Optional[str] = None
    embedding_quantize: Optional[str] = None
    pt2e_quantize: Optional[Pt2eQuantize] = None
    group_size: Optional[int] = None
    use_spin_quant: Optional[SpinQuant] = None
    use_qat: bool = False
    calibration_tasks: Optional[List[str]] = None
    calibration_limit: Optional[int] = None
    calibration_seq_length: Optional[int] = None
    calibration_data: str = "Once upon a time"

    def __post_init__(self):
        if self.qmode:
            self._validate_qmode()

    def _validate_qmode(self) -> None:
        choices = ["int8", "8da4w", "8da4w-gptq", "vulkan_4w"]
        patterns = [r"torchao:8da(\d+)w", r"torchao:fpa(\d+)w"]

        if self.qmode in choices:
            return

        for pattern in patterns:
            matches = re.findall(pattern, self.qmode)
            if len(matches) == 1:
                return

        raise ValueError(
            f"Got qmode {self.qmode}, but expected one of {choices}, or one of the regex patterns {patterns}."
        )


################################################################################
############################### BackendConfig ##################################
################################################################################


@dataclass
class XNNPackConfig:
    enabled: bool = False
    extended_ops: bool = False


class CoreMLQuantize(str, Enum):
    B4W = "b4w"
    C4W = "c4w"


class CoreMLComputeUnit(str, Enum):
    CPU_ONLY = "cpu_only"
    CPU_AND_GPU = "cpu_and_gpu"
    CPU_AND_NE = "cpu_and_ne"
    ALL = "all"


@dataclass
class CoreMLConfig:
    enabled: bool = False
    enable_state: bool = False
    preserve_sdpa: bool = False
    quantize: Optional[CoreMLQuantize] = None
    ios: int = 15
    compute_units: CoreMLComputeUnit = CoreMLComputeUnit.CPU_ONLY

    def __post_init__(self):
        if self.ios not in (15, 16, 17, 18):
            raise ValueError(f"Invalid coreml ios version: {self.ios}")


@dataclass
class VulkanConfig:
    enabled: bool = False


@dataclass
class QNNConfig:
    enabled: bool = False
    use_sha: bool = False
    soc_model: str = "SM8650"
    use_qnn_sha: bool = False
    optimized_rotation_path: Optional[str] = None
    num_sharding: int = 0


@dataclass
class MPSConfig:
    enabled: bool = False


@dataclass
class BackendConfig:
    xnnpack: XNNPackConfig = field(default_factory=XNNPackConfig)
    coreml: CoreMLConfig = field(default_factory=CoreMLConfig)
    vulkan: VulkanConfig = field(default_factory=VulkanConfig)
    qnn: QNNConfig = field(default_factory=QNNConfig)
    mps: MPSConfig = field(default_factory=MPSConfig)


################################################################################
################################## LlmConfig ###################################
################################################################################


@dataclass
class LlmConfig:
    base: BaseConfig = field(default_factory=BaseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
