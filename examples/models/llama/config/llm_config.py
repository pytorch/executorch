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

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BaseConfig:
    """
    These are specific to the specific model, e.g. whether it’s Qwen3 0.6B or Phi-4-mini.
    for each of these different models, you can expect each of these fields to change.
    """
    model_class: str = "llama"
    params: Optional[str] = None
    checkpoint: Optional[str] = None
    checkpoint_dir: Optional[str] = None  # For sharded checkpoint
    tokenizer_path: Optional[str] = None
    metadata: Optional[str] = None
    fairseq2: bool = False  # For legacy internal use cases


@dataclass
class ModelConfig:
    """
    These are not necessarily specific to the model, but are needed to finish off
    the rest of the model configuration in eager. You can think of these like
    optimizations / actual configurations. The same ModelConfig can be applied
    to different models.
    """
    dtype_override: str = "fp32"
    enable_dynamic_shape: bool = True
    use_shared_embedding: bool = False
    use_lora: bool = False
    use_sdpa_with_kv_cache: bool = False
    expand_rope_table: bool = False
    output_prune_map: Optional[str] = None
    input_prune_map: Optional[str] = None

    # Below are config options relating to kv cache.
    use_kv_cache: Optional[bool] = None
    quantize_kv_cache: Optional[bool] = None
    local_global_attention: List[int] = None


@dataclass
class ExportConfig:
    max_seq_length: Optional[int] = None
    max_context_length: Optional[int] = None
    output_dir: Optional[str] = None
    output_name: Optional[str] = None
    so_library: Optional[str] = None
    export_only: Optional[bool] = None


@dataclass
<<<<<<< HEAD
=======
class KVCacheConfig:
    use_kv_cache: Optional[bool] = None
    quantize_kv_cache: Optional[bool] = None
    local_global_attention: List[int] = None
    # ...potentially more in the future such as cache eviction strategy


@dataclass
>>>>>>> ec85c4be2 (Add new export LLM config)
class DebugConfig:
    profile_memory: bool = False
    profile_path: Optional[str] = None
    generate_etrecord: bool = False
    generate_full_logits: bool = False
    verbose: bool = False  # Would be good to remove this from the config eventually


########################################################################
#### The below config can eventually be replaced by export recipes #####
########################################################################


@dataclass
class QuantizationConfig:
    qmode: Optional[str] = None
    embedding_quantize: Optional[bool] = None
    pt2e_quantize: Optional[bool] = None
    group_size: Optional[int] = None
    use_spin_quant: Optional[bool] = None
    use_qat: Optional[bool] = None
    preq_mode: Optional[str] = None
    preq_group_size: Optional[int] = None
    preq_embedding_quantize: Optional[bool] = None
    calibration_tasks: Optional[str] = None
    calibration_limit: Optional[int] = None
    calibration_seq_length: Optional[int] = None
    calibration_data: Optional[str] = None


@dataclass
class XNNPackConfig:
    enabled: Optional[bool] = None
    extended_ops: Optional[bool] = None


@dataclass
class CoreMLConfig:  # coreML recipe?
    enabled: Optional[bool] = None
    enable_state: Optional[bool] = None
    preserve_sdpa: Optional[bool] = None
    quantize: Optional[bool] = None
    ios: Optional[bool] = None
    compute_units: Optional[str] = None


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
    enabled: Optional[bool] = False


@dataclass
class BackendConfig:
    xnnpack: XNNPackConfig = field(default_factory=XNNPackConfig)
    coreml: CoreMLConfig = field(default_factory=CoreMLConfig)
    vulkan: VulkanConfig = field(default_factory=VulkanConfig)
    qnn: QNNConfig = field(default_factory=QNNConfig)
    mps: MPSConfig = field(default_factory=MPSConfig)


@dataclass
class LlmConfig:
    base: BaseConfig = field(default_factory=BaseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
