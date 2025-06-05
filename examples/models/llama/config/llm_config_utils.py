# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse

from executorch.examples.models.llama.config.llm_config import (
    CoreMLComputeUnit,
    CoreMLQuantize,
    DtypeOverride,
    LlmConfig,
    ModelType,
    PreqMode,
    Pt2eQuantize,
    SpinQuant,
)


def convert_args_to_llm_config(args: argparse.Namespace) -> LlmConfig:
    """
    To support legacy purposes, this function converts CLI args from
    argparse to an LlmConfig, which is used by the LLM export process.
    """
    llm_config = LlmConfig()

    # BaseConfig
    if hasattr(args, "model"):
        llm_config.base.model_class = ModelType(args.model)
    if hasattr(args, "params"):
        llm_config.base.params = args.params
    if hasattr(args, "checkpoint"):
        llm_config.base.checkpoint = args.checkpoint
    if hasattr(args, "checkpoint_dir"):
        llm_config.base.checkpoint_dir = args.checkpoint_dir
    if hasattr(args, "tokenizer_path"):
        llm_config.base.tokenizer_path = args.tokenizer_path
    if hasattr(args, "metadata"):
        llm_config.base.metadata = args.metadata
    if hasattr(args, "use_lora"):
        llm_config.base.use_lora = args.use_lora
    if hasattr(args, "fairseq2"):
        llm_config.base.fairseq2 = args.fairseq2

    # PreqMode settings
    if hasattr(args, "preq_mode") and args.preq_mode:
        llm_config.base.preq_mode = PreqMode(args.preq_mode)
        if hasattr(args, "preq_group_size"):
            llm_config.base.preq_group_size = args.preq_group_size
        if hasattr(args, "preq_embedding_quantize"):
            llm_config.base.preq_embedding_quantize = args.preq_embedding_quantize

    # ModelConfig
    if hasattr(args, "dtype_override"):
        llm_config.model.dtype_override = DtypeOverride(args.dtype_override)
    if hasattr(args, "enable_dynamic_shape"):
        llm_config.model.enable_dynamic_shape = args.enable_dynamic_shape
    if hasattr(args, "use_shared_embedding"):
        llm_config.model.use_shared_embedding = args.use_shared_embedding
    if hasattr(args, "use_sdpa_with_kv_cache"):
        llm_config.model.use_sdpa_with_kv_cache = args.use_sdpa_with_kv_cache
    if hasattr(args, "expand_rope_table"):
        llm_config.model.expand_rope_table = args.expand_rope_table
    if hasattr(args, "use_attention_sink"):
        llm_config.model.use_attention_sink = args.use_attention_sink
    if hasattr(args, "output_prune_map"):
        llm_config.model.output_prune_map = args.output_prune_map
    if hasattr(args, "input_prune_map"):
        llm_config.model.input_prune_map = args.input_prune_map
    if hasattr(args, "use_kv_cache"):
        llm_config.model.use_kv_cache = args.use_kv_cache
    if hasattr(args, "quantize_kv_cache"):
        llm_config.model.quantize_kv_cache = args.quantize_kv_cache
    if hasattr(args, "local_global_attention"):
        llm_config.model.local_global_attention = args.local_global_attention

    # ExportConfig
    if hasattr(args, "max_seq_length"):
        llm_config.export.max_seq_length = args.max_seq_length
    if hasattr(args, "max_context_length"):
        llm_config.export.max_context_length = args.max_context_length
    if hasattr(args, "output_dir"):
        llm_config.export.output_dir = args.output_dir
    if hasattr(args, "output_name"):
        llm_config.export.output_name = args.output_name
    if hasattr(args, "so_library"):
        llm_config.export.so_library = args.so_library
    if hasattr(args, "export_only"):
        llm_config.export.export_only = args.export_only

    # QuantizationConfig
    if hasattr(args, "quantization_mode"):
        llm_config.quantization.qmode = args.quantization_mode
    if hasattr(args, "embedding_quantize"):
        llm_config.quantization.embedding_quantize = args.embedding_quantize
    if hasattr(args, "pt2e_quantize") and args.pt2e_quantize:
        llm_config.quantization.pt2e_quantize = Pt2eQuantize(args.pt2e_quantize)
    if hasattr(args, "group_size"):
        llm_config.quantization.group_size = args.group_size
    if hasattr(args, "use_spin_quant") and args.use_spin_quant:
        llm_config.quantization.use_spin_quant = SpinQuant(args.use_spin_quant)
    if hasattr(args, "use_qat"):
        llm_config.quantization.use_qat = args.use_qat
    if hasattr(args, "calibration_tasks"):
        llm_config.quantization.calibration_tasks = args.calibration_tasks
    if hasattr(args, "calibration_limit"):
        llm_config.quantization.calibration_limit = args.calibration_limit
    if hasattr(args, "calibration_seq_length"):
        llm_config.quantization.calibration_seq_length = args.calibration_seq_length
    if hasattr(args, "calibration_data"):
        llm_config.quantization.calibration_data = args.calibration_data

    # BackendConfig - XNNPack
    if hasattr(args, "xnnpack"):
        llm_config.backend.xnnpack.enabled = args.xnnpack
    if hasattr(args, "xnnpack_extended_ops"):
        llm_config.backend.xnnpack.extended_ops = args.xnnpack_extended_ops

    # CoreML
    if hasattr(args, "coreml"):
        llm_config.backend.coreml.enabled = args.coreml
    llm_config.backend.coreml.enable_state = getattr(args, "coreml_enable_state", False)
    llm_config.backend.coreml.preserve_sdpa = getattr(args, "coreml_preserve_sdpa", False)
    if hasattr(args, "coreml_quantize") and args.coreml_quantize:
        llm_config.backend.coreml.quantize = CoreMLQuantize(args.coreml_quantize)
    if hasattr(args, "coreml_ios"):
        llm_config.backend.coreml.ios = args.coreml_ios
    if hasattr(args, "coreml_compute_units"):
        llm_config.backend.coreml.compute_units = CoreMLComputeUnit(args.coreml_compute_units)

    # Vulkan
    if hasattr(args, "vulkan"):
        llm_config.backend.vulkan.enabled = args.vulkan

    # QNN
    if hasattr(args, "qnn"):
        llm_config.backend.qnn.enabled = args.qnn
    if hasattr(args, "use_qnn_sha"):
        llm_config.backend.qnn.use_sha = args.use_qnn_sha
    if hasattr(args, "soc_model"):
        llm_config.backend.qnn.soc_model = args.soc_model
    if hasattr(args, "optimized_rotation_path"):
        llm_config.backend.qnn.optimized_rotation_path = args.optimized_rotation_path
    if hasattr(args, "num_sharding"):
        llm_config.backend.qnn.num_sharding = args.num_sharding

    # MPS
    if hasattr(args, "mps"):
        llm_config.backend.mps.enabled = args.mps

    # DebugConfig
    if hasattr(args, "profile_memory"):
        llm_config.debug.profile_memory = args.profile_memory
    if hasattr(args, "profile_path"):
        llm_config.debug.profile_path = args.profile_path
    if hasattr(args, "generate_etrecord"):
        llm_config.debug.generate_etrecord = args.generate_etrecord
    if hasattr(args, "generate_full_logits"):
        llm_config.debug.generate_full_logits = args.generate_full_logits
    if hasattr(args, "verbose"):
        llm_config.debug.verbose = args.verbose

    return llm_config
