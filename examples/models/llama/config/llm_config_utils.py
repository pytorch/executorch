# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
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
    llm_config.base.model_class = ModelType(args.model)
    llm_config.base.params = args.params
    llm_config.base.checkpoint = args.checkpoint
    llm_config.base.checkpoint_dir = args.checkpoint_dir
    llm_config.base.tokenizer_path = args.tokenizer_path
    llm_config.base.metadata = args.metadata
    llm_config.base.use_lora = bool(args.use_lora)
    llm_config.base.fairseq2 = args.fairseq2

    # PreqMode settings
    if args.preq_mode:
        llm_config.base.preq_mode = PreqMode(args.preq_mode)
        llm_config.base.preq_group_size = args.preq_group_size
        llm_config.base.preq_embedding_quantize = args.preq_embedding_quantize

    # ModelConfig
    llm_config.model.dtype_override = DtypeOverride(args.dtype_override)
    llm_config.model.enable_dynamic_shape = args.enable_dynamic_shape
    llm_config.model.use_shared_embedding = args.use_shared_embedding
    llm_config.model.use_sdpa_with_kv_cache = args.use_sdpa_with_kv_cache
    llm_config.model.expand_rope_table = args.expand_rope_table
    llm_config.model.use_attention_sink = args.use_attention_sink
    llm_config.model.output_prune_map = args.output_prune_map
    llm_config.model.input_prune_map = args.input_prune_map
    llm_config.model.use_kv_cache = args.use_kv_cache
    llm_config.model.quantize_kv_cache = args.quantize_kv_cache
    llm_config.model.local_global_attention = args.local_global_attention

    # ExportConfig
    llm_config.export.max_seq_length = args.max_seq_length
    llm_config.export.max_context_length = args.max_context_length
    llm_config.export.output_dir = args.output_dir
    llm_config.export.output_name = args.output_name
    llm_config.export.so_library = args.so_library
    llm_config.export.export_only = args.export_only

    # QuantizationConfig
    llm_config.quantization.qmode = args.quantization_mode
    llm_config.quantization.embedding_quantize = args.embedding_quantize
    if args.pt2e_quantize:
        llm_config.quantization.pt2e_quantize = Pt2eQuantize(args.pt2e_quantize)
    llm_config.quantization.group_size = args.group_size
    if args.use_spin_quant:
        llm_config.quantization.use_spin_quant = SpinQuant(args.use_spin_quant)
    llm_config.quantization.use_qat = args.use_qat
    llm_config.quantization.calibration_tasks = args.calibration_tasks
    llm_config.quantization.calibration_limit = args.calibration_limit
    llm_config.quantization.calibration_seq_length = args.calibration_seq_length
    llm_config.quantization.calibration_data = args.calibration_data

    # BackendConfig
    # XNNPack
    llm_config.backend.xnnpack.enabled = args.xnnpack
    llm_config.backend.xnnpack.extended_ops = args.xnnpack_extended_ops

    # CoreML
    llm_config.backend.coreml.enabled = args.coreml
    llm_config.backend.coreml.enable_state = getattr(args, "coreml_enable_state", False)
    llm_config.backend.coreml.preserve_sdpa = getattr(
        args, "coreml_preserve_sdpa", False
    )
    if args.coreml_quantize:
        llm_config.backend.coreml.quantize = CoreMLQuantize(args.coreml_quantize)
    llm_config.backend.coreml.ios = args.coreml_ios
    llm_config.backend.coreml.compute_units = CoreMLComputeUnit(
        args.coreml_compute_units
    )

    # Vulkan
    llm_config.backend.vulkan.enabled = args.vulkan

    # QNN
    llm_config.backend.qnn.enabled = args.qnn
    llm_config.backend.qnn.use_sha = args.use_qnn_sha
    llm_config.backend.qnn.soc_model = args.soc_model
    llm_config.backend.qnn.optimized_rotation_path = args.optimized_rotation_path
    llm_config.backend.qnn.num_sharding = args.num_sharding

    # MPS
    llm_config.backend.mps.enabled = args.mps

    # DebugConfig
    llm_config.debug.profile_memory = args.profile_memory
    llm_config.debug.profile_path = args.profile_path
    llm_config.debug.generate_etrecord = args.generate_etrecord
    llm_config.debug.generate_full_logits = args.generate_full_logits
    llm_config.debug.verbose = args.verbose

    return llm_config
