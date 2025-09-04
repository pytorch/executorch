/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
// constants for LLM runtime
namespace executorch::extension::llm {

// Runtime metadata key constants
inline constexpr auto kEnableDynamicShape = "enable_dynamic_shape";
inline constexpr auto kBosId = "get_bos_id";
inline constexpr auto kEosIds = "get_eos_ids";
inline constexpr auto kMaxSeqLen = "get_max_seq_len";
inline constexpr auto kMaxContextLen = "get_max_context_len";
inline constexpr auto kVocabSize = "get_vocab_size";
inline constexpr auto kUseKVCache = "use_kv_cache";
inline constexpr auto kUseSDPAWithKVCache = "use_sdpa_with_kv_cache";

// Multimodal method name conventions
inline constexpr auto kImageEncoderMethod = "image_encoder";
inline constexpr auto kTokenEmbeddingMethod = "token_embedding";
inline constexpr auto kTextModelMethod = "text_model";

} // namespace executorch::extension::llm
