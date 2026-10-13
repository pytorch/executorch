/*
 * Copyright (c) Samsung Electronics Co. LTD
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
// constants for LLM runtime
namespace example {

// Exynos Runtime metadata key constants
inline constexpr auto kPrefillArLen = "get_prefill_ar_len";
inline constexpr auto kNumOfLayers = "get_n_layers";
inline constexpr auto kNumOfHiddenDim = "get_hidden_dim";
inline constexpr auto kNumOfHeads = "get_n_heads";
inline constexpr auto kNumOfKVHeads = "get_n_kv_heads";

static constexpr auto default_method_name = "forward";
static constexpr auto default_method_name_prefill = "prefill_forward";
static constexpr auto default_method_name_decode = "decode_forward";
} // namespace executorch::extension::llm
