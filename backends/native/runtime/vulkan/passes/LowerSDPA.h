// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <array>
#include <cstddef>
#include <string_view>

#include <executorch/backends/native/runtime/graph/Graph.h>

namespace ptn::vulkan {

// The parameters of aten.scaled_dot_product_attention.
inline constexpr std::array<std::string_view, 8> kSDPAParams{
    "query",
    "key",
    "value",
    "attn_mask",
    "dropout_p",
    "is_causal",
    "scale",
    "enable_gqa"};

// Rewrites cache-free aten.scaled_dot_product_attention nodes (encoder
// self-attention and decoder cross-attention, e.g. Whisper) into ET-VK's
// fused et_vk.sdpa. Only the forms the fused kernel honors are rewritten:
// 4D float/half query/key/value of one dtype with matching batch sizes, head
// counts and widths, dropout 0, is_causal false, and enable_gqa absent/false.
// The mask may be absent, an additive bias in the query's dtype (passed
// through), or a bool keep-mask (converted once per mask to an additive bias,
// since the kernel only adds its mask operand). Masks may broadcast over batch,
// heads and queries but must span the key axis. Cached decoder-only attention
// is handled by lower_hf_attention, which must run first; anything this pass
// leaves behind fails dispatch by name.
size_t lower_sdpa(Graph& graph);

} // namespace ptn::vulkan
