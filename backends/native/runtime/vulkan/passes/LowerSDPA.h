// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>

#include <executorch/backends/native/runtime/graph/Graph.h>

namespace ptn::vulkan {

// Rewrites cache-free aten.scaled_dot_product_attention nodes (encoder
// self-attention and decoder cross-attention, e.g. Whisper) into ET-VK's
// fused et_vk.sdpa. Only the forms the fused kernel honors are rewritten:
// 4D float/half query/key/value with matching head counts and widths, dropout
// 0, is_causal false, and enable_gqa absent/false. The mask may be absent, a
// floating-point additive bias (passed through), or a bool keep-mask
// (converted to an additive bias, since the kernel only adds its mask
// operand). Masks must span the full key axis. Cached decoder-only attention is
// handled by lower_hf_attention, which must run first; anything this pass
// leaves behind fails dispatch by name.
size_t lower_sdpa(Graph& graph);

} // namespace ptn::vulkan
