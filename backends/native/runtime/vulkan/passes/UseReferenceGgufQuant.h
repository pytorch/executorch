// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <set>
#include <string>

#include <executorch/backends/native/runtime/Method.h>

namespace ptn::vulkan {

// Holds the std::string codec of a constant the engine decodes on load with
// the reference GGUF decoder.
inline constexpr char kReferenceGgufAttr[] = "vulkan_reference_gguf_codec";

struct ReferenceGgufFallback {
  size_t constants = 0;
  size_t decoded_bytes = 0;
  std::set<std::string> codecs;
};

// The Vulkan engine does not lower PackedQuant codecs yet. As a temporary,
// slow fallback, retype every stored constant of a GGUF codec the reference
// decoder handles as the fp32 tensor it denotes and mark it with
// kReferenceGgufAttr, so the reference decoder expands it on load and it takes
// the unquantized path. Throws std::runtime_error for any other PackedQuant
// constant, naming its codec.
ReferenceGgufFallback use_reference_gguf(Method& method);

} // namespace ptn::vulkan
