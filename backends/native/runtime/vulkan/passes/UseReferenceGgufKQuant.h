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
// the reference GGUF K-quant decoder.
inline constexpr char kReferenceGgufKQuantAttr[] =
    "vulkan_reference_gguf_kquant_codec";

struct ReferenceGgufKQuantFallback {
  size_t constants = 0;
  size_t decoded_bytes = 0;
  std::set<std::string> codecs;
};

// The Vulkan engine does not lower GGUF K-quant codecs yet. As a temporary,
// slow fallback, retype every stored GGUF K-quant constant as the fp32 tensor
// it denotes and mark it with kReferenceGgufKQuantAttr, so the reference
// decoder expands it on load and it takes the unquantized path.
ReferenceGgufKQuantFallback use_reference_gguf_kquant(Method& method);

} // namespace ptn::vulkan
