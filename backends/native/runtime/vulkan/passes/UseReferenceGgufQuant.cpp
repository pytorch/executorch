// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/vulkan/passes/UseReferenceGgufQuant.h>

#include <stdexcept>
#include <variant>

#include <executorch/backends/native/runtime/graph/ScalarType.h>
#include <executorch/backends/native/runtime/graph/TensorMeta.h>
#include <executorch/backends/native/runtime/graph/Value.h>
#include <executorch/backends/native/runtime/utils/ReferenceGgufQuant.h>

namespace ptn::vulkan {

// cppcheck-suppress unusedFunction
ReferenceGgufFallback use_reference_gguf(Method& method) {
  ReferenceGgufFallback fallback;
  for (const DataBinding& binding : method.data_bindings) {
    if (!binding.has_data || binding.mutated) {
      continue;
    }
    Value& value = method.graph.value(binding.value_id);
    if (!value.is_tensor() || !value.tensor_meta().quant.has_value()) {
      continue;
    }
    TensorMeta& meta = value.tensor_meta();
    const auto* packed = std::get_if<PackedQuant>(&*meta.quant);
    if (packed == nullptr) {
      continue;
    }
    if (!reference::can_decode_gguf(packed->codec)) {
      throw std::runtime_error(
          "vulkan: constant '" + binding.key + "' has PackedQuant codec '" +
          packed->codec + "', which the Vulkan engine cannot decode");
    }
    fallback.codecs.insert(packed->codec);
    value.attrs[kReferenceGgufAttr] = packed->codec;
    meta.dtype = ScalarType::Float;
    meta.quant.reset();
    ++fallback.constants;
    fallback.decoded_bytes += static_cast<size_t>(meta.numel()) * sizeof(float);
  }
  return fallback;
}

} // namespace ptn::vulkan
