/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <limits>

#include <executorch/extension/llm/batching/types.h>

namespace executorch {
namespace extension {
namespace llm {
namespace batching {
namespace detail {

inline bool valid_position_count(std::size_t count) {
  return count > 0 &&
      count <= static_cast<std::size_t>(std::numeric_limits<Position>::max());
}

// Metadata only; concrete storage remains executor-private.
inline bool valid_prepared(const PreparedInputPtr& input) {
  return input && valid_position_count(input->size());
}

inline bool valid_payload(const InputPayload& payload) {
  if (const auto* tokens = std::get_if<TokenInputPtr>(&payload)) {
    return *tokens && valid_position_count((*tokens)->size());
  }
  const auto* prepared = std::get_if<PreparedInputPtr>(&payload);
  return prepared && valid_prepared(*prepared);
}

inline bool validate_batch(const BatchInput& batch) {
  for (const auto& input : batch.inputs) {
    if (!valid_payload(input.payload) || input.size == 0) {
      return false;
    }
    const auto count = std::holds_alternative<TokenInputPtr>(input.payload)
        ? std::get<TokenInputPtr>(input.payload)->size()
        : std::get<PreparedInputPtr>(input.payload)->size();
    if (input.offset > count || input.size > count - input.offset) {
      return false;
    }
    // The backing count bounds offset before conversion to signed arithmetic.
    const auto start = static_cast<std::int64_t>(input.position) +
        static_cast<std::int64_t>(input.offset);
    if (start < 0 || start > std::numeric_limits<Position>::max() ||
        input.size > static_cast<std::size_t>(
                         std::numeric_limits<Position>::max() - start)) {
      return false;
    }
  }
  return true;
}

} // namespace detail
} // namespace batching
} // namespace llm
} // namespace extension
} // namespace executorch
