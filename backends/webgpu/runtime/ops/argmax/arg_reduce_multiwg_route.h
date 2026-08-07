/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

namespace executorch::backends::webgpu {

constexpr uint32_t kArgReduceElemsPerWorkgroup = 1024u;
constexpr uint32_t kArgReduceMaxParts = 256u;
constexpr uint32_t kArgReduceMinReduceSize = 4096u;
constexpr uint32_t kArgReduceMaxPartials = 1u << 20;

constexpr bool arg_reduce_partial_slots_fit(
    uint64_t max_partial_slots,
    uint32_t num_rows,
    uint32_t num_parts) {
  return max_partial_slots != 0u && num_rows != 0u && num_parts != 0u &&
      num_rows <= max_partial_slots / num_parts;
}

constexpr uint32_t select_arg_reduce_parts(
    uint32_t num_rows,
    uint32_t reduce_size,
    uint32_t max_workgroups_per_dim) {
  if (num_rows == 0u || reduce_size < kArgReduceMinReduceSize) {
    return 0u;
  }
  uint32_t parts = reduce_size / kArgReduceElemsPerWorkgroup +
      static_cast<uint32_t>(reduce_size % kArgReduceElemsPerWorkgroup != 0u);
  if (parts > kArgReduceMaxParts) {
    parts = kArgReduceMaxParts;
  }
  if (parts < 2u || num_rows > kArgReduceMaxPartials / parts ||
      max_workgroups_per_dim == 0u) {
    return 0u;
  }
  const uint64_t total = static_cast<uint64_t>(num_rows) * parts;
  const uint64_t grid_capacity =
      static_cast<uint64_t>(max_workgroups_per_dim) * max_workgroups_per_dim;
  return total <= grid_capacity ? parts : 0u;
}

} // namespace executorch::backends::webgpu
