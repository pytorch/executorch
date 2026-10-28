/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace executorch {
namespace backends {
namespace vgf {

inline std::optional<uint32_t> vgf_find_mutable_alias_group_conflict(
    const std::vector<std::optional<uint32_t>>& alias_groups,
    uint32_t input_mrt,
    uint32_t output_mrt) {
  const auto input_group = alias_groups.at(input_mrt);
  const auto output_group = alias_groups.at(output_mrt);
  for (size_t resource = 0; resource < alias_groups.size(); ++resource) {
    if (resource == input_mrt || resource == output_mrt ||
        !alias_groups[resource].has_value()) {
      continue;
    }
    if (alias_groups[resource] == input_group ||
        alias_groups[resource] == output_group) {
      return static_cast<uint32_t>(resource);
    }
  }
  return std::nullopt;
}

} // namespace vgf
} // namespace backends
} // namespace executorch
