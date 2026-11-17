// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/cpu/runtime/KernelProvider.h>

#include <vector>

namespace executorch::backends::cpu::xnnpack {

// Reads a literal optional clamp bound; None selects `fallback`.
bool clamp_bound(const ptn::Argument& argument, float fallback, float& bound);

// The decomposition `x * clamp(x + 3, 0, 6) / 6` with intermediates private to
// one region. `nodes` ends with the node that produces `output`.
struct HardswishChain {
  ptn::ValueId input = ptn::kInvalid;
  ptn::ValueId output = ptn::kInvalid;
  std::vector<KernelId> nodes;
};

// Requires bound constant buffers to read the scalar operands.
std::vector<HardswishChain> find_hardswish_chains(
    const KernelRegion& region,
    const ptn::Graph& graph,
    const std::vector<Buffer>& buffers);

} // namespace executorch::backends::cpu::xnnpack
