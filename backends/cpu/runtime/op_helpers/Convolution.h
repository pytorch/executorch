// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/cpu/runtime/KernelProvider.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace executorch::backends::cpu {

/// Operands and static attributes of `aten.convolution.default`. bias is
/// ptn::kInvalid when absent.
struct Convolution {
  ptn::ValueId input;
  ptn::ValueId weight;
  ptn::ValueId bias;
  ptn::ValueId output;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  std::vector<int64_t> output_padding;
  bool transposed;
  int64_t groups;
};

/// Returns nullopt unless node is an unmutated convolution with tensor operands
/// and compile-time attributes. Shape and dtype checks remain with the caller.
std::optional<Convolution> parse_convolution(const Kernel& node);

} // namespace executorch::backends::cpu
