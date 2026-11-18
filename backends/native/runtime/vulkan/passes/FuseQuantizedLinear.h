// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <executorch/backends/native/runtime/Method.h>

namespace ptn::vulkan {

inline constexpr char kQ4ConstantTransformAttr[] =
    "vulkan_q4_constant_transform";

enum class Q4ConstantTransformKind {
  PackWeight,
  TransposeScales,
  WeightSums,
};

struct Q4ConstantTransform {
  Q4ConstantTransformKind kind;
  ValueId source_id;
  ValueId zero_points_id = kInvalid; // kInvalid for symmetric weights
  int64_t group_size = 0;
};

struct Q4GroupSumsLayout {
  size_t rows;
  size_t cols;
  size_t group_size;
  size_t output_cols;
  // Two weights per byte, low nibble first, each stored as value + 8.
  // Otherwise one int8 weight per byte.
  bool packed;
};

// Returns the int32 sum of each [row, group] of `weight` as
// [cols / group_size, output_cols]. Columns past `rows` are zero.
std::vector<int32_t> q4_group_sums(
    std::span<const uint8_t> weight,
    const Q4GroupSumsLayout& layout);

// Rewrites aten.linear over a q4 weight into a Native-VK runtime kernel. The
// weight is either a torchao dequantize_affine output or a constant carrying
// AffineGroupQuant read directly. A torchao q8 dynamic activation selects
// linear_dq8ca_q4gsw, and a floating-point activation linear_q4gsw. This runs
// after PTN deserialization; PTNs therefore contain neither et_vk operators nor
// Vulkan AOT transformations.
size_t fuse_quantized_linears(Method& method);

} // namespace ptn::vulkan
