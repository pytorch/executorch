// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>

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
  ValueId zero_points_id = kInvalid;
  int64_t group_size = 0;
};

// Rewrites portable torchao q8-dynamic/q4-weight linear patterns into the
// Native-VK runtime kernel. This runs after PTN deserialization; PTNs therefore
// contain neither et_vk operators nor Vulkan AOT transformations.
size_t fuse_quantized_linears(Method& method);

} // namespace ptn::vulkan
