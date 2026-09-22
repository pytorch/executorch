// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>

namespace ptn::detail {

// Bound work and allocation performed by the PTN deserialization layer.
// Exceeding a cap is reported as ResourceLimitError at the call site.
inline constexpr uint64_t kMaxPackageBytes = uint64_t{1} << 40;
inline constexpr size_t kMaxJsonBytes = size_t{64} << 20;
inline constexpr size_t kMaxJsonValues = size_t{1} << 20;
inline constexpr int kMaxJsonDepth = 64;
inline constexpr size_t kMaxPackageMembers = size_t{1} << 20;
inline constexpr size_t kMaxTensorCount = size_t{1} << 20;
inline constexpr size_t kMaxAliasCount = size_t{1} << 20;
inline constexpr size_t kMaxProgramMethods = size_t{1} << 16;
inline constexpr size_t kMaxTensorRank = 64;
inline constexpr uint64_t kMaxTensorDimension = (1ULL << 31) - 1;
inline constexpr uint64_t kMaxTensorBytes = uint64_t{1} << 40;
inline constexpr uint64_t kMaxConstantBytes = uint64_t{1} << 40;
inline constexpr size_t kMaxProgramBytes = size_t{1} << 30;

} // namespace ptn::detail
