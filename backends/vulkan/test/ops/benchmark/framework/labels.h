// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/vulkan/runtime/api/api.h>

#include <cstdint>
#include <string>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {

// NOLINTNEXTLINE(google-build-using-namespace)
using namespace vkcompute;

//
// String utilities
//

// Helper function to get abbreviated layout names for test case naming
inline std::string layout_abbrev(utils::GPUMemoryLayout layout) {
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (layout) {
    case utils::kWidthPacked:
      return "WP";
    case utils::kChannelsPacked:
      return "CP";
    case utils::kPackedInt8_4W:
      return "4W";
    case utils::kPackedInt8_4C:
      return "4C";
    case utils::kPackedInt8_4W4C:
      return "4W4C";
    case utils::kPackedInt8_4H4W:
      return "4H4W";
    case utils::kPackedInt8_4C1W:
      return "4C1W";
    default:
      return "UNK";
  }
}

// Helper function to get abbreviated storage type names for test case naming
inline std::string storage_type_abbrev(utils::StorageType storage_type) {
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (storage_type) {
    case utils::kTexture3D:
      return "Tex";
    case utils::kBuffer:
      return "Buf";
    default:
      return "UNK";
  }
}

// Helper function to get combined storage type and layout representation
// Example: (kBuffer, kPackedInt8_4W4C) -> "Buf_4W4C"
inline std::string repr_str(
    utils::StorageType storage_type,
    utils::GPUMemoryLayout layout) {
  return storage_type_abbrev(storage_type) + "(" + layout_abbrev(layout) + ")";
}

// Helper function to generate comma-separated shape string for test case naming
// Example: {1, 128, 56, 56} -> "1,128,56,56"
inline std::string shape_string(const std::vector<int64_t>& shape) {
  std::string result;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      result += ",";
    }
    result += std::to_string(shape[i]);
  }
  return result;
}

// Helper function to generate a bracketed shape string for test case naming
// Example: {1, 128, 56, 56} -> "[1,128,56,56]"
inline std::string shape_bracket(const std::vector<int64_t>& shape) {
  return "[" + shape_string(shape) + "]";
}

// Short dtype symbol used in standardized test case labels
// Example: kFloat -> "f32", kHalf -> "f16", kChar -> "i8", kInt -> "i32"
inline std::string dtype_short(vkapi::ScalarType dtype) {
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (dtype) {
    case vkapi::kFloat:
      return "f32";
    case vkapi::kHalf:
      return "f16";
    case vkapi::kChar:
      return "i8";
    case vkapi::kByte:
      return "u8";
    case vkapi::kInt:
      return "i32";
    case vkapi::kBool:
      return "b";
    default:
      return "?";
  }
}

// Build a standardized test label of the form:
//   "<prefix>  <dtype_str>  <shape_str>  <storage_str>[ <suffix>]"
// where <dtype_str> is "<in_dtype>-><out_dtype>" when the two differ, or just
// "<in_dtype>" when they match. Sections are separated by two spaces. If
// suffix is non-empty it is appended after a single space (allowing callers
// to pass e.g. "[general]" or "[gemv] +bias").
inline std::string make_test_label(
    const std::string& prefix,
    const std::string& in_dtype,
    const std::string& out_dtype,
    const std::string& shape_str,
    const std::string& storage_str,
    const std::string& suffix = "") {
  const std::string dtype_str =
      (in_dtype == out_dtype) ? in_dtype : in_dtype + "->" + out_dtype;
  std::string label =
      prefix + "  " + dtype_str + "  " + shape_str + "  " + storage_str;
  if (!suffix.empty()) {
    label += " " + suffix;
  }
  return label;
}

} // namespace prototyping
} // namespace vulkan
} // namespace executorch
