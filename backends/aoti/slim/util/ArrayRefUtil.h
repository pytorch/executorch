#pragma once

// Utilities for working with ExecuTorch's ArrayRef in SlimTensor code.
// This header provides helper functions for creating ArrayRefs from
// std::vector and std::initializer_list, and for converting ArrayRefs
// back to std::vector.

#include <executorch/runtime/core/array_ref.h>

#include <initializer_list>
#include <vector>

namespace executorch::backends::aoti::slim {

// Bring ExecuTorch's ArrayRef types into the SlimTensor namespace
using ::executorch::runtime::ArrayRef;
using ::executorch::runtime::IntArrayRef;
using ::executorch::runtime::makeArrayRef;

/// Helper function to construct an ArrayRef from a std::vector.
template <typename T>
inline ArrayRef<T> makeArrayRef(const std::vector<T>& Vec) {
  return ArrayRef<T>(Vec.data(), Vec.size());
}

/// Helper function to construct an ArrayRef from a std::initializer_list.
template <typename T>
inline ArrayRef<T> makeArrayRef(std::initializer_list<T> list) {
  return ArrayRef<T>(list.begin(), list.size());
}

/// Helper function to convert ArrayRef to std::vector.
template <typename T>
inline std::vector<T> toVec(ArrayRef<T> arr) {
  return std::vector<T>(arr.begin(), arr.end());
}

} // namespace executorch::backends::aoti::slim
