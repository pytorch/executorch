#pragma once

#include <vector>
#include <cstdint>
#include <initializer_list>
#include <executorch/runtime/core/array_ref.h>
#include <executorch/backends/cuda/runtime/c10/util/ArrayRef.h>

namespace executorch::backends::cuda::slim {

/**
 * Utility functions to extend ET IntArrayRef functionality without modifying
 * the original ET IntArrayRef implementation.
 * 
 * These utilities provide missing methods that are present in C10's IntArrayRef
 * but not in ET's IntArrayRef, such as .vec() for converting to std::vector.
 */

/**
 * Convert an ET IntArrayRef to a std::vector<int64_t>
 * This is equivalent to C10's IntArrayRef::vec() method
 */
inline std::vector<int64_t> to_vec(executorch::aten::IntArrayRef arr) {
  return std::vector<int64_t>(arr.data(), arr.data() + arr.size());
}

/**
 * Convert a C10 IntArrayRef to a std::vector<int64_t>
 */
inline std::vector<int64_t> to_vec(executorch::backends::cuda::c10::IntArrayRef arr) {
  return std::vector<int64_t>(arr.data(), arr.data() + arr.size());
}

/**
 * Template version for generic ArrayRef types
 */
template <typename T>
inline std::vector<T> to_vec(executorch::runtime::ArrayRef<T> arr) {
  return std::vector<T>(arr.data(), arr.data() + arr.size());
}

/**
 * Convert C10 IntArrayRef to ET IntArrayRef
 */
inline executorch::aten::IntArrayRef c10_to_et(executorch::backends::cuda::c10::IntArrayRef arr) {
  return executorch::aten::IntArrayRef(arr.data(), arr.size());
}

/**
 * Convert ET IntArrayRef to C10 IntArrayRef
 */
inline executorch::backends::cuda::c10::IntArrayRef et_to_c10(executorch::aten::IntArrayRef arr) {
  return executorch::backends::cuda::c10::IntArrayRef(arr.data(), arr.size());
}

/**
 * Convert std::vector to ET IntArrayRef
 * Note: The returned ArrayRef references the vector's data, so the vector
 * must outlive the ArrayRef.
 */
inline executorch::aten::IntArrayRef vec_to_et(const std::vector<int64_t>& vec) {
  return executorch::aten::IntArrayRef(vec.data(), vec.size());
}

/**
 * Convert std::initializer_list to ET IntArrayRef
 * Note: The returned ArrayRef references the initializer_list's data, so the
 * initializer_list must outlive the ArrayRef.
 */
inline executorch::aten::IntArrayRef initlist_to_et(std::initializer_list<int64_t> list) {
  return executorch::aten::IntArrayRef(list.begin(), list.size());
}

} // namespace executorch::backends::cuda::slim
