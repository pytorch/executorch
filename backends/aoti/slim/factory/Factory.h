/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/aoti/slim/factory/Empty.h>
#include <executorch/backends/aoti/slim/util/ArrayRefUtil.h>

namespace executorch::backends::aoti::slim {
inline SlimTensor zeros(
    IntArrayRef sizes,
    executorch::backends::aoti::slim::c10::ScalarType dtype,
    const executorch::backends::aoti::slim::c10::Device& device = CPU_DEVICE) {
  SlimTensor tensor = empty(sizes, dtype, device);
  tensor.fill_(executorch::backends::aoti::slim::c10::Scalar(0));
  return tensor;
}

inline SlimTensor zeros(
    std::initializer_list<int64_t> sizes,
    executorch::backends::aoti::slim::c10::ScalarType dtype,
    const executorch::backends::aoti::slim::c10::Device& device = CPU_DEVICE) {
  return zeros(makeArrayRef(sizes), dtype, device);
}

inline SlimTensor zeros_like(const SlimTensor& other) {
  return zeros(other.sizes(), other.dtype(), other.device());
}

inline SlimTensor ones(
    IntArrayRef sizes,
    executorch::backends::aoti::slim::c10::ScalarType dtype,
    const executorch::backends::aoti::slim::c10::Device& device = CPU_DEVICE) {
  SlimTensor tensor = empty(sizes, dtype, device);
  tensor.fill_(executorch::backends::aoti::slim::c10::Scalar(1));
  return tensor;
}

inline SlimTensor ones(
    std::initializer_list<int64_t> sizes,
    executorch::backends::aoti::slim::c10::ScalarType dtype,
    const executorch::backends::aoti::slim::c10::Device& device = CPU_DEVICE) {
  return ones(makeArrayRef(sizes), dtype, device);
}

inline SlimTensor ones_like(const SlimTensor& other) {
  return ones(other.sizes(), other.dtype(), other.device());
}

} // namespace executorch::backends::aoti::slim
