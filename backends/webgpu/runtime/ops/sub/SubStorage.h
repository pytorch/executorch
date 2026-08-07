/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <stdexcept>

namespace executorch::backends::webgpu {

enum class SubStorage { Float32, Int32 };

template <typename Tensor>
SubStorage
classify_sub_storage(const Tensor& in1, const Tensor& in2, const Tensor& out) {
  const bool all_int32 = in1.is_int && !in1.is_bool &&
      in1.elem_size == sizeof(int32_t) && in2.is_int && !in2.is_bool &&
      in2.elem_size == sizeof(int32_t) && out.is_int && !out.is_bool &&
      out.elem_size == sizeof(int32_t);
  const bool any_integer = in1.is_int || in1.is_bool || in2.is_int ||
      in2.is_bool || out.is_int || out.is_bool;
  if (any_integer && !all_int32) {
    throw std::runtime_error(
        "sub: integer operands must all be signed int32 tensors");
  }
  return all_int32 ? SubStorage::Int32 : SubStorage::Float32;
}

} // namespace executorch::backends::webgpu
