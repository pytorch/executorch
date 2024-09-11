/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/tensor/tensor_ptr_maker.h>

namespace executorch {
namespace extension {
namespace {
template <
    typename INT_T,
    typename std::enable_if<
        std::is_integral<INT_T>::value && !std::is_same<INT_T, bool>::value,
        bool>::type = true>
bool extract_scalar(exec_aten::Scalar scalar, INT_T* out_val) {
  if (!scalar.isIntegral(/*includeBool=*/false)) {
    return false;
  }
  int64_t val = scalar.to<int64_t>();
  if (val < std::numeric_limits<INT_T>::lowest() ||
      val > std::numeric_limits<INT_T>::max()) {
    return false;
  }
  *out_val = static_cast<INT_T>(val);
  return true;
}

template <
    typename FLOAT_T,
    typename std::enable_if<std::is_floating_point<FLOAT_T>::value, bool>::
        type = true>
bool extract_scalar(exec_aten::Scalar scalar, FLOAT_T* out_val) {
  double val;
  if (scalar.isFloatingPoint()) {
    val = scalar.to<double>();
    if (std::isfinite(val) &&
        (val < std::numeric_limits<FLOAT_T>::lowest() ||
         val > std::numeric_limits<FLOAT_T>::max())) {
      return false;
    }
  } else if (scalar.isIntegral(/*includeBool=*/false)) {
    val = static_cast<double>(scalar.to<int64_t>());
  } else {
    return false;
  }
  *out_val = static_cast<FLOAT_T>(val);
  return true;
}

template <
    typename BOOL_T,
    typename std::enable_if<std::is_same<BOOL_T, bool>::value, bool>::type =
        true>
bool extract_scalar(exec_aten::Scalar scalar, BOOL_T* out_val) {
  if (scalar.isIntegral(false)) {
    *out_val = static_cast<bool>(scalar.to<int64_t>());
    return true;
  }
  if (scalar.isBoolean()) {
    *out_val = scalar.to<bool>();
    return true;
  }
  return false;
}

#define ET_EXTRACT_SCALAR(scalar, out_val) \
  ET_CHECK_MSG(                            \
      extract_scalar(scalar, &out_val),    \
      #scalar " could not be extracted: wrong type or out of range");

} // namespace

TensorPtr empty_strided(
    std::vector<exec_aten::SizesType> sizes,
    std::vector<exec_aten::StridesType> strides,
    exec_aten::ScalarType type,
    exec_aten::TensorShapeDynamism dynamism) {
  std::vector<uint8_t> data(
      exec_aten::compute_numel(sizes.data(), sizes.size()) *
      exec_aten::elementSize(type));
  return make_tensor_ptr(
      type,
      std::move(sizes),
      std::move(data),
      {},
      std::move(strides),
      dynamism);
}

TensorPtr full_strided(
    std::vector<exec_aten::SizesType> sizes,
    std::vector<exec_aten::StridesType> strides,
    exec_aten::Scalar fill_value,
    exec_aten::ScalarType type,
    exec_aten::TensorShapeDynamism dynamism) {
  auto tensor =
      empty_strided(std::move(sizes), std::move(strides), type, dynamism);
  ET_SWITCH_REALB_TYPES(type, nullptr, "full_strided", CTYPE, [&] {
    CTYPE value;
    ET_EXTRACT_SCALAR(fill_value, value);
    std::fill(
        tensor->mutable_data_ptr<CTYPE>(),
        tensor->mutable_data_ptr<CTYPE>() + tensor->numel(),
        value);
  });
  return tensor;
}

} // namespace extension
} // namespace executorch
