/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/tensor/tensor_ptr_maker.h>

#include <random>

namespace executorch {
namespace extension {
namespace {

template <
    typename INT_T,
    typename std::enable_if<
        std::is_integral<INT_T>::value && !std::is_same<INT_T, bool>::value,
        bool>::type = true>
bool extract_scalar(executorch::aten::Scalar scalar, INT_T* out_val) {
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
    typename std::enable_if<
        std::is_floating_point_v<FLOAT_T> ||
            std::is_same_v<FLOAT_T, executorch::aten::BFloat16> ||
            std::is_same_v<FLOAT_T, executorch::aten::Half>,
        bool>::type = true>
bool extract_scalar(executorch::aten::Scalar scalar, FLOAT_T* out_val) {
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
bool extract_scalar(executorch::aten::Scalar scalar, BOOL_T* out_val) {
  if (scalar.isIntegral(/*includeBool=*/false)) {
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

template <typename Distribution>
TensorPtr random_strided(
    std::vector<executorch::aten::SizesType> sizes,
    std::vector<executorch::aten::StridesType> strides,
    executorch::aten::ScalarType type,
    executorch::aten::TensorShapeDynamism dynamism,
    Distribution&& distribution) {
  auto tensor =
      empty_strided(std::move(sizes), std::move(strides), type, dynamism);
  std::default_random_engine gen{std::random_device{}()};

  ET_SWITCH_REALHBBF16_TYPES(type, nullptr, "random_strided", CTYPE, [&] {
    std::generate_n(tensor->mutable_data_ptr<CTYPE>(), tensor->numel(), [&]() {
      return static_cast<CTYPE>(distribution(gen));
    });
  });
  return tensor;
}

} // namespace

TensorPtr empty_strided(
    std::vector<executorch::aten::SizesType> sizes,
    std::vector<executorch::aten::StridesType> strides,
    executorch::aten::ScalarType type,
    executorch::aten::TensorShapeDynamism dynamism) {
  std::vector<uint8_t> data(
      executorch::aten::compute_numel(sizes.data(), sizes.size()) *
      executorch::aten::elementSize(type));
  return make_tensor_ptr(
      std::move(sizes),
      std::move(data),
      {},
      std::move(strides),
      type,
      dynamism);
}

TensorPtr full_strided(
    std::vector<executorch::aten::SizesType> sizes,
    std::vector<executorch::aten::StridesType> strides,
    executorch::aten::Scalar fill_value,
    executorch::aten::ScalarType type,
    executorch::aten::TensorShapeDynamism dynamism) {
  auto tensor =
      empty_strided(std::move(sizes), std::move(strides), type, dynamism);
  ET_SWITCH_REALHBBF16_TYPES(type, nullptr, "full_strided", CTYPE, [&] {
    CTYPE value;
    ET_EXTRACT_SCALAR(fill_value, value);
    std::fill(
        tensor->mutable_data_ptr<CTYPE>(),
        tensor->mutable_data_ptr<CTYPE>() + tensor->numel(),
        value);
  });
  return tensor;
}

TensorPtr rand_strided(
    std::vector<executorch::aten::SizesType> sizes,
    std::vector<executorch::aten::StridesType> strides,
    executorch::aten::ScalarType type,
    executorch::aten::TensorShapeDynamism dynamism) {
  auto upper_bound = 1.0f;
  // Adjusts the upper bound to prevent rounding to 1.0 when converting to
  // lower-precision types.
  if (type == executorch::aten::ScalarType::Half) {
    upper_bound -=
        float(std::numeric_limits<executorch::aten::Half>::epsilon()) / 2;
  } else if (type == executorch::aten::ScalarType::BFloat16) {
    upper_bound -=
        float(std::numeric_limits<executorch::aten::BFloat16>::epsilon()) / 2;
  }
  return random_strided(
      std::move(sizes),
      std::move(strides),
      type,
      dynamism,
      std::uniform_real_distribution<float>(0.0f, upper_bound));
}

TensorPtr randn_strided(
    std::vector<executorch::aten::SizesType> sizes,
    std::vector<executorch::aten::StridesType> strides,
    executorch::aten::ScalarType type,
    executorch::aten::TensorShapeDynamism dynamism) {
  return random_strided(
      std::move(sizes),
      std::move(strides),
      type,
      dynamism,
      std::normal_distribution<float>(0.0f, 1.0f));
}

TensorPtr randint_strided(
    int64_t low,
    int64_t high,
    std::vector<executorch::aten::SizesType> sizes,
    std::vector<executorch::aten::StridesType> strides,
    executorch::aten::ScalarType type,
    executorch::aten::TensorShapeDynamism dynamism) {
  return random_strided(
      std::move(sizes),
      std::move(strides),
      type,
      dynamism,
      std::uniform_int_distribution<int64_t>(low, high - 1));
}

} // namespace extension
} // namespace executorch
