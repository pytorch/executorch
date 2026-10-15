// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "value_spec.h"
#include "config.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {

int get_seed() {
  static int seed = 42;
  return seed++;
}

int get_seed_or_explicit(int explicit_seed) {
  if (explicit_seed >= 0) {
    return explicit_seed;
  }
  return get_seed();
}

// Forward declarations for data generation utilities
void generate_random_float_data(
    std::vector<float>& data,
    float min_val = -1.0f,
    float max_val = 1.0f,
    int explicit_seed = -1);
void generate_random_int_data(
    std::vector<int32_t>& data,
    int min_val = -10,
    int max_val = 10,
    int explicit_seed = -1);
void generate_randint_float_data(
    std::vector<float>& data,
    int min_val = -10,
    int max_val = 10,
    int explicit_seed = -1);
void generate_randint_half_data(
    std::vector<uint16_t>& data,
    int min_val = -10,
    int max_val = 10,
    int explicit_seed = -1);
void generate_random_int8_data(
    std::vector<int8_t>& data,
    int8_t min_val = -10,
    int8_t max_val = 10,
    int explicit_seed = -1);
void generate_random_uint8_data(
    std::vector<uint8_t>& data,
    uint8_t min_val = 0,
    uint8_t max_val = 255,
    int explicit_seed = -1);
void generate_random_2xint4_data(
    std::vector<uint8_t>& data,
    int explicit_seed = -1);
void generate_random_2xint4_data(
    std::vector<int8_t>& data,
    int explicit_seed = -1);
void generate_random_int4_data(
    std::vector<int8_t>& data,
    int8_t min_val = -8,
    int8_t max_val = 7,
    int explicit_seed = -1);
void generate_ones_data(std::vector<float>& data);
void generate_zeros_data(std::vector<float>& data);

// Convert a float32 value to IEEE 754 half-precision (uint16_t)
uint16_t float_to_half(float value) {
  uint32_t float_bits;
  std::memcpy(&float_bits, &value, sizeof(float));
  uint32_t sign = (float_bits >> 31) & 0x1;
  int32_t exponent = static_cast<int32_t>((float_bits >> 23) & 0xFF) - 127;
  uint32_t mantissa = float_bits & 0x7FFFFF;

  uint16_t half_val;
  if (exponent > 15) {
    half_val = static_cast<uint16_t>((sign << 15) | 0x7C00); // Inf
  } else if (exponent < -14) {
    half_val = static_cast<uint16_t>(sign << 15); // Zero / subnormal
  } else {
    half_val = static_cast<uint16_t>(
        (sign << 15) | (static_cast<uint32_t>(exponent + 15) << 10) |
        (mantissa >> 13));
  }
  return half_val;
}

// Convert a IEEE 754 half-precision (uint16_t) value to float32
float half_to_float(uint16_t half_val) {
  uint32_t sign = (half_val >> 15) & 0x1;
  uint32_t exponent = (half_val >> 10) & 0x1F;
  uint32_t mantissa = half_val & 0x3FF;

  float result;
  if (exponent == 0) {
    result = std::ldexp(static_cast<float>(mantissa), -24);
  } else if (exponent == 31) {
    result = mantissa ? std::numeric_limits<float>::quiet_NaN()
                      : std::numeric_limits<float>::infinity();
  } else {
    result = std::ldexp(
        1.0f + static_cast<float>(mantissa) / 1024.0f, exponent - 15);
  }
  if (sign) {
    result = -result;
  }
  return result;
}

// ValueSpec implementation
void ValueSpec::ensure_unique_data() const {
  if (data_.use_count() != 1) {
    data_ = std::make_shared<TensorData>(*data_);
  }
}

void ValueSpec::ensure_unique_reference_data() const {
  if (reference_data_.use_count() != 1) {
    reference_data_ = std::make_shared<TensorData>(*reference_data_);
  }
}

void ValueSpec::generate_tensor_data(int seed) const {
  if (spec_type != SpecType::Tensor) {
    return;
  }

  ensure_unique_data();
  auto& float_data = data_->float_data;
  auto& int32_data = data_->int32_data;
  auto& half_data = data_->half_data;
  auto& int8_data = data_->int8_data;
  auto& uint8_data = data_->uint8_data;

  int64_t num_elements = numel();

  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (dtype) {
    case vkapi::kFloat: {
      float_data.resize(num_elements);
      if (data_gen_type == DataGenType::RANDOM) {
        generate_random_float_data(float_data, -1.0f, 1.0f, seed);
      } else if (data_gen_type == DataGenType::RANDOM_SCALES) {
        generate_random_float_data(float_data, 0.005, 0.015, seed);
      } else if (data_gen_type == DataGenType::RANDINT) {
        generate_randint_float_data(float_data, -10, 10, seed);
      } else if (data_gen_type == DataGenType::RANDINT8) {
        generate_randint_float_data(float_data, -128, 127, seed);
      } else if (data_gen_type == DataGenType::RANDINT4) {
        generate_randint_float_data(float_data, -8, 7, seed);
      } else if (data_gen_type == DataGenType::ONES) {
        generate_ones_data(float_data);
      } else if (data_gen_type == DataGenType::ZEROS) {
        generate_zeros_data(float_data);
      } else {
        generate_zeros_data(float_data);
      }
      break;
    }
    case vkapi::kHalf: {
      half_data.resize(num_elements);
      if (data_gen_type == DataGenType::RANDOM) {
        // Generate random float data first, then convert to IEEE 754 half.
        std::vector<float> temp_data(num_elements);
        generate_random_float_data(temp_data, -1.0f, 1.0f, seed);
        for (size_t i = 0; i < temp_data.size(); ++i) {
          half_data[i] = float_to_half(temp_data[i]);
        }
      } else if (data_gen_type == DataGenType::RANDOM_SCALES) {
        // Generate random scales in float, then convert to proper fp16
        std::vector<float> temp_data(num_elements);
        generate_random_float_data(temp_data, 0.005f, 0.015f, seed);
        for (size_t i = 0; i < temp_data.size(); ++i) {
          half_data[i] = float_to_half(temp_data[i]);
        }
      } else if (data_gen_type == DataGenType::RANDINT) {
        generate_randint_half_data(half_data, -10, 10, seed);
      } else if (data_gen_type == DataGenType::RANDINT8) {
        generate_randint_half_data(half_data, -128, 127, seed);
      } else if (data_gen_type == DataGenType::RANDINT4) {
        generate_randint_half_data(half_data, -8, 7, seed);
      } else if (data_gen_type == DataGenType::ONES) {
        std::fill(half_data.begin(), half_data.end(), float_to_half(1.0f));
      } else if (data_gen_type == DataGenType::ZEROS) {
        std::fill(
            half_data.begin(),
            half_data.end(),
            static_cast<uint16_t>(0)); // 0.0 in half
      } else {
        std::fill(
            half_data.begin(),
            half_data.end(),
            static_cast<uint16_t>(0)); // 0.0 in half
      }
      break;
    }
    case vkapi::kInt: {
      int32_data.resize(num_elements);
      if (data_gen_type == DataGenType::RANDOM) {
        generate_random_int_data(int32_data, -10, 10, seed);
      } else if (data_gen_type == DataGenType::RANDINT) {
        generate_random_int_data(
            int32_data,
            -10,
            10,
            seed); // For int type, RANDINT is same as RANDOM
      } else if (data_gen_type == DataGenType::RANDINT8) {
        generate_random_int_data(int32_data, -128, 127, seed);
      } else if (data_gen_type == DataGenType::RANDINT4) {
        generate_random_int_data(int32_data, -8, 7, seed);
      } else if (data_gen_type == DataGenType::ONES) {
        std::fill(int32_data.begin(), int32_data.end(), 1);
      } else if (data_gen_type == DataGenType::ZEROS) {
        std::fill(int32_data.begin(), int32_data.end(), 0);
      } else {
        std::fill(int32_data.begin(), int32_data.end(), 0);
      }
      break;
    }
    case vkapi::kChar: {
      int8_data.resize(num_elements);
      if (data_gen_type == DataGenType::RANDOM) {
        generate_random_int8_data(int8_data, -10, 10, seed);
      } else if (data_gen_type == DataGenType::RANDINT) {
        generate_random_int8_data(int8_data, -10, 10, seed);
      } else if (data_gen_type == DataGenType::RANDINT8) {
        generate_random_int8_data(int8_data, -128, 127, seed);
      } else if (data_gen_type == DataGenType::RANDINT4) {
        generate_random_2xint4_data(int8_data, seed);
      } else if (data_gen_type == DataGenType::ONES) {
        std::fill(int8_data.begin(), int8_data.end(), 1);
      } else if (data_gen_type == DataGenType::ONES_INT4) {
        int8_t packed_data = (1 << 4) | 1;
        std::fill(int8_data.begin(), int8_data.end(), packed_data);
      } else if (data_gen_type == DataGenType::ZEROS) {
        std::fill(int8_data.begin(), int8_data.end(), 0);
      } else {
        std::fill(int8_data.begin(), int8_data.end(), 0);
      }
      break;
    }
    case vkapi::kByte: {
      uint8_data.resize(num_elements);
      if (data_gen_type == DataGenType::RANDOM) {
        generate_random_uint8_data(uint8_data, 0, 255, seed);
      } else if (data_gen_type == DataGenType::RANDINT) {
        generate_random_uint8_data(uint8_data, 0, 255, seed);
      } else if (data_gen_type == DataGenType::RANDINT8) {
        generate_random_uint8_data(uint8_data, 0, 255, seed);
      } else if (data_gen_type == DataGenType::RANDINT4) {
        generate_random_2xint4_data(uint8_data, seed);
      } else if (data_gen_type == DataGenType::ONES) {
        std::fill(uint8_data.begin(), uint8_data.end(), 1);
      } else if (data_gen_type == DataGenType::ONES_INT4) {
        uint8_t packed_data = (9 << 4) | 9;
        std::fill(uint8_data.begin(), uint8_data.end(), packed_data);
      } else if (data_gen_type == DataGenType::ZEROS) {
        std::fill(uint8_data.begin(), uint8_data.end(), 0);
      } else {
        std::fill(uint8_data.begin(), uint8_data.end(), 0);
      }
      break;
    }
    default:
      // Default to float
      float_data.resize(num_elements);
      if (data_gen_type == DataGenType::RANDOM) {
        generate_random_float_data(float_data, -1.0f, 1.0f, seed);
      } else if (data_gen_type == DataGenType::RANDINT) {
        generate_randint_float_data(float_data, -10, 10, seed);
      } else if (data_gen_type == DataGenType::ONES) {
        generate_ones_data(float_data);
      } else if (data_gen_type == DataGenType::ZEROS) {
        generate_zeros_data(float_data);
      } else {
        generate_zeros_data(float_data);
      }
      break;
  }
}

int64_t ValueSpec::numel() const {
  if (spec_type == SpecType::Int || spec_type == SpecType::Float ||
      spec_type == SpecType::Bool) {
    return 1;
  } else if (spec_type == SpecType::IntList) {
    return sizes.empty() ? 0 : sizes[0];
  } else { // Tensor
    int64_t total = 1;
    for (int64_t size : sizes) {
      total *= size;
    }
    return total;
  }
}

size_t ValueSpec::nbytes() const {
  size_t element_size = 0;
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (dtype) {
    case vkapi::kFloat:
      element_size = sizeof(float);
      break;
    case vkapi::kHalf:
      element_size = sizeof(uint16_t);
      break;
    case vkapi::kInt:
      element_size = sizeof(int32_t);
      break;
    case vkapi::kChar:
      element_size = sizeof(int8_t);
      break;
    case vkapi::kByte:
      element_size = sizeof(uint8_t);
      break;
    default:
      element_size = sizeof(float); // Default fallback
      break;
  }
  return numel() * element_size;
}

std::string ValueSpec::to_string() const {
  std::string result = "ValueSpec(";

  switch (spec_type) {
    case SpecType::Tensor:
      result += "type=Tensor, sizes=[";
      break;
    case SpecType::IntList:
      result += "type=IntList, count=";
      result += std::to_string(sizes.empty() ? 0 : sizes[0]);
      result += ", data_gen=";
      result += (data_gen_type == DataGenType::FIXED) ? "FIXED" : "RANDOM";
      result += ")";
      return result;
    case SpecType::Int:
      result += "type=Int, value=";
      result += std::to_string(get_int_value());
      result += ", data_gen=";
      result += (data_gen_type == DataGenType::FIXED) ? "FIXED" : "RANDOM";
      result += ")";
      return result;
    case SpecType::Float:
      result += "type=Float, value=";
      result += std::to_string(get_float_value());
      result += ", data_gen=";
      result += (data_gen_type == DataGenType::FIXED) ? "FIXED" : "RANDOM";
      result += ")";
      return result;
    case SpecType::Bool:
      result += "type=Bool, value=";
      result += get_bool_value() ? "true" : "false";
      result += ", data_gen=";
      result += (data_gen_type == DataGenType::FIXED) ? "FIXED" : "RANDOM";
      result += ")";
      return result;
    case SpecType::String:
      result += "type=String, value=\"";
      result += get_string_value();
      result += "\")";
      return result;
  }

  for (size_t i = 0; i < sizes.size(); ++i) {
    result += std::to_string(sizes[i]);
    if (i < sizes.size() - 1) {
      result += ", ";
    }
  }
  result += "]";

  if (spec_type == SpecType::Tensor) {
    result += ", dtype=";
    // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
    switch (dtype) {
      case vkapi::kFloat:
        result += "float";
        break;
      case vkapi::kHalf:
        result += "half";
        break;
      case vkapi::kInt:
        result += "int32";
        break;
      case vkapi::kChar:
        result += "int8";
        break;
      case vkapi::kByte:
        result += "uint8";
        break;
      default:
        result += "unknown";
        break;
    }

    result += ", memory_layout=";
    // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
    switch (memory_layout) {
      case utils::kWidthPacked:
        result += "WidthPacked";
        break;
      case utils::kHeightPacked:
        result += "HeightPacked";
        break;
      case utils::kChannelsPacked:
        result += "ChannelsPacked";
        break;
      default:
        result += "unknown";
        break;
    }

    result += ", storage_type=";
    // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
    switch (storage_type) {
      case utils::kTexture3D:
        result += "Texture3D";
        break;
      case utils::kBuffer:
        result += "Buffer";
        break;
      default:
        result += "unknown";
        break;
    }
  }

  result += ", data_gen=";
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (data_gen_type) {
    case DataGenType::FIXED:
      result += "FIXED";
      break;
    case DataGenType::RANDOM:
      result += "RANDOM";
      break;
    case DataGenType::RANDINT:
      result += "RANDINT";
      break;
    case DataGenType::RANDINT8:
      result += "RANDINT8";
      break;
    case DataGenType::RANDINT4:
      result += "RANDINT4";
      break;
    case DataGenType::ONES:
      result += "ONES";
      break;
    case DataGenType::ZEROS:
      result += "ZEROS";
      break;
    default:
      result += "unknown";
      break;
  }
  result += ")";
  return result;
}

// Additional ValueSpec methods
void ValueSpec::resize_data(size_t new_size) {
  // Generate first so a deferred tensor keeps its data-gen pattern (resized,
  // not pinned to zeros by the data_generated_ flag set below).
  ensure_data_generated();
  ensure_unique_data();
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (dtype) {
    case vkapi::kFloat:
      data_->float_data.resize(new_size);
      break;
    case vkapi::kHalf:
      data_->half_data.resize(new_size);
      break;
    case vkapi::kInt:
      data_->int32_data.resize(new_size);
      break;
    case vkapi::kChar:
      data_->int8_data.resize(new_size);
      break;
    case vkapi::kByte:
      data_->uint8_data.resize(new_size);
      break;
    default:
      data_->float_data.resize(new_size);
      break;
  }
  data_generated_ = true;
}

void* ValueSpec::get_mutable_data_ptr() {
  ensure_data_generated();
  ensure_unique_data();
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (dtype) {
    case vkapi::kFloat:
      return data_->float_data.data();
    case vkapi::kHalf:
      return data_->half_data.data();
    case vkapi::kInt:
      return data_->int32_data.data();
    case vkapi::kChar:
      return data_->int8_data.data();
    case vkapi::kByte:
      return data_->uint8_data.data();
    default:
      return data_->float_data.data();
  }
}

float ValueSpec::get_element(size_t index) const {
  ensure_data_generated();
  if (index >= static_cast<size_t>(numel())) {
    return 0.0f;
  }

  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (dtype) {
    case vkapi::kFloat:
      return index < data_->float_data.size() ? data_->float_data[index] : 0.0f;
    case vkapi::kHalf:
      return index < data_->half_data.size()
          ? half_to_float(data_->half_data[index])
          : 0.0f;
    case vkapi::kInt:
      return index < data_->int32_data.size()
          ? static_cast<float>(data_->int32_data[index])
          : 0.0f;
    case vkapi::kChar:
      return index < data_->int8_data.size()
          ? static_cast<float>(data_->int8_data[index])
          : 0.0f;
    case vkapi::kByte:
      return index < data_->uint8_data.size()
          ? static_cast<float>(data_->uint8_data[index])
          : 0.0f;
    default:
      return 0.0f;
  }
}

const void* ValueSpec::get_data_ptr() const {
  ensure_data_generated();
  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (dtype) {
    case vkapi::kFloat:
      return data_->float_data.data();
    case vkapi::kHalf:
      return data_->half_data.data();
    case vkapi::kInt:
      return data_->int32_data.data();
    case vkapi::kChar:
      return data_->int8_data.data();
    case vkapi::kByte:
      return data_->uint8_data.data();
    default:
      throw std::runtime_error("Unsupported data type for get_data_ptr");
  }
}

void generate_random_float_data(
    std::vector<float>& data,
    float min_val,
    float max_val,
    int explicit_seed) {
  std::mt19937 gen(get_seed_or_explicit(explicit_seed));
  std::uniform_real_distribution<float> dis(min_val, max_val);
  for (auto& val : data) {
    val = dis(gen);
  }
}

void generate_random_int_data(
    std::vector<int32_t>& data,
    int min_val,
    int max_val,
    int explicit_seed) {
  std::mt19937 gen(get_seed_or_explicit(explicit_seed));
  std::uniform_int_distribution<int32_t> dis(min_val, max_val);
  for (auto& val : data) {
    val = dis(gen);
  }
}

void generate_randint_float_data(
    std::vector<float>& data,
    int min_val,
    int max_val,
    int explicit_seed) {
  std::mt19937 gen(get_seed_or_explicit(explicit_seed));
  std::uniform_int_distribution<int32_t> dis(min_val, max_val);
  for (auto& val : data) {
    val = static_cast<float>(dis(gen));
  }
}

void generate_randint_half_data(
    std::vector<uint16_t>& data,
    int min_val,
    int max_val,
    int explicit_seed) {
  std::mt19937 gen(get_seed_or_explicit(explicit_seed));
  std::uniform_int_distribution<int32_t> dis(min_val, max_val);
  for (auto& val : data) {
    val = float_to_half(static_cast<float>(dis(gen)));
  }
}

void generate_ones_data(std::vector<float>& data) {
  std::fill(data.begin(), data.end(), 1.0f);
}

void generate_random_int8_data(
    std::vector<int8_t>& data,
    int8_t min_val,
    int8_t max_val,
    int explicit_seed) {
  std::mt19937 gen(get_seed_or_explicit(explicit_seed));
  std::uniform_int_distribution<int16_t> dis(min_val, max_val);
  for (auto& val : data) {
    val = static_cast<int8_t>(dis(gen));
  }
}

void generate_random_uint8_data(
    std::vector<uint8_t>& data,
    uint8_t min_val,
    uint8_t max_val,
    int explicit_seed) {
  std::mt19937 gen(get_seed_or_explicit(explicit_seed));
  std::uniform_int_distribution<uint16_t> dis(min_val, max_val);
  for (auto& val : data) {
    val = static_cast<uint8_t>(dis(gen));
  }
}

void generate_random_int4_data(
    std::vector<int8_t>& data,
    int8_t min_val,
    int8_t max_val,
    int explicit_seed) {
  std::mt19937 gen(get_seed_or_explicit(explicit_seed));
  std::uniform_int_distribution<int16_t> dis(min_val, max_val);
  for (auto& val : data) {
    val = static_cast<int8_t>(dis(gen));
  }
}

void generate_random_2xint4_data(std::vector<int8_t>& data, int explicit_seed) {
  std::mt19937 gen(get_seed_or_explicit(explicit_seed));
  std::uniform_int_distribution<int16_t> dis(-8, 7); // Signed 4-bit range
  for (auto& val : data) {
    // Generate two separate 4-bit values
    int8_t lower_4bits = static_cast<int8_t>(dis(gen)) & 0x0F;
    int8_t upper_4bits = static_cast<int8_t>(dis(gen)) & 0x0F;
    // Pack them into a single 8-bit value
    val = (upper_4bits << 4) | lower_4bits;
  }
}

void generate_random_2xint4_data(
    std::vector<uint8_t>& data,
    int explicit_seed) {
  std::mt19937 gen(get_seed_or_explicit(explicit_seed));
  std::uniform_int_distribution<uint16_t> dis(0, 15); // Unsigned 4-bit range
  for (auto& val : data) {
    // Generate two separate 4-bit values
    uint8_t lower_4bits = static_cast<uint8_t>(dis(gen)) & 0x0F;
    uint8_t upper_4bits = static_cast<uint8_t>(dis(gen)) & 0x0F;
    // Pack them into a single 8-bit value
    val = (upper_4bits << 4) | lower_4bits;
  }
}

void generate_zeros_data(std::vector<float>& data) {
  std::fill(data.begin(), data.end(), 0.0f);
}

// Correctness checking against reference data
bool ValueSpec::validate_against_reference(
    float abs_tolerance,
    float rel_tolerance) const {
  // Only validate float and half tensors. For half tensors, convert the
  // computed half data to float for comparison against the fp32 reference.
  if (!is_tensor() || (dtype != vkapi::kFloat && dtype != vkapi::kHalf)) {
    return true; // Skip validation for non-float/half or non-tensor types
  }

  // For kHalf, materialize the GPU output as float so the same tolerance
  // machinery can compare against the (always-float) reference data.
  std::vector<float> half_as_float;
  if (dtype == vkapi::kHalf) {
    const auto& half_bits = get_half_data();
    half_as_float.resize(half_bits.size());
    for (size_t i = 0; i < half_bits.size(); ++i) {
      half_as_float[i] = half_to_float(half_bits[i]);
    }
  }
  // Materialize computed data as float32 for comparison. The dtype is
  // guaranteed to be float or half by the early-out above.
  const std::vector<float>& computed_data =
      (dtype == vkapi::kHalf) ? half_as_float : get_float_data();
  const auto& reference_data = get_ref_float_data();

  // Skip validation if no reference data is available
  if (reference_data.empty()) {
    return true;
  }

  // Check if sizes match
  if (computed_data.size() != reference_data.size()) {
    if (debugging()) {
      std::cout << "Size mismatch: computed=" << computed_data.size()
                << ", reference=" << reference_data.size() << std::endl;
    }
    return false;
  }

  // Element-wise comparison with both absolute and relative tolerance
  size_t num_mismatched = 0;
  size_t first_mismatch = 0;
  for (size_t i = 0; i < computed_data.size(); ++i) {
    float diff = std::abs(computed_data[i] - reference_data[i]);
    float abs_ref = std::abs(reference_data[i]);

    // Check if either absolute or relative tolerance condition is satisfied
    bool abs_tolerance_ok = diff <= abs_tolerance;
    bool rel_tolerance_ok = diff <= rel_tolerance * abs_ref;

    if (!abs_tolerance_ok && !rel_tolerance_ok) {
      if (num_mismatched == 0) {
        first_mismatch = i;
        std::cout << "Mismatch at element " << i
                  << ": computed=" << computed_data[i]
                  << ", reference=" << reference_data[i] << ", diff=" << diff
                  << ", abs_tolerance=" << abs_tolerance
                  << ", rel_tolerance=" << rel_tolerance
                  << ", rel_threshold=" << (rel_tolerance * abs_ref)
                  << std::endl;
      }
      num_mismatched++;
    }
  }
  if (num_mismatched > 0) {
    std::cout << "  total mismatched: " << num_mismatched << " / "
              << computed_data.size() << " (first at " << first_mismatch << ")"
              << std::endl;
    // For 2D outputs, print a per-16x16-tile mismatch-count map to expose
    // the spatial structure of the failure (e.g. zeroed MMA subtiles).
    if (sizes.size() == 2) {
      const int64_t Mr = sizes[0];
      const int64_t Nc = sizes[1];
      std::cout << "  16x16-tile mismatch counts (rows=M/16, cols=N/16):"
                << std::endl;
      for (int64_t ti = 0; ti < (Mr + 15) / 16; ++ti) {
        std::cout << "    ";
        for (int64_t tj = 0; tj < (Nc + 15) / 16; ++tj) {
          int count = 0;
          for (int64_t r = ti * 16; r < std::min(Mr, (ti + 1) * 16); ++r) {
            for (int64_t c = tj * 16; c < std::min(Nc, (tj + 1) * 16); ++c) {
              float diff = std::abs(
                  computed_data[r * Nc + c] - reference_data[r * Nc + c]);
              float abs_ref = std::abs(reference_data[r * Nc + c]);
              if (diff > abs_tolerance && diff > rel_tolerance * abs_ref) {
                count++;
              }
            }
          }
          std::cout << std::setw(4) << count;
        }
        std::cout << std::endl;
      }
    }
    return false;
  }

  if (debugging()) {
    std::cout << "Correctness validation PASSED" << std::endl;
  }
  return true;
}

// Ensure data is generated for this ValueSpec
void ValueSpec::ensure_data_generated(int seed) const {
  if (data_generated_) {
    return;
  }
  generate_tensor_data(seed);
  data_generated_ = true;
}

void ValueSpec::share_data_from(const ValueSpec& other) {
  if (!is_tensor() || !other.is_tensor()) {
    return;
  }
  // Materialize the source first: sharing an ungenerated payload would let a
  // later access materialize each spec independently under different seeds.
  other.ensure_data_generated();
  data_ = other.data_;
  data_generated_ = true;
}

void ValueSpec::share_reference_from(const ValueSpec& other) {
  if (!is_tensor() || !other.is_tensor()) {
    return;
  }
  reference_data_ = other.reference_data_;
}

// ValueSpec data printing utilities
void print_valuespec_data(
    const ValueSpec& spec,
    const std::string& name,
    const bool print_ref_data,
    size_t max_elements,
    int precision) {
  std::cout << "\n" << name << " Data:" << std::endl;
  std::cout << "  Type: " << spec.to_string() << std::endl;

  if (!spec.is_tensor()) {
    if (spec.is_int()) {
      std::cout << "  Value: " << spec.get_int_value() << std::endl;
    } else if (spec.is_int_list()) {
      const auto& int_list = spec.get_int_list();
      std::cout << "  Values: [";
      size_t print_count = std::min(max_elements, int_list.size());
      for (size_t i = 0; i < print_count; ++i) {
        std::cout << int_list[i];
        if (i < print_count - 1) {
          std::cout << ", ";
        }
      }
      if (int_list.size() > max_elements) {
        std::cout << ", ... (" << (int_list.size() - max_elements) << " more)";
      }
      std::cout << "]" << std::endl;
    }
    return;
  }

  // Print tensor data
  size_t total_elements = spec.numel();
  size_t print_count = std::min(max_elements, total_elements);

  std::cout << "  Total elements: " << total_elements << std::endl;
  std::cout << "  Data (first " << print_count << " elements): [";

  std::cout << std::fixed << std::setprecision(precision);

  // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
  switch (spec.dtype) {
    case vkapi::kFloat: {
      auto data = spec.get_float_data().data();
      if (print_ref_data) {
        data = spec.get_ref_float_data().data();
      }
      for (size_t i = 0; i < print_count; ++i) {
        std::cout << data[i];
        if (i < print_count - 1) {
          std::cout << ", ";
        }
      }
      break;
    }
    case vkapi::kHalf: {
      if (print_ref_data) {
        const auto& ref = spec.get_ref_float_data();
        for (size_t i = 0; i < print_count; ++i) {
          std::cout << ref[i];
          if (i < print_count - 1) {
            std::cout << ", ";
          }
        }
        break;
      }
      const auto& data = spec.get_half_data();
      for (size_t i = 0; i < print_count; ++i) {
        // Convert IEEE 754 half-precision bit pattern back to float.
        float value = half_to_float(data[i]);
        std::cout << value;
        if (i < print_count - 1) {
          std::cout << ", ";
        }
      }
      break;
    }
    case vkapi::kInt: {
      const auto& data = spec.get_int32_data();
      for (size_t i = 0; i < print_count; ++i) {
        std::cout << data[i];
        if (i < print_count - 1) {
          std::cout << ", ";
        }
      }
      break;
    }
    case vkapi::kChar: {
      const auto& data = spec.get_int8_data();
      if (spec.is_int4()) {
        // Print each 4-bit value individually
        size_t element_count = 0;
        for (size_t i = 0; i < data.size() && element_count < print_count;
             ++i) {
          // Extract lower 4 bits (signed)
          int8_t lower_4bits = data[i] & 0x0F;
          if (lower_4bits > 7) {
            lower_4bits -= 16; // Convert to signed
          }
          std::cout << static_cast<int>(lower_4bits);
          element_count++;

          if (element_count < print_count) {
            std::cout << ", ";
            // Extract upper 4 bits (signed)
            int8_t upper_4bits = (data[i] >> 4) & 0x0F;
            if (upper_4bits > 7) {
              upper_4bits -= 16; // Convert to signed
            }
            std::cout << static_cast<int>(upper_4bits);
            element_count++;

            if (element_count < print_count) {
              std::cout << ", ";
            }
          }
        }
      } else {
        for (size_t i = 0; i < print_count; ++i) {
          std::cout << static_cast<int>(data[i]);
          if (i < print_count - 1) {
            std::cout << ", ";
          }
        }
      }
      break;
    }
    case vkapi::kByte: {
      const auto& data = spec.get_uint8_data();
      if (spec.is_int4()) {
        // Print each 4-bit value individually
        size_t element_count = 0;
        for (size_t i = 0; i < data.size() && element_count < print_count;
             ++i) {
          // Extract lower 4 bits
          uint8_t lower_4bits = data[i] & 0x0F;
          std::cout << static_cast<unsigned int>(lower_4bits);
          element_count++;

          if (element_count < print_count) {
            std::cout << ", ";
            // Extract upper 4 bits
            uint8_t upper_4bits = (data[i] >> 4) & 0x0F;
            std::cout << static_cast<unsigned int>(upper_4bits);
            element_count++;

            if (element_count < print_count) {
              std::cout << ", ";
            }
          }
        }
      } else {
        for (size_t i = 0; i < print_count; ++i) {
          std::cout << static_cast<unsigned int>(data[i]);
          if (i < print_count - 1) {
            std::cout << ", ";
          }
        }
      }
      break;
    }
    default:
      std::cout << "unsupported data type";
      break;
  }

  if (total_elements > max_elements) {
    std::cout << ", ... (" << (total_elements - max_elements) << " more)";
  }
  std::cout << "]" << std::endl;

  // Print some statistics for tensor data
  if (total_elements > 0) {
    float min_val = 0.0f, max_val = 0.0f, sum = 0.0f;
    bool first = true;

    for (size_t i = 0; i < total_elements; ++i) {
      float val = spec.get_element(i);
      if (first) {
        min_val = max_val = val;
        first = false;
      } else {
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
      }
      sum += val;
    }

    float mean = sum / total_elements;
    std::cout << "  Statistics: min=" << std::setprecision(precision) << min_val
              << ", max=" << max_val << ", mean=" << mean << ", sum=" << sum
              << std::endl;
  }
}

} // namespace prototyping
} // namespace vulkan
} // namespace executorch
