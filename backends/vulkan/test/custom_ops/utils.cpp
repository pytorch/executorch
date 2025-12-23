// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "utils.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace executorch {
namespace vulkan {
namespace prototyping {

int get_seed() {
  static int seed = 42;
  return seed++;
}

// Forward declarations for data generation utilities
void generate_random_float_data(
    std::vector<float>& data,
    float min_val = -1.0f,
    float max_val = 1.0f);
void generate_random_int_data(
    std::vector<int32_t>& data,
    int min_val = -10,
    int max_val = 10);
void generate_randint_float_data(
    std::vector<float>& data,
    int min_val = -10,
    int max_val = 10);
void generate_randint_half_data(
    std::vector<uint16_t>& data,
    int min_val = -10,
    int max_val = 10);
void generate_random_int8_data(
    std::vector<int8_t>& data,
    int8_t min_val = -10,
    int8_t max_val = 10);
void generate_random_uint8_data(
    std::vector<uint8_t>& data,
    uint8_t min_val = 0,
    uint8_t max_val = 255);
void generate_random_2xint4_data(std::vector<uint8_t>& data);
void generate_random_2xint4_data(std::vector<int8_t>& data);
void generate_random_int4_data(
    std::vector<int8_t>& data,
    int8_t min_val = -8,
    int8_t max_val = 7);
void generate_ones_data(std::vector<float>& data);
void generate_zeros_data(std::vector<float>& data);

// Output and latency printing utilities
namespace {
static int print_output_enabled = 0;
static int print_latencies_enabled = 0;
static int gpu_timestamps_enabled = 0;
static int debugging_enabled = 0;
} // namespace

bool print_output() {
  return print_output_enabled > 0;
}

void set_print_output(bool print_output) {
  print_output_enabled = print_output ? 1 : 0;
}

bool print_latencies() {
  return print_latencies_enabled > 0;
}

void set_print_latencies(bool print_latencies) {
  print_latencies_enabled = print_latencies ? 1 : 0;
}

bool use_gpu_timestamps() {
  return gpu_timestamps_enabled > 0;
}

void set_use_gpu_timestamps(bool use_timestamps) {
  gpu_timestamps_enabled = use_timestamps ? 1 : 0;
}

bool debugging() {
  return debugging_enabled > 0;
}

void set_debugging(bool enable_debugging) {
  debugging_enabled = enable_debugging ? 1 : 0;
}

// ValueSpec implementation
void ValueSpec::generate_tensor_data() {
  if (spec_type != SpecType::Tensor) {
    return;
  }

  int64_t num_elements = numel();

  switch (dtype) {
    case vkapi::kFloat: {
      float_data.resize(num_elements);
      if (data_gen_type == DataGenType::RANDOM) {
        generate_random_float_data(float_data);
      } else if (data_gen_type == DataGenType::RANDOM_SCALES) {
        generate_random_float_data(float_data, 0.005, 0.015);
      } else if (data_gen_type == DataGenType::RANDINT) {
        generate_randint_float_data(float_data);
      } else if (data_gen_type == DataGenType::RANDINT8) {
        generate_randint_float_data(float_data, -128, 127);
      } else if (data_gen_type == DataGenType::RANDINT4) {
        generate_randint_float_data(float_data, -8, 7);
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
        // Generate random float data first, then convert to half
        std::vector<float> temp_data(num_elements);
        generate_random_float_data(temp_data);
        for (size_t i = 0; i < temp_data.size(); ++i) {
          // Simple conversion to uint16_t representation of half
          half_data[i] = static_cast<uint16_t>(temp_data[i] * 32767.0f);
        }
      } else if (data_gen_type == DataGenType::RANDINT) {
        generate_randint_half_data(half_data);
      } else if (data_gen_type == DataGenType::RANDINT8) {
        generate_randint_half_data(half_data, -128, 127);
      } else if (data_gen_type == DataGenType::RANDINT4) {
        generate_randint_half_data(half_data, -8, 7);
      } else if (data_gen_type == DataGenType::ONES) {
        std::fill(
            half_data.begin(),
            half_data.end(),
            static_cast<uint16_t>(32767)); // 1.0 in half
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
        generate_random_int_data(int32_data);
      } else if (data_gen_type == DataGenType::RANDINT) {
        generate_random_int_data(
            int32_data); // For int type, RANDINT is same as RANDOM
      } else if (data_gen_type == DataGenType::RANDINT8) {
        generate_random_int_data(int32_data, -128, 127);
      } else if (data_gen_type == DataGenType::RANDINT4) {
        generate_random_int_data(int32_data, -8, 7);
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
        generate_random_int8_data(int8_data);
      } else if (data_gen_type == DataGenType::RANDINT) {
        generate_random_int8_data(int8_data);
      } else if (data_gen_type == DataGenType::RANDINT8) {
        generate_random_int8_data(int8_data, -128, 127);
      } else if (data_gen_type == DataGenType::RANDINT4) {
        generate_random_2xint4_data(int8_data);
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
        generate_random_uint8_data(uint8_data);
      } else if (data_gen_type == DataGenType::RANDINT) {
        generate_random_uint8_data(uint8_data);
      } else if (data_gen_type == DataGenType::RANDINT8) {
        generate_random_uint8_data(uint8_data, 0, 255);
      } else if (data_gen_type == DataGenType::RANDINT4) {
        generate_random_2xint4_data(uint8_data);
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
        generate_random_float_data(float_data);
      } else if (data_gen_type == DataGenType::RANDINT) {
        generate_randint_float_data(float_data);
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
  }

  for (size_t i = 0; i < sizes.size(); ++i) {
    result += std::to_string(sizes[i]);
    if (i < sizes.size() - 1)
      result += ", ";
  }
  result += "]";

  if (spec_type == SpecType::Tensor) {
    result += ", dtype=";
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
  switch (dtype) {
    case vkapi::kFloat:
      float_data.resize(new_size);
      break;
    case vkapi::kHalf:
      half_data.resize(new_size);
      break;
    case vkapi::kInt:
      int32_data.resize(new_size);
      break;
    case vkapi::kChar:
      int8_data.resize(new_size);
      break;
    case vkapi::kByte:
      uint8_data.resize(new_size);
      break;
    default:
      float_data.resize(new_size);
      break;
  }
}

void* ValueSpec::get_mutable_data_ptr() {
  switch (dtype) {
    case vkapi::kFloat:
      return float_data.data();
    case vkapi::kHalf:
      return half_data.data();
    case vkapi::kInt:
      return int32_data.data();
    case vkapi::kChar:
      return int8_data.data();
    case vkapi::kByte:
      return uint8_data.data();
    default:
      return float_data.data();
  }
}

float ValueSpec::get_element(size_t index) const {
  if (index >= static_cast<size_t>(numel())) {
    return 0.0f;
  }

  switch (dtype) {
    case vkapi::kFloat:
      return index < float_data.size() ? float_data[index] : 0.0f;
    case vkapi::kHalf:
      return index < half_data.size() ? (half_data[index] / 32767.0f) : 0.0f;
    case vkapi::kInt:
      return index < int32_data.size() ? static_cast<float>(int32_data[index])
                                       : 0.0f;
    case vkapi::kChar:
      return index < int8_data.size() ? static_cast<float>(int8_data[index])
                                      : 0.0f;
    case vkapi::kByte:
      return index < uint8_data.size() ? static_cast<float>(uint8_data[index])
                                       : 0.0f;
    default:
      return 0.0f;
  }
}

const void* ValueSpec::get_data_ptr() const {
  switch (dtype) {
    case vkapi::kFloat:
      return float_data.data();
    case vkapi::kHalf:
      return half_data.data();
    case vkapi::kInt:
      return int32_data.data();
    case vkapi::kChar:
      return int8_data.data();
    case vkapi::kByte:
      return uint8_data.data();
    default:
      throw std::runtime_error("Unsupported data type for get_data_ptr");
  }
}

void generate_random_float_data(
    std::vector<float>& data,
    float min_val,
    float max_val) {
  std::mt19937 gen(get_seed());
  std::uniform_real_distribution<float> dis(min_val, max_val);
  for (auto& val : data) {
    val = dis(gen);
  }
}

void generate_random_int_data(
    std::vector<int32_t>& data,
    int min_val,
    int max_val) {
  std::mt19937 gen(get_seed());
  std::uniform_int_distribution<int32_t> dis(min_val, max_val);
  for (auto& val : data) {
    val = dis(gen);
  }
}

void generate_randint_float_data(
    std::vector<float>& data,
    int min_val,
    int max_val) {
  std::mt19937 gen(get_seed());
  std::uniform_int_distribution<int32_t> dis(min_val, max_val);
  for (auto& val : data) {
    val = static_cast<float>(dis(gen));
  }
}

void generate_randint_half_data(
    std::vector<uint16_t>& data,
    int min_val,
    int max_val) {
  std::mt19937 gen(get_seed());
  std::uniform_int_distribution<int32_t> dis(min_val, max_val);
  for (auto& val : data) {
    val = static_cast<uint16_t>(std::abs(dis(gen)) % 65536);
  }
}

void generate_ones_data(std::vector<float>& data) {
  std::fill(data.begin(), data.end(), 1.0f);
}

void generate_random_int8_data(
    std::vector<int8_t>& data,
    int8_t min_val,
    int8_t max_val) {
  std::mt19937 gen(get_seed());
  std::uniform_int_distribution<int16_t> dis(min_val, max_val);
  for (auto& val : data) {
    val = static_cast<int8_t>(dis(gen));
  }
}

void generate_random_uint8_data(
    std::vector<uint8_t>& data,
    uint8_t min_val,
    uint8_t max_val) {
  std::mt19937 gen(get_seed());
  std::uniform_int_distribution<uint16_t> dis(min_val, max_val);
  for (auto& val : data) {
    val = static_cast<uint8_t>(dis(gen));
  }
}

void generate_random_int4_data(
    std::vector<int8_t>& data,
    int8_t min_val,
    int8_t max_val) {
  std::mt19937 gen(get_seed());
  std::uniform_int_distribution<int16_t> dis(min_val, max_val);
  for (auto& val : data) {
    val = static_cast<int8_t>(dis(gen));
  }
}

void generate_random_2xint4_data(std::vector<int8_t>& data) {
  std::mt19937 gen(get_seed());
  std::uniform_int_distribution<int16_t> dis(-8, 7); // Signed 4-bit range
  for (auto& val : data) {
    // Generate two separate 4-bit values
    int8_t lower_4bits = static_cast<int8_t>(dis(gen)) & 0x0F;
    int8_t upper_4bits = static_cast<int8_t>(dis(gen)) & 0x0F;
    // Pack them into a single 8-bit value
    val = (upper_4bits << 4) | lower_4bits;
  }
}

void generate_random_2xint4_data(std::vector<uint8_t>& data) {
  std::mt19937 gen(get_seed());
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
  // Only validate float tensors as specified in requirements
  if (dtype != vkapi::kFloat || !is_tensor()) {
    return true; // Skip validation for non-float or non-tensor types
  }

  const auto& computed_data = get_float_data();
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
  for (size_t i = 0; i < computed_data.size(); ++i) {
    float diff = std::abs(computed_data[i] - reference_data[i]);
    float abs_ref = std::abs(reference_data[i]);

    // Check if either absolute or relative tolerance condition is satisfied
    bool abs_tolerance_ok = diff <= abs_tolerance;
    bool rel_tolerance_ok = diff <= rel_tolerance * abs_ref;

    if (!abs_tolerance_ok && !rel_tolerance_ok) {
      std::cout << "Mismatch at element " << i
                << ": computed=" << computed_data[i]
                << ", reference=" << reference_data[i] << ", diff=" << diff
                << ", abs_tolerance=" << abs_tolerance
                << ", rel_tolerance=" << rel_tolerance
                << ", rel_threshold=" << (rel_tolerance * abs_ref) << std::endl;
      return false;
    }
  }

  if (debugging()) {
    std::cout << "Correctness validation PASSED" << std::endl;
  }
  return true;
}

// Helper function to collect GPU timing from querypool
float collect_gpu_timing_us(ComputeGraph& graph) {
  graph.context()->querypool().extract_results();
  const auto results = graph.context()->querypool().get_shader_timestamp_data();
  if (!results.empty()) {
    // Sum durations of all shaders that don't contain nchw_to or to_nchw
    float total_duration_us = 0.0f;
    for (const auto& shader_result : results) {
      if (shader_result.kernel_name.find("nchw_to") == std::string::npos &&
          shader_result.kernel_name.find("to_nchw") == std::string::npos &&
          shader_result.kernel_name.find("quantize_and_pack_4w4c") ==
              std::string::npos &&
          shader_result.kernel_name.find("unpack_4w4c_and_dequantize") ==
              std::string::npos) {
        // Calculate duration from start and end times, convert from ns to μs
        uint64_t duration_ns =
            shader_result.end_time_ns - shader_result.start_time_ns;
        total_duration_us += static_cast<float>(duration_ns) / 1000.0f;
      }
    }
    return total_duration_us;
  }
  return 0.0f;
}

// BenchmarkResult implementation
void BenchmarkResult::add_iter_timing(float time_us) {
  iter_timings.push_back(time_us);
}

float BenchmarkResult::get_avg_time_us() const {
  if (iter_timings.empty()) {
    return 0.0f;
  }

  float sum = 0.0f;
  for (float timing : iter_timings) {
    sum += timing;
  }
  return sum / iter_timings.size();
}

float BenchmarkResult::get_min_time_us() const {
  if (iter_timings.empty()) {
    return 0.0f;
  }

  return *std::min_element(iter_timings.begin(), iter_timings.end());
}

float BenchmarkResult::get_max_time_us() const {
  if (iter_timings.empty()) {
    return 0.0f;
  }

  return *std::max_element(iter_timings.begin(), iter_timings.end());
}

float BenchmarkResult::get_std_dev_us() const {
  if (iter_timings.size() <= 1) {
    return 0.0f;
  }

  float mean = get_avg_time_us();
  float sum_sq_diff = 0.0f;

  for (float timing : iter_timings) {
    float diff = timing - mean;
    sum_sq_diff += diff * diff;
  }

  return std::sqrt(sum_sq_diff / (iter_timings.size() - 1));
}

void BenchmarkResult::print_summary(
    int case_number,
    const std::string& size_info,
    float total_gflops) const {
  static constexpr int OPERATOR_NAME_WIDTH = 50;
  static constexpr int KERNEL_NAME_WIDTH = 70;
  static constexpr int SIZE_INFO_WIDTH = 20;
  static constexpr int TIMING_WIDTH = 20;
  static constexpr int GFLOPS_WIDTH = 20;
  static constexpr int CORRECTNESS_WIDTH = 10;

  std::string correctness_str;
  switch (correctness_status_) {
    case CorrectnessStatus::SKIPPED:
      correctness_str = "SKIPPED";
      break;
    case CorrectnessStatus::PASSED:
      correctness_str = "PASSED";
      break;
    case CorrectnessStatus::FAILED:
      correctness_str = "FAILED";
      break;
  }

  std::cout << std::left << std::setw(OPERATOR_NAME_WIDTH)
            << get_operator_name() << " " << std::left
            << std::setw(KERNEL_NAME_WIDTH) << get_kernel_name() << std::right
            << " " << std::setw(SIZE_INFO_WIDTH) << size_info
            << std::setw(TIMING_WIDTH) << std::fixed << std::setprecision(3)
            << get_avg_time_us() << " μs " << std::setw(GFLOPS_WIDTH)
            << std::fixed << std::setprecision(3) << total_gflops << " GFLOP/s "
            << std::setw(CORRECTNESS_WIDTH) << correctness_str << std::endl;
}

// TestResult implementation
void TestResult::add_result(const BenchmarkResult& result) {
  results_.push_back(result);
}

void TestResult::add_result(BenchmarkResult&& result) {
  results_.push_back(std::move(result));
}

void TestResult::print_summary() const {
  static constexpr int CASE_WIDTH = 80;
  static constexpr int KERNEL_NAME_WIDTH = 20;
  static constexpr int TIMING_WIDTH = 12;
  static constexpr int PASS_WIDTH = 8;

  if (results_.empty()) {
    std::cout << "No results to display" << std::endl;
    return;
  }

  std::cout << "\n=== " << operation_name_
            << " Performance Summary ===" << std::endl;
  print_separator();

  std::cout << std::left << std::setw(CASE_WIDTH) << "Case" << std::left
            << std::setw(KERNEL_NAME_WIDTH) << "Kernel Name" << std::left
            << std::setw(TIMING_WIDTH) << "Avg (μs)" << std::left
            << std::setw(TIMING_WIDTH) << "Min (μs)" << std::left
            << std::setw(TIMING_WIDTH) << "Max (μs)" << std::left
            << std::setw(TIMING_WIDTH) << "Std Dev" << std::left
            << std::setw(PASS_WIDTH) << "Pass" << std::endl;
  print_separator();

  for (size_t i = 0; i < results_.size(); ++i) {
    const auto& result = results_[i];
    bool vulkan_execute_succeeded =
        result.get_num_iterations() > 0 && result.get_avg_time_us() > 0.0f;
    std::cout << std::left << std::setw(CASE_WIDTH) << i + 1 << std::left
              << std::setw(KERNEL_NAME_WIDTH)
              << result.get_kernel_name().substr(0, KERNEL_NAME_WIDTH - 1)
              << std::left << std::setw(TIMING_WIDTH) << std::fixed
              << std::setprecision(3) << result.get_avg_time_us() << std::left
              << std::setw(TIMING_WIDTH) << std::fixed << std::setprecision(3)
              << result.get_min_time_us() << std::left
              << std::setw(TIMING_WIDTH) << std::fixed << std::setprecision(3)
              << result.get_max_time_us() << std::left
              << std::setw(TIMING_WIDTH) << std::fixed << std::setprecision(3)
              << result.get_std_dev_us() << std::left << std::setw(PASS_WIDTH)
              << (vulkan_execute_succeeded ? "✓" : "✗") << std::endl;
  }

  print_separator();
  std::cout << "Total cases: " << results_.size()
            << ", Passed: " << get_passed_count()
            << ", Failed: " << get_failed_count() << std::endl;
  std::cout << "Overall GFLOP/s: " << std::fixed << std::setprecision(3)
            << gflops_ << std::endl;
  std::cout << "Overall correctness: "
            << (correctness_passed_ ? "PASSED" : "FAILED") << std::endl;
}

void TestResult::print_detailed_results() const {
  if (results_.empty()) {
    std::cout << "No results to display" << std::endl;
    return;
  }

  std::cout << "\n=== " << operation_name_
            << " Detailed Results ===" << std::endl;

  for (size_t i = 0; i < results_.size(); ++i) {
    const auto& result = results_[i];
    bool vulkan_execute_succeeded =
        result.get_num_iterations() > 0 && result.get_avg_time_us() > 0.0f;
    std::cout << "\nCase " << i + 1 << ": " << result.get_kernel_name()
              << std::endl;
    std::cout << "  Iterations: " << result.get_num_iterations() << std::endl;
    std::cout << "  Average: " << std::fixed << std::setprecision(3)
              << result.get_avg_time_us() << " μs" << std::endl;
    std::cout << "  Min: " << std::fixed << std::setprecision(3)
              << result.get_min_time_us() << " μs" << std::endl;
    std::cout << "  Max: " << std::fixed << std::setprecision(3)
              << result.get_max_time_us() << " μs" << std::endl;
    std::cout << "  Std Dev: " << std::fixed << std::setprecision(3)
              << result.get_std_dev_us() << " μs" << std::endl;
    std::cout << "  Correctness: "
              << (vulkan_execute_succeeded ? "PASSED" : "FAILED") << std::endl;

    if (result.get_num_iterations() > 0) {
      std::cout << "  Individual timings (μs): ";
      const auto& timings = result.get_iter_timings();
      for (size_t j = 0; j < std::min(size_t(10), timings.size()); ++j) {
        std::cout << std::fixed << std::setprecision(1) << timings[j];
        if (j < std::min(size_t(10), timings.size()) - 1)
          std::cout << ", ";
      }
      if (timings.size() > 10) {
        std::cout << " ... (" << (timings.size() - 10) << " more)";
      }
      std::cout << std::endl;
    }
  }

  std::cout << "\nOverall Results:" << std::endl;
  std::cout << "  Total GFLOP/s: " << std::fixed << std::setprecision(3)
            << gflops_ << std::endl;
  std::cout << "  Overall correctness: "
            << (correctness_passed_ ? "PASSED" : "FAILED") << std::endl;
}

void TestResult::print_statistics() const {
  if (results_.empty()) {
    std::cout << "No results to display statistics for" << std::endl;
    return;
  }

  std::cout << "\n=== " << operation_name_ << " Statistics ===" << std::endl;
  std::cout << "Total test cases: " << results_.size() << std::endl;
  std::cout << "Passed: " << get_passed_count() << std::endl;
  std::cout << "Failed: " << get_failed_count() << std::endl;
  std::cout << "Success rate: " << std::fixed << std::setprecision(1)
            << (100.0f * get_passed_count() / results_.size()) << "%"
            << std::endl;

  if (get_passed_count() > 0) {
    std::cout << "Total average time: " << std::fixed << std::setprecision(3)
              << get_total_avg_time_us() << " μs" << std::endl;
    std::cout << "Total GFLOP/s: " << std::fixed << std::setprecision(3)
              << get_total_gflops() << std::endl;

    const auto* fastest = get_fastest_result();
    const auto* slowest = get_slowest_result();
    const auto* highest_gflops = get_highest_gflops_result();

    if (fastest) {
      std::cout << "Fastest case: " << fastest->get_kernel_name() << " ("
                << std::fixed << std::setprecision(3)
                << fastest->get_avg_time_us() << " μs)" << std::endl;
    }

    if (slowest) {
      std::cout << "Slowest case: " << slowest->get_kernel_name() << " ("
                << std::fixed << std::setprecision(3)
                << slowest->get_avg_time_us() << " μs)" << std::endl;
    }

    if (highest_gflops) {
      std::cout << "Best performing case: " << highest_gflops->get_kernel_name()
                << " (" << std::fixed << std::setprecision(3)
                << highest_gflops->get_avg_time_us() << " μs)" << std::endl;
    }
  }
}

void TestResult::print_brief_summary() const {
  print_separator();
  std::cout << "Summary Statistics:" << std::endl;

  if (get_passed_count() > 0) {
    std::cout << "Average execution time: " << std::fixed
              << std::setprecision(3) << get_total_avg_time_us() << " μs"
              << std::endl;
    std::cout << "Total throughput: " << std::fixed << std::setprecision(3)
              << get_gflops() << " GFLOP/s" << std::endl;
    std::cout << "Successful test cases: " << get_passed_count() << "/"
              << size() << std::endl;
    std::cout << "Overall correctness: "
              << (get_correctness_passed() ? "PASSED" : "FAILED") << std::endl;
  } else {
    std::cout << "No successful test cases to report" << std::endl;
  }
}

float TestResult::get_total_avg_time_us() const {
  if (results_.empty()) {
    return 0.0f;
  }

  float sum = 0.0f;
  size_t count = 0;

  for (const auto& result : results_) {
    bool vulkan_execute_succeeded =
        result.get_num_iterations() > 0 && result.get_avg_time_us() > 0.0f;
    if (vulkan_execute_succeeded) {
      sum += result.get_avg_time_us();
      count++;
    }
  }

  return count > 0 ? sum / count : 0.0f;
}

float TestResult::get_total_gflops() const {
  return gflops_;
}

size_t TestResult::get_passed_count() const {
  size_t count = 0;
  for (const auto& result : results_) {
    bool vulkan_execute_succeeded =
        result.get_num_iterations() > 0 && result.get_avg_time_us() > 0.0f;
    if (vulkan_execute_succeeded) {
      count++;
    }
  }
  return count;
}

size_t TestResult::get_failed_count() const {
  return results_.size() - get_passed_count();
}

const BenchmarkResult* TestResult::get_fastest_result() const {
  const BenchmarkResult* fastest = nullptr;

  for (const auto& result : results_) {
    bool vulkan_execute_succeeded =
        result.get_num_iterations() > 0 && result.get_avg_time_us() > 0.0f;
    if (vulkan_execute_succeeded) {
      if (!fastest || result.get_avg_time_us() < fastest->get_avg_time_us()) {
        fastest = &result;
      }
    }
  }

  return fastest;
}

const BenchmarkResult* TestResult::get_slowest_result() const {
  const BenchmarkResult* slowest = nullptr;

  for (const auto& result : results_) {
    bool vulkan_execute_succeeded =
        result.get_num_iterations() > 0 && result.get_avg_time_us() > 0.0f;
    if (vulkan_execute_succeeded) {
      if (!slowest || result.get_avg_time_us() > slowest->get_avg_time_us()) {
        slowest = &result;
      }
    }
  }

  return slowest;
}

const BenchmarkResult* TestResult::get_highest_gflops_result() const {
  // Since GFLOPS is now a TestResult-level metric rather than per-case,
  // this method returns the fastest result as a proxy for highest performance
  return get_fastest_result();
}

// Default FLOP calculation function (assumes 1 FLOP per element)
int64_t default_flop_calculator(const TestCase& test_case) {
  // Calculate total elements from the first input tensor
  int64_t total_elements = 1;
  if (!test_case.empty() && test_case.num_inputs() > 0 &&
      test_case.inputs()[0].is_tensor()) {
    const auto& sizes = test_case.inputs()[0].get_tensor_sizes();
    for (int64_t size : sizes) {
      total_elements *= size;
    }
  }

  // Assume 1 FLOP per element for basic operations
  return total_elements;
}

ComputeGraph setup_compute_graph(TestCase& test_case, std::string op_name) {
  GraphConfig config;
  config.enable_querypool = true;
  ComputeGraph graph(config);

  std::vector<ValueRef> input_values;

  // Process input ValueSpecs
  for (size_t i = 0; i < test_case.num_inputs(); ++i) {
    const ValueSpec& input_spec = test_case.inputs()[i];

    if (input_spec.is_none()) {
      input_values.push_back(graph.add_none());
    } else if (input_spec.is_float()) {
      ValueRef input_value =
          graph.add_scalar(static_cast<double>(input_spec.get_float_value()));
      input_values.push_back(input_value);
    } else if (input_spec.is_int()) {
      ValueRef input_value =
          graph.add_scalar(static_cast<int64_t>(input_spec.get_int_value()));
      input_values.push_back(input_value);
    } else if (input_spec.is_bool()) {
      ValueRef input_value = graph.add_scalar(input_spec.get_bool_value());
      input_values.push_back(input_value);
    } else if (input_spec.is_int_list()) {
      // Convert int32_t list to int64_t list for ComputeGraph
      const auto& int32_list = input_spec.get_int_list();
      std::vector<int64_t> int64_list;
      int64_list.reserve(int32_list.size());
      for (int32_t val : int32_list) {
        int64_list.push_back(static_cast<int64_t>(val));
      }
      ValueRef input_value = graph.add_scalar_list(std::move(int64_list));
      input_values.push_back(input_value);
    } else if (input_spec.is_constant()) {
      ValueRef input_value = graph.add_tensorref(
          input_spec.get_tensor_sizes(),
          input_spec.dtype,
          input_spec.get_data_ptr());
      input_values.push_back(input_value);
    } else {
      IOValueRef input_io = graph.add_input_tensor(
          input_spec.get_tensor_sizes(),
          input_spec.dtype,
          input_spec.storage_type,
          input_spec.memory_layout);
      input_values.push_back(input_io.value);
    }
  }

  std::vector<ValueRef> output_values;

  // Process output ValueSpecs
  for (size_t i = 0; i < test_case.num_outputs(); ++i) {
    const ValueSpec& output_spec = test_case.outputs()[i];

    if (!output_spec.is_tensor()) {
      throw std::runtime_error("All output specifications must be tensors");
    }

    // Create output tensor
    ValueRef output_value = graph.add_tensor(
        output_spec.get_tensor_sizes(),
        output_spec.dtype,
        output_spec.storage_type,
        output_spec.memory_layout);

    output_values.push_back(output_value);
  }

  // Get the operator function and call it
  auto opFn = VK_GET_OP_FN(op_name);

  // Create arguments vector for the operator function
  std::vector<ValueRef> op_args = input_values;
  op_args.insert(op_args.end(), output_values.begin(), output_values.end());

  opFn(graph, op_args);

  for (size_t i = 0; i < output_values.size(); ++i) {
    graph.set_output_value(output_values[i]);
  }
  return graph;
}

// Test execution utilities
BenchmarkResult
execute_test_case(TestCase& test_case, int warmup_runs, int benchmark_runs) {
  BenchmarkResult result(
      test_case.name().empty() ? "unnamed_test_case" : test_case.name());

  // Initialize querypool if using GPU timestamps
  if (use_gpu_timestamps()) {
    api::context()->initialize_querypool();
  }

  // Create the compute graph for this test case using setup_compute_graph
  ComputeGraph graph =
      setup_compute_graph(test_case, test_case.operator_name());

  // Prepare the graph
  graph.prepare();
  graph.prepack();

  // Copy input data into the graph's staging buffers
  for (size_t i = 0; i < test_case.num_inputs(); ++i) {
    const ValueSpec& input_spec = test_case.inputs()[i];
    if (input_spec.is_tensor() && i < graph.inputs().size()) {
      // Skip copying data for constant tensors
      if (input_spec.is_constant()) {
        continue;
      }

      const auto& input_ref = graph.inputs()[i];

      // Get the appropriate data based on dtype
      const void* data_ptr = nullptr;
      size_t data_numel = input_spec.numel();

      switch (input_spec.dtype) {
        case vkapi::kFloat:
          data_ptr = input_spec.get_float_data().data();
          break;
        case vkapi::kHalf:
          data_ptr = input_spec.get_half_data().data();
          break;
        case vkapi::kInt:
          data_ptr = input_spec.get_int32_data().data();
          break;
        case vkapi::kChar:
          data_ptr = input_spec.get_int8_data().data();
          break;
        case vkapi::kByte:
          data_ptr = input_spec.get_uint8_data().data();
          break;
        default:
          throw std::runtime_error("Unsupported data type for input tensor");
      }

      // Copy data into staging buffer
      graph.copy_into_staging(input_ref.staging, data_ptr, data_numel);
    }
  }

  // Warmup runs
  for (int run = 0; run < warmup_runs; ++run) {
    graph.execute();
  }

  // Benchmark runs - collect individual iteration timings
  float total_cpu_time_us = 0.0f;
  float total_gpu_time_us = 0.0f;

  for (int run = 0; run < benchmark_runs; ++run) {
    // Measure CPU time for each execute() call
    auto cpu_start = std::chrono::high_resolution_clock::now();
    graph.execute();
    auto cpu_end = std::chrono::high_resolution_clock::now();

    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        cpu_end - cpu_start);
    float cpu_time_us = static_cast<float>(cpu_duration.count());
    total_cpu_time_us += cpu_time_us;

    // Collect GPU timing using helper function
    float gpu_time_us = collect_gpu_timing_us(graph);
    total_gpu_time_us += gpu_time_us;

    // Add the appropriate timing based on the flag
    float iter_time_us = use_gpu_timestamps() ? gpu_time_us : cpu_time_us;
    result.add_iter_timing(iter_time_us);
  }

  // Calculate averages for display
  float avg_cpu_time_us = total_cpu_time_us / benchmark_runs;
  float avg_gpu_time_us = total_gpu_time_us / benchmark_runs;

  // Print both timings if latency printing is enabled
  if (print_latencies()) {
    if (use_gpu_timestamps()) {
      graph.context()->querypool().print_results();
    }
    std::cout << "  CPU timing: " << std::fixed << std::setprecision(3)
              << avg_cpu_time_us << " μs" << std::endl;
    std::cout << "  GPU timing: " << std::fixed << std::setprecision(3)
              << avg_gpu_time_us << " μs" << std::endl;
    std::cout << "  Using " << (use_gpu_timestamps() ? "GPU" : "CPU")
              << " timing for result" << std::endl;
  }

  // Copy output data from the graph's staging buffers
  for (size_t i = 0; i < test_case.num_outputs(); ++i) {
    ValueSpec& output_spec = test_case.outputs()[i];

    if (output_spec.is_tensor() && i < graph.outputs().size()) {
      const auto& output_ref = graph.outputs()[i];

      // Ensure output data vector is properly sized
      size_t data_numel = output_spec.numel();
      output_spec.resize_data(data_numel);

      // Get mutable data pointer for the output
      void* data_ptr = output_spec.get_mutable_data_ptr();

      if (data_ptr != nullptr) {
        // Copy data from staging buffer to output spec
        graph.copy_from_staging(output_ref.staging, data_ptr, data_numel);
      }

      // Print output tensor data if output printing is enabled
      if (print_output()) {
        std::string output_name = "Output[" + std::to_string(i) + "]";
        print_valuespec_data(output_spec, output_name);
      }
    }
  }

  return result;
}

TestResult execute_test_cases(
    std::function<std::vector<TestCase>()> test_case_generator,
    FlopCalculatorFunc flop_calculator,
    const std::string& operation_name,
    int warmup_runs,
    int benchmark_runs,
    ReferenceComputeFunc reference_compute_func) {
  TestResult results(operation_name);

  // Generate all test cases
  std::vector<TestCase> test_cases = test_case_generator();

  std::cout << "Executing " << test_cases.size() << " test cases for "
            << operation_name << std::endl;
  print_separator();

  bool any_correctness_failed = false;
  float total_gflops = 0.0f;

  for (size_t i = 0; i < test_cases.size(); ++i) {
    TestCase& test_case = test_cases[i];

    // Compute reference data if reference function is provided
    bool skipped_reference_fn = true;
    if (reference_compute_func) {
      try {
        reference_compute_func(test_case);
        skipped_reference_fn = false;
      } catch (const std::invalid_argument& e) {
        if (debugging()) {
          std::cout << "Compute reference skipped: " << e.what() << std::endl;
        }
      }
    }

    // Execute single test case
    BenchmarkResult result;
    bool shader_not_supported = false;
    try {
      result = execute_test_case(test_case, warmup_runs, benchmark_runs);
      result.set_operator_name(test_case.operator_name());
    } catch (const vkcompute::vkapi::ShaderNotSupportedError&) {
      result = BenchmarkResult(
          test_case.name().empty() ? "unnamed_test_case" : test_case.name(),
          test_case.operator_name());
      shader_not_supported = true;
    }

    // Determine if this test case passed (has valid timing data)
    bool vulkan_execute_succeeded =
        result.get_num_iterations() > 0 && result.get_avg_time_us() > 0.0f;

    if (shader_not_supported) {
      result.set_correctness_status(CorrectnessStatus::SKIPPED);
    } else if (!vulkan_execute_succeeded) {
      result.set_correctness_status(CorrectnessStatus::FAILED);
    } else if (skipped_reference_fn) {
      result.set_correctness_status(CorrectnessStatus::SKIPPED);
    } else {
      // Reference function provided and succeeded - validate outputs
      bool correctness_passed = true;

      for (size_t output_idx = 0; output_idx < test_case.num_outputs();
           ++output_idx) {
        const ValueSpec& output_spec = test_case.outputs()[output_idx];

        if (!output_spec.validate_against_reference(
                test_case.get_abs_tolerance(), test_case.get_rel_tolerance())) {
          correctness_passed = false;
          std::cout << "  Correctness validation FAILED for test "
                    << result.get_kernel_name() << std::endl;
          print_valuespec_data(output_spec, "vulkan output");
          print_valuespec_data(output_spec, "ref output", true);

          throw std::runtime_error("Correctness validation failed");
        }
      }

      if (correctness_passed) {
        result.set_correctness_status(CorrectnessStatus::PASSED);
      } else {
        any_correctness_failed = true;
        result.set_correctness_status(CorrectnessStatus::FAILED);
      }
    }

    // Calculate GFLOPS for this test case using the provided FLOP calculator
    float case_gflops = 0.0f;
    if (vulkan_execute_succeeded) {
      // Use the provided FLOP calculator to get total FLOPs for this test case
      int64_t total_flops = flop_calculator(test_case);
      float flops = static_cast<float>(total_flops);
      float avg_time_us = result.get_avg_time_us();
      if (avg_time_us > 0.0f && total_flops > 0) {
        case_gflops = (flops / 1e9f) / (avg_time_us / 1e6f);
      }

      total_gflops += case_gflops;
    } else {
      case_gflops = -1.0f; // Indicate failure
    }

    // Calculate tensor info for display
    std::string size_info = "[";
    if (!test_case.empty() && test_case.num_inputs() > 0 &&
        test_case.inputs()[0].is_tensor()) {
      const auto& sizes = test_case.inputs()[0].get_tensor_sizes();
      for (size_t j = 0; j < sizes.size(); ++j) {
        size_info += std::to_string(sizes[j]);
        if (j < sizes.size() - 1)
          size_info += "x";
      }
    }
    size_info += "]";

    // Print progress using the BenchmarkResult member function
    result.print_summary(i + 1, size_info, case_gflops);

    // Add result to collection
    results.add_result(std::move(result));
  }

  // Set the overall results on the TestResult
  results.set_correctness_passed(!any_correctness_failed);
  results.set_gflops(total_gflops);

  print_separator();
  std::cout << "Completed " << results.size() << " test cases" << std::endl;

  return results;
}

// Convenience overload that uses the default FLOP calculator
TestResult execute_test_cases(
    std::function<std::vector<TestCase>()> test_case_generator,
    const std::string& operation_name,
    int warmup_runs,
    int benchmark_runs,
    ReferenceComputeFunc reference_compute_func) {
  return execute_test_cases(
      test_case_generator,
      default_flop_calculator,
      operation_name,
      warmup_runs,
      benchmark_runs,
      reference_compute_func);
}

// Utility functions for printing
void print_performance_header() {
  std::cout << "\n=== Compute Shader Performance Benchmark ===" << std::endl;
}

void print_separator() {
  std::cout << std::string(70, '-') << std::endl;
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
        if (i < print_count - 1)
          std::cout << ", ";
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

  switch (spec.dtype) {
    case vkapi::kFloat: {
      auto data = spec.get_float_data().data();
      if (print_ref_data) {
        data = spec.get_ref_float_data().data();
      }
      for (size_t i = 0; i < print_count; ++i) {
        std::cout << data[i];
        if (i < print_count - 1)
          std::cout << ", ";
      }
      break;
    }
    case vkapi::kHalf: {
      const auto& data = spec.get_half_data();
      for (size_t i = 0; i < print_count; ++i) {
        // Convert uint16_t back to float for display
        float value = data[i] / 32767.0f;
        std::cout << value;
        if (i < print_count - 1)
          std::cout << ", ";
      }
      break;
    }
    case vkapi::kInt: {
      const auto& data = spec.get_int32_data();
      for (size_t i = 0; i < print_count; ++i) {
        std::cout << data[i];
        if (i < print_count - 1)
          std::cout << ", ";
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
          if (lower_4bits > 7)
            lower_4bits -= 16; // Convert to signed
          std::cout << static_cast<int>(lower_4bits);
          element_count++;

          if (element_count < print_count) {
            std::cout << ", ";
            // Extract upper 4 bits (signed)
            int8_t upper_4bits = (data[i] >> 4) & 0x0F;
            if (upper_4bits > 7)
              upper_4bits -= 16; // Convert to signed
            std::cout << static_cast<int>(upper_4bits);
            element_count++;

            if (element_count < print_count)
              std::cout << ", ";
          }
        }
      } else {
        for (size_t i = 0; i < print_count; ++i) {
          std::cout << static_cast<int>(data[i]);
          if (i < print_count - 1)
            std::cout << ", ";
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

            if (element_count < print_count)
              std::cout << ", ";
          }
        }
      } else {
        for (size_t i = 0; i < print_count; ++i) {
          std::cout << static_cast<unsigned int>(data[i]);
          if (i < print_count - 1)
            std::cout << ", ";
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

ValueRef quantized_weights_canvas(
    ComputeGraph& graph,
    const ValueRef weight_ref) {
  const auto original_sizes = graph.sizes_of(weight_ref);

  // Get the 2 highest values of original_sizes
  std::vector<int64_t> sorted_sizes = original_sizes;
  std::sort(sorted_sizes.begin(), sorted_sizes.end(), std::greater<int64_t>());
  int64_t largest1 = sorted_sizes.size() > 0 ? sorted_sizes[0] : 0;
  int64_t largest2 = sorted_sizes.size() > 1 ? sorted_sizes[1] : 0;

  std::vector<int64_t> final_sizes = {1, largest1, largest1};

  // Debug logging if debugging flag is set
  if (debugging()) {
    std::cout << "Debug: Creating quantized weights canvas tensor" << std::endl;
    std::cout << "Debug: Original sizes: [";
    for (size_t i = 0; i < original_sizes.size(); ++i) {
      std::cout << original_sizes[i];
      if (i < original_sizes.size() - 1)
        std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Debug: Canvas sizes: [";
    for (size_t i = 0; i < final_sizes.size(); ++i) {
      std::cout << final_sizes[i];
      if (i < final_sizes.size() - 1)
        std::cout << ", ";
    }
    std::cout << "]" << std::endl;
  }

  ValueRef packed_weight = graph.add_tensor(
      final_sizes, vkapi::kInt, utils::kTexture3D, utils::kWidthPacked);

  utils::uvec3 global_wg_size{
      utils::div_up(utils::safe_downcast<uint32_t>(largest1), uint32_t(4)),
      utils::safe_downcast<uint32_t>(largest2),
      utils::safe_downcast<uint32_t>(std::min(largest1, int64_t(2048)))};

  std::string kernel_name = "packed_int32_canvas";
  add_storage_type_suffix(kernel_name, graph.storage_type_of(packed_weight));

  graph.prepack_nodes().emplace_back(new PrepackNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      graph.create_global_wg_size(packed_weight),
      graph.create_local_wg_size(packed_weight),
      weight_ref,
      packed_weight,
      // UBOs
      {graph.logical_limits_ubo(packed_weight)},
      // Specialization constants
      {},
      // Push Constants
      {}));

  return packed_weight;
}

ValueRef float_tensor_canvas(ComputeGraph& graph, const ValueRef weight_ref) {
  const auto original_sizes = graph.sizes_of(weight_ref);

  // Get the 2 highest values of original_sizes
  std::vector<int64_t> sorted_sizes = original_sizes;
  std::sort(sorted_sizes.begin(), sorted_sizes.end(), std::greater<int64_t>());
  int64_t largest1 = sorted_sizes.size() > 0 ? sorted_sizes[0] : 0;
  int64_t largest2 = sorted_sizes.size() > 1 ? sorted_sizes[1] : 0;

  std::vector<int64_t> final_sizes = {1, largest1, largest1};

  // Debug logging if debugging flag is set
  if (debugging()) {
    std::cout << "Debug: Creating float tensor canvas" << std::endl;
    std::cout << "Debug: Original sizes: [";
    for (size_t i = 0; i < original_sizes.size(); ++i) {
      std::cout << original_sizes[i];
      if (i < original_sizes.size() - 1)
        std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Debug: Canvas sizes: [";
    for (size_t i = 0; i < final_sizes.size(); ++i) {
      std::cout << final_sizes[i];
      if (i < final_sizes.size() - 1)
        std::cout << ", ";
    }
    std::cout << "]" << std::endl;
  }

  ValueRef packed_weight = graph.add_tensor(
      final_sizes, vkapi::kFloat, utils::kTexture3D, utils::kWidthPacked);

  utils::uvec3 global_wg_size{
      utils::div_up(utils::safe_downcast<uint32_t>(largest1), uint32_t(4)),
      utils::safe_downcast<uint32_t>(largest2),
      utils::safe_downcast<uint32_t>(std::min(largest1, int64_t(2048)))};

  graph.prepack_nodes().emplace_back(new PrepackNode(
      graph,
      VK_KERNEL_FROM_STR("float_canvas"),
      graph.create_global_wg_size(packed_weight),
      graph.create_local_wg_size(packed_weight),
      weight_ref,
      packed_weight,
      // UBOs
      {graph.logical_limits_ubo(packed_weight)},
      // Specialization constants
      {},
      // Push Constants
      {}));

  return packed_weight;
}

// Compute weight sums for quantized operations (linear and convolution)
void compute_weight_sums(
    ValueSpec& weight_sums,
    const ValueSpec& quantized_weight,
    int64_t out_features,
    int64_t elements_per_output_feature) {
  auto& weight_sums_data = weight_sums.get_int32_data();
  auto& quantized_weight_data = quantized_weight.get_int8_data();

  weight_sums_data.resize(out_features);

  // For each output feature, compute the sum of quantized weights
  for (int64_t out_f = 0; out_f < out_features; ++out_f) {
    int32_t sum = 0;
    for (int64_t elem = 0; elem < elements_per_output_feature; ++elem) {
      // Weight indexing depends on the layout:
      // For linear: [out_features, in_features] -> out_f *
      // elements_per_output_feature + elem For conv2d: [C_out, C_in * K_h *
      // K_w] -> out_f * elements_per_output_feature + elem
      int64_t weight_idx = out_f * elements_per_output_feature + elem;
      sum += static_cast<int32_t>(quantized_weight_data[weight_idx]);
    }
    weight_sums_data[out_f] = sum;
  }
}

// Compute weight sums for 4D quantized conv2d operations
// Weight layout: [C_out, K_h, K_w, align_up_4(C_in_per_group)]
void compute_weight_sums_4d(
    ValueSpec& weight_sums,
    const ValueSpec& quantized_weight,
    int64_t out_channels,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t aligned_in_channels) {
  auto& weight_sums_data = weight_sums.get_int32_data();
  auto& quantized_weight_data = quantized_weight.get_int8_data();

  weight_sums_data.resize(out_channels);

  // For each output channel, compute the sum of quantized weights
  for (int64_t out_c = 0; out_c < out_channels; ++out_c) {
    int32_t sum = 0;

    for (int64_t kh = 0; kh < kernel_h; ++kh) {
      for (int64_t kw = 0; kw < kernel_w; ++kw) {
        for (int64_t in_c = 0; in_c < aligned_in_channels; ++in_c) {
          // Weight indexing: [out_c, kh, kw, in_c]
          int64_t weight_idx =
              out_c * (kernel_h * kernel_w * aligned_in_channels) +
              kh * (kernel_w * aligned_in_channels) + kw * aligned_in_channels +
              in_c;
          sum += static_cast<int32_t>(quantized_weight_data[weight_idx]);
        }
      }
    }

    weight_sums_data[out_c] = sum;
  }
}

// Helper function to unpack 4-bit values from uint8 (same as in
// q4gsw_linear.cpp)
std::pair<int8_t, int8_t> unpack_4bit_utils(uint8_t packed) {
  // Extract lower 4 bits and upper 4 bits
  int8_t lower = packed & 0x0F;
  int8_t upper = (packed >> 4) & 0x0F;

  // Subtract 8 from unpacked 4-bit values
  lower -= 8;
  upper -= 8;

  return std::make_pair(lower, upper);
}

// Compute weight sums for 4-bit group symmetric quantized weights
void compute_weight_sums_4bit_grouped(
    ValueSpec& weight_sums,
    const ValueSpec& quantized_weight,
    int64_t num_groups,
    int64_t out_features,
    int64_t group_size) {
  auto& weight_sums_data = weight_sums.get_int32_data();
  auto& quantized_weight_data = quantized_weight.get_uint8_data();

  // Resize to [num_groups, out_features]
  weight_sums_data.resize(num_groups * out_features);

  // For each group and each output feature, compute the sum of quantized
  // weights in that group
  for (int64_t group_idx = 0; group_idx < num_groups; ++group_idx) {
    for (int64_t out_f = 0; out_f < out_features; ++out_f) {
      int32_t sum = 0;

      // Sum weights for this group and output feature
      for (int64_t in_group = 0; in_group < group_size; ++in_group) {
        int64_t in_f = group_idx * group_size + in_group;

        // Get packed weight value - weight matrix is [N, K/2]
        int64_t weight_idx =
            out_f * ((num_groups * group_size) / 2) + (in_f / 2);
        uint8_t packed_weight = quantized_weight_data[weight_idx];

        // Unpack 4-bit weight
        auto unpacked = unpack_4bit_utils(packed_weight);
        int8_t weight_4bit = (in_f % 2 == 0) ? unpacked.first : unpacked.second;

        sum += static_cast<int32_t>(weight_4bit);
      }

      // Store sum for this group and output feature
      int64_t sums_idx = group_idx * out_features + out_f;
      weight_sums_data[sums_idx] = sum;
    }
  }
}

} // namespace prototyping
} // namespace vulkan
} // namespace executorch
