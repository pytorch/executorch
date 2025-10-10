// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/vulkan/runtime/api/api.h>
#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>
#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {

using namespace vkcompute;

//
// Global configuration options
//

bool print_output();
void set_print_output(bool print_output);

bool print_latencies();
void set_print_latencies(bool print_latencies);

bool use_gpu_timestamps();
void set_use_gpu_timestamps(bool use_timestamps);

bool debugging();
void set_debugging(bool enable_debugging);

//
// ValueSpec class
//

enum class SpecType { Tensor, IntList, Int, Float, Bool };

// Data generation types
enum class DataGenType {
  FIXED,
  RANDOM,
  RANDOM_SCALES,
  RANDINT,
  RANDINT8,
  RANDINT4,
  ONES,
  ONES_INT4,
  ZEROS
};

// Value specification struct
struct ValueSpec {
  std::vector<int64_t> sizes;
  vkapi::ScalarType dtype;
  utils::GPUMemoryLayout memory_layout;
  utils::StorageType storage_type;
  SpecType spec_type;
  DataGenType data_gen_type;
  bool is_constant_tensor;
  bool is_none_flag;
  bool is_int4_tensor;

  std::vector<float> float_data;
  std::vector<int32_t> int32_data;
  std::vector<uint16_t> half_data; // Using uint16_t as substitute for half
  std::vector<int8_t> int8_data; // For kChar (signed 8-bit)
  std::vector<uint8_t> uint8_data; // For kByte (unsigned 8-bit)

  std::vector<float> ref_float_data;
  std::vector<int32_t> ref_int32_data;
  std::vector<uint16_t> ref_half_data;
  std::vector<int8_t> ref_int8_data;
  std::vector<uint8_t> ref_uint8_data;

  ValueSpec(
      const std::vector<int64_t>& sizes,
      vkapi::ScalarType dtype,
      utils::StorageType storage_type = utils::kTexture3D,
      utils::GPUMemoryLayout memory_layout = utils::kWidthPacked)
      : sizes(sizes),
        dtype(dtype),
        memory_layout(memory_layout),
        storage_type(storage_type),
        spec_type(SpecType::Tensor),
        data_gen_type(DataGenType::ZEROS),
        is_constant_tensor(false),
        is_none_flag(false),
        is_int4_tensor(false) {
    generate_tensor_data();
  }

  // Constructor for tensor with custom data generation type
  ValueSpec(
      const std::vector<int64_t>& sizes,
      vkapi::ScalarType dtype,
      utils::StorageType storage_type,
      utils::GPUMemoryLayout memory_layout,
      DataGenType data_gen_type)
      : sizes(sizes),
        dtype(dtype),
        memory_layout(memory_layout),
        storage_type(storage_type),
        spec_type(SpecType::Tensor),
        data_gen_type(data_gen_type),
        is_constant_tensor(false),
        is_none_flag(false),
        is_int4_tensor(false) {
    generate_tensor_data();
  }

  // Constructor for single int
  ValueSpec(int32_t value)
      : sizes({1}),
        dtype(vkapi::kInt),
        memory_layout(utils::kWidthPacked),
        storage_type(utils::kTexture3D),
        spec_type(SpecType::Int),
        data_gen_type(DataGenType::FIXED),
        is_constant_tensor(false),
        is_none_flag(false),
        is_int4_tensor(false) {
    int32_data.push_back(value);
  }

  // Constructor for single float
  ValueSpec(float value)
      : sizes({1}),
        dtype(vkapi::kFloat),
        memory_layout(utils::kWidthPacked),
        storage_type(utils::kTexture3D),
        spec_type(SpecType::Float),
        data_gen_type(DataGenType::FIXED),
        is_constant_tensor(false),
        is_none_flag(false),
        is_int4_tensor(false) {
    float_data.push_back(value);
  }

  // Constructor for single bool
  ValueSpec(bool value)
      : sizes({1}),
        dtype(vkapi::kInt),
        memory_layout(utils::kWidthPacked),
        storage_type(utils::kTexture3D),
        spec_type(SpecType::Bool),
        data_gen_type(DataGenType::FIXED),
        is_constant_tensor(false),
        is_none_flag(false),
        is_int4_tensor(false) {
    int32_data.push_back(value ? 1 : 0);
  }

  // Constructor for int list
  ValueSpec(const std::vector<int32_t>& values)
      : sizes({static_cast<int64_t>(values.size())}),
        dtype(vkapi::kInt),
        memory_layout(utils::kWidthPacked),
        storage_type(utils::kTexture3D),
        spec_type(SpecType::IntList),
        data_gen_type(DataGenType::FIXED),
        is_constant_tensor(false),
        is_none_flag(false),
        is_int4_tensor(false),
        int32_data(values) {}

  // Default constructor
  ValueSpec()
      : dtype(vkapi::kFloat),
        memory_layout(utils::kWidthPacked),
        storage_type(utils::kTexture3D),
        spec_type(SpecType::Tensor),
        data_gen_type(DataGenType::ZEROS),
        is_constant_tensor(false),
        is_none_flag(false),
        is_int4_tensor(false) {}

  int64_t numel() const;
  size_t nbytes() const;
  std::string to_string() const;

  bool is_tensor() const {
    return spec_type == SpecType::Tensor;
  }
  bool is_int_list() const {
    return spec_type == SpecType::IntList;
  }
  bool is_int() const {
    return spec_type == SpecType::Int;
  }
  bool is_float() const {
    return spec_type == SpecType::Float;
  }
  bool is_bool() const {
    return spec_type == SpecType::Bool;
  }

  int32_t get_int_value() const {
    return int32_data.empty() ? 0 : int32_data[0];
  }
  float get_float_value() const {
    return float_data.empty() ? 0.0f : float_data[0];
  }
  bool get_bool_value() const {
    return int32_data.empty() ? false : (int32_data[0] != 0);
  }
  const std::vector<int32_t>& get_int_list() const {
    return int32_data;
  }
  const std::vector<int64_t>& get_tensor_sizes() const {
    return sizes;
  }

  const std::vector<float>& get_float_data() const {
    return float_data;
  }
  const std::vector<int32_t>& get_int32_data() const {
    return int32_data;
  }
  const std::vector<uint16_t>& get_half_data() const {
    return half_data;
  }
  const std::vector<int8_t>& get_int8_data() const {
    return int8_data;
  }
  const std::vector<uint8_t>& get_uint8_data() const {
    return uint8_data;
  }

  std::vector<float>& get_float_data() {
    return float_data;
  }
  std::vector<int32_t>& get_int32_data() {
    return int32_data;
  }
  std::vector<uint16_t>& get_half_data() {
    return half_data;
  }
  std::vector<int8_t>& get_int8_data() {
    return int8_data;
  }
  std::vector<uint8_t>& get_uint8_data() {
    return uint8_data;
  }

  const std::vector<float>& get_ref_float_data() const {
    return ref_float_data;
  }
  const std::vector<int32_t>& get_ref_int32_data() const {
    return ref_int32_data;
  }
  const std::vector<uint16_t>& get_ref_half_data() const {
    return ref_half_data;
  }
  const std::vector<int8_t>& get_ref_int8_data() const {
    return ref_int8_data;
  }
  const std::vector<uint8_t>& get_ref_uint8_data() const {
    return ref_uint8_data;
  }

  std::vector<float>& get_ref_float_data() {
    return ref_float_data;
  }
  std::vector<int32_t>& get_ref_int32_data() {
    return ref_int32_data;
  }
  std::vector<uint16_t>& get_ref_half_data() {
    return ref_half_data;
  }
  std::vector<int8_t>& get_ref_int8_data() {
    return ref_int8_data;
  }
  std::vector<uint8_t>& get_ref_uint8_data() {
    return ref_uint8_data;
  }

  void resize_data(size_t new_size);
  void* get_mutable_data_ptr();
  float get_element(size_t index) const;

  // Set/get constant flag
  bool is_constant() const {
    return is_constant_tensor;
  }
  void set_constant(bool is_constant) {
    is_constant_tensor = is_constant;
  }

  // Set/get none flag
  bool is_none() const {
    return is_none_flag;
  }

  void set_none(bool is_none) {
    is_none_flag = is_none;
  }

  // Set/get int4 flag
  bool is_int4() const {
    return is_int4_tensor;
  }
  void set_int4(bool is_int4) {
    is_int4_tensor = is_int4;
  }

  const void* get_data_ptr() const;

  // Correctness checking against reference data
  // Returns true if computed data matches reference data within tolerance
  // Only validates float tensors as specified in requirements
  bool validate_against_reference(
      float abs_tolerance = 2e-3f,
      float rel_tolerance = 1e-3f) const;

 private:
  void generate_tensor_data();
};

//
// TestCase
//

class TestCase {
 public:
  TestCase() : abs_tolerance_(2e-3f), rel_tolerance_(1e-3f) {}
  TestCase(const std::string& name)
      : name_(name), abs_tolerance_(2e-3f), rel_tolerance_(1e-3f) {}

  void set_name(const std::string& name) {
    name_ = name;
  }
  const std::string& name() const {
    return name_;
  }

  void set_operator_name(const std::string& op_name) {
    operator_name_ = op_name;
  }
  const std::string& operator_name() const {
    return operator_name_;
  }

  // Tolerance settings
  void set_abs_tolerance(float abs_tolerance) {
    abs_tolerance_ = abs_tolerance;
  }
  float get_abs_tolerance() const {
    return abs_tolerance_;
  }

  void set_rel_tolerance(float rel_tolerance) {
    rel_tolerance_ = rel_tolerance;
  }
  float get_rel_tolerance() const {
    return rel_tolerance_;
  }

  void add_input_spec(const ValueSpec& spec) {
    inputs_.push_back(spec);
  }

  const std::vector<ValueSpec>& inputs() const {
    return inputs_;
  }

  std::vector<ValueSpec>& inputs() {
    return inputs_;
  }

  size_t num_inputs() const {
    return inputs_.size();
  }

  void add_output_spec(const ValueSpec& spec) {
    outputs_.push_back(spec);
  }

  const std::vector<ValueSpec>& outputs() const {
    return outputs_;
  }

  std::vector<ValueSpec>& outputs() {
    return outputs_;
  }

  size_t num_outputs() const {
    return outputs_.size();
  }

  bool empty() const {
    return inputs_.empty() && outputs_.empty();
  }
  void clear() {
    inputs_.clear();
    outputs_.clear();
    name_.clear();
    operator_name_.clear();
    abs_tolerance_ = 2e-3f;
    rel_tolerance_ = 1e-3f;
  }

 private:
  std::string name_;
  std::string operator_name_;
  std::vector<ValueSpec> inputs_;
  std::vector<ValueSpec> outputs_;
  float abs_tolerance_;
  float rel_tolerance_;
};

//
// BenchmarkResult
//

enum class CorrectnessStatus {
  SKIPPED, // No reference function provided
  PASSED, // Reference function provided and validation passed
  FAILED // Reference function provided but validation failed
};

class BenchmarkResult {
 public:
  BenchmarkResult() : correctness_status_(CorrectnessStatus::SKIPPED) {}

  BenchmarkResult(const std::string& name)
      : kernel_name(name), correctness_status_(CorrectnessStatus::SKIPPED) {}

  BenchmarkResult(
      const std::string& kernel_name,
      const std::string& operator_name)
      : kernel_name(kernel_name),
        operator_name(operator_name),
        correctness_status_(CorrectnessStatus::SKIPPED) {}

  // Add timing for a single iteration
  void add_iter_timing(float time_us);

  // Getters
  const std::string& get_kernel_name() const {
    return kernel_name;
  }
  const std::string& get_operator_name() const {
    return operator_name;
  }
  float get_avg_time_us() const;
  size_t get_num_iterations() const {
    return iter_timings.size();
  }
  const std::vector<float>& get_iter_timings() const {
    return iter_timings;
  }
  CorrectnessStatus get_correctness_status() const {
    return correctness_status_;
  }

  // Setters
  void set_kernel_name(const std::string& name) {
    kernel_name = name;
  }
  void set_operator_name(const std::string& name) {
    operator_name = name;
  }
  void set_correctness_status(CorrectnessStatus status) {
    correctness_status_ = status;
  }

  // Statistics
  float get_min_time_us() const;
  float get_max_time_us() const;
  float get_std_dev_us() const;

  // Clear all timings
  void clear_timings() {
    iter_timings.clear();
  }

  // Print progress for this benchmark result
  void print_summary(
      int case_number,
      const std::string& size_info,
      float total_gflops) const;

 private:
  std::string kernel_name;
  std::string operator_name;
  std::vector<float>
      iter_timings; // Individual iteration timings in microseconds
  CorrectnessStatus correctness_status_;
};

// Test result collection and processing
class TestResult {
 public:
  TestResult() : gflops_(0.0f), correctness_passed_(true) {}
  TestResult(const std::string& operation_name)
      : operation_name_(operation_name),
        gflops_(0.0f),
        correctness_passed_(true) {}

  // Add a benchmark result
  void add_result(const BenchmarkResult& result);
  void add_result(BenchmarkResult&& result);

  // Getters
  const std::string& get_operation_name() const {
    return operation_name_;
  }
  float get_gflops() const {
    return gflops_;
  }
  bool get_correctness_passed() const {
    return correctness_passed_;
  }
  size_t size() const {
    return results_.size();
  }
  bool empty() const {
    return results_.empty();
  }

  // Setters
  void set_gflops(float gflops_val) {
    gflops_ = gflops_val;
  }
  void set_correctness_passed(bool passed) {
    correctness_passed_ = passed;
  }

  // Access results
  const BenchmarkResult& operator[](size_t index) const {
    return results_[index];
  }
  BenchmarkResult& operator[](size_t index) {
    return results_[index];
  }
  const std::vector<BenchmarkResult>& get_results() const {
    return results_;
  }

  // Iterator support
  std::vector<BenchmarkResult>::iterator begin() {
    return results_.begin();
  }
  std::vector<BenchmarkResult>::iterator end() {
    return results_.end();
  }
  std::vector<BenchmarkResult>::const_iterator begin() const {
    return results_.begin();
  }
  std::vector<BenchmarkResult>::const_iterator end() const {
    return results_.end();
  }

  // Processing and analysis
  void print_summary() const;
  void print_detailed_results() const;
  void print_statistics() const;
  void print_brief_summary() const;

  // Get aggregate statistics
  float get_total_avg_time_us() const;
  float get_total_gflops() const;
  size_t get_passed_count() const;
  size_t get_failed_count() const;

  // Find best/worst performing results
  const BenchmarkResult* get_fastest_result() const;
  const BenchmarkResult* get_slowest_result() const;
  const BenchmarkResult* get_highest_gflops_result() const;

  // Clear all results
  void clear() {
    results_.clear();
  }

  // Set operation name
  void set_operation_name(const std::string& name) {
    operation_name_ = name;
  }

 private:
  std::string operation_name_;
  std::vector<BenchmarkResult> results_;
  float gflops_;
  bool correctness_passed_;
};

//
// Test case execution
//

using FlopCalculatorFunc = std::function<int64_t(const TestCase&)>;

// Default FLOP calculation function (assumes 1 FLOP per element)
int64_t default_flop_calculator(const TestCase& test_case);

using ReferenceComputeFunc = std::function<void(TestCase&)>;

BenchmarkResult execute_test_case(
    TestCase& test_case,
    int warmup_runs = 3,
    int benchmark_runs = 10);

TestResult execute_test_cases(
    std::function<std::vector<TestCase>()> test_case_generator,
    FlopCalculatorFunc flop_calculator,
    const std::string& operation_name = "Operation",
    int warmup_runs = 3,
    int benchmark_runs = 10,
    ReferenceComputeFunc reference_compute_func = nullptr);

TestResult execute_test_cases(
    std::function<std::vector<TestCase>()> test_case_generator,
    const std::string& operation_name = "Operation",
    int warmup_runs = 3,
    int benchmark_runs = 10,
    ReferenceComputeFunc reference_compute_func = nullptr);

//
// Print utilities
//

void print_performance_header();
void print_separator();

void print_valuespec_data(
    const ValueSpec& spec,
    const std::string& name = "ValueSpec",
    const bool print_ref_data = false,
    size_t max_elements = 20,
    int precision = 6);

ValueRef quantized_weights_canvas(
    ComputeGraph& graph,
    const ValueRef weight_ref);

ValueRef float_tensor_canvas(ComputeGraph& graph, const ValueRef weight_ref);

// Compute weight sums for quantized operations (linear and convolution)
void compute_weight_sums(
    ValueSpec& weight_sums,
    const ValueSpec& quantized_weight,
    int64_t out_features,
    int64_t elements_per_output_feature);

// Compute weight sums for 4D quantized conv2d operations
// Weight layout: [C_out, K_h, K_w, align_up_4(C_in_per_group)]
void compute_weight_sums_4d(
    ValueSpec& weight_sums,
    const ValueSpec& quantized_weight,
    int64_t out_channels,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t aligned_in_channels);

// Compute weight sums for 4-bit group symmetric quantized weights
void compute_weight_sums_4bit_grouped(
    ValueSpec& weight_sums,
    const ValueSpec& quantized_weight,
    int64_t num_groups,
    int64_t out_features,
    int64_t group_size);

// Setup compute graph based on TestCase and operation name
ComputeGraph setup_compute_graph(TestCase& test_case, std::string op_name);

} // namespace prototyping
} // namespace vulkan
} // namespace executorch
