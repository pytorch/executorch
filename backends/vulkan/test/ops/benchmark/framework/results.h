// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {

//
// BenchmarkResult
//

enum class CorrectnessStatus {
  SKIPPED, // No reference function provided
  PASSED, // Reference function provided and validation passed
  FAILED // Reference function provided but validation failed
};

// Per-shader timing data for detailed reporting
struct ShaderTiming {
  std::string shader_name;
  std::vector<float> iter_timings_us; // Individual iteration timings
  uint32_t gwg[3] = {0, 0, 0};
  uint32_t lwg[3] = {0, 0, 0};

  float get_avg_time_us() const {
    if (iter_timings_us.empty()) {
      return 0.0f;
    }
    float sum = 0.0f;
    for (float t : iter_timings_us) {
      sum += t;
    }
    return sum / iter_timings_us.size();
  }
};

class BenchmarkResult {
 public:
  BenchmarkResult() : correctness_status_(CorrectnessStatus::SKIPPED) {}

  explicit BenchmarkResult(const std::string& name)
      : kernel_name(name), correctness_status_(CorrectnessStatus::SKIPPED) {}

  BenchmarkResult(
      const std::string& kernel_name,
      const std::string& operator_name)
      : kernel_name(kernel_name),
        operator_name(operator_name),
        correctness_status_(CorrectnessStatus::SKIPPED) {}

  // Add timing for a single iteration
  void add_iter_timing(float time_us);

  // Add per-shader timing for a single iteration
  void add_shader_timing(
      const std::string& shader_name,
      float time_us,
      const uint32_t gwg[3],
      const uint32_t lwg[3]);

  // Get per-shader timing data
  const std::vector<ShaderTiming>& get_shader_timings() const {
    return shader_timings_;
  }

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
  std::vector<ShaderTiming> shader_timings_; // Per-shader timing data
  CorrectnessStatus correctness_status_;
};

// Test result collection and processing
class TestResult {
 public:
  TestResult() : gflops_(0.0f), correctness_passed_(true) {}
  explicit TestResult(const std::string& operation_name)
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
// Print utilities
//

void print_performance_header();
void print_separator();

} // namespace prototyping
} // namespace vulkan
} // namespace executorch
