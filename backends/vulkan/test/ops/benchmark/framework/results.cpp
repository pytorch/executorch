// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "results.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>

namespace executorch {
namespace vulkan {
namespace prototyping {

// BenchmarkResult implementation
void BenchmarkResult::add_iter_timing(float time_us) {
  iter_timings.push_back(time_us);
}

void BenchmarkResult::add_shader_timing(
    const std::string& shader_name,
    float time_us,
    const uint32_t gwg[3],
    const uint32_t lwg[3]) {
  // Find existing shader timing or create new one
  for (auto& st : shader_timings_) {
    if (st.shader_name == shader_name) {
      st.iter_timings_us.push_back(time_us);
      // Work group sizes should be consistent across iterations
      return;
    }
  }
  // Not found, create new entry
  ShaderTiming new_timing;
  new_timing.shader_name = shader_name;
  new_timing.iter_timings_us.push_back(time_us);
  new_timing.gwg[0] = gwg[0];
  new_timing.gwg[1] = gwg[1];
  new_timing.gwg[2] = gwg[2];
  new_timing.lwg[0] = lwg[0];
  new_timing.lwg[1] = lwg[1];
  new_timing.lwg[2] = lwg[2];
  shader_timings_.push_back(std::move(new_timing));
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
  (void)case_number;
  static constexpr int OPERATOR_NAME_WIDTH = 50;
  static constexpr int GLOBAL_WG_WIDTH = 16;
  static constexpr int LOCAL_WG_WIDTH = 12;
  static constexpr int KERNEL_NAME_WIDTH = 80;
  static constexpr int SIZE_INFO_WIDTH = 20;
  static constexpr int TIMING_WIDTH = 16;
  static constexpr int GFLOPS_WIDTH = 14;
  static constexpr int CORRECTNESS_WIDTH = 8;

  // Helper to truncate shader names longer than 46 chars to 44 chars + ".."
  auto truncate_shader_name = [](const std::string& name) -> std::string {
    if (name.length() > 46) {
      return name.substr(0, 44) + "..";
    }
    return name;
  };

  // Helper to format work group size as (x,y,z)
  auto format_wg_size = [](const uint32_t wg[3]) -> std::string {
    return "(" + std::to_string(wg[0]) + "," + std::to_string(wg[1]) + "," +
        std::to_string(wg[2]) + ")";
  };

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

  // If we have per-shader timing data, print one line per shader plus overall
  if (!shader_timings_.empty()) {
    // If only one shader, print a single combined row
    if (shader_timings_.size() == 1) {
      const auto& st = shader_timings_[0];
      std::cout << std::left << std::setw(OPERATOR_NAME_WIDTH)
                << truncate_shader_name(st.shader_name) << " " << std::left
                << std::setw(GLOBAL_WG_WIDTH) << format_wg_size(st.gwg)
                << std::left << std::setw(LOCAL_WG_WIDTH)
                << format_wg_size(st.lwg) << std::left
                << std::setw(KERNEL_NAME_WIDTH) << get_kernel_name()
                << std::right << " " << std::setw(SIZE_INFO_WIDTH) << size_info
                << std::setw(TIMING_WIDTH) << std::fixed << std::setprecision(3)
                << get_avg_time_us() << " μs " << std::setw(GFLOPS_WIDTH)
                << std::fixed << std::setprecision(3) << total_gflops
                << " GFLOP/s " << std::setw(CORRECTNESS_WIDTH)
                << correctness_str << std::endl;
    } else {
      // Multiple shaders: print individual shader lines (without GFLOP/s)
      for (size_t i = 0; i < shader_timings_.size(); ++i) {
        const auto& st = shader_timings_[i];
        float shader_avg_time = st.get_avg_time_us();

        // Shader lines don't show test case info
        std::cout << std::left << std::setw(OPERATOR_NAME_WIDTH)
                  << truncate_shader_name(st.shader_name) << " " << std::left
                  << std::setw(GLOBAL_WG_WIDTH) << format_wg_size(st.gwg)
                  << std::left << std::setw(LOCAL_WG_WIDTH)
                  << format_wg_size(st.lwg) << std::left
                  << std::setw(KERNEL_NAME_WIDTH) << "" << std::right << " "
                  << std::setw(SIZE_INFO_WIDTH) << "" << std::setw(TIMING_WIDTH)
                  << std::fixed << std::setprecision(3) << shader_avg_time
                  << " μs " << std::setw(GFLOPS_WIDTH) << "" << "          "
                  << std::setw(CORRECTNESS_WIDTH) << "" << std::endl;
      }

      // Print overall row with operator name, test case info, total time, and
      // GFLOP/s
      std::cout << std::left << std::setw(OPERATOR_NAME_WIDTH)
                << get_operator_name() << " " << std::left
                << std::setw(GLOBAL_WG_WIDTH) << "" << std::left
                << std::setw(LOCAL_WG_WIDTH) << "" << std::left
                << std::setw(KERNEL_NAME_WIDTH) << get_kernel_name()
                << std::right << " " << std::setw(SIZE_INFO_WIDTH) << size_info
                << std::setw(TIMING_WIDTH) << std::fixed << std::setprecision(3)
                << get_avg_time_us() << " μs " << std::setw(GFLOPS_WIDTH)
                << std::fixed << std::setprecision(3) << total_gflops
                << " GFLOP/s " << std::setw(CORRECTNESS_WIDTH)
                << correctness_str << std::endl;
    }

    // Print separator line between test cases
  } else {
    // No per-shader timing data, use the original format
    std::cout << std::left << std::setw(OPERATOR_NAME_WIDTH)
              << get_operator_name() << " " << std::left
              << std::setw(GLOBAL_WG_WIDTH) << "" << std::left
              << std::setw(LOCAL_WG_WIDTH) << "" << std::left
              << std::setw(KERNEL_NAME_WIDTH) << get_kernel_name() << std::right
              << " " << std::setw(SIZE_INFO_WIDTH) << size_info
              << std::setw(TIMING_WIDTH) << std::fixed << std::setprecision(3)
              << get_avg_time_us() << " μs " << std::setw(GFLOPS_WIDTH)
              << std::fixed << std::setprecision(3) << total_gflops
              << " GFLOP/s " << std::setw(CORRECTNESS_WIDTH) << correctness_str
              << std::endl;
  }
}

// TestResult implementation
void TestResult::add_result(const BenchmarkResult& result) {
  results_.push_back(result);
}

void TestResult::add_result(BenchmarkResult&& result) {
  results_.push_back(std::move(result));
}

void TestResult::print_summary() const {
  static constexpr int CASE_WIDTH = 100;
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
        if (j < std::min(size_t(10), timings.size()) - 1) {
          std::cout << ", ";
        }
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

// Utility functions for printing
void print_performance_header() {
  std::cout << "\n=== Compute Shader Performance Benchmark ===" << std::endl;
}

void print_separator() {
  std::cout << std::string(70, '-') << std::endl;
}

} // namespace prototyping
} // namespace vulkan
} // namespace executorch
