// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "runner.h"
#include "config.h"

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {

namespace {

// Upper bound on the auto-sized chained_dispatches factor. Caps the factor
// when an op is fast enough that target_us / probe_us would otherwise
// exceed this; past this point, stacking more dispatches doesn't materially
// improve measurement precision.
constexpr int kMaxChainedDispatches = 200;

// Floor for probe_us to avoid division-by-zero / a runaway chained_dispatches
// factor when an op finishes faster than the wall clock can resolve.
constexpr float kMinProbeTimeUs = 1.0f;

} // namespace

// Benchmark graphs are immutable after input upload, so the operator nodes can
// be recorded repeatedly. Staging uploads/downloads are encoded once:
// repeating them would redo host-visible copies on every repetition and bias
// per-invocation timings.
RepeatedGraphExecutor::RepeatedGraphExecutor(
    ComputeGraph& graph,
    int repetitions,
    OpNodeRange op_nodes)
    : graph_(graph) {
  VK_CHECK_COND(repetitions > 0);
  VK_CHECK_COND(op_nodes.begin <= op_nodes.end);
  VK_CHECK_COND(op_nodes.end <= graph_.execute_nodes().size());

  api::Context* const context = graph_.context();
  context->flush();
  context->set_cmd(/*reusable=*/true);
  context->cmd_reset_querypool();

  for (size_t i = 0; i < op_nodes.begin; ++i) {
    graph_.execute_nodes()[i]->encode(&graph_);
  }
  for (int i = 0; i < repetitions; ++i) {
    for (size_t j = op_nodes.begin; j < op_nodes.end; ++j) {
      graph_.execute_nodes()[j]->encode(&graph_);
    }
  }
  for (size_t i = op_nodes.end; i < graph_.execute_nodes().size(); ++i) {
    graph_.execute_nodes()[i]->encode(&graph_);
  }

  command_ =
      std::make_unique<vkapi::CommandBuffer>(std::move(context->extract_cmd()));
}

void RepeatedGraphExecutor::execute() {
  api::Context* const context = graph_.context();
  command_->end();

  // Intentionally bypasses Context::submit_cmd_to_gpu(): its submit-count
  // bookkeeping and threshold-split path only serve submit_compute_job
  // recording, which benchmarks don't use.
  vkapi::VulkanFence fence = context->fences().get_fence();
  context->adapter_ptr()->submit_cmd(
      context->queue(),
      command_->get_submit_handle(/*final_use=*/false),
      fence.get_submit_handle());
  fence.wait();
  context->fences().return_fence(fence);
}

// Helper function to collect GPU timing from querypool
float collect_gpu_timing_us(
    ComputeGraph& graph,
    const std::vector<std::string>& shader_filter) {
  graph.context()->querypool().extract_results();
  const auto results = graph.context()->querypool().get_shader_timestamp_data();
  if (!results.empty()) {
    // Sum durations of all shaders that don't match any pattern in
    // shader_filter
    float total_duration_us = 0.0f;
    for (const auto& shader_result : results) {
      bool filtered = false;
      // Check if this shader matches any filter pattern
      for (const auto& filter_pattern : shader_filter) {
        if (shader_result.kernel_name.find(filter_pattern) !=
            std::string::npos) {
          filtered = true;
          break;
        }
      }

      if (!filtered) {
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

// Helper function to collect per-shader GPU timing from querypool
// Returns a map of shader_name -> timing_us for non-filtered shaders
std::unordered_map<std::string, float> collect_per_shader_timing_us(
    ComputeGraph& graph,
    const std::vector<std::string>& shader_filter) {
  std::unordered_map<std::string, float> shader_timings;

  graph.context()->querypool().extract_results();
  const auto results = graph.context()->querypool().get_shader_timestamp_data();
  for (const auto& shader_result : results) {
    bool filtered = false;
    // Check if this shader matches any filter pattern
    for (const auto& filter_pattern : shader_filter) {
      if (shader_result.kernel_name.find(filter_pattern) != std::string::npos) {
        filtered = true;
        break;
      }
    }

    if (!filtered) {
      // Calculate duration from start and end times, convert from ns to μs
      uint64_t duration_ns =
          shader_result.end_time_ns - shader_result.start_time_ns;
      float duration_us = static_cast<float>(duration_ns) / 1000.0f;
      // Accumulate timing for shaders with the same name
      shader_timings[shader_result.kernel_name] += duration_us;
    }
  }
  return shader_timings;
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

BenchmarkGraph setup_compute_graph(
    TestCase& test_case,
    const std::string& op_name,
    int op_invocations_per_execute) {
  GraphConfig config;
  config.enable_querypool = true;
  // Pool sizing takes max(execute, prepack) * factor, so scaling the factor
  // also over-reserves the prepack side (encoded once). Accepted: precise
  // execute-only sizing would need runtime changes.
  config.descriptor_pool_safety_factor *=
      std::max(1, op_invocations_per_execute);
  // Default-on (opt-out via TestCase::set_force_resize(false)): force every
  // DynamicDispatchNode to run its resize function when execute_test_case
  // runs propagate_resize() after prepack, exercising the op's resize formula
  // even when input shapes are unchanged.
  config.force_resize = test_case.get_force_resize();
  config.force_narrow_int4_tile = test_case.get_force_narrow_int4_tile();
  auto graph = std::make_unique<ComputeGraph>(config);

  std::vector<ValueRef> input_values;

  // Process input ValueSpecs
  for (size_t i = 0; i < test_case.num_inputs(); ++i) {
    const ValueSpec& input_spec = test_case.inputs()[i];

    if (input_spec.is_none()) {
      input_values.push_back(graph->add_none());
    } else if (input_spec.is_float()) {
      ValueRef input_value =
          graph->add_scalar(static_cast<double>(input_spec.get_float_value()));
      input_values.push_back(input_value);
    } else if (input_spec.is_int()) {
      ValueRef input_value =
          graph->add_scalar(static_cast<int64_t>(input_spec.get_int_value()));
      input_values.push_back(input_value);
    } else if (input_spec.is_bool()) {
      ValueRef input_value = graph->add_scalar(input_spec.get_bool_value());
      input_values.push_back(input_value);
    } else if (input_spec.is_int_list()) {
      // Convert int32_t list to int64_t list for ComputeGraph
      const auto& int32_list = input_spec.get_int_list();
      std::vector<int64_t> int64_list;
      int64_list.reserve(int32_list.size());
      for (int32_t val : int32_list) {
        int64_list.push_back(static_cast<int64_t>(val));
      }
      ValueRef input_value = graph->add_scalar_list(std::move(int64_list));
      input_values.push_back(input_value);
    } else if (input_spec.is_string()) {
      std::string str_copy = input_spec.get_string_value();
      ValueRef input_value = graph->add_string(std::move(str_copy));
      input_values.push_back(input_value);
    } else if (input_spec.is_constant()) {
      ValueRef input_value = graph->add_tensorref(
          input_spec.get_tensor_sizes(),
          input_spec.dtype,
          input_spec.get_data_ptr());
      input_values.push_back(input_value);
    } else {
      IOValueRef input_io = graph->add_input_tensor(
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
    ValueRef output_value = graph->add_tensor(
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

  // Nodes added before the op are staging uploads; nodes added after are
  // staging downloads. Only the op's own nodes are repeated by benchmarks.
  const size_t op_begin = graph->execute_nodes().size();
  opFn(*graph, op_args);
  const size_t op_end = graph->execute_nodes().size();

  for (size_t i = 0; i < output_values.size(); ++i) {
    graph->set_output_value(output_values[i]);
  }
  return {std::move(graph), {op_begin, op_end}};
}

// Test execution utilities
BenchmarkResult execute_test_case(
    TestCase& test_case,
    int warmup_runs,
    int benchmark_runs,
    int chained_dispatches,
    bool write_outputs) {
  BenchmarkResult result(
      test_case.name().empty() ? "unnamed_test_case" : test_case.name());

  // Initialize querypool if using GPU timestamps
  if (use_gpu_timestamps()) {
    api::context()->initialize_querypool();
  }

  // Build the operator once. Benchmark repetition is encoded separately so
  // persistent graph allocations are not duplicated.
  BenchmarkGraph benchmark = setup_compute_graph(
      test_case, test_case.operator_name(), chained_dispatches);
  ComputeGraph& graph = *benchmark.graph;

  // Prepare the graph
  graph.prepare();
  graph.prepack();

  // Run resize functions once so force_resize exercises resize formulas even
  // though the record/replay path below never calls propagate_resize().
  graph.propagate_resize();

  // Copy input data into the graph's staging buffers
  size_t graph_input_idx = 0;
  for (size_t i = 0; i < test_case.num_inputs(); ++i) {
    const ValueSpec& input_spec = test_case.inputs()[i];

    // Only non-constant tensor inputs correspond to graph.inputs() entries
    bool is_graph_input = input_spec.is_tensor() && !input_spec.is_constant() &&
        !input_spec.is_none();
    if (!is_graph_input) {
      continue;
    }

    if (graph_input_idx < graph.inputs().size()) {
      const auto& input_ref = graph.inputs()[graph_input_idx];

      // Get the appropriate data based on dtype
      const void* data_ptr = nullptr;
      size_t data_numel = input_spec.numel();

      // NOLINTNEXTLINE(clang-diagnostic-switch-enum)
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
      graph.maybe_cast_and_copy_into_staging(
          input_ref.staging, data_ptr, data_numel, input_spec.dtype);
    }
    ++graph_input_idx;
  }

  RepeatedGraphExecutor graph_executor(
      graph, chained_dispatches, benchmark.op_nodes);

  // Warmup runs
  for (int run = 0; run < warmup_runs; ++run) {
    graph_executor.execute();
  }

  // Benchmark runs - collect individual iteration timings
  float total_cpu_time_us = 0.0f;
  float total_gpu_time_us = 0.0f;

  const float chained_dispatches_f = static_cast<float>(chained_dispatches);

  for (int run = 0; run < benchmark_runs; ++run) {
    // Measure CPU time for each execute() call
    auto cpu_start = std::chrono::high_resolution_clock::now();
    graph_executor.execute();
    auto cpu_end = std::chrono::high_resolution_clock::now();

    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        cpu_end - cpu_start);
    // Each execute() ran chained_dispatches op invocations; report
    // per-invocation time.
    float cpu_time_us =
        static_cast<float>(cpu_duration.count()) / chained_dispatches_f;
    total_cpu_time_us += cpu_time_us;

    // Collect per-shader GPU timing - get raw shader results to preserve
    // metadata
    graph.context()->querypool().extract_results();
    const auto shader_results =
        graph.context()->querypool().get_shader_timestamp_data();

    // Calculate total GPU time from per-shader timings
    float gpu_time_us = 0.0f;
    for (const auto& shader_result : shader_results) {
      // Check if this shader matches any filter pattern
      bool filtered = false;
      for (const auto& filter_pattern : test_case.get_shader_filter()) {
        if (shader_result.kernel_name.find(filter_pattern) !=
            std::string::npos) {
          filtered = true;
          break;
        }
      }

      if (!filtered) {
        uint64_t duration_ns =
            shader_result.end_time_ns - shader_result.start_time_ns;
        float duration_us = static_cast<float>(duration_ns) / 1000.0f;
        gpu_time_us += duration_us;
        // Per-shader sample is already one querypool entry per dispatch (at
        // chained_dispatches>1 we get that many entries per shader), so do NOT
        // divide here — the aggregator (get_avg_time_us) averages across them
        // naturally.
        result.add_shader_timing(
            shader_result.kernel_name,
            duration_us,
            shader_result.metadata.gwg,
            shader_result.metadata.lwg);
      }
    }
    // gpu_time_us aggregates chained_dispatches worth of shader runs; divide
    // it down to per-invocation time.
    gpu_time_us /= chained_dispatches_f;
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

  // Copy output data from the graph's staging buffers. The benchmarking path
  // skips this work since it doesn't need correctness data — the probe pass
  // has already populated test_case.outputs() at chained_dispatches=1.
  if (write_outputs) {
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
          graph.maybe_cast_and_copy_from_staging(
              output_ref.staging, data_ptr, data_numel, output_spec.dtype);
        }

        // Print output tensor data if output printing is enabled
        if (print_output()) {
          std::string output_name = "Output[" + std::to_string(i) + "]";
          print_valuespec_data(output_spec, output_name);
        }
      }
    }
  }

  return result;
}

TestResult execute_test_cases(
    const std::function<std::vector<TestCase>()>& test_case_generator,
    const FlopCalculatorFunc& flop_calculator,
    const std::string& operation_name,
    int warmup_runs,
    int benchmark_runs,
    const ReferenceComputeFunc& reference_compute_func) {
  TestResult results(operation_name);

  // Generate all test cases
  std::vector<TestCase> test_cases = test_case_generator();

  std::cout << "Executing " << test_cases.size() << " test cases for "
            << operation_name << std::endl;
  print_separator();

  // Group test cases by ReferenceKey for caching reference computations
  // Use a vector to preserve the order in which groups first appear
  std::vector<ReferenceKey> group_order;
  std::unordered_map<ReferenceKey, std::vector<size_t>, ReferenceKeyHash>
      groups;
  for (size_t i = 0; i < test_cases.size(); ++i) {
    ReferenceKey key = ReferenceKey::from_test_case(test_cases[i]);
    if (groups.find(key) == groups.end()) {
      group_order.push_back(key);
    }
    groups[key].push_back(i);
  }

  bool any_correctness_failed = false;
  float total_gflops = 0.0f;
  size_t test_case_counter = 0;

  // Process each group: generate data, compute reference, execute, and print
  // Iterate in the order groups first appeared in test_cases
  for (const auto& key : group_order) {
    const auto& indices = groups[key];
    if (indices.empty()) {
      continue;
    }

    // Get first test case as the "prototype"
    size_t prototype_idx = indices[0];
    TestCase& prototype = test_cases[prototype_idx];

    // Generate data for prototype with deterministic seed based on key
    int group_seed =
        static_cast<int>(std::hash<std::string>{}(key.key_string) % 10000);
    for (auto& input : prototype.inputs()) {
      input.ensure_data_generated(group_seed++);
    }

    // Compute reference once for prototype
    bool ref_computed = false;
    if (reference_compute_func) {
      try {
        reference_compute_func(prototype);
        ref_computed = true;
      } catch (const std::invalid_argument&) {
        // Reference computation skipped for this group
      }
    }

    // Copy data and reference to other test cases in group
    for (size_t i = 1; i < indices.size(); ++i) {
      size_t tc_idx = indices[i];
      TestCase& tc = test_cases[tc_idx];

      // Copy input data from prototype
      for (size_t j = 0;
           j < tc.inputs().size() && j < prototype.inputs().size();
           ++j) {
        auto& dest = tc.inputs()[j];
        const auto& src = prototype.inputs()[j];
        if (dest.is_tensor() && src.is_tensor() && dest.sizes == src.sizes &&
            dest.dtype == src.dtype) {
          dest.share_data_from(src);
        }
      }

      // Copy reference output data if available
      if (ref_computed) {
        for (size_t j = 0;
             j < tc.outputs().size() && j < prototype.outputs().size();
             ++j) {
          const auto& src = prototype.outputs()[j];
          auto& dest = tc.outputs()[j];
          if (dest.is_tensor() && src.is_tensor() && dest.sizes == src.sizes &&
              dest.dtype == src.dtype) {
            dest.share_reference_from(src);
          }
        }
      }
    }

    // Execute and print results for all test cases in this group
    for (size_t tc_idx : indices) {
      TestCase& test_case = test_cases[tc_idx];
      ++test_case_counter;

      // Execute single test case
      BenchmarkResult result;
      bool shader_not_supported = false;
      try {
        // Always run a probe pass at chained_dispatches=1 with
        // write_outputs=true. This populates test_case.outputs() for the
        // downstream correctness check with a clean single-dispatch result,
        // and also gives us probe_us for adaptive sizing of the measurement
        // run's chained_dispatches.
        //
        // probe_then_scale (Google Benchmark style): size chained_dispatches
        // so each measurement execute() takes ~target_us. Tiny ops get a
        // large factor (driving GPU governor escalation); heavy ops get a
        // small factor (bounded wall time). Caveat: on Adreno 740 the probe
        // runs at the DCVS-pinned clock (~220 MHz), so probe_us is inflated
        // and the computed factor comes out under-sized for boost clock — but
        // the default target is generous enough that an under-sized factor
        // still drives sustained activity.
        BenchmarkResult probe_result = execute_test_case(
            test_case,
            /*warmup_runs=*/1,
            /*benchmark_runs=*/1,
            /*chained_dispatches=*/1,
            /*write_outputs=*/true);
        float probe_us = probe_result.get_avg_time_us();
        if (probe_us < kMinProbeTimeUs) {
          probe_us = kMinProbeTimeUs;
        }

        int chained_dispatches;
        const int manual_n = test_case.get_op_invocations_per_execute();
        if (manual_n > 0) {
          chained_dispatches = manual_n;
        } else {
          const int target_us = test_case.get_target_execute_time_us();
          chained_dispatches =
              std::max(1, static_cast<int>(target_us / probe_us));
          chained_dispatches =
              std::min(chained_dispatches, kMaxChainedDispatches);
        }
        if (debugging()) {
          std::cout << "[probe] " << test_case.name()
                    << ": probe_us=" << probe_us
                    << ", chained_dispatches=" << chained_dispatches
                    << std::endl;
        }

        // Measurement pass: chained_dispatches stacking, skip output copy
        // since the probe already wrote the correctness data.
        result = execute_test_case(
            test_case,
            warmup_runs,
            benchmark_runs,
            chained_dispatches,
            /*write_outputs=*/false);
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
      } else if (!ref_computed) {
        result.set_correctness_status(CorrectnessStatus::SKIPPED);
      } else {
        // Reference function provided and succeeded - validate outputs
        bool correctness_passed = true;

        for (size_t output_idx = 0; output_idx < test_case.num_outputs();
             ++output_idx) {
          const ValueSpec& output_spec = test_case.outputs()[output_idx];

          if (!output_spec.validate_against_reference(
                  test_case.get_abs_tolerance(),
                  test_case.get_rel_tolerance())) {
            correctness_passed = false;
            std::cout << "  Correctness validation FAILED for test "
                      << result.get_kernel_name() << std::endl;
            print_valuespec_data(output_spec, "vulkan output");
            print_valuespec_data(output_spec, "ref output", true);
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
        // Use the provided FLOP calculator to get total FLOPs for this test
        // case
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
          if (j < sizes.size() - 1) {
            size_info += "x";
          }
        }
      }
      size_info += "]";

      // Print progress using the BenchmarkResult member function
      result.print_summary(
          utils::safe_downcast<int>(test_case_counter), size_info, case_gflops);

      // Add result to collection
      results.add_result(std::move(result));

      test_case.clear();
    }
  }

  // Set the overall results on the TestResult
  results.set_correctness_passed(!any_correctness_failed);
  results.set_gflops(total_gflops);

  print_separator();
  std::cout << "Completed " << results.size() << " test cases" << std::endl;

  // Hard-fail if any correctness check failed. The per-case loop above keeps
  // running after a mismatch (so the full pass/fail matrix and per-16x16-tile
  // mismatch maps are printed for debugging) rather than aborting on the first
  // failure; throwing here preserves the original abort-on-mismatch behavior.
  if (any_correctness_failed) {
    throw std::runtime_error("Correctness validation failed");
  }

  return results;
}

// Convenience overload that uses the default FLOP calculator
TestResult execute_test_cases(
    const std::function<std::vector<TestCase>()>& test_case_generator,
    const std::string& operation_name,
    int warmup_runs,
    int benchmark_runs,
    const ReferenceComputeFunc& reference_compute_func) {
  return execute_test_cases(
      test_case_generator,
      default_flop_calculator,
      operation_name,
      warmup_runs,
      benchmark_runs,
      reference_compute_func);
}

} // namespace prototyping
} // namespace vulkan
} // namespace executorch
