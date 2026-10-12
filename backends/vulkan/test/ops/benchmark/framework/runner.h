// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

#include "results.h"
#include "test_case.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {

//
// Test case execution
//

using FlopCalculatorFunc = std::function<int64_t(const TestCase&)>;

// Default FLOP calculation function (assumes 1 FLOP per element)
int64_t default_flop_calculator(const TestCase& test_case);

using ReferenceComputeFunc = std::function<void(TestCase&)>;

// Half-open index range of the operator's own dispatch nodes within a
// benchmark graph's execute_nodes(). Staging upload nodes precede it, staging
// download nodes follow it.
struct OpNodeRange {
  size_t begin = 0;
  size_t end = 0;
};

// A benchmark graph plus the location of its repeatable operator nodes. The
// graph is heap-held: ComputeGraph owns its Context and must never be moved
// (a moved-from graph's destructor dereferences a null context).
struct BenchmarkGraph {
  std::unique_ptr<ComputeGraph> graph;
  OpNodeRange op_nodes;
};

// Benchmark-only executor that records a graph's execute nodes into a single
// reusable command buffer, then replays it on every execute(). Staging uploads
// are encoded once, operator nodes N times, staging downloads once, so each
// iteration performs the same work as the old stacked-nodes layout (1 upload +
// N ops + 1 download) and the per-invocation divisor is unchanged. Production
// ComputeGraph::execute() behavior is unchanged.
//
// Notes for interpreting benchmark numbers:
// - Submit granularity differs from stacking N distinct nodes: all encodings
//   live in one command buffer with one submit per iteration (the old path
//   could split across command buffers at the node-count threshold), so
//   per-dispatch times may shift systematically against older data.
// - Resize functions run once via propagate_resize() in execute_test_case
//   before recording; replay itself never re-triggers resize.
// - Repeated encodings share one node/dispatch id, so per-repetition
//   querypool attribution is unavailable (aggregation keys on kernel name).
class RepeatedGraphExecutor final {
 public:
  RepeatedGraphExecutor(
      ComputeGraph& graph,
      int repetitions,
      OpNodeRange op_nodes);
  void execute();

 private:
  ComputeGraph& graph_;
  std::unique_ptr<vkapi::CommandBuffer> command_;
};

// Runs a measurement at the given chained_dispatches factor. The operator is
// built once, then its execute nodes are encoded that many times into a
// benchmark-only reusable command buffer. The probe-then-scale orchestration
// lives in execute_test_cases().
//
// write_outputs controls whether the graph's staging output buffers are copied
// back into test_case.outputs() at the end of the run. The probe path needs
// write_outputs=true so the correctness check has a clean single-dispatch
// reference. The benchmarking path passes write_outputs=false to avoid the
// per-iter GPU->CPU copy cost.
BenchmarkResult execute_test_case(
    TestCase& test_case,
    int warmup_runs = 1,
    int benchmark_runs = 1,
    int chained_dispatches = 1,
    bool write_outputs = true);

TestResult execute_test_cases(
    const std::function<std::vector<TestCase>()>& test_case_generator,
    const FlopCalculatorFunc& flop_calculator,
    const std::string& operation_name = "Operation",
    int warmup_runs = 1,
    int benchmark_runs = 1,
    const ReferenceComputeFunc& reference_compute_func = nullptr);

TestResult execute_test_cases(
    const std::function<std::vector<TestCase>()>& test_case_generator,
    const std::string& operation_name = "Operation",
    int warmup_runs = 1,
    int benchmark_runs = 1,
    const ReferenceComputeFunc& reference_compute_func = nullptr);

// Setup compute graph based on TestCase and operation name. The op function is
// invoked once. op_invocations_per_execute is used only to reserve enough
// descriptor capacity for benchmark-only repeated command encoding. Returns
// the graph plus the range of the operator's own nodes for repeated encoding.
BenchmarkGraph setup_compute_graph(
    TestCase& test_case,
    const std::string& op_name,
    int op_invocations_per_execute = 1);

} // namespace prototyping
} // namespace vulkan
} // namespace executorch
