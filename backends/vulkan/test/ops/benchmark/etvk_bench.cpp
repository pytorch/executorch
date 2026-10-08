// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Runs the test case sets registered under cases/. See --helpshort for usage.

#include <executorch/backends/vulkan/test/ops/benchmark/framework/config.h>
#include <executorch/backends/vulkan/test/ops/benchmark/framework/device_info.h>
#include <executorch/backends/vulkan/test/ops/benchmark/framework/registry.h>
#include <executorch/backends/vulkan/test/ops/benchmark/framework/results.h>

#include <gflags/gflags.h>

#include <algorithm>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

DEFINE_bool(list, false, "List the selected sets as <op>:<set> and exit.");
DEFINE_string(
    op,
    "",
    "Comma-separated operators whose sets to run. An operator is its "
    "directory under cases/; a parent directory such as q8ta selects every "
    "operator beneath it. Empty selects all operators.");
DEFINE_string(
    set,
    "",
    "Comma-separated set names to run. Empty selects all sets.");
DEFINE_string(
    filter,
    "",
    "Run only test cases whose name matches the filter: '|'-separated "
    "alternatives, each a ';'-separated list of substrings that must all "
    "appear. For example, 'f16;+bias|f32' selects f16 cases with bias and all "
    "f32 cases.");
DEFINE_int32(
    chain,
    0,
    "Dispatch the operator this many times per measured execute. 0 sizes the "
    "chain adaptively.");
DEFINE_bool(
    gpu_timestamps,
    true,
    "Time with GPU timestamps. --nogpu_timestamps uses the wall clock.");
DEFINE_bool(print_output, false, "Print output tensor data.");
DEFINE_bool(print_latencies, false, "Print per-iteration latencies.");
DEFINE_bool(debug, false, "Print debugging information.");
DEFINE_bool(device_info, false, "Print the Vulkan device properties and exit.");

using namespace executorch::vulkan::prototyping;

namespace {

std::vector<std::string> split(const std::string& value, char delimiter) {
  std::vector<std::string> items;
  std::stringstream stream(value);
  std::string item;
  while (std::getline(stream, item, delimiter)) {
    if (!item.empty()) {
      items.push_back(item);
    }
  }
  return items;
}

bool op_selected(const std::vector<std::string>& ops, const std::string& op) {
  if (ops.empty()) {
    return true;
  }
  for (const std::string& sel : ops) {
    if (op == sel || op.rfind(sel + "/", 0) == 0) {
      return true;
    }
  }
  return false;
}

bool set_selected(
    const std::vector<std::string>& sets,
    const std::string& name) {
  return sets.empty() ||
      std::find(sets.begin(), sets.end(), name) != sets.end();
}

// One list of required substrings per alternative.
using NameFilter = std::vector<std::vector<std::string>>;

NameFilter parse_filter(const std::string& value) {
  NameFilter filter;
  for (const std::string& alternative : split(value, '|')) {
    std::vector<std::string> substrings = split(alternative, ';');
    if (!substrings.empty()) {
      filter.push_back(std::move(substrings));
    }
  }
  return filter;
}

bool contains_all(
    const std::string& name,
    const std::vector<std::string>& substrings) {
  for (const std::string& substring : substrings) {
    if (name.find(substring) == std::string::npos) {
      return false;
    }
  }
  return true;
}

bool name_matches(const NameFilter& filter, const std::string& name) {
  if (filter.empty()) {
    return true;
  }
  for (const std::vector<std::string>& substrings : filter) {
    if (contains_all(name, substrings)) {
      return true;
    }
  }
  return false;
}

bool set_precedes(
    const RegisteredTestCaseSet* a,
    const RegisteredTestCaseSet* b) {
  return a->op != b->op ? a->op < b->op : a->name < b->name;
}

std::vector<const RegisteredTestCaseSet*> sorted_sets() {
  std::vector<const RegisteredTestCaseSet*> sets;
  for (const RegisteredTestCaseSet& set : registered_test_case_sets()) {
    sets.push_back(&set);
  }
  std::sort(sets.begin(), sets.end(), set_precedes);
  return sets;
}

// Returns true if the set passed.
bool run_set(
    const RegisteredTestCaseSet& registered,
    const NameFilter& filter) {
  const std::string id = registered.op + ":" + registered.name;
  print_performance_header();
  std::cout << id << std::endl;
  print_separator();

  try {
    TestCaseSet set = registered.make();
    if (set.custom_test) {
      set.custom_test();
      return true;
    }
    std::vector<TestCase> test_cases;
    for (TestCase& test_case : set.test_cases) {
      if (!name_matches(filter, test_case.name())) {
        continue;
      }
      if (FLAGS_chain > 0) {
        test_case.set_op_invocations_per_execute(FLAGS_chain);
      }
      test_cases.push_back(std::move(test_case));
    }
    execute_test_cases(
        std::move(test_cases),
        set.flop_calculator,
        id,
        set.warmup_runs,
        set.benchmark_runs,
        set.reference);
  } catch (const std::exception& e) {
    std::cerr << id << " failed: " << e.what() << std::endl;
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "Runs the test case sets registered under cases/. With no selection "
      "flags, every set runs.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc > 1) {
    std::cerr << "Unexpected argument: " << argv[1] << std::endl;
    return 2;
  }
  if (FLAGS_chain < 0) {
    std::cerr << "--chain must not be negative" << std::endl;
    return 2;
  }

  if (FLAGS_device_info) {
    print_device_info();
    return 0;
  }

  const std::vector<std::string> ops = split(FLAGS_op, ',');
  const std::vector<std::string> set_names = split(FLAGS_set, ',');
  std::vector<const RegisteredTestCaseSet*> selected;
  for (const RegisteredTestCaseSet* set : sorted_sets()) {
    if (op_selected(ops, set->op) && set_selected(set_names, set->name)) {
      selected.push_back(set);
    }
  }
  if (selected.empty()) {
    std::cerr << "No test case sets match the selection" << std::endl;
    return 2;
  }

  if (FLAGS_list) {
    for (const RegisteredTestCaseSet* set : selected) {
      std::cout << set->op << ":" << set->name << std::endl;
    }
    return 0;
  }

  set_debugging(FLAGS_debug);
  set_print_output(FLAGS_print_output);
  set_print_latencies(FLAGS_print_latencies);
  set_use_gpu_timestamps(FLAGS_gpu_timestamps);

  const NameFilter filter = parse_filter(FLAGS_filter);
  std::vector<std::string> failed;
  for (const RegisteredTestCaseSet* set : selected) {
    if (!run_set(*set, filter)) {
      failed.push_back(set->op + ":" + set->name);
    }
  }

  print_separator();
  std::cout << "Ran " << selected.size() << " test case sets, " << failed.size()
            << " failed" << std::endl;
  for (const std::string& id : failed) {
    std::cout << "  FAILED: " << id << std::endl;
  }
  return failed.empty() ? 0 : 1;
}
