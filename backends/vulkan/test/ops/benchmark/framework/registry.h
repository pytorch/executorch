// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "runner.h"
#include "test_case.h"

#include <functional>
#include <string>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {

struct TestCaseSet {
  std::vector<TestCase> test_cases;
  ReferenceComputeFunc reference = nullptr;
  FlopCalculatorFunc flop_calculator = default_flop_calculator;
  int warmup_runs = 1;
  int benchmark_runs = 1;
  // Runs instead of test_cases for checks that are not expressible as test
  // cases. Throws on failure.
  std::function<void()> custom_test = nullptr;
};

struct RegisteredTestCaseSet {
  // Path of the operator's directory under cases/, e.g. "q8ta/conv2d".
  std::string op;
  std::string name;
  // Invoked only when the set is selected, since building test cases may
  // query the device.
  std::function<TestCaseSet()> make;
};

const std::vector<RegisteredTestCaseSet>& registered_test_case_sets();

class TestCaseSetRegistrar final {
 public:
  TestCaseSetRegistrar(
      std::string op,
      std::string name,
      std::function<TestCaseSet()> make);
};

#define ETVK_TEST_CASE_SET_CONCAT_IMPL(a, b) a##b
#define ETVK_TEST_CASE_SET_CONCAT(a, b) ETVK_TEST_CASE_SET_CONCAT_IMPL(a, b)

// Registers the TestCaseSet returned by the function body that follows.
#define REGISTER_TEST_CASE_SET(op, name)                               \
  static ::executorch::vulkan::prototyping::TestCaseSet                \
      ETVK_TEST_CASE_SET_CONCAT(make_test_case_set_, __LINE__)();      \
  static const ::executorch::vulkan::prototyping::TestCaseSetRegistrar \
      ETVK_TEST_CASE_SET_CONCAT(test_case_set_registrar_, __LINE__)(   \
          op,                                                          \
          name,                                                        \
          &ETVK_TEST_CASE_SET_CONCAT(make_test_case_set_, __LINE__));  \
  static ::executorch::vulkan::prototyping::TestCaseSet                \
  ETVK_TEST_CASE_SET_CONCAT(make_test_case_set_, __LINE__)()

} // namespace prototyping
} // namespace vulkan
} // namespace executorch
