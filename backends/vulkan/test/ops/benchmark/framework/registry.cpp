// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "registry.h"

#include <executorch/backends/vulkan/runtime/vk_api/Exception.h>

#include <utility>

namespace executorch {
namespace vulkan {
namespace prototyping {

namespace {

std::vector<RegisteredTestCaseSet>& registry() {
  static std::vector<RegisteredTestCaseSet> sets;
  return sets;
}

} // namespace

const std::vector<RegisteredTestCaseSet>& registered_test_case_sets() {
  return registry();
}

TestCaseSetRegistrar::TestCaseSetRegistrar(
    std::string op,
    std::string name,
    std::function<TestCaseSet()> make) {
  for (const RegisteredTestCaseSet& set : registry()) {
    VK_CHECK_COND(
        set.op != op || set.name != name,
        "Test case set ",
        op,
        ":",
        name,
        " is registered more than once");
  }
  registry().push_back({std::move(op), std::move(name), std::move(make)});
}

} // namespace prototyping
} // namespace vulkan
} // namespace executorch
