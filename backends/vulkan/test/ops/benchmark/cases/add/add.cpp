// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/add/add.h>

#include <stdexcept>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace add {

// Custom FLOP calculator for add operation
// Add operation performs 1 FLOP (addition) per element
int64_t add_flop_calculator(const TestCase& test_case) {
  // Calculate total elements from the first input tensor
  int64_t total_elements = 1;
  if (!test_case.empty() && test_case.num_inputs() > 0 &&
      test_case.inputs()[0].is_tensor()) {
    const auto& sizes = test_case.inputs()[0].get_tensor_sizes();
    for (int64_t size : sizes) {
      total_elements *= size;
    }
  }

  // Add operation: 1 FLOP per element (one addition)
  return total_elements;
}

// Reference implementation for add operator
void add_reference_compute(TestCase& test_case) {
  const ValueSpec& input_a = test_case.inputs().at(0);
  const ValueSpec& input_b = test_case.inputs().at(1);

  ValueSpec& output = test_case.outputs().at(0);

  if (input_a.dtype != vkapi::kFloat) {
    throw std::invalid_argument("Unsupported dtype");
  }

  // Calculate number of elements
  int64_t num_elements = input_a.numel();

  auto& input_a_data = input_a.get_float_data();
  auto& input_b_data = input_b.get_float_data();

  auto& ref_data = output.get_ref_float_data();
  ref_data.resize(num_elements);
  for (int64_t i = 0; i < num_elements; ++i) {
    ref_data[i] = input_a_data[i] + input_b_data[i];
  }
}

} // namespace add
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
