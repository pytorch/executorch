// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>
#include <iostream>
#include <vector>
#include "utils.h"

using namespace executorch::vulkan::prototyping;

// Generate test cases for any operation
std::vector<TestCase> generate_any_test_cases() {
  std::vector<TestCase> test_cases;

  // Set the data generation type
  DataGenType data_gen_type = DataGenType::RANDOM;

  // Define different input size configurations and reduction dimensions
  std::vector<std::pair<std::vector<int64_t>, int64_t>> configs = {
      {{1, 4, 32, 128}, 3}, // reduce on innermost dim
      {{1, 4, 32, 128}, 2}, // reduce on seq dim
  };

  std::vector<utils::StorageType> storage_types = {
      utils::kTexture3D, utils::kBuffer};

  std::vector<vkapi::ScalarType> data_types = {vkapi::kFloat};

  for (const auto& config : configs) {
    const auto& sizes = config.first;
    int64_t dim = config.second;

    for (const auto& storage_type : storage_types) {
      for (const auto& data_type : data_types) {
        // Vulkan reduction on buffers only supports reducing the innermost dimension
        if (storage_type == utils::kBuffer && dim != static_cast<int64_t>(sizes.size() - 1)) {
          continue;
        }

        TestCase test_case;

        std::string shape_str = shape_bracket(sizes) + "_dim" + std::to_string(dim);
        std::string storage_str = repr_str(storage_type, utils::kWidthPacked);
        std::string test_name = make_test_label("ANY", "f32", "f32", shape_str, storage_str);
        test_case.set_name(test_name);

        test_case.set_operator_name("aten.any.dim");

        ValueSpec input(sizes, data_type, storage_type, utils::kWidthPacked, data_gen_type);

        std::vector<int64_t> out_sizes = sizes;
        out_sizes[dim] = 1; // reduced dim becomes 1

        ValueSpec output(out_sizes, data_type, storage_type, utils::kWidthPacked, DataGenType::ZEROS);

        ValueSpec dim_spec; // Create spec for dim. Dim is int list or scalar int.
        // Wait, aten.any.dim takes (Tensor self, int dim, bool keepdim)
        // Since we are dispatching via graph, the operator implementation expects `dim_ref`
        // In Vulkan executorch, ATen reduce ops usually take dim and keepdim.
        // Wait, let's look at `Reduce.cpp`.
        // `int32_t reduce_dim = graph.extract_scalar<int32_t>(dim_ref);`
        // So `dim_ref` is a scalar!
        
        test_case.add_input_spec(input);
        
        // Add dim as integer scalar
        ValueSpec dim_val(static_cast<int32_t>(dim));
        test_case.add_input_spec(dim_val);

        // Add keepdim as boolean scalar (true)
        ValueSpec keepdim_val(true);
        test_case.add_input_spec(keepdim_val);

        test_case.add_output_spec(output);

        test_cases.push_back(test_case);
      }
    }
  }

  return test_cases;
}

int64_t any_flop_calculator(const TestCase& test_case) {
  int64_t total_elements = test_case.inputs()[0].numel();
  return total_elements;
}

void any_reference_compute(TestCase& test_case) {
  const ValueSpec& input = test_case.inputs().at(0);
  const ValueSpec& dim_val = test_case.inputs().at(1);
  ValueSpec& output = test_case.outputs().at(0);

  int64_t dim = dim_val.get_int32_data()[0];
  const auto& sizes = input.sizes;

  // We only support dim=3 or 2 for this simple reference compute
  if (dim != 3 && dim != 2) {
    throw std::runtime_error("Reference compute only supports dim 2 or 3, got: " + std::to_string(dim));
  }

  int64_t B = sizes[0];
  int64_t H = sizes[1];
  int64_t Seq = sizes[2];
  int64_t Dim = sizes[3];

  auto& in_data = input.get_float_data();
  auto& out_data = output.get_ref_float_data();
  out_data.resize(output.numel());

  for (auto& val : out_data) val = 0.0f;

  for (int64_t b = 0; b < B; ++b) {
    for (int64_t h = 0; h < H; ++h) {
      for (int64_t seq = 0; seq < Seq; ++seq) {
        for (int64_t d = 0; d < Dim; ++d) {
          int64_t in_idx = b * (H * Seq * Dim) + h * (Seq * Dim) + seq * Dim + d;
          
          int64_t out_seq = (dim == 2) ? 0 : seq;
          int64_t out_d = (dim == 3) ? 0 : d;
          
          int64_t out_idx = b * (H * ((dim==2)?1:Seq) * ((dim==3)?1:Dim)) + h * (((dim==2)?1:Seq) * ((dim==3)?1:Dim)) + out_seq * ((dim==3)?1:Dim) + out_d;
          
          if (in_data[in_idx] != 0.0f) {
            out_data[out_idx] = 1.0f;
          }
        }
      }
    }
  }
}

int main(int argc, char* argv[]) {
  set_print_output(false); 
  set_print_latencies(false); 
  set_use_gpu_timestamps(true);

  print_performance_header();
  std::cout << "Any Operation Prototyping Framework" << std::endl;
  print_separator();

  try {
    api::context()->initialize_querypool();
  } catch (const std::exception& e) {
    std::cerr << "Failed to initialize Vulkan context: " << e.what() << std::endl;
    return 1;
  }

  auto results = execute_test_cases(
      generate_any_test_cases,
      any_flop_calculator,
      "Any",
      1,
      1,
      any_reference_compute);

  return 0;
}
