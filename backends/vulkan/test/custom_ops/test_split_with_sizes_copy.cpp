// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>
#include <iostream>
#include <vector>
#include "utils.h"

using namespace executorch::vulkan::prototyping;
using namespace vkcompute;

int main(int argc, char* argv[]) {
  // Initialize Vulkan context
  try {
    api::context()->initialize_querypool();
  } catch (const std::exception& e) {
    std::cerr << "Failed to initialize Vulkan context: " << e.what()
              << std::endl;
    return 1;
  }

  std::cout << "Testing aten.split_with_sizes_copy.default natively..." << std::endl;

  // Build the compute graph manually
  GraphConfig config;
  ComputeGraph graph(config);
  
  std::vector<int64_t> sizes = {1, 11, 6144};
  ValueRef in = graph.add_tensor(sizes, vkapi::kFloat);
  
  std::vector<int64_t> split_sizes_vec = {2048, 2048, 2048};
  ValueRef split_sizes = graph.add_scalar_list<int64_t>(std::move(split_sizes_vec));
  
  ValueRef dim = graph.add_scalar<int64_t>(-1);
  
  std::vector<int64_t> out_sizes = {1, 11, 2048};
  ValueRef out1 = graph.add_tensor(out_sizes, vkapi::kFloat);
  ValueRef out2 = graph.add_tensor(out_sizes, vkapi::kFloat);
  ValueRef out3 = graph.add_tensor(out_sizes, vkapi::kFloat);
  
  std::vector<ValueRef> outputs = {out1, out2, out3};
  ValueRef out_list = graph.add_value_list(std::move(outputs));
  
  std::vector<ValueRef> args = {in, split_sizes, dim, out_list};
  VK_GET_OP_FN("aten.split_with_sizes_copy.default")(graph, args);
  
  graph.set_input_tensor(in);
  graph.set_output_tensor(out1);
  graph.set_output_tensor(out2);
  graph.set_output_tensor(out3);

  std::cout << "Graph built successfully. Compiling..." << std::endl;
  graph.prepare();
  graph.prepack();

  std::cout << "Graph compiled! Executing on GPU..." << std::endl;
  
  try {
    graph.execute();
    graph.context()->flush();
    std::cout << "SUCCESS! aten.split_with_sizes_copy.default executed natively!" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Failed to execute: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
