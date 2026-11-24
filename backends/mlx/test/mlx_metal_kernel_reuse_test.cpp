/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Tests for MetalKernelNode kernel-function reuse: nodes that run the same
// kernel share one function, executions after the first reuse it, and
// metal_kernel_key() separates kernels on every field that shapes them.
//
// Must run on Apple Silicon: MLX needs the Metal backend.

#include "MLXInterpreter.h"

#include <mlx/mlx.h>

#include <gtest/gtest.h>

#include <memory>
#include <set>
#include <string>
#include <vector>

using namespace ::executorch::backends::mlx;

namespace {

constexpr uint32_t kA = 0; // input tids
constexpr uint32_t kB = 1;
constexpr int kN = 64;

const char* const kAdd =
    "uint i = thread_position_in_grid.x; out[i] = a[i] + b[i];";
const char* const kSub =
    "uint i = thread_position_in_grid.x; out[i] = a[i] - b[i];";

MetalKernelNode make_node(const char* source, uint32_t out) {
  MetalKernelNode n;
  n.name = "binary";
  n.source = std::make_shared<const std::string>(source);
  n.inputs = {Tid{kA}, Tid{kB}};
  n.outputs = {Tid{out}};
  n.grid = {int64_t{kN}, int64_t{1}, int64_t{1}};
  n.threadgroup = {int64_t{kN}, int64_t{1}, int64_t{1}};
  n.input_names = {"a", "b"};
  n.output_names = {"out"};
  n.ensure_row_contiguous = true;
  n.atomic_outputs = false;
  n.output_shapes_flat = {int64_t{kN}};
  n.output_shape_lengths = {1};
  n.output_dtypes = {6}; // ET ScalarType::Float
  return n;
}

Instruction make_instruction(MetalKernelNode node) {
  Instruction instr;
  instr.op = OpCode::METAL_KERNEL;
  instr.node = std::move(node);
  return instr;
}

} // namespace

// Two nodes run one kernel and a third another. The first execution builds two
// functions for the three nodes, the second reuses each node's function, and
// binding the program again drops them.
TEST(MLXMetalKernelReuse, ExecutionsReuseSharedFunctions) {
#ifndef ET_MLX_ALLOW_CUSTOM_KERNEL_EXECUTION
  GTEST_SKIP() << "custom kernel execution is disabled";
#else
  MLXProgram program;
  program.num_input_tensors = 2;
  program.num_output_tensors = 3;
  program.instruction_chains.push_back({
      make_instruction(make_node(kAdd, 2)),
      make_instruction(make_node(kAdd, 3)),
      make_instruction(make_node(kSub, 4)),
  });
  program.main_chain_idx = 0;

  ConstantData constants;
  MutableBufferData bufs;
  ExecutionState st;
  st.bind(program, constants, bufs);
  Interpreter interp;

  std::vector<const ::mlx::core::fast::CustomKernelFunction*> first;
  for (int run = 0; run < 2; ++run) {
    st.reset();
    st.set_tensor(Tid{kA}, ::mlx::core::full({kN}, 3.0f, ::mlx::core::float32));
    st.set_tensor(Tid{kB}, ::mlx::core::full({kN}, 1.0f, ::mlx::core::float32));
    interp.run(program, st);

    EXPECT_EQ(st.metal_kernels.size(), 2u);
    EXPECT_EQ(st.metal_kernel_of_node.size(), 3u);
    std::vector<const ::mlx::core::fast::CustomKernelFunction*> functions;
    for (const auto& instr : program.instruction_chains[0]) {
      functions.push_back(
          st.metal_kernel_of_node.at(&std::get<MetalKernelNode>(instr.node)));
    }
    EXPECT_EQ(functions[0], functions[1]);
    EXPECT_NE(functions[0], functions[2]);
    if (run == 0) {
      first = functions;
    } else {
      EXPECT_EQ(functions, first);
    }

    for (const auto& [out, expected] :
         {std::pair{2u, 4.0f}, std::pair{3u, 4.0f}, std::pair{4u, 2.0f}}) {
      const auto& y = st.tensors[st.tensor_index(Tid{out})].value();
      ::mlx::core::eval(y);
      EXPECT_EQ(y.data<float>()[0], expected);
      EXPECT_EQ(y.data<float>()[kN - 1], expected);
    }
  }

  st.bind(program, constants, bufs);
  EXPECT_TRUE(st.metal_kernels.empty());
  EXPECT_TRUE(st.metal_kernel_of_node.empty());
#endif
}

// Nodes that differ in any field metal_kernel() reads, or in the order of
// their template argument names, get separate keys; launch arguments don't
// split a kernel.
TEST(MLXMetalKernelReuse, KeyCoversEveryKernelField) {
  const MetalKernelNode base = make_node(kAdd, 2);
  std::vector<MetalKernelNode> variants(9, base);
  variants[0].name = "other";
  variants[1].input_names = {"b", "a"};
  variants[2].output_names = {"y"};
  variants[3].source = std::make_shared<const std::string>(kSub);
  variants[4].header = std::make_shared<const std::string>("// header");
  variants[5].ensure_row_contiguous = false;
  variants[6].atomic_outputs = true;
  variants[7].template_arg_names = {"X", "Y"};
  variants[8].template_arg_names = {"Y", "X"};

  std::set<std::string> keys{ops::metal_kernel_key(base)};
  for (const auto& variant : variants) {
    keys.insert(ops::metal_kernel_key(variant));
  }
  EXPECT_EQ(keys.size(), variants.size() + 1);

  MetalKernelNode launch = base;
  launch.outputs = {Tid{3}};
  launch.grid = {int64_t{32}, int64_t{2}, int64_t{1}};
  launch.output_shapes_flat = {int64_t{32}, int64_t{2}};
  launch.output_shape_lengths = {2};
  EXPECT_EQ(ops::metal_kernel_key(launch), ops::metal_kernel_key(base));
}
