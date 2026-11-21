/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "MLXInterpreter.h"

#include <gtest/gtest.h>

namespace mx = ::mlx::core;
using namespace ::executorch::backends::mlx;

namespace {

Instruction make_kernel(uint32_t out, const std::string& expression) {
  MetalKernelNode node{};
  node.name = "shared_name";
  node.source = std::make_shared<const std::string>(
      "uint i = thread_position_in_grid.x; out[i] = " + expression + ";");
  node.inputs = {Tid{0}};
  node.outputs = {Tid{out}};
  node.input_names = {"x"};
  node.output_names = {"out"};
  node.grid = {Vid{0}, int64_t{1}, int64_t{1}};
  node.threadgroup = {int64_t{32}, int64_t{1}, int64_t{1}};
  node.ensure_row_contiguous = true;
  node.output_shapes_flat = {Vid{0}};
  node.output_shape_lengths = {1};
  node.output_dtypes = {6}; // Float
  return {OpCode::METAL_KERNEL, std::move(node)};
}

MLXProgram make_program() {
  MLXProgram program;
  program.num_input_tensors = 1;
  program.num_output_tensors = 2;
  program.num_values = 1;
  program.instruction_chains = {
      {make_kernel(1, "float(x[i]) + 1.0f"),
       make_kernel(2, "float(x[i]) * 3.0f")}};
  return program;
}

void expect_values(const mx::array& output, int size, float value) {
  mx::eval(output);
  ASSERT_EQ(output.size(), size);
  for (int i = 0; i < size; ++i) {
    EXPECT_FLOAT_EQ(output.data<float>()[i], value);
  }
}

} // namespace

TEST(MLXCustomKernel, CachedCallablesKeepDynamicInputsAndDistinctSources) {
  auto program = make_program();
  ConstantData constants;
  MutableBufferData buffers;
  ExecutionState state;
  state.bind(program, constants, buffers);
  auto stream = mx::new_stream(mx::Device::gpu);
  for (int round = 0; round < 3; ++round) {
    for (int size : {1, 7, 8, 17}) {
      for (auto dtype : {mx::float16, mx::float32}) {
        state.reset();
        state.set_tensor(Tid{0}, mx::full({size}, float(round), dtype));
        state.set_value(Vid{0}, size);
        Interpreter().run(program, state, stream);
        expect_values(state.const_tensor_ref(Tid{1}), size, float(round + 1));
        expect_values(state.const_tensor_ref(Tid{2}), size, float(round * 3));
        EXPECT_EQ(state.metal_kernels.size(), 2);
      }
    }
  }
}

TEST(MLXCustomKernel, RebindingDiscardsCachedCallables) {
  auto program = make_program();
  ConstantData constants;
  MutableBufferData buffers;
  ExecutionState state;
  state.bind(program, constants, buffers);
  state.set_tensor(Tid{0}, mx::full({8}, 2.0f, mx::float32));
  state.set_value(Vid{0}, 8);
  Interpreter().run(program, state);
  auto old_output = state.const_tensor_ref(Tid{1});

  program.instruction_chains[0][0] = make_kernel(1, "float(x[i]) + 9.0f");
  state.bind(program, constants, buffers);
  EXPECT_TRUE(state.metal_kernels.empty());
  state.set_tensor(Tid{0}, mx::full({8}, 4.0f, mx::float32));
  state.set_value(Vid{0}, 8);
  Interpreter().run(program, state);
  expect_values(old_output, 8, 3.0f);
  expect_values(state.const_tensor_ref(Tid{1}), 8, 13.0f);
  expect_values(state.const_tensor_ref(Tid{2}), 8, 12.0f);
}
