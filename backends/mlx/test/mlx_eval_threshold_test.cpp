/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Tests for the lazy-graph evaluation threshold (kEvalThresholdBytesKey).
//
// Drives Interpreter::run_chain directly over hand-built programs (no .pte),
// covering: that the disabled path does no accounting at all, that nested
// IF/SCAN chains accumulate through the caller's counter without being
// double-counted, that crossing the threshold actually materializes the live
// tensors mid-chain, and that results are unchanged by any of it.
//
// Must run on Apple Silicon: MLX needs the Metal backend.

#include "MLXInterpreter.h"

#include <mlx/mlx.h>

#include <gtest/gtest.h>

#include <vector>

using namespace ::executorch::backends::mlx;
using ::mlx::core::array;

namespace {

constexpr uint32_t kIn = 0; // input tid
constexpr uint32_t kOut = 1; // output tid
constexpr uint32_t kTemp0 = 2; // first temp tid

Instruction make_add(uint32_t a, uint32_t b, uint32_t out) {
  Instruction instr;
  instr.op = OpCode::ADD;
  AddNode node;
  node.a = Tid{a};
  node.b = Tid{b};
  node.out = Tid{out};
  instr.node = node;
  return instr;
}

// A chain of `n` ADDs: in -> temp0 -> temp1 -> ... -> out.
std::vector<Instruction> make_add_chain(uint32_t n) {
  std::vector<Instruction> chain;
  uint32_t prev = kIn;
  for (uint32_t i = 0; i < n; ++i) {
    const bool last = (i + 1 == n);
    const uint32_t out = last ? kOut : (kTemp0 + i);
    chain.push_back(make_add(prev, prev, out));
    prev = out;
  }
  return chain;
}

// Program whose main chain is `n` ADDs and nothing else.
MLXProgram make_flat_program(uint32_t n) {
  MLXProgram program;
  program.num_input_tensors = 1;
  program.num_output_tensors = 1;
  program.num_temp_tensors = n; // generous; unused slots stay nullopt
  program.instruction_chains.push_back(make_add_chain(n));
  program.main_chain_idx = 0;
  return program;
}

// Program whose main chain is a single IF; the taken branch is `n` ADDs.
// Exercises nested accumulation: the branch's instructions must be charged to
// the caller's counter, and the IF itself must not be charged again.
MLXProgram make_if_program(uint32_t n) {
  MLXProgram program;
  program.num_input_tensors = 1;
  program.num_output_tensors = 1;
  program.num_temp_tensors = n;

  Instruction if_instr;
  if_instr.op = OpCode::IF;
  IfNode node;
  node.cond = static_cast<int64_t>(1); // always take then_chain
  node.then_chain_idx = 1;
  node.else_chain_idx = 2;
  if_instr.node = node;

  program.instruction_chains.push_back({if_instr}); // chain 0: main
  program.instruction_chains.push_back(make_add_chain(n)); // chain 1: then
  program.instruction_chains.push_back(make_add_chain(1)); // chain 2: else
  program.main_chain_idx = 0;
  return program;
}

// Bind a state for `program` and seed the input with a non-trivial tensor.
// `mb` sizes the input so byte thresholds are easy to reason about.
void bind_state(
    ExecutionState& st,
    const MLXProgram& program,
    ConstantData& constants,
    MutableBufferData& bufs,
    int floats) {
  st.bind(program, constants, bufs);
  st.set_tensor(
      Tid{kIn}, ::mlx::core::full({floats}, 1.0f, ::mlx::core::float32));
}

// Bytes of one input tensor of `floats` elements.
size_t input_bytes(int floats) {
  return static_cast<size_t>(floats) * sizeof(float);
}

std::vector<float> read_output(ExecutionState& st) {
  const array& out = st.tensors[st.tensor_index(Tid{kOut})].value();
  ::mlx::core::eval(out);
  return std::vector<float>(out.data<float>(), out.data<float>() + out.size());
}

} // namespace

// The whole mechanism must be inert when the option is unset. Not merely "no
// evaluation happens": no traversal, no nbytes() queries, no accumulation.
TEST(MLXEvalThreshold, DisabledDoesNoAccountingAtAll) {
  const uint32_t kN = 8;
  MLXProgram program = make_flat_program(kN);
  ConstantData constants;
  MutableBufferData bufs;
  ExecutionState st;
  bind_state(st, program, constants, bufs, 1024);

  Interpreter interp; // default: disabled
  ASSERT_EQ(interp.eval_threshold_bytes(), 0u);

  Interpreter::reset_accounting_calls();
  interp.run(program, st);
  EXPECT_EQ(Interpreter::accounting_calls(), 0u);
}

// Enabled but with a threshold no run can reach: every instruction is still
// accounted, and nothing is evaluated early.
TEST(MLXEvalThreshold, EnabledAccountsEveryInstruction) {
  const uint32_t kN = 8;
  MLXProgram program = make_flat_program(kN);
  ConstantData constants;
  MutableBufferData bufs;
  ExecutionState st;
  bind_state(st, program, constants, bufs, 1024);

  Interpreter interp;
  interp.set_eval_threshold_bytes(std::numeric_limits<size_t>::max());
  EXPECT_EQ(interp.eval_threshold_bytes(), std::numeric_limits<size_t>::max());

  Interpreter::reset_accounting_calls();
  interp.run(program, st);
  EXPECT_EQ(Interpreter::accounting_calls(), kN);
  for (uint32_t i = 0; i < kN - 1; ++i) {
    const array& temp = st.tensors[st.tensor_index(Tid{kTemp0 + i})].value();
    EXPECT_FALSE(temp.is_available()) << "temp=" << i;
  }
  const array& out = st.tensors[st.tensor_index(Tid{kOut})].value();
  EXPECT_FALSE(out.is_available());
}

// An IF's branch instructions are charged to the caller's counter, and the IF
// instruction itself is not charged on top of them.
TEST(MLXEvalThreshold, NestedIfAccumulatesWithoutDoubleCounting) {
  const uint32_t kN = 6;
  MLXProgram program = make_if_program(kN);
  ConstantData constants;
  MutableBufferData bufs;
  ExecutionState st;
  bind_state(st, program, constants, bufs, 1024);

  Interpreter interp;
  interp.set_eval_threshold_bytes(std::numeric_limits<size_t>::max());

  Interpreter::reset_accounting_calls();
  interp.run(program, st);
  // kN branch instructions, and NOT kN + 1: the IF is excluded at the parent.
  EXPECT_EQ(Interpreter::accounting_calls(), kN);
}

// Crossing the threshold must actually materialize the live tensors. Without a
// forced evaluation an MLX array built by dispatch alone is not available.
TEST(MLXEvalThreshold, CrossingThresholdEvaluatesLiveTensors) {
  const int kFloats = 4096;
  const uint32_t kN = 9;
  MLXProgram program = make_flat_program(kN);
  ConstantData constants;
  MutableBufferData bufs;

  // Threshold of two instructions' worth: evaluation must fire mid-chain.
  {
    ExecutionState st;
    bind_state(st, program, constants, bufs, kFloats);
    Interpreter interp;
    interp.set_eval_threshold_bytes(2 * input_bytes(kFloats));
    interp.run(program, st);
    // The eighth ADD triggers a barrier; the ninth stays below the threshold.
    for (uint32_t i = 0; i < kN - 1; ++i) {
      const array& temp = st.tensors[st.tensor_index(Tid{kTemp0 + i})].value();
      EXPECT_TRUE(temp.is_available()) << "temp=" << i;
    }
    const array& out = st.tensors[st.tensor_index(Tid{kOut})].value();
    EXPECT_FALSE(out.is_available());
  }

  // Disabled: the same slot is still an unevaluated graph node.
  {
    ExecutionState st;
    bind_state(st, program, constants, bufs, kFloats);
    Interpreter interp;
    interp.run(program, st);
    const array& first_temp = st.tensors[st.tensor_index(Tid{kTemp0})].value();
    EXPECT_FALSE(first_temp.is_available());
  }
}

// Evaluating early must not change results, on a flat chain or through a
// nested one.
TEST(MLXEvalThreshold, OutputsAreUnchangedByTheThreshold) {
  const int kFloats = 2048;
  const uint32_t kN = 8;

  for (bool nested : {false, true}) {
    MLXProgram program = nested ? make_if_program(kN) : make_flat_program(kN);
    ConstantData constants;
    MutableBufferData bufs;

    ExecutionState off;
    bind_state(off, program, constants, bufs, kFloats);
    Interpreter disabled;
    disabled.run(program, off);
    const std::vector<float> expected = read_output(off);

    ExecutionState on;
    bind_state(on, program, constants, bufs, kFloats);
    Interpreter enabled;
    enabled.set_eval_threshold_bytes(input_bytes(kFloats)); // fires often
    enabled.run(program, on);
    const std::vector<float> actual = read_output(on);

    ASSERT_EQ(actual.size(), expected.size()) << "nested=" << nested;
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_EQ(actual[i], expected[i]) << "nested=" << nested << " i=" << i;
    }
  }
}

// The setting lives on the interpreter, so two handles configured differently
// do not interfere.
TEST(MLXEvalThreshold, SettingsArePerInterpreter) {
  MLXProgram program = make_flat_program(4);
  ConstantData constants;
  MutableBufferData bufs;

  Interpreter enabled;
  enabled.set_eval_threshold_bytes(1024);
  Interpreter disabled;

  EXPECT_EQ(enabled.eval_threshold_bytes(), 1024u);
  EXPECT_EQ(disabled.eval_threshold_bytes(), 0u);

  ExecutionState st_off;
  bind_state(st_off, program, constants, bufs, 256);
  Interpreter::reset_accounting_calls();
  disabled.run(program, st_off);
  EXPECT_EQ(Interpreter::accounting_calls(), 0u);

  ExecutionState st_on;
  bind_state(st_on, program, constants, bufs, 256);
  Interpreter::reset_accounting_calls();
  enabled.run(program, st_on);
  EXPECT_GT(Interpreter::accounting_calls(), 0u);
}
