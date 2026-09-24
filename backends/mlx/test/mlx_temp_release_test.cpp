/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Tests for temp-slot release: the interpreter drops a temp as soon as the
// chain running it has no further use for it, so a long chain does not finish
// holding every intermediate it produced.
//
// The case that matters is a temp that is NOT dead at the end of the chain that
// produced it: a branch or scan body runs through run_chain of its own, so a
// temp it writes for its caller would otherwise die at its last use inside the
// body.
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

bool slot_is_live(const ExecutionState& st, uint32_t tid) {
  return st.tensors[st.tensor_index(Tid{tid})].has_value();
}

} // namespace

// A temp that only the running chain names is dropped once its last reader has
// run, and the chain still produces the right answer.
TEST(MLXTempRelease, SingleChainTempIsDroppedAfterLastUse) {
  MLXProgram program;
  program.num_input_tensors = 1;
  program.num_output_tensors = 1;
  program.num_temp_tensors = 2;
  program.instruction_chains.push_back({
      make_add(kIn, kIn, kTemp0), // temp0 = 2
      make_add(kTemp0, kTemp0, kTemp0 + 1), // temp1 = 4, last use of temp0
      make_add(kTemp0 + 1, kTemp0 + 1, kOut), // out   = 8, last use of temp1
  });
  program.main_chain_idx = 0;

  ConstantData constants;
  MutableBufferData bufs;
  ExecutionState st;
  bind_state(st, program, constants, bufs, 4);

  Interpreter interp;
  interp.run(program, st);

  EXPECT_FALSE(slot_is_live(st, kTemp0));
  EXPECT_FALSE(slot_is_live(st, kTemp0 + 1));

  const array& out = st.tensors[st.tensor_index(Tid{kOut})].value();
  ::mlx::core::eval(out);
  EXPECT_EQ(out.data<float>()[0], 8.0f);
}

// A temp another chain names is never dropped, whichever chain is running.
TEST(MLXTempRelease, TempSharedWithAnotherChainIsKept) {
  const uint32_t kShared = kTemp0; // written by chain 1, read by chain 0

  MLXProgram program;
  program.num_input_tensors = 1;
  program.num_output_tensors = 1;
  program.num_temp_tensors = 1;
  // chain 0 (main): out = shared + shared
  program.instruction_chains.push_back({make_add(kShared, kShared, kOut)});
  // chain 1 (init): shared = in + in
  program.instruction_chains.push_back({make_add(kIn, kIn, kShared)});
  program.main_chain_idx = 0;
  program.init_chain_idx = 1;

  ConstantData constants;
  MutableBufferData bufs;
  ExecutionState st;
  bind_state(st, program, constants, bufs, 4);

  Interpreter interp;
  interp.run_chain(program, 1, st); // producer chain
  ASSERT_TRUE(slot_is_live(st, kShared))
      << "the producing chain dropped a temp another chain still reads";

  for (int i = 0; i < 3; ++i) {
    interp.run(program, st);
    ASSERT_TRUE(slot_is_live(st, kShared)) << "dropped on execute " << i;
    const array& out = st.tensors[st.tensor_index(Tid{kOut})].value();
    ::mlx::core::eval(out);
    EXPECT_EQ(out.data<float>()[0], 4.0f) << "execute " << i;
  }
}

// A branch chain must not drop a temp its caller reads afterwards. The caller
// opts out of release because it holds an IF, but the branch is itself run
// through run_chain and would otherwise see the temp die at its last use
// inside the branch.
TEST(MLXTempRelease, TempWrittenInABranchOutlivesIt) {
  const uint32_t kShared = kTemp0;

  Instruction if_instr;
  if_instr.op = OpCode::IF;
  IfNode node;
  node.cond = static_cast<int64_t>(1); // always take then_chain
  node.then_chain_idx = 1;
  node.else_chain_idx = 2;
  if_instr.node = node;

  MLXProgram program;
  program.num_input_tensors = 1;
  program.num_output_tensors = 1;
  program.num_temp_tensors = 1;
  // chain 0 (main): if (...) { shared = in + in }  then  out = shared + shared
  program.instruction_chains.push_back(
      {if_instr, make_add(kShared, kShared, kOut)});
  program.instruction_chains.push_back({make_add(kIn, kIn, kShared)}); // then
  program.instruction_chains.push_back({make_add(kIn, kIn, kShared)}); // else
  program.main_chain_idx = 0;

  ConstantData constants;
  MutableBufferData bufs;
  ExecutionState st;
  bind_state(st, program, constants, bufs, 4);

  Interpreter interp;
  interp.run(program, st);

  ASSERT_TRUE(slot_is_live(st, kShared))
      << "the branch dropped a temp its caller still reads";
  const array& out = st.tensors[st.tensor_index(Tid{kOut})].value();
  ::mlx::core::eval(out);
  EXPECT_EQ(out.data<float>()[0], 4.0f);
}

// A scan body must not drop a carry it reads but never writes: the tid is named
// by the parent's SCAN instruction, so it is shared and stays live across
// iterations.
TEST(MLXTempRelease, ScanBodyKeepsACarryItOnlyReads) {
  const uint32_t kCarry = kTemp0;
  const uint32_t kBodyTmp = kTemp0 + 1;

  Instruction scan_instr;
  scan_instr.op = OpCode::SCAN;
  ScanNode node;
  node.carry = {Tid{kCarry}};
  node.body_chain_idx = 1;
  node.scan_axis = 0;
  scan_instr.node = node;

  MLXProgram program;
  program.num_input_tensors = 1;
  program.num_output_tensors = 1;
  program.num_temp_tensors = 2;
  // chain 0 (main): carry = in + in ; SCAN ; out = carry + carry
  program.instruction_chains.push_back({
      make_add(kIn, kIn, kCarry),
      scan_instr,
      make_add(kCarry, kCarry, kOut),
  });
  // chain 1 (body): reads the carry, writes only its own temp
  program.instruction_chains.push_back({make_add(kCarry, kCarry, kBodyTmp)});
  program.main_chain_idx = 0;

  ConstantData constants;
  MutableBufferData bufs;
  ExecutionState st;
  bind_state(st, program, constants, bufs, 4);

  // The body names kCarry and the parent does too, so the body must not own it.
  const auto& body_table = st.temp_last_use[1];
  ASSERT_FALSE(body_table.empty()) << "the body chain should not opt out";
  EXPECT_EQ(
      body_table[st.tensor_index(Tid{kCarry})], ExecutionState::kNoTempLastUse)
      << "the body claimed a carry its caller also names";

  // The parent holds a SCAN, so it opts out entirely.
  EXPECT_TRUE(st.temp_last_use[0].empty());
}
