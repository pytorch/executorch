/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Tests for temp-slot release: the interpreter drops a temp as soon as the
// program has no further use for it, so a long chain does not finish holding
// every intermediate it produced.
//
// The cases that matter are temps that cross a chain boundary: a branch or scan
// body runs through run_chain of its own, so a temp it writes for its caller
// would otherwise die at its last use inside the body, and a chain holding an
// IF must count the IF as a use of every temp its branches name.
//
// Must run on Apple Silicon: MLX needs the Metal backend.

#include "MLXInterpreter.h"

#include <mlx/mlx.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <variant>
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

Instruction make_full(int floats, float value, uint32_t out) {
  Instruction instr;
  instr.op = OpCode::FULL;
  FullNode node;
  node.out = Tid{out};
  node.shape = {static_cast<int64_t>(floats)};
  node.v = static_cast<double>(value);
  node.scalar_type = 6; // ET ScalarType::Float
  instr.node = node;
  return instr;
}

Instruction make_sum_all(uint32_t x, uint32_t out) {
  Instruction instr;
  instr.op = OpCode::SUM;
  SumNode node;
  node.x = Tid{x};
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

Instruction make_if(
    std::variant<int64_t, Vid> cond,
    uint32_t then_chain,
    uint32_t else_chain) {
  Instruction instr;
  instr.op = OpCode::IF;
  IfNode node;
  node.cond = cond;
  node.then_chain_idx = then_chain;
  node.else_chain_idx = else_chain;
  instr.node = node;
  return instr;
}

// Whether `chain` drops `tid` once its instruction `instr` has run.
bool dropped_after(
    const ExecutionState& st,
    uint32_t chain,
    uint32_t instr,
    uint32_t tid) {
  const auto& table = st.temp_releases[chain];
  if (table.empty()) {
    return false;
  }
  const auto& slots = table[instr];
  return std::find(slots.begin(), slots.end(), st.tensor_index(Tid{tid})) !=
      slots.end();
}

float first_output(const ExecutionState& st) {
  const array& out = st.tensors[st.tensor_index(Tid{kOut})].value();
  ::mlx::core::eval(out);
  return out.data<float>()[0];
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

// A shrinking op is charged for the input it was the last to read. FULL makes a
// 16 KB temp and SUM reduces it to a scalar: 16 KB + 16 KB crosses a 24 KB
// threshold only if SUM is accounted before the temp is released.
TEST(MLXTempRelease, ShrinkingOpIsAccountedBeforeRelease) {
  const int kFloats = 4096;

  MLXProgram program;
  program.num_input_tensors = 1;
  program.num_output_tensors = 1;
  program.num_temp_tensors = 1;
  program.instruction_chains.push_back({
      make_full(kFloats, 1.0f, kTemp0), // temp0 = ones(4096), 16 KB
      make_sum_all(kTemp0, kOut), // out = 4096, last use of temp0
  });
  program.main_chain_idx = 0;

  ConstantData constants;
  MutableBufferData bufs;
  ExecutionState st;
  bind_state(st, program, constants, bufs, 1);

  Interpreter interp;
  interp.set_eval_threshold_bytes(24576);
  interp.run(program, st);

  EXPECT_FALSE(slot_is_live(st, kTemp0));
  const array& out = st.tensors[st.tensor_index(Tid{kOut})].value();
  EXPECT_TRUE(out.is_available()) << "SUM did not trigger the barrier";
  EXPECT_EQ(out.item<float>(), static_cast<float>(kFloats));
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

// A branch chain must not drop a temp its caller reads afterwards; the caller
// drops it after its own last read.
TEST(MLXTempRelease, TempWrittenInABranchIsDroppedByItsCaller) {
  const uint32_t kShared = kTemp0;

  MLXProgram program;
  program.num_input_tensors = 1;
  program.num_output_tensors = 1;
  program.num_temp_tensors = 1;
  // chain 0 (main): if (...) { shared = in + in }  then  out = shared + shared
  program.instruction_chains.push_back(
      {make_if(int64_t{1}, 1, 2), make_add(kShared, kShared, kOut)});
  program.instruction_chains.push_back({make_add(kIn, kIn, kShared)}); // then
  program.instruction_chains.push_back({make_add(kIn, kIn, kShared)}); // else
  program.main_chain_idx = 0;

  ConstantData constants;
  MutableBufferData bufs;
  ExecutionState st;
  bind_state(st, program, constants, bufs, 4);

  EXPECT_FALSE(dropped_after(st, 1, 0, kShared))
      << "the branch claimed a temp its caller reads";
  EXPECT_TRUE(dropped_after(st, 0, 1, kShared));

  Interpreter interp;
  interp.run(program, st);

  EXPECT_EQ(first_output(st), 4.0f);
  EXPECT_FALSE(slot_is_live(st, kShared));
}

// A chain holding an IF drops its own temps: one read inside the branches at
// the IF, one written there at its caller's last read.
TEST(MLXTempRelease, ChainHoldingAnIfDropsItsTemps) {
  const uint32_t kT0 = kTemp0;
  const uint32_t kT1 = kTemp0 + 1;
  const uint32_t kT2 = kTemp0 + 2;

  MLXProgram program;
  program.num_input_tensors = 1;
  program.num_output_tensors = 1;
  program.num_temp_tensors = 3;
  // chain 0 (main): t0 = in + in ; if (...) { t1 = ... } ; t2 = t1 + t1 ;
  //                 out = t2 + t2
  program.instruction_chains.push_back({
      make_add(kIn, kIn, kT0),
      make_if(int64_t{1}, 1, 2),
      make_add(kT1, kT1, kT2),
      make_add(kT2, kT2, kOut),
  });
  program.instruction_chains.push_back({make_add(kT0, kT0, kT1)}); // then
  program.instruction_chains.push_back({make_add(kT0, kIn, kT1)}); // else
  program.main_chain_idx = 0;

  ConstantData constants;
  MutableBufferData bufs;
  ExecutionState st;
  bind_state(st, program, constants, bufs, 4);

  EXPECT_TRUE(dropped_after(st, 0, 1, kT0));
  EXPECT_TRUE(dropped_after(st, 0, 2, kT1));
  EXPECT_TRUE(dropped_after(st, 0, 3, kT2));

  for (size_t threshold : {size_t{0}, size_t{1}}) {
    Interpreter interp;
    interp.set_eval_threshold_bytes(threshold);
    bind_state(st, program, constants, bufs, 4);
    interp.run(program, st);
    EXPECT_EQ(first_output(st), 16.0f) << "threshold " << threshold;
    for (uint32_t tid : {kT0, kT1, kT2}) {
      EXPECT_FALSE(slot_is_live(st, tid))
          << "temp " << tid << " threshold " << threshold;
    }
  }
}

// A temp only one branch names is dropped inside that branch.
TEST(MLXTempRelease, BranchScratchIsDroppedInsideTheBranch) {
  const uint32_t kScratch = kTemp0;
  const uint32_t kResult = kTemp0 + 1;

  MLXProgram program;
  program.num_input_tensors = 1;
  program.num_output_tensors = 1;
  program.num_temp_tensors = 2;
  program.instruction_chains.push_back(
      {make_if(int64_t{1}, 1, 2), make_add(kResult, kResult, kOut)});
  program.instruction_chains.push_back({
      make_add(kIn, kIn, kScratch),
      make_add(kScratch, kScratch, kResult),
  }); // then
  program.instruction_chains.push_back({make_add(kIn, kIn, kResult)}); // else
  program.main_chain_idx = 0;

  ConstantData constants;
  MutableBufferData bufs;
  ExecutionState st;
  bind_state(st, program, constants, bufs, 4);

  EXPECT_TRUE(dropped_after(st, 1, 1, kScratch));
  EXPECT_TRUE(dropped_after(st, 0, 1, kResult));

  Interpreter interp;
  interp.run(program, st);
  EXPECT_EQ(first_output(st), 8.0f);
  EXPECT_FALSE(slot_is_live(st, kScratch));
  EXPECT_FALSE(slot_is_live(st, kResult));
}

// A temp named in a nested IF's branches and its holder belongs to the holder,
// not to the outer chain.
TEST(MLXTempRelease, NestedBranchTempBelongsToTheCommonAncestor) {
  const uint32_t kX = kTemp0;
  const uint32_t kY = kTemp0 + 1;

  MLXProgram program;
  program.num_input_tensors = 1;
  program.num_output_tensors = 1;
  program.num_temp_tensors = 2;
  program.instruction_chains.push_back(
      {make_if(int64_t{1}, 1, 2), make_add(kY, kY, kOut)}); // 0: main
  program.instruction_chains.push_back(
      {make_if(int64_t{1}, 3, 4), make_add(kX, kX, kY)}); // 1
  program.instruction_chains.push_back({make_add(kIn, kIn, kY)}); // 2
  program.instruction_chains.push_back({make_add(kIn, kIn, kX)}); // 3
  program.instruction_chains.push_back({make_add(kIn, kIn, kX)}); // 4
  program.main_chain_idx = 0;

  ConstantData constants;
  MutableBufferData bufs;
  ExecutionState st;
  bind_state(st, program, constants, bufs, 4);

  EXPECT_TRUE(dropped_after(st, 1, 1, kX));
  EXPECT_TRUE(dropped_after(st, 0, 1, kY));

  Interpreter interp;
  interp.run(program, st);
  EXPECT_EQ(first_output(st), 8.0f);
  EXPECT_FALSE(slot_is_live(st, kX));
  EXPECT_FALSE(slot_is_live(st, kY));
}

// A slot the caller uses before an IF and a branch reuses as scratch is only
// dropped after the IF.
TEST(MLXTempRelease, SlotReusedInsideABranchIsKeptUntilTheIf) {
  const uint32_t kS = kTemp0;
  const uint32_t kT = kTemp0 + 1;
  const uint32_t kU = kTemp0 + 2;

  MLXProgram program;
  program.num_input_tensors = 1;
  program.num_output_tensors = 1;
  program.num_temp_tensors = 3;
  // chain 0 (main): s = in + in ; t = s + s ; if (...) { u = ... } ;
  //                 out = u + u
  program.instruction_chains.push_back({
      make_add(kIn, kIn, kS),
      make_add(kS, kS, kT),
      make_if(int64_t{1}, 1, 2),
      make_add(kU, kU, kOut),
  });
  program.instruction_chains.push_back({
      make_add(kIn, kIn, kS),
      make_add(kS, kT, kU),
  }); // then: s = 2, u = 6
  program.instruction_chains.push_back({make_add(kT, kT, kU)}); // else
  program.main_chain_idx = 0;

  ConstantData constants;
  MutableBufferData bufs;
  ExecutionState st;
  bind_state(st, program, constants, bufs, 4);

  EXPECT_FALSE(dropped_after(st, 0, 1, kS));
  EXPECT_TRUE(dropped_after(st, 0, 2, kS));
  EXPECT_TRUE(dropped_after(st, 0, 2, kT));

  Interpreter interp;
  interp.run(program, st);
  EXPECT_EQ(first_output(st), 12.0f);
}

// Taking a different branch on each execute leaves every result right.
TEST(MLXTempRelease, AlternatingBranchesStayCorrect) {
  const uint32_t kScratch = kTemp0;
  const uint32_t kResult = kTemp0 + 1;

  MLXProgram program;
  program.num_input_tensors = 1;
  program.num_output_tensors = 1;
  program.num_temp_tensors = 2;
  program.num_values = 1;
  program.instruction_chains.push_back(
      {make_if(Vid{0}, 1, 2), make_add(kResult, kResult, kOut)});
  program.instruction_chains.push_back({make_add(kIn, kIn, kResult)}); // then
  program.instruction_chains.push_back({
      make_add(kIn, kIn, kScratch),
      make_add(kScratch, kIn, kResult),
  }); // else
  program.main_chain_idx = 0;

  ConstantData constants;
  MutableBufferData bufs;
  ExecutionState st;
  bind_state(st, program, constants, bufs, 4);

  Interpreter interp;
  for (int32_t take_then : {1, 0, 1, 0}) {
    st.reset();
    st.set_tensor(Tid{kIn}, ::mlx::core::full({4}, 1.0f, ::mlx::core::float32));
    st.set_value(Vid{0}, take_then);
    interp.run(program, st);
    EXPECT_EQ(first_output(st), take_then ? 4.0f : 6.0f);
    EXPECT_FALSE(slot_is_live(st, kScratch));
    EXPECT_FALSE(slot_is_live(st, kResult));
  }
}

// A chain run by two IFs is a tree of its own: what it shares with its callers
// is never dropped, and its callers still drop their own temps.
TEST(MLXTempRelease, ChainRunByTwoIfsKeepsWhatItShares) {
  const uint32_t kT = kTemp0;
  const uint32_t kU = kTemp0 + 1;

  MLXProgram program;
  program.num_input_tensors = 1;
  program.num_output_tensors = 1;
  program.num_temp_tensors = 2;
  program.instruction_chains.push_back({
      make_if(int64_t{1}, 1, 2),
      make_add(kT, kT, kU),
      make_if(int64_t{1}, 1, 2),
      make_add(kT, kU, kOut),
  });
  program.instruction_chains.push_back({make_add(kIn, kIn, kT)}); // then
  program.instruction_chains.push_back({make_add(kIn, kIn, kT)}); // else
  program.main_chain_idx = 0;

  ConstantData constants;
  MutableBufferData bufs;
  ExecutionState st;
  bind_state(st, program, constants, bufs, 4);

  EXPECT_TRUE(dropped_after(st, 0, 3, kU));

  Interpreter interp;
  interp.run(program, st);
  EXPECT_EQ(first_output(st), 6.0f);
  EXPECT_TRUE(slot_is_live(st, kT));
  EXPECT_FALSE(slot_is_live(st, kU));
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

  // The body names kCarry and the parent does too, so neither drops it.
  EXPECT_FALSE(dropped_after(st, 1, 0, kCarry))
      << "the body claimed a carry its caller also names";
  EXPECT_TRUE(dropped_after(st, 1, 0, kBodyTmp));

  // The parent holds a SCAN, so it drops nothing.
  EXPECT_TRUE(st.temp_releases[0].empty());
}
