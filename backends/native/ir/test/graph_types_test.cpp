/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <backends/native/ir/GraphTypes.h>
#include <executorch/schema/program_generated.h>

#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>

using namespace executorch::backends::portable;

namespace {

std::vector<uint8_t> load_file(const char* path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  EXPECT_TRUE(f.is_open()) << "Cannot open " << path;
  auto size = f.tellg();
  f.seekg(0);
  std::vector<uint8_t> buf(size);
  f.read(reinterpret_cast<char*>(buf.data()), size);
  return buf;
}

std::string testdata_path(const char* filename) {
  const char* base = std::getenv("ET_NATIVE_TEST_DATA");
  EXPECT_NE(base, nullptr) << "Set ET_NATIVE_TEST_DATA env var";
  std::string p(base);
  auto slash = p.rfind('/');
  EXPECT_NE(slash, std::string::npos);
  return p.substr(0, slash + 1) + filename;
}

std::unique_ptr<Graph> load_graph(
    const std::string& path,
    std::vector<uint8_t>& data_out,
    const executorch_flatbuffer::Program** prog_out = nullptr) {
  data_out = load_file(path.c_str());
  auto* program =
      flatbuffers::GetRoot<executorch_flatbuffer::Program>(data_out.data());
  if (prog_out)
    *prog_out = program;
  return std::make_unique<Graph>(program->execution_plan()->Get(0), program);
}

// Verifies that every op's outputs are recorded as produced by that op,
// and every op's inputs list that op as a user.
void verify_producer_user_consistency(const Graph& g) {
  for (size_t ci = 0; ci < g.num_chains(); ++ci) {
    for (size_t ii = 0; ii < g.num_ops_in_chain(ci); ++ii) {
      OperatorCall op = g.get_op(ci, ii);
      for (size_t k = 0; k < op.num_outputs(); ++k) {
        auto* p = g.producer(op.output(k));
        ASSERT_NE(p, nullptr) << "output vid=" << op.output(k);
        EXPECT_EQ(p->chain_idx, ci);
        EXPECT_EQ(p->instr_idx, ii);
      }
      for (size_t k = 0; k < op.num_inputs(); ++k) {
        auto u = g.users(op.input(k));
        bool found = false;
        for (size_t j = 0; j < u.size(); ++j) {
          if (u[j].chain_idx == ci && u[j].instr_idx == ii) {
            found = true;
            break;
          }
        }
        EXPECT_TRUE(found) << "op(" << ci << "," << ii
                           << ") not in users of vid=" << op.input(k);
      }
    }
  }
}

} // namespace

// =============================================================================
// Linear model (nn.Linear(4,4)): single op, covers core API surface
// =============================================================================

class LinearGraphTest : public ::testing::Test {
 protected:
  void SetUp() override {
    graph_ = load_graph(testdata_path("linear_4x4.bin"), data_, &program_);
  }
  std::vector<uint8_t> data_;
  const executorch_flatbuffer::Program* program_ = nullptr;
  std::unique_ptr<Graph> graph_;
};

TEST_F(LinearGraphTest, StructureAndMetadata) {
  EXPECT_STREQ(graph_->version(), "1.0");
  EXPECT_GT(graph_->num_values(), 0u);
  EXPECT_EQ(graph_->num_input_ids(), 1u);
  EXPECT_EQ(graph_->num_output_ids(), 1u);
  EXPECT_LT(graph_->input_id(0), graph_->num_values());
  EXPECT_LT(graph_->output_id(0), graph_->num_values());
  EXPECT_NE(graph_->input_id(0), graph_->output_id(0));
  EXPECT_GE(graph_->num_chains(), 1u);
  EXPECT_GE(graph_->num_operators(), 1u);
  EXPECT_EQ(graph_->main_chain_idx(), 0);
}

TEST_F(LinearGraphTest, ValueKindsAndTypes) {
  // Input
  uint32_t in_vid = graph_->input_id(0);
  EXPECT_EQ(graph_->value_kind(in_vid), ValueKind::INPUT);
  EXPECT_EQ(graph_->value_type(in_vid), ValueType::Tensor);
  EXPECT_FALSE(graph_->is_constant(in_vid));
  EXPECT_EQ(graph_->tensor_constant_data_key(in_vid), nullptr);

  // Output
  uint32_t out_vid = graph_->output_id(0);
  EXPECT_EQ(graph_->value_kind(out_vid), ValueKind::OUTPUT);
  EXPECT_EQ(graph_->value_type(out_vid), ValueType::Tensor);

  // Constants exist (weight + bias)
  bool has_constant = false;
  for (uint32_t i = 0; i < graph_->num_values(); ++i) {
    EXPECT_NE(graph_->value_meta(i), nullptr) << "vid=" << i;
    auto kind = graph_->value_kind(i);
    EXPECT_TRUE(
        kind == ValueKind::INPUT || kind == ValueKind::OUTPUT ||
        kind == ValueKind::CONSTANT || kind == ValueKind::MUTABLE_BUFFER ||
        kind == ValueKind::INTERMEDIATE);
    if (kind == ValueKind::CONSTANT)
      has_constant = true;
  }
  EXPECT_TRUE(has_constant);
}

TEST_F(LinearGraphTest, TensorAccessors) {
  uint32_t vid = graph_->input_id(0);
  EXPECT_EQ(graph_->tensor_dtype(vid), ::executorch::aten::ScalarType::Float);

  auto sizes = graph_->tensor_sizes(vid);
  EXPECT_EQ(sizes.size(), 2u);
  EXPECT_EQ(sizes[0], 1);
  EXPECT_EQ(sizes[1], 4);

  auto dim_order = graph_->tensor_dim_order(vid);
  EXPECT_EQ(dim_order.size(), 2u);
  EXPECT_EQ(dim_order[0], 0);
  EXPECT_EQ(dim_order[1], 1);

  EXPECT_EQ(graph_->tensor_nbytes_max(vid), 16u); // 1*4*sizeof(float)
  EXPECT_EQ(graph_->tensor_nbytes_max(graph_->output_id(0)), 16u);

  graph_->tensor_shape_dynamism(vid); // shouldn't crash
}

TEST_F(LinearGraphTest, MemObjId) {
  for (uint32_t i = 0; i < graph_->num_values(); ++i) {
    EXPECT_GE(graph_->mem_obj_id(i), -1);
  }
  for (size_t i = 0; i < graph_->num_mutable_buffer_ids(); ++i) {
    EXPECT_LT(graph_->mutable_buffer_id(i), graph_->num_values());
  }
}

TEST_F(LinearGraphTest, OperatorTable) {
  bool found_linear = false;
  for (size_t i = 0; i < graph_->num_operators(); ++i) {
    const char* name = graph_->operator_name(i);
    EXPECT_NE(name, nullptr);
    graph_->operator_overload(i); // shouldn't crash
    if (name && std::string(name).find("linear") != std::string::npos)
      found_linear = true;
  }
  EXPECT_TRUE(found_linear);
}

TEST_F(LinearGraphTest, InstructionAccess) {
  EXPECT_EQ(
      graph_->num_instructions(),
      graph_->num_ops_in_chain(graph_->main_chain_idx()));

  for (size_t ci = 0; ci < graph_->num_chains(); ++ci) {
    for (size_t ii = 0; ii < graph_->num_ops_in_chain(ci); ++ii) {
      EXPECT_EQ(graph_->instruction_kind(ci, ii), InstructionKind::Kernel);
      OperatorCall a = graph_->get_op(ci, ii);
      OperatorCall b = graph_->get_kernel_call(ci, ii);
      EXPECT_STREQ(a.name(), b.name());
      EXPECT_EQ(a.num_inputs(), b.num_inputs());
    }
  }

  for (size_t ii = 0; ii < graph_->num_instructions(); ++ii) {
    EXPECT_EQ(graph_->instruction_kind(ii), InstructionKind::Kernel);
    OperatorCall a = graph_->get_instruction(ii);
    OperatorCall b = graph_->get_op(graph_->main_chain_idx(), ii);
    EXPECT_STREQ(a.name(), b.name());
  }
}

TEST_F(LinearGraphTest, OperatorCallAPI) {
  auto refs = graph_->find_ops("aten::linear");
  ASSERT_EQ(refs.size(), 1u);
  OperatorCall op = graph_->get_op(refs[0].chain_idx, refs[0].instr_idx);

  EXPECT_STREQ(op.name(), "aten::linear");
  EXPECT_TRUE(
      std::string(op.full_name()).find("aten::linear") != std::string::npos);

  EXPECT_GE(op.num_inputs(), 1u);
  EXPECT_EQ(op.num_outputs(), 1u);
  EXPECT_EQ(op.args().size(), op.inputs().size() + op.num_outputs());

  for (size_t i = 0; i < op.num_inputs(); ++i)
    EXPECT_LT(op.input(i), graph_->num_values());
  EXPECT_LT(op.output(0), graph_->num_values());

  // linear's first input is graph input, output is graph output
  EXPECT_EQ(op.input(0), graph_->input_id(0));
  EXPECT_EQ(op.output(0), graph_->output_id(0));

  // node_id is mutable
  EXPECT_EQ(op.node_id(), 0u);
  OperatorCall op2 = graph_->get_instruction(0);
  op2.set_node_id(42);
  EXPECT_EQ(op2.node_id(), 42u);
}

TEST_F(LinearGraphTest, FindOps) {
  EXPECT_EQ(graph_->find_ops("aten::linear").size(), 1u);
  EXPECT_EQ(graph_->find_ops("nonexistent::op").size(), 0u);
  EXPECT_EQ(graph_->find_ops("").size(), 0u);
}

TEST_F(LinearGraphTest, ProducerAndUsers) {
  // Input: no producer, has users
  uint32_t in_vid = graph_->input_id(0);
  EXPECT_EQ(graph_->producer(in_vid), nullptr);
  EXPECT_GT(graph_->num_users(in_vid), 0u);

  // Output: has producer (linear op), no users
  uint32_t out_vid = graph_->output_id(0);
  auto* p = graph_->producer(out_vid);
  ASSERT_NE(p, nullptr);
  auto refs = graph_->find_ops("aten::linear");
  EXPECT_EQ(p->chain_idx, refs[0].chain_idx);
  EXPECT_EQ(p->instr_idx, refs[0].instr_idx);
  EXPECT_EQ(graph_->num_users(out_vid), 0u);

  // Constants: no producer
  for (uint32_t i = 0; i < graph_->num_values(); ++i) {
    if (graph_->value_kind(i) == ValueKind::CONSTANT)
      EXPECT_EQ(graph_->producer(i), nullptr) << "vid=" << i;
  }

  // num_users == users().size() for all values
  for (uint32_t i = 0; i < graph_->num_values(); ++i)
    EXPECT_EQ(graph_->num_users(i), graph_->users(i).size()) << "vid=" << i;

  // User refs are in range
  for (uint32_t i = 0; i < graph_->num_values(); ++i) {
    auto u = graph_->users(i);
    for (size_t j = 0; j < u.size(); ++j) {
      EXPECT_LT(u[j].chain_idx, graph_->num_chains());
      EXPECT_LT(u[j].instr_idx, graph_->num_ops_in_chain(u[j].chain_idx));
    }
  }
}

TEST_F(LinearGraphTest, OutOfRangeSafety) {
  EXPECT_EQ(graph_->value_meta(UINT32_MAX), nullptr);
  EXPECT_EQ(graph_->value_type(UINT32_MAX), ValueType::None);
  EXPECT_EQ(graph_->mem_obj_id(UINT32_MAX), -1);
  EXPECT_EQ(graph_->tensor_nbytes_max(UINT32_MAX), 0u);
  EXPECT_EQ(graph_->producer(UINT32_MAX), nullptr);
  EXPECT_EQ(graph_->users(UINT32_MAX).size(), 0u);
  EXPECT_EQ(graph_->num_users(UINT32_MAX), 0u);
  EXPECT_EQ(graph_->operator_name(999), nullptr);
  EXPECT_EQ(graph_->operator_overload(999), nullptr);
}

TEST_F(LinearGraphTest, ConstructWithPlanOnly) {
  Graph plan_only(program_->execution_plan()->Get(0));
  EXPECT_GT(plan_only.num_values(), 0u);
  EXPECT_EQ(plan_only.num_input_ids(), 1u);
  EXPECT_TRUE(plan_only.tensor_inline_data(plan_only.input_id(0)).empty());
}

TEST_F(LinearGraphTest, ProducerUserConsistency) {
  verify_producer_user_consistency(*graph_);
}

// =============================================================================
// Diamond model: multi-user values and producer chains
//
// x -> add(x,x) -> a -> mul(a,2) -> b
//                     -> add(a,1) -> c -> add(b,c) -> out
// =============================================================================

class DiamondGraphTest : public ::testing::Test {
 protected:
  void SetUp() override {
    graph_ = load_graph(testdata_path("diamond.bin"), data_);
  }
  std::vector<uint8_t> data_;
  std::unique_ptr<Graph> graph_;
};

TEST_F(DiamondGraphTest, Structure) {
  EXPECT_EQ(graph_->num_instructions(), 4u);
  EXPECT_EQ(graph_->find_ops("aten::add").size(), 3u);
  EXPECT_EQ(graph_->find_ops("aten::mul").size(), 1u);
}

TEST_F(DiamondGraphTest, MultiUserAndProducerChain) {
  // x (input 0) used twice by add[0]
  uint32_t x_vid = graph_->input_id(0);
  auto x_users = graph_->users(x_vid);
  EXPECT_GE(x_users.size(), 2u);
  EXPECT_EQ(x_users[0].instr_idx, 0u);
  EXPECT_EQ(x_users[1].instr_idx, 0u);

  // 'a' (output of add[0]) has 2 distinct user instructions
  OperatorCall first_add = graph_->get_instruction(0);
  uint32_t a_vid = first_add.output(0);
  EXPECT_GE(graph_->num_users(a_vid), 2u);
  std::unordered_set<uint32_t> a_user_instrs;
  for (auto& ref : graph_->users(a_vid))
    a_user_instrs.insert(ref.instr_idx);
  EXPECT_TRUE(a_user_instrs.count(1)); // mul
  EXPECT_TRUE(a_user_instrs.count(2)); // add(a,1)

  // Trace from output back to input through the diamond
  uint32_t out_vid = graph_->output_id(0);
  auto* p_final = graph_->producer(out_vid);
  ASSERT_NE(p_final, nullptr);
  EXPECT_EQ(p_final->instr_idx, 3u);

  OperatorCall final_add =
      graph_->get_op(p_final->chain_idx, p_final->instr_idx);
  auto* p_b = graph_->producer(final_add.input(0));
  auto* p_c = graph_->producer(final_add.input(1));
  ASSERT_NE(p_b, nullptr);
  ASSERT_NE(p_c, nullptr);
  EXPECT_EQ(p_b->instr_idx, 1u);
  EXPECT_EQ(p_c->instr_idx, 2u);

  // Both fan-in paths converge at 'a'
  OperatorCall mul_op = graph_->get_op(p_b->chain_idx, p_b->instr_idx);
  OperatorCall add_a1 = graph_->get_op(p_c->chain_idx, p_c->instr_idx);
  EXPECT_EQ(mul_op.input(0), add_a1.input(0)); // same 'a'
  EXPECT_EQ(graph_->producer(mul_op.input(0))->instr_idx, 0u);

  // Output has no users
  EXPECT_EQ(graph_->num_users(out_vid), 0u);
}

TEST_F(DiamondGraphTest, ProducerUserConsistency) {
  verify_producer_user_consistency(*graph_);
}

// =============================================================================
// KV cache model: inplace ops and cache buffer pattern
//
// k_cache [1,8,4] updated via index_copy_ (lowered to index_put_), then
// reduced via sum. index_put_ aliases input[0] and output[0] (inplace).
// =============================================================================

class KVCacheGraphTest : public ::testing::Test {
 protected:
  void SetUp() override {
    graph_ = load_graph(testdata_path("kv_cache.bin"), data_);
    // Find the cache: intermediate tensor with allocation, consumed by 2+ ops
    for (uint32_t i = 0; i < graph_->num_values(); ++i) {
      if (graph_->value_kind(i) == ValueKind::INTERMEDIATE &&
          graph_->value_type(i) == ValueType::Tensor &&
          graph_->mem_obj_id(i) >= 0 && graph_->num_users(i) >= 2) {
        cache_vid_ = i;
        break;
      }
    }
  }
  std::vector<uint8_t> data_;
  std::unique_ptr<Graph> graph_;
  uint32_t cache_vid_ = UINT32_MAX;
};

TEST_F(KVCacheGraphTest, Structure) {
  EXPECT_EQ(graph_->num_instructions(), 2u);
  EXPECT_EQ(graph_->find_ops("aten::index_put_").size(), 1u);
  EXPECT_EQ(graph_->find_ops("aten::sum").size(), 1u);
}

TEST_F(KVCacheGraphTest, InplaceCachePattern) {
  ASSERT_NE(cache_vid_, UINT32_MAX) << "cache tensor not found";

  // Cache is not an input, output, or constant
  EXPECT_EQ(graph_->value_kind(cache_vid_), ValueKind::INTERMEDIATE);
  EXPECT_FALSE(graph_->is_constant(cache_vid_));
  EXPECT_GE(graph_->mem_obj_id(cache_vid_), 0);

  // Cache shape: [1, 8, 4], 128 bytes
  auto sizes = graph_->tensor_sizes(cache_vid_);
  EXPECT_EQ(sizes.size(), 3u);
  EXPECT_EQ(sizes[0], 1);
  EXPECT_EQ(sizes[1], 8);
  EXPECT_EQ(sizes[2], 4);
  EXPECT_EQ(graph_->tensor_nbytes_max(cache_vid_), 128u);

  // index_put_ is inplace: input[0] == output[0] == cache
  auto refs = graph_->find_ops("aten::index_put_");
  ASSERT_EQ(refs.size(), 1u);
  OperatorCall ip = graph_->get_op(refs[0].chain_idx, refs[0].instr_idx);
  EXPECT_EQ(ip.input(0), ip.output(0));
  EXPECT_EQ(ip.output(0), cache_vid_);

  // Cache consumed by both index_put_ and sum
  std::unordered_set<uint32_t> user_instrs;
  for (auto& ref : graph_->users(cache_vid_))
    user_instrs.insert(ref.instr_idx);
  EXPECT_TRUE(user_instrs.count(0)); // index_put_
  EXPECT_TRUE(user_instrs.count(1)); // sum

  // Producer is index_put_ (inplace writes back to same value)
  auto* p = graph_->producer(cache_vid_);
  ASSERT_NE(p, nullptr);
  EXPECT_STREQ(
      graph_->get_op(p->chain_idx, p->instr_idx).name(), "aten::index_put_");

  // Cache and output have different mem_obj_ids (128B vs 16B)
  EXPECT_NE(
      graph_->mem_obj_id(cache_vid_), graph_->mem_obj_id(graph_->output_id(0)));
}

TEST_F(KVCacheGraphTest, ProducerUserConsistency) {
  verify_producer_user_consistency(*graph_);
}
