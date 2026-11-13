// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/cpu/runtime/providers/xnnpack/XNNPACKPasses.h>
#include <executorch/backends/cpu/runtime/providers/xnnpack/XNNPACKProvider.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>

namespace executorch::backends::cpu {
namespace {
using namespace executorch::runtime;

// Values of `x * clamp(clamp(x + 3, 0), max=6) / 6`, as ATen decomposes
// hardswish.
enum : ptn::ValueId {
  kX,
  kThree,
  kSix,
  kAdd,
  kClampMin,
  kClampMax,
  kMul,
  kDiv
};
constexpr size_t kCount = 8;

class XNNPACKPassesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    for (const auto& [name, shape] :
         std::array<std::pair<const char*, std::vector<int64_t>>, 8>{{
             {"x", {kCount}},
             {"three", {}},
             {"six", {}},
             {"add", {kCount}},
             {"clamp_min", {kCount}},
             {"clamp_max", {kCount}},
             {"mul", {kCount}},
             {"div", {kCount}},
         }}) {
      graph.values.emplace_back(name, ptn::ScalarType::Float, shape);
    }
    graph.values[kX].role = ptn::ValueRole::UserInput;
    graph.values[kThree].role = ptn::ValueRole::ConstantTensor;
    graph.values[kSix].role = ptn::ValueRole::ConstantTensor;
    add_node(
        "torch.ops.aten.add.Tensor",
        {ptn::TensorArg{kX}, ptn::TensorArg{kThree}, ptn::IntArg{1}},
        kAdd);
    add_node(
        "torch.ops.aten.clamp.default",
        {ptn::TensorArg{kAdd}, ptn::IntArg{0}, ptn::NoneArg{}},
        kClampMin);
    add_node(
        "torch.ops.aten.clamp.default",
        {ptn::TensorArg{kClampMin}, ptn::NoneArg{}, ptn::IntArg{6}},
        kClampMax);
    add_node(
        "torch.ops.aten.mul.Tensor",
        {ptn::TensorArg{kX}, ptn::TensorArg{kClampMax}},
        kMul);
    add_node(
        "torch.ops.aten.div.Tensor",
        {ptn::TensorArg{kMul}, ptn::TensorArg{kSix}},
        kDiv);
    graph.input_ids = {kX};
    graph.output_ids = {kDiv};
    graph.initialize_schedule();
    graph.rebuild_def_use();
    for (const auto& value : graph.values) {
      auto buffer = allocate_buffer(allocator, value.tensor_meta().numel() * 4);
      ASSERT_TRUE(buffer.ok());
      buffers.push_back(buffer.get());
    }
    *static_cast<float*>(buffers[kThree].data) = 3;
    *static_cast<float*>(buffers[kSix].data) = 6;
  }

  void add_node(
      const char* target,
      std::vector<ptn::Argument> arguments,
      ptn::ValueId output) {
    ptn::Node node;
    node.name = graph.values[output].name;
    node.target = target;
    for (auto& argument : arguments) {
      node.inputs.push_back({"", std::move(argument)});
    }
    node.outputs = {{ptn::OutputValueKind::Tensor, output, {}}};
    graph.nodes.push_back(std::move(node));
  }

  KernelRegion region() const {
    KernelRegion region{{}, {kX}, {kDiv}};
    region.nodes.resize(graph.nodes.size());
    std::iota(region.nodes.begin(), region.nodes.end(), 0);
    return region;
  }

  bool matches_whole_chain(const KernelRegion& region) const {
    const auto chains = xnnpack::find_hardswish_chains(region, graph, buffers);
    if (chains.size() != 1) {
      return false;
    }
    auto nodes = chains[0].nodes;
    std::sort(nodes.begin(), nodes.end());
    return chains[0].input == kX && chains[0].output == kDiv &&
        chains[0].nodes.back() == graph.value(kDiv).producer_id &&
        nodes == region.nodes;
  }

  alignas(64) std::array<uint8_t, 64 * 1024> storage{};
  MemoryAllocator allocator{
      static_cast<uint32_t>(storage.size()),
      storage.data()};
  ptn::Graph graph;
  std::vector<Buffer> buffers;
};

// Cppcheck cannot expand GTest fixture macros without build headers.
// cppcheck-suppress-begin syntaxError
TEST_F(XNNPACKPassesTest, FindsDecomposedHardswish) {
  EXPECT_TRUE(matches_whole_chain(region()));
}

TEST_F(XNNPACKPassesTest, FindsCommutedMulAndOverlappingClamps) {
  auto& mul = graph.nodes[3];
  std::swap(mul.inputs[0], mul.inputs[1]);
  graph.nodes[1].inputs[2].arg = ptn::IntArg{6};
  graph.nodes[2].inputs[2].arg = ptn::FloatArg{8.0};
  graph.rebuild_def_use();
  EXPECT_TRUE(matches_whole_chain(region()));
}

TEST_F(XNNPACKPassesTest, KeepsChainWhoseIntermediateLeavesRegion) {
  auto escaping = region();
  escaping.outputs.push_back(kClampMin);
  EXPECT_TRUE(xnnpack::find_hardswish_chains(escaping, graph, buffers).empty());
}

TEST_F(XNNPACKPassesTest, KeepsChainWithDifferentScalar) {
  *static_cast<float*>(buffers[kSix].data) = 5;
  EXPECT_TRUE(xnnpack::find_hardswish_chains(region(), graph, buffers).empty());
}

TEST_F(XNNPACKPassesTest, ProviderExecutesRecomposedHardswish) {
  auto* x = static_cast<float*>(buffers[kX].data);
  const std::array<float, kCount> input{-5, -3, -1.5f, -0.25f, 0, 1, 3, 5};
  std::copy(input.begin(), input.end(), x);
  size_t private_bytes = 0;
  PreparationContext context{
      graph, buffers, allocator, private_bytes, ExecutionContext{}};
  auto provider = create_xnnpack_provider();
  auto executable = provider->implementations()[0]->compile(region(), context);
  ASSERT_TRUE(executable.ok());
  ASSERT_EQ(provider->finish(), Error::Ok);
  ASSERT_EQ(executable.get()->reshape(), Error::Ok);
  ASSERT_EQ(executable.get()->bind(), Error::Ok);
  ASSERT_EQ(executable.get()->run(ExecutionContext{}), Error::Ok);
  const auto* result = static_cast<const float*>(buffers[kDiv].data);
  for (size_t index = 0; index < kCount; ++index) {
    SCOPED_TRACE(index);
    EXPECT_NEAR(
        result[index],
        input[index] * std::clamp(input[index] + 3, 0.0f, 6.0f) / 6,
        1e-6);
  }
}
// cppcheck-suppress-end syntaxError

} // namespace
} // namespace executorch::backends::cpu
