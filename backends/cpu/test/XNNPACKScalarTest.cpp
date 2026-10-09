// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/cpu/runtime/CPUPlan.h>
#include <executorch/backends/cpu/runtime/providers/xnnpack/XNNPACKProvider.h>
#include <executorch/runtime/platform/runtime.h>

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace executorch::backends::cpu {
namespace {
using runtime::Error;

class XNNPACKScalarTest : public ::testing::Test {
 protected:
  void SetUp() override {
    runtime::runtime_init();
    graph_.values.emplace_back(
        "input", ptn::ScalarType::Float, std::vector<int64_t>{});
    graph_.values.emplace_back(
        "result", ptn::ScalarType::Float, std::vector<int64_t>{});
    graph_.values[0].role = ptn::ValueRole::UserInput;
    graph_.input_ids = {0};
    graph_.output_ids = {1};
    graph_.nodes.resize(3);
    graph_.nodes[0].name = "input";
    graph_.nodes[0].op_kind = ptn::OpKind::Placeholder;
    graph_.nodes[0].outputs = {{ptn::OutputValueKind::Tensor, 0, {}}};
    graph_.nodes[1].name = "result";
    graph_.nodes[1].outputs = {{ptn::OutputValueKind::Tensor, 1, {}}};
    graph_.nodes[2].name = "output";
    graph_.nodes[2].op_kind = ptn::OpKind::Output;
    graph_.nodes[2].inputs = {{"", ptn::TensorArg{1}}};
    graph_.initialize_schedule();
  }

  void check_operation(
      const std::string& target,
      const std::vector<ptn::NamedArgument>& attributes,
      const std::vector<float>& input,
      const std::vector<float>& expected) {
    auto& operation = graph_.nodes[1];
    operation.target = target;
    operation.inputs = {{"self", ptn::TensorArg{0}}};
    operation.inputs.insert(
        operation.inputs.end(), attributes.begin(), attributes.end());
    graph_.rebuild_def_use();

    std::array<uint8_t, 4096> storage{};
    runtime::MemoryAllocator allocator(storage.size(), storage.data());
    std::vector<Buffer> buffers(graph_.values.size());
    const std::array<ProviderFactory, 1> factories{create_xnnpack_provider};
    CPUPlan plan(
        graph_, buffers, {{factories.data(), factories.size()}, {}, false}, {});
    ASSERT_EQ(plan.select(), Error::Ok);
    ASSERT_EQ(plan.prepare(allocator), Error::Ok);
    std::copy(input.begin(), input.end(), static_cast<float*>(buffers[0].data));
    ASSERT_EQ(plan.execute({}), Error::Ok);
    const auto* output = static_cast<const float*>(buffers[1].data);
    EXPECT_EQ(std::vector<float>(output, output + expected.size()), expected);
  }

  void set_linear(int64_t channels, const std::vector<int64_t>& bias_shape) {
    graph_.values[0].tensor_meta().sizes = {2, 3};
    graph_.values[1].tensor_meta().sizes = {2, channels};
    graph_.values.emplace_back(
        "weight", ptn::ScalarType::Float, std::vector<int64_t>{channels, 3});
    graph_.values.emplace_back("bias", ptn::ScalarType::Float, bias_shape);
    graph_.values[2].role = ptn::ValueRole::Parameter;
    graph_.values[3].role = ptn::ValueRole::Parameter;
    auto& operation = graph_.nodes[1];
    operation.target = "torch.ops.aten.linear.default";
    operation.inputs = {
        {"input", ptn::TensorArg{0}},
        {"weight", ptn::TensorArg{2}},
        {"bias", ptn::TensorArg{3}}};
    graph_.rebuild_def_use();
  }

  ptn::Graph graph_;
};

// Cppcheck cannot expand GTest fixture macros without build headers.
// cppcheck-suppress-begin syntaxError
TEST_F(XNNPACKScalarTest, Mean_ScalarAxesAndKeepdim_PreservesValue) {
  for (const int64_t axis : {0, -1}) {
    for (const bool keepdim : {false, true}) {
      SCOPED_TRACE(
          ::testing::Message() << "axis=" << axis << " keepdim=" << keepdim);
      check_operation(
          "torch.ops.aten.mean.dim",
          {{"dim", ptn::IntListArg{{axis}, {}}},
           {"keepdim", ptn::BoolArg{keepdim}},
           {"dtype", ptn::NoneArg{}}},
          {-3.25f},
          {-3.25f});
    }
  }
}

TEST_F(XNNPACKScalarTest, Permute_ScalarEmptyOrder_PreservesValue) {
  check_operation(
      "torch.ops.aten.permute_copy.default",
      {{"dims", ptn::IntListArg{}}},
      {7.5f},
      {7.5f});
}

TEST_F(XNNPACKScalarTest, Mean_VectorInput_ProducesScalar) {
  graph_.values[0].tensor_meta().sizes = {4};
  check_operation(
      "torch.ops.aten.mean.dim",
      {{"dim", ptn::IntListArg{{0}, {}}},
       {"keepdim", ptn::BoolArg{false}},
       {"dtype", ptn::NoneArg{}}},
      {1.0f, 2.0f, 3.0f, 4.0f},
      {2.5f});
}

TEST_F(XNNPACKScalarTest, Permute_MatrixInput_TransposesValues) {
  graph_.values[0].tensor_meta().sizes = {2, 3};
  graph_.values[1].tensor_meta().sizes = {3, 2};
  check_operation(
      "torch.ops.aten.permute_copy.default",
      {{"dims", ptn::IntListArg{{1, 0}, {}}}},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f});
}

TEST_F(XNNPACKScalarTest, Linear_ConstantBias_BroadcastsAcrossOutputChannels) {
  constexpr int64_t channels = 32;
  set_linear(channels, {});

  for (const auto& bias_shape :
       std::vector<std::vector<int64_t>>{{}, {1}, {channels}}) {
    SCOPED_TRACE(::testing::PrintToString(bias_shape));
    graph_.values[3].tensor_meta().sizes = bias_shape;
    std::array<uint8_t, 4096> storage{};
    runtime::MemoryAllocator allocator(storage.size(), storage.data());
    std::vector<Buffer> buffers(graph_.values.size());
    for (const ptn::ValueId id : {2, 3}) {
      auto bytes = tensor_bytes(graph_.value(id));
      ASSERT_TRUE(bytes.ok());
      auto buffer = allocate_buffer(allocator, bytes.get());
      ASSERT_TRUE(buffer.ok());
      std::fill_n(
          static_cast<float*>(buffer->data),
          bytes.get() / sizeof(float),
          id == 2 ? 1.0f : 2.0f);
      buffer->writable_bytes = 0;
      buffers[id] = buffer.get();
    }
    const std::array<ProviderFactory, 1> factories{create_xnnpack_provider};
    CPUPlan plan(
        graph_, buffers, {{factories.data(), factories.size()}, {}, false}, {});
    ASSERT_EQ(plan.select(), Error::Ok);
    ASSERT_EQ(plan.prepare(allocator), Error::Ok);
    std::fill_n(static_cast<float*>(buffers[0].data), 6, 1.0f);
    ASSERT_EQ(plan.execute({}), Error::Ok);
    const auto* output = static_cast<const float*>(buffers[1].data);
    EXPECT_EQ(
        std::vector<float>(output, output + 2 * channels),
        std::vector<float>(2 * channels, 5.0f));
  }
}

TEST_F(XNNPACKScalarTest, Linear_PerBatchBias_IsIneligible) {
  set_linear(4, {2, 4});
  std::vector<Buffer> buffers(graph_.values.size());
  const std::array<ProviderFactory, 1> factories{create_xnnpack_provider};
  CPUPlan plan(
      graph_, buffers, {{factories.data(), factories.size()}, {}, false}, {});
  EXPECT_EQ(plan.select(), Error::NotSupported);
}
// cppcheck-suppress-end syntaxError
} // namespace
} // namespace executorch::backends::cpu
