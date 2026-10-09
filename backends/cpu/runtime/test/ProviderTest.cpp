// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/cpu/runtime/providers/executorch/ETProvider.h>
#include <executorch/backends/cpu/runtime/providers/xnnpack/XNNPACKProvider.h>
#include <executorch/backends/cpu/runtime/test/BufferTestUtil.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>

namespace executorch::backends::cpu {
namespace {
using namespace executorch::runtime;
using torch::executor::testing::TensorFactory;

class ProviderTest : public ::testing::Test {
 protected:
  void gelu_graph(size_t count) {
    graph.values.emplace_back(
        "input",
        ptn::ScalarType::Float,
        std::vector<int64_t>{static_cast<int64_t>(count)});
    graph.values.back().role = ptn::ValueRole::UserInput;
    graph.values.emplace_back(
        "output",
        ptn::ScalarType::Float,
        std::vector<int64_t>{static_cast<int64_t>(count)});
    ptn::Node node;
    node.name = "gelu";
    node.target = "torch.ops.aten.gelu.default";
    node.inputs = {
        {"self", ptn::TensorArg{0}}, {"approximate", ptn::StringArg{"none"}}};
    node.outputs = {{ptn::OutputValueKind::Tensor, 1, {}}};
    graph.nodes.push_back(std::move(node));
    graph.input_ids = {0};
    graph.output_ids = {1};
    graph.initialize_schedule();
    graph.rebuild_def_use();
    for (const auto& value : graph.values) {
      const auto& shape = value.tensor_meta().sizes;
      values.emplace_back(
          factory.zeros(std::vector<int32_t>(shape.begin(), shape.end())));
    }
    buffers.resize(2);
  }
  void bind_value(size_t id, Buffer buffer) {
    buffers[id] = buffer;
    values[id].toTensor().unsafeGetTensorImpl()->set_data(buffer.data);
  }
  alignas(64) std::array<uint8_t, 1024 * 1024> storage{};
  MemoryAllocator allocator{
      static_cast<uint32_t>(storage.size()),
      storage.data()};
  TensorFactory<aten::ScalarType::Float> factory;
  ptn::Graph graph;
  std::vector<EValue> values;
  std::vector<Buffer> buffers;
  size_t private_bytes = 0;
  ExecutionContext execution_context{1, true};
};

// Cppcheck cannot expand GTest fixture macros without build headers.
// cppcheck-suppress-begin syntaxError
TEST_F(ProviderTest, ActualProvidersRespectGuardPageAndRebindingBounds) {
  gelu_graph(16);
  GuardedBuffer input(64);
  GuardedBuffer output(64);
  bind_value(0, input.buffer);
  bind_value(1, output.buffer);
  auto* source = static_cast<float*>(input.buffer.data);
  for (size_t i = 0; i < 16; ++i) {
    source[i] = (static_cast<float>(i) - 8) / 4;
  }
  PreparationContext context{
      graph, buffers, allocator, private_bytes, execution_context};
  ExecutionContext execution = execution_context;
  for (auto create : {create_xnnpack_provider, create_et_provider}) {
    auto provider = create();
    auto* implementation = provider->implementations()[0];
    ASSERT_TRUE(
        implementation->supports(graph.node(0), graph, execution_context)
            .supported);
    auto executable = implementation->compile({{0}, {0}, {1}}, context);
    ASSERT_TRUE(executable.ok());
    ASSERT_EQ(provider->finish(), Error::Ok);
    ASSERT_EQ(executable.get()->reshape(), Error::Ok);
    ASSERT_EQ(executable.get()->bind(), Error::Ok);
    ASSERT_EQ(executable.get()->run(execution), Error::Ok);
    auto* result = static_cast<float*>(output.buffer.data);
    for (size_t i = 0; i < 16; ++i) {
      EXPECT_NEAR(
          result[i],
          0.5f * source[i] * (1 + std::erf(source[i] / std::sqrt(2.0f))),
          1e-6);
    }
    const float before = result[0];
    buffers[0].readable_bytes = 64;
    EXPECT_EQ(executable.get()->bind(), Error::InvalidArgument);
    EXPECT_EQ(result[0], before);
    buffers[0] = input.buffer;
    buffers[0].data = static_cast<uint8_t*>(buffers[0].data) + 4;
    EXPECT_EQ(executable.get()->bind(), Error::InvalidArgument);
    buffers[0] = input.buffer;
    buffers[0].readable_bytes = 0;
    EXPECT_EQ(executable.get()->bind(), Error::InvalidArgument);
    buffers[0] = input.buffer;
  }
}

TEST_F(ProviderTest, ProvidersHandleNonVectorMultipleAndPointerReplacement) {
  gelu_graph(11);
  GuardedBuffer first(44);
  GuardedBuffer second(44);
  GuardedBuffer output(44);
  bind_value(0, first.buffer);
  bind_value(1, output.buffer);
  std::fill_n(static_cast<float*>(first.buffer.data), 11, 1);
  std::fill_n(static_cast<float*>(second.buffer.data), 11, 2);
  PreparationContext context{
      graph, buffers, allocator, private_bytes, execution_context};
  for (auto create : {create_xnnpack_provider, create_et_provider}) {
    buffers[0] = first.buffer;
    auto provider = create();
    auto executable =
        provider->implementations()[0]->compile({{0}, {0}, {1}}, context);
    ASSERT_TRUE(executable.ok());
    ASSERT_EQ(provider->finish(), Error::Ok);
    ASSERT_EQ(executable.get()->reshape(), Error::Ok);
    ExecutionContext execution = execution_context;
    ASSERT_EQ(executable.get()->bind(), Error::Ok);
    ASSERT_EQ(executable.get()->run(execution), Error::Ok);
    EXPECT_NEAR(static_cast<float*>(output.buffer.data)[10], 0.84134475f, 1e-6);
    buffers[0] = second.buffer;
    ASSERT_EQ(executable.get()->bind(), Error::Ok);
    ASSERT_EQ(executable.get()->run(execution), Error::Ok);
    EXPECT_NEAR(static_cast<float*>(output.buffer.data)[10], 1.95449974f, 1e-6);
  }
}

class XNNActivationTest : public ProviderTest,
                          public ::testing::WithParamInterface<const char*> {
 protected:
  void activation_graph(size_t count) {
    gelu_graph(count);
    graph.nodes[0].name = "activation";
    graph.nodes[0].target = GetParam();
    graph.nodes[0].inputs.resize(1);
  }
};

TEST_P(XNNActivationTest, MatchesReferenceAndRebindsNonVectorMultiple) {
  const float infinity = std::numeric_limits<float>::infinity();
  const std::array<float, 13> input{
      -infinity, -100, -10, -1, -0.0f, 0, 1e-6f, 1, 10, 100, infinity,
      std::numeric_limits<float>::quiet_NaN(), -0.125f};
  activation_graph(input.size());
  GuardedBuffer first(sizeof(input)), second(sizeof(input)), output(sizeof(input));
  std::copy(input.begin(), input.end(), static_cast<float*>(first.buffer.data));
  std::reverse_copy(
      input.begin(), input.end(), static_cast<float*>(second.buffer.data));
  bind_value(0, first.buffer);
  bind_value(1, output.buffer);
  PreparationContext context{
      graph, buffers, allocator, private_bytes, execution_context};
  auto provider = create_xnnpack_provider();
  auto* implementation = provider->implementations()[0];
  ASSERT_TRUE(
      implementation->supports(graph.node(0), graph, execution_context).supported);
  auto executable = implementation->compile({{0}, {0}, {1}}, context);
  ASSERT_TRUE(executable.ok());
  ASSERT_EQ(provider->finish(), Error::Ok);
  ASSERT_EQ(executable.get()->reshape(), Error::Ok);
  const bool relu = graph.nodes[0].target == "torch.ops.aten.relu.default";
  for (const auto& source : {first.buffer, second.buffer}) {
    buffers[0] = source;
    ASSERT_EQ(executable.get()->bind(), Error::Ok);
    ASSERT_EQ(executable.get()->run(execution_context), Error::Ok);
    const auto* data = static_cast<const float*>(source.data);
    const auto* result = static_cast<const float*>(output.buffer.data);
    for (size_t index = 0; index < input.size(); ++index) {
      SCOPED_TRACE(index);
      if (std::isnan(data[index])) {
        EXPECT_TRUE(std::isnan(result[index]));
      } else if (relu) {
        EXPECT_FLOAT_EQ(result[index], data[index] < 0 ? 0 : data[index]);
      } else {
        EXPECT_NEAR(
            result[index], 1.0 / (1.0 + std::exp(-double(data[index]))), 1e-6);
      }
    }
  }
  buffers[0].readable_bytes = sizeof(input);
  EXPECT_EQ(executable.get()->bind(), Error::InvalidArgument);
}

TEST_P(XNNActivationTest, RejectsUnsupportedDtypesShapesAndSchemas) {
  activation_graph(13);
  auto provider = create_xnnpack_provider();
  auto* implementation = provider->implementations()[0];
  const auto supported = [&] {
    return implementation->supports(graph.node(0), graph, execution_context)
        .supported;
  };
  ASSERT_TRUE(supported());
  for (auto dtype : {ptn::ScalarType::Half, ptn::ScalarType::Char}) {
    graph.values[0] = ptn::Value("input", dtype, {13});
    EXPECT_FALSE(supported());
    graph.values[0] = ptn::Value("input", ptn::ScalarType::Float, {13});
    graph.values[1] = ptn::Value("output", dtype, {13});
    EXPECT_FALSE(supported());
    graph.values[1] = ptn::Value("output", ptn::ScalarType::Float, {13});
  }
  graph.values[1] = ptn::Value("output", ptn::ScalarType::Float, {1, 13});
  EXPECT_FALSE(supported());
  graph.values[1] = ptn::Value("output", ptn::ScalarType::Float, {13});
  auto& node = graph.nodes[0];
  node.inputs[0].arg = ptn::IntArg{0};
  EXPECT_FALSE(supported());
  node.inputs.clear();
  EXPECT_FALSE(supported());
  node.inputs = {{"self", ptn::TensorArg{0}}, {"extra", ptn::IntArg{0}}};
  EXPECT_FALSE(supported());
  node.inputs.resize(1);
  node.outputs.clear();
  EXPECT_FALSE(supported());
}

INSTANTIATE_TEST_SUITE_P(
    ReluAndSigmoid,
    XNNActivationTest,
    ::testing::Values("torch.ops.aten.relu.default", "torch.ops.aten.sigmoid.default"));

TEST_F(ProviderTest, XNNPACKValidatesFinalConstantBeforePacking) {
  graph.values.emplace_back(
      "input", ptn::ScalarType::Float, std::vector<int64_t>{1, 3});
  graph.values[0].role = ptn::ValueRole::UserInput;
  graph.values.emplace_back(
      "weight", ptn::ScalarType::Float, std::vector<int64_t>{2, 3});
  graph.values[1].role = ptn::ValueRole::Parameter;
  graph.values.emplace_back(
      "output", ptn::ScalarType::Float, std::vector<int64_t>{1, 2});
  ptn::Node node;
  node.name = "linear";
  node.target = "torch.ops.aten.linear.default";
  node.inputs = {
      {"input", ptn::TensorArg{0}},
      {"weight", ptn::TensorArg{1}},
      {"bias", ptn::NoneArg{}}};
  node.outputs = {{ptn::OutputValueKind::Tensor, 2, {}}};
  graph.nodes.push_back(std::move(node));
  for (const auto& value : graph.values) {
    const auto& shape = value.tensor_meta().sizes;
    values.emplace_back(
        factory.zeros(std::vector<int32_t>(shape.begin(), shape.end())));
  }
  buffers.resize(3);
  GuardedBuffer input(12), weight(24), output(8);
  bind_value(0, input.buffer);
  bind_value(1, weight.buffer);
  bind_value(2, output.buffer);
  std::fill_n(static_cast<float*>(input.buffer.data), 3, 1);
  std::fill_n(static_cast<float*>(weight.buffer.data), 6, 2);
  PreparationContext context{
      graph, buffers, allocator, private_bytes, execution_context};
  auto provider = create_xnnpack_provider();
  buffers[1].readable_bytes = 24;
  auto rejected =
      provider->implementations()[0]->compile({{0}, {0, 1}, {2}}, context);
  EXPECT_EQ(rejected.error(), Error::InvalidArgument);
  buffers[1] = weight.buffer;
  auto executable =
      provider->implementations()[0]->compile({{0}, {0, 1}, {2}}, context);
  ASSERT_TRUE(executable.ok());
  ASSERT_EQ(provider->finish(), Error::Ok);
  ASSERT_EQ(executable.get()->reshape(), Error::Ok);
  ASSERT_EQ(executable.get()->bind(), Error::Ok);
  ExecutionContext execution = execution_context;
  ASSERT_EQ(executable.get()->run(execution), Error::Ok);
  EXPECT_FLOAT_EQ(static_cast<float*>(output.buffer.data)[0], 6);
  EXPECT_FLOAT_EQ(static_cast<float*>(output.buffer.data)[1], 6);
}

TEST_F(ProviderTest, XNNPACKSharesWeightConversionsAcrossRegions) {
  for (const auto* name : {"input", "weight", "middle", "output"}) {
    graph.values.emplace_back(
        name, ptn::ScalarType::Float, std::vector<int64_t>{1, 2, 3, 3});
  }
  graph.values[0].role = ptn::ValueRole::UserInput;
  graph.values[1] = ptn::Value("weight", ptn::ScalarType::Float, {2, 2, 3, 3});
  graph.values[1].role = ptn::ValueRole::Parameter;
  for (auto id : {0, 1}) {
    ptn::Node node;
    node.name = id == 0 ? "first" : "second";
    node.target = "torch.ops.aten.convolution.default";
    node.inputs = {
        {"input", ptn::TensorArg{id == 0 ? 0 : 2}},
        {"weight", ptn::TensorArg{1}},
        {"bias", ptn::NoneArg{}},
        {"stride", ptn::IntListArg{{1, 1}, {}}},
        {"padding", ptn::IntListArg{{1, 1}, {}}},
        {"dilation", ptn::IntListArg{{1, 1}, {}}},
        {"transposed", ptn::BoolArg{false}},
        {"output_padding", ptn::IntListArg{{0, 0}, {}}},
        {"groups", ptn::IntArg{1}}};
    node.outputs = {{ptn::OutputValueKind::Tensor, id == 0 ? 2 : 3, {}}};
    graph.nodes.push_back(std::move(node));
  }
  for (const auto& value : graph.values) {
    auto storage = allocate_buffer(allocator, tensor_bytes(value).get());
    ASSERT_TRUE(storage.ok());
    buffers.push_back(storage.get());
  }
  std::fill_n(static_cast<float*>(buffers[0].data), 18, 1);
  std::fill_n(static_cast<float*>(buffers[1].data), 36, 2);
  PreparationContext context{
      graph, buffers, allocator, private_bytes, execution_context};
  auto provider = create_xnnpack_provider();
  auto first =
      provider->implementations()[0]->compile({{0}, {0, 1}, {2}}, context);
  ASSERT_TRUE(first.ok());
  EXPECT_EQ(private_bytes, 36 * sizeof(float) + kReadableTail);
  auto second =
      provider->implementations()[0]->compile({{1}, {1, 2}, {3}}, context);
  ASSERT_TRUE(second.ok());
  EXPECT_EQ(private_bytes, 36 * sizeof(float) + kReadableTail);
  ASSERT_EQ(provider->finish(), Error::Ok);
  for (auto* executable : {first.get().get(), second.get().get()}) {
    ASSERT_EQ(executable->reshape(), Error::Ok);
    ASSERT_EQ(executable->bind(), Error::Ok);
    ASSERT_EQ(executable->run(execution_context), Error::Ok);
  }
  EXPECT_FLOAT_EQ(static_cast<float*>(buffers[3].data)[0], 400);
  EXPECT_FLOAT_EQ(static_cast<float*>(buffers[3].data)[4], 784);

  first.get().reset();
  second.get().reset();
  provider.reset();
  graph.values[1] = ptn::Value("weight", ptn::ScalarType::Float, {2, 2, 1, 1});
  graph.values[1].role = ptn::ValueRole::Parameter;
  graph.nodes[0].inputs[4].arg = ptn::IntListArg{{0, 0}, {}};
  private_bytes = 0;
  provider = create_xnnpack_provider();
  auto compatible =
      provider->implementations()[0]->compile({{0}, {0, 1}, {2}}, context);
  ASSERT_TRUE(compatible.ok());
  EXPECT_EQ(private_bytes, 0);
  ASSERT_EQ(provider->finish(), Error::Ok);
  ASSERT_EQ(compatible.get()->reshape(), Error::Ok);
  ASSERT_EQ(compatible.get()->bind(), Error::Ok);
  ASSERT_EQ(compatible.get()->run(execution_context), Error::Ok);
  EXPECT_FLOAT_EQ(static_cast<float*>(buffers[2].data)[0], 4);
}

TEST_F(ProviderTest, BufferAllocationAndOffsetExtentAreExplicit) {
  auto allocated = allocate_buffer(allocator, 36);
  ASSERT_TRUE(allocated.ok());
  EXPECT_TRUE(allocated->accepts(36, true));
  for (size_t i = 36; i < 100; ++i) {
    EXPECT_EQ(static_cast<uint8_t*>(allocated->data)[i], 0);
  }
  EXPECT_FALSE(allocated->accepts(37, false));
  EXPECT_FALSE(allocated->accepts(std::numeric_limits<size_t>::max(), false));
  Buffer view{storage.data() + 64, 128, 64, 64};
  EXPECT_TRUE(view.accepts(64, true));
  view.readable_bytes = 127;
  EXPECT_FALSE(view.accepts(64, false));
}

TEST_F(ProviderTest, XNNPACKAcceptsOnlyContiguousFullSpanStridedCopies) {
  gelu_graph(6);
  auto& node = graph.node(0);
  node.target = "torch.ops.aten.as_strided_copy.default";
  node.inputs = {
      {"self", ptn::TensorArg{0}},
      {"size", ptn::IntListArg{{6}, {}}},
      {"stride", ptn::IntListArg{{1}, {}}},
      {"storage_offset", ptn::IntArg{0}}};
  GuardedBuffer input(24), output(24);
  bind_value(0, input.buffer);
  bind_value(1, output.buffer);
  std::iota(
      static_cast<float*>(input.buffer.data),
      static_cast<float*>(input.buffer.data) + 6,
      1);
  auto provider = create_xnnpack_provider();
  auto* implementation = provider->implementations()[0];
  ASSERT_TRUE(
      implementation->supports(node, graph, execution_context).supported);
  PreparationContext context{
      graph, buffers, allocator, private_bytes, execution_context};
  auto executable = implementation->compile({{0}, {0}, {1}}, context);
  ASSERT_TRUE(executable.ok());
  ASSERT_EQ(provider->finish(), Error::Ok);
  ASSERT_EQ(executable.get()->reshape(), Error::Ok);
  ASSERT_EQ(executable.get()->bind(), Error::Ok);
  ExecutionContext execution = execution_context;
  ASSERT_EQ(executable.get()->run(execution), Error::Ok);
  EXPECT_EQ(std::memcmp(input.buffer.data, output.buffer.data, 24), 0);

  graph.values[1] = ptn::Value("output", ptn::ScalarType::Float, {3, 2});
  node.inputs[1].arg = ptn::IntListArg{{3, 2}, {}};
  node.inputs[2].arg = ptn::IntListArg{{1, 3}, {}};
  EXPECT_FALSE(
      implementation->supports(node, graph, execution_context).supported);

  graph.values[1] = ptn::Value("output", ptn::ScalarType::Float, {3});
  node.inputs[1].arg = ptn::IntListArg{{3}, {}};
  node.inputs[2].arg = ptn::IntListArg{{1}, {}};
  EXPECT_FALSE(
      implementation->supports(node, graph, execution_context).supported);
  node.inputs[3].arg = ptn::IntArg{1};
  EXPECT_FALSE(
      implementation->supports(node, graph, execution_context).supported);
}

TEST_F(ProviderTest, XNNPACKBMMRejectsIncompatibleShapesAndDtypes) {
  graph.values.emplace_back(
      "lhs", ptn::ScalarType::Float, std::vector<int64_t>{2, 2, 3});
  graph.values.emplace_back(
      "rhs", ptn::ScalarType::Float, std::vector<int64_t>{2, 3, 2});
  graph.values.emplace_back(
      "output", ptn::ScalarType::Float, std::vector<int64_t>{2, 2, 2});
  ptn::Node node;
  node.name = "bmm";
  node.target = "torch.ops.aten.bmm.default";
  node.inputs = {{"self", ptn::TensorArg{0}}, {"mat2", ptn::TensorArg{1}}};
  node.outputs = {{ptn::OutputValueKind::Tensor, 2, {}}};
  graph.nodes.push_back(std::move(node));
  auto provider = create_xnnpack_provider();
  auto* implementation = provider->implementations()[0];
  EXPECT_TRUE(
      implementation->supports(graph.node(0), graph, execution_context)
          .supported);
  graph.values[1] = ptn::Value("rhs", ptn::ScalarType::Float, {2, 4, 2});
  EXPECT_FALSE(
      implementation->supports(graph.node(0), graph, execution_context)
          .supported);
  graph.values[1] = ptn::Value("rhs", ptn::ScalarType::Float, {3, 3, 2});
  EXPECT_FALSE(
      implementation->supports(graph.node(0), graph, execution_context)
          .supported);
  graph.values[1] = ptn::Value("rhs", ptn::ScalarType::Float, {2, 3, 2});
  graph.values[2] = ptn::Value("output", ptn::ScalarType::Float, {2, 2, 3});
  EXPECT_FALSE(
      implementation->supports(graph.node(0), graph, execution_context)
          .supported);
  graph.values[2] = ptn::Value("output", ptn::ScalarType::Float, {2, 2, 2});
  graph.values[0] = ptn::Value("lhs", ptn::ScalarType::Float, {2, 6});
  EXPECT_FALSE(
      implementation->supports(graph.node(0), graph, execution_context)
          .supported);
  graph.values[0] = ptn::Value("lhs", ptn::ScalarType::Int, {2, 2, 3});
  EXPECT_FALSE(
      implementation->supports(graph.node(0), graph, execution_context)
          .supported);
}

TEST_F(ProviderTest, XNNPACKSoftmaxAcceptsOnlyFP32LastDimension) {
  graph.values.emplace_back(
      "input", ptn::ScalarType::Float, std::vector<int64_t>{2, 3, 4});
  graph.values.emplace_back(
      "output", ptn::ScalarType::Float, std::vector<int64_t>{2, 3, 4});
  ptn::Node node;
  node.name = "softmax";
  node.target = "torch.ops.aten._softmax.default";
  node.inputs = {
      {"self", ptn::TensorArg{0}},
      {"dim", ptn::IntArg{-1}},
      {"half_to_float", ptn::BoolArg{false}}};
  node.outputs = {{ptn::OutputValueKind::Tensor, 1}};
  graph.nodes.push_back(std::move(node));
  auto provider = create_xnnpack_provider();
  auto* implementation = provider->implementations()[0];
  auto supported = [&] {
    return implementation->supports(graph.node(0), graph, execution_context)
        .supported;
  };
  EXPECT_TRUE(supported());
  graph.node(0).inputs[1].arg = ptn::IntArg{2};
  EXPECT_TRUE(supported());
  graph.node(0).inputs[1].arg = ptn::IntArg{1};
  EXPECT_FALSE(supported());
  graph.node(0).inputs[1].arg = ptn::IntArg{-2};
  EXPECT_FALSE(supported());
  graph.node(0).inputs[1].arg = ptn::IntArg{-1};
  graph.node(0).inputs[2].arg = ptn::BoolArg{true};
  EXPECT_FALSE(supported());
  graph.node(0).inputs[2].arg = ptn::BoolArg{false};
  graph.values[1] = ptn::Value("output", ptn::ScalarType::Float, {2, 3, 5});
  EXPECT_FALSE(supported());
  graph.values[1] = ptn::Value("output", ptn::ScalarType::Int, {2, 3, 4});
  EXPECT_FALSE(supported());
  graph.values[1] = ptn::Value("output", ptn::ScalarType::Float, {2, 3, 4});
  graph.values[0] = ptn::Value("input", ptn::ScalarType::Int, {2, 3, 4});
  EXPECT_FALSE(supported());
  graph.values[0] = ptn::Value("input", ptn::ScalarType::Float, {});
  graph.values[1] = ptn::Value("output", ptn::ScalarType::Float, {});
  EXPECT_FALSE(supported());
}

TEST_F(ProviderTest, XNNPACKAndETRegionsShareBoundaryAndSkipBuffers) {
  for (size_t id = 0; id < 5; ++id) {
    graph.values.emplace_back(
        std::to_string(id), ptn::ScalarType::Float, std::vector<int64_t>{11});
    values.emplace_back(factory.zeros({11}));
    auto buffer = allocate_buffer(allocator, 44);
    ASSERT_TRUE(buffer.ok());
    buffers.push_back(buffer.get());
    values.back().toTensor().unsafeGetTensorImpl()->set_data(buffer->data);
  }
  graph.values[0].role = ptn::ValueRole::UserInput;
  graph.values[1].role = ptn::ValueRole::UserInput;
  ptn::Node add;
  add.name = "add";
  add.target = "torch.ops.aten.add.Tensor";
  add.inputs = {
      {"self", ptn::TensorArg{0}},
      {"other", ptn::TensorArg{1}},
      {"alpha", ptn::IntArg{1}}};
  add.outputs = {{ptn::OutputValueKind::Tensor, 2, {}}};
  graph.nodes.push_back(add);
  ptn::Node gelu;
  gelu.name = "gelu";
  gelu.target = "torch.ops.aten.gelu.default";
  gelu.inputs = {
      {"self", ptn::TensorArg{2}}, {"approximate", ptn::StringArg{"none"}}};
  gelu.outputs = {{ptn::OutputValueKind::Tensor, 3, {}}};
  graph.nodes.push_back(gelu);
  add.name = "skip_add";
  add.inputs[0].arg = ptn::TensorArg{2};
  add.inputs[1].arg = ptn::TensorArg{3};
  add.outputs[0].value_id = 4;
  graph.nodes.push_back(add);
  std::fill_n(static_cast<float*>(buffers[0].data), 11, 1);
  std::fill_n(static_cast<float*>(buffers[1].data), 11, 1);
  PreparationContext context{
      graph, buffers, allocator, private_bytes, execution_context};
  auto xnn = create_xnnpack_provider();
  auto et = create_et_provider();
  auto first = xnn->implementations()[0]->compile({{0}, {0, 1}, {2}}, context);
  auto middle = et->implementations()[0]->compile({{1}, {2}, {3}}, context);
  auto last = xnn->implementations()[0]->compile({{2}, {2, 3}, {4}}, context);
  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(middle.ok());
  ASSERT_TRUE(last.ok());
  ASSERT_EQ(xnn->finish(), Error::Ok);
  for (auto* executable :
       {first.get().get(), middle.get().get(), last.get().get()}) {
    ASSERT_EQ(executable->reshape(), Error::Ok);
    ASSERT_EQ(executable->bind(), Error::Ok);
  }
  ExecutionContext execution = execution_context;
  for (size_t iteration = 0; iteration < 2; ++iteration) {
    for (auto* executable :
         {first.get().get(), middle.get().get(), last.get().get()}) {
      ASSERT_EQ(executable->run(execution), Error::Ok);
    }
    EXPECT_NEAR(static_cast<float*>(buffers[4].data)[10], 3.95449974f, 1e-6);
    EXPECT_FLOAT_EQ(static_cast<float*>(buffers[2].data)[10], 2);
  }
}

TEST_F(ProviderTest, ETChainsReluIntoDimOrderCopy) {
  for (const auto* name : {"input", "mid", "output"}) {
    graph.values.emplace_back(
        name, ptn::ScalarType::Float, std::vector<int64_t>{4});
    values.emplace_back(factory.zeros({4}));
    auto buffer = allocate_buffer(allocator, 16);
    ASSERT_TRUE(buffer.ok());
    buffers.push_back(buffer.get());
    values.back().toTensor().unsafeGetTensorImpl()->set_data(buffer->data);
  }
  graph.values[0].role = ptn::ValueRole::UserInput;
  ptn::Node relu;
  relu.name = "relu";
  relu.target = "torch.ops.aten.relu.default";
  relu.inputs = {{"self", ptn::TensorArg{0}}};
  relu.outputs = {{ptn::OutputValueKind::Tensor, 1, {}}};
  graph.nodes.push_back(relu);
  ptn::Node clone;
  clone.name = "clone_dim_order";
  clone.target = "torch.ops.dim_order_ops._clone_dim_order.default";
  clone.inputs = {
      {"self", ptn::TensorArg{1}},
      {"non_blocking", ptn::BoolArg{false}},
      {"dim_order", ptn::NoneArg{}}};
  clone.outputs = {{ptn::OutputValueKind::Tensor, 2, {}}};
  graph.nodes.push_back(clone);
  auto source = static_cast<float*>(buffers[0].data);
  source[0] = -2;
  source[1] = -0.5f;
  source[2] = 1;
  source[3] = 3;
  PreparationContext context{
      graph, buffers, allocator, private_bytes, execution_context};
  auto provider = create_et_provider();
  auto* implementation = provider->implementations()[0];
  ASSERT_TRUE(implementation->supports(graph.node(0), graph, execution_context)
                  .supported);
  ASSERT_TRUE(implementation->supports(graph.node(1), graph, execution_context)
                  .supported);
  auto executable = implementation->compile({{0, 1}, {0}, {2}}, context);
  ASSERT_TRUE(executable.ok());
  ASSERT_EQ(provider->finish(), Error::Ok);
  ASSERT_EQ(executable.get()->reshape(), Error::Ok);
  ASSERT_EQ(executable.get()->bind(), Error::Ok);
  ExecutionContext execution = execution_context;
  ASSERT_EQ(executable.get()->run(execution), Error::Ok);
  auto* result = static_cast<float*>(buffers[2].data);
  EXPECT_FLOAT_EQ(result[0], 0);
  EXPECT_FLOAT_EQ(result[1], 0);
  EXPECT_FLOAT_EQ(result[2], 1);
  EXPECT_FLOAT_EQ(result[3], 3);
}

TEST_F(ProviderTest, ETFillsWithFactoryMapAndValidatesTargetStrings) {
  graph.values.emplace_back(
      "output", ptn::ScalarType::Float, std::vector<int64_t>{2, 2});
  values.emplace_back(factory.zeros({2, 2}));
  buffers.resize(1);
  GuardedBuffer output(16);
  bind_value(0, output.buffer);
  ptn::Node node;
  node.name = "full";
  node.target = "torch.ops.aten.full.default";
  node.inputs = {
      {"size", ptn::IntListArg{{2, 2}, {}}},
      {"fill_value", ptn::FloatArg{2.5}},
      {"dtype", ptn::ScalarTypeArg{ptn::ScalarType::Float}},
      {"layout", ptn::StringArg{"torch.strided"}},
      {"device", ptn::StringArg{"cpu"}},
      {"pin_memory", ptn::BoolArg{false}}};
  node.outputs = {{ptn::OutputValueKind::Tensor, 0, {}}};
  graph.nodes.push_back(node);
  PreparationContext context{
      graph, buffers, allocator, private_bytes, execution_context};
  auto provider = create_et_provider();
  auto* implementation = provider->implementations()[0];
  ASSERT_TRUE(implementation->supports(graph.node(0), graph, execution_context)
                  .supported);
  auto executable = implementation->compile({{0}, {}, {0}}, context);
  ASSERT_TRUE(executable.ok());
  ASSERT_EQ(provider->finish(), Error::Ok);
  ASSERT_EQ(executable.get()->reshape(), Error::Ok);
  ASSERT_EQ(executable.get()->bind(), Error::Ok);
  ExecutionContext execution = execution_context;
  ASSERT_EQ(executable.get()->run(execution), Error::Ok);
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(static_cast<float*>(output.buffer.data)[i], 2.5f);
  }
  node.inputs[4].arg = ptn::StringArg{"cuda:0"};
  EXPECT_FALSE(
      implementation->supports(node, graph, execution_context).supported);
  node.inputs[4].arg = ptn::StringArg{"cpu"};
  node.inputs[3].arg = ptn::StringArg{"torch.sparse_coo"};
  EXPECT_FALSE(
      implementation->supports(node, graph, execution_context).supported);
}

TEST_F(ProviderTest, ETExecutesFunctionalDimOrderSchemas) {
  for (const auto* name : {"empty", "filled", "copy"}) {
    graph.values.emplace_back(
        name, ptn::ScalarType::Float, std::vector<int64_t>{2, 2});
    auto buffer = allocate_buffer(allocator, 16);
    ASSERT_TRUE(buffer.ok());
    buffers.push_back(buffer.get());
  }
  ptn::Node empty;
  empty.name = "empty";
  empty.target = "torch.ops.dim_order_ops._empty_dim_order.default";
  empty.inputs = {
      {"size", ptn::IntListArg{{2, 2}, {}}},
      {"dtype", ptn::ScalarTypeArg{ptn::ScalarType::Float}},
      {"layout", ptn::StringArg{"torch.strided"}},
      {"device", ptn::StringArg{"cpu"}},
      {"pin_memory", ptn::BoolArg{false}},
      {"dim_order", ptn::IntListArg{{0, 1}, {}}}};
  empty.outputs = {{ptn::OutputValueKind::Tensor, 0, {}}};
  graph.nodes.push_back(std::move(empty));
  ptn::Node fill;
  fill.name = "fill";
  fill.target = "torch.ops.aten.fill.Scalar";
  fill.inputs = {{"self", ptn::TensorArg{0}}, {"value", ptn::FloatArg{2.5}}};
  fill.outputs = {{ptn::OutputValueKind::Tensor, 1, {}}};
  graph.nodes.push_back(std::move(fill));
  ptn::Node copy;
  copy.name = "copy";
  copy.target = "torch.ops.dim_order_ops._to_dim_order_copy.default";
  copy.inputs = {
      {"self", ptn::TensorArg{1}},
      {"dtype", ptn::ScalarTypeArg{ptn::ScalarType::Float}},
      {"layout", ptn::NoneArg{}},
      {"device", ptn::StringArg{"cpu"}},
      {"pin_memory", ptn::NoneArg{}},
      {"non_blocking", ptn::BoolArg{false}},
      {"dim_order", ptn::IntListArg{{0, 1}, {}}}};
  copy.outputs = {{ptn::OutputValueKind::Tensor, 2, {}}};
  graph.nodes.push_back(std::move(copy));
  PreparationContext context{
      graph, buffers, allocator, private_bytes, execution_context};
  auto provider = create_et_provider();
  auto* implementation = provider->implementations()[0];
  for (const auto& node : graph.nodes) {
    ASSERT_TRUE(
        implementation->supports(node, graph, execution_context).supported);
  }
  auto executable = implementation->compile({{0, 1, 2}, {}, {2}}, context);
  ASSERT_TRUE(executable.ok());
  ASSERT_EQ(provider->finish(), Error::Ok);
  ASSERT_EQ(executable.get()->reshape(), Error::Ok);
  ASSERT_EQ(executable.get()->bind(), Error::Ok);
  ASSERT_EQ(executable.get()->run(execution_context), Error::Ok);
  const auto* result = static_cast<const float*>(buffers[2].data);
  EXPECT_EQ(
      std::vector<float>(4, 2.5f), std::vector<float>(result, result + 4));
}

TEST_F(ProviderTest, ETUpsamplesWithScaleFactorsOrOutputSize) {
  struct Case {
    const char* target;
    std::vector<ptn::NamedArgument> args;
    std::vector<int64_t> output_shape;
    std::vector<float> expected;
  };
  const std::vector<float> nearest{
      0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3};
  const std::vector<Case> cases = {
      {"torch.ops.aten.upsample_nearest2d.vec",
       {{"output_size", ptn::NoneArg{}},
        {"scale_factors", ptn::FloatListArg{{2.0, 2.0}}}},
       {1, 1, 4, 4},
       nearest},
      {"torch.ops.aten.upsample_nearest2d.vec",
       {{"output_size", ptn::IntListArg{{4, 4}, {}}},
        {"scale_factors", ptn::NoneArg{}}},
       {1, 1, 4, 4},
       nearest},
      {"torch.ops.aten.upsample_bilinear2d.vec",
       {{"output_size", ptn::NoneArg{}},
        {"align_corners", ptn::BoolArg{true}},
        {"scale_factors", ptn::FloatListArg{{1.5, 1.5}}}},
       {1, 1, 3, 3},
       {0, 0.5f, 1, 1, 1.5f, 2, 2, 2.5f, 3}},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.target);
    graph = ptn::Graph();
    buffers.clear();
    graph.values.emplace_back(
        "input", ptn::ScalarType::Float, std::vector<int64_t>{1, 1, 2, 2});
    graph.values[0].role = ptn::ValueRole::UserInput;
    graph.values.emplace_back(
        "output", ptn::ScalarType::Float, test_case.output_shape);
    for (const auto& value : graph.values) {
      auto buffer = allocate_buffer(allocator, tensor_bytes(value).get());
      ASSERT_TRUE(buffer.ok());
      buffers.push_back(buffer.get());
    }
    ptn::Node node;
    node.name = "upsample";
    node.target = test_case.target;
    node.inputs = {{"input", ptn::TensorArg{0}}};
    node.inputs.insert(
        node.inputs.end(), test_case.args.begin(), test_case.args.end());
    node.outputs = {{ptn::OutputValueKind::Tensor, 1, {}}};
    graph.nodes.push_back(std::move(node));
    auto* input = static_cast<float*>(buffers[0].data);
    std::iota(input, input + 4, 0.0f);
    PreparationContext context{
        graph, buffers, allocator, private_bytes, execution_context};
    auto provider = create_et_provider();
    auto* implementation = provider->implementations()[0];
    ASSERT_TRUE(
        implementation->supports(graph.node(0), graph, execution_context)
            .supported);
    auto executable = implementation->compile({{0}, {0}, {1}}, context);
    ASSERT_TRUE(executable.ok());
    ASSERT_EQ(provider->finish(), Error::Ok);
    // Bound lists must not borrow from the graph's argument payloads.
    for (size_t index = 1; index < graph.node(0).inputs.size(); ++index) {
      graph.node(0).inputs[index].arg = ptn::NoneArg{};
    }
    ASSERT_EQ(executable.get()->reshape(), Error::Ok);
    ASSERT_EQ(executable.get()->bind(), Error::Ok);
    ExecutionContext execution = execution_context;
    for (const float offset : {0.0f, 10.0f}) {
      std::iota(input, input + 4, offset);
      ASSERT_EQ(executable.get()->run(execution), Error::Ok);
      std::vector<float> expected = test_case.expected;
      for (auto& element : expected) {
        element += offset;
      }
      const auto* result = static_cast<const float*>(buffers[1].data);
      EXPECT_EQ(std::vector<float>(result, result + expected.size()), expected);
    }
  }
}

TEST_F(ProviderTest, ETUpsamplesWithFloatingScaleFactors) {
  graph.values.emplace_back(
      "input", ptn::ScalarType::Float, std::vector<int64_t>{1, 1, 2, 2});
  graph.values.emplace_back(
      "output", ptn::ScalarType::Float, std::vector<int64_t>{1, 1, 4, 4});
  graph.values[0].role = ptn::ValueRole::UserInput;
  GuardedBuffer input(16), output(64);
  buffers = {input.buffer, output.buffer};
  const std::array<float, 4> source{1, 2, 3, 4};
  std::memcpy(input.buffer.data, source.data(), sizeof(source));
  graph.nodes.resize(1);
  auto& node = graph.nodes[0];
  node.name = "upsample";
  node.outputs = {{ptn::OutputValueKind::Tensor, 1, {}}};
  PreparationContext context{
      graph, buffers, allocator, private_bytes, execution_context};
  auto provider = create_et_provider();
  auto* implementation = provider->implementations()[0];
  for (bool bilinear : {false, true}) {
    node.target = bilinear ? "torch.ops.aten.upsample_bilinear2d.vec"
                           : "torch.ops.aten.upsample_nearest2d.vec";
    node.inputs = {
        {"input", ptn::TensorArg{0}}, {"output_size", ptn::NoneArg{}}};
    if (bilinear) {
      node.inputs.push_back({"align_corners", ptn::BoolArg{false}});
    }
    node.inputs.push_back({"scale_factors", ptn::FloatListArg{{2.0, 2.0}}});
    ASSERT_TRUE(
        implementation->supports(node, graph, execution_context).supported);
    auto executable = implementation->compile({{0}, {0}, {1}}, context);
    ASSERT_TRUE(executable.ok());
    ASSERT_EQ(executable.get()->reshape(), Error::Ok);
    ASSERT_EQ(executable.get()->bind(), Error::Ok);
    ASSERT_EQ(executable.get()->run(execution_context), Error::Ok);
    const std::vector<float> expected = bilinear
        ? std::vector<
              float>{1, 1.25f, 1.75f, 2, 1.5f, 1.75f, 2.25f, 2.5f, 2.5f, 2.75f, 3.25f, 3.5f, 3, 3.25f, 3.75f, 4}
        : std::vector<float>{1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4};
    const auto* result = static_cast<const float*>(output.buffer.data);
    EXPECT_EQ(expected, std::vector<float>(result, result + 16));
  }
}

TEST_F(ProviderTest, ETBoxesIntegerLiteralsForFloatParameters) {
  graph.values.emplace_back(
      "input", ptn::ScalarType::Float, std::vector<int64_t>{2, 2});
  graph.values.emplace_back(
      "output", ptn::ScalarType::Float, std::vector<int64_t>{1});
  graph.values[0].role = ptn::ValueRole::UserInput;
  GuardedBuffer input(16), output(4);
  buffers = {input.buffer, output.buffer};
  const std::array<float, 4> source{0, 0, 3, 4};
  std::memcpy(input.buffer.data, source.data(), sizeof(source));
  graph.nodes.resize(1);
  auto& node = graph.nodes[0];
  node.name = "distance";
  node.target = "torch.ops.aten._pdist_forward.default";
  node.outputs = {{ptn::OutputValueKind::Tensor, 1, {}}};
  PreparationContext context{
      graph, buffers, allocator, private_bytes, execution_context};
  auto provider = create_et_provider();
  auto* implementation = provider->implementations()[0];
  const std::array<ptn::Argument, 2> exponents{
      ptn::IntArg{2}, ptn::FloatArg{2.0}};
  for (const auto& exponent : exponents) {
    SCOPED_TRACE(static_cast<int>(exponent.kind()));
    node.inputs = {{"self", ptn::TensorArg{0}}, {"p", exponent}};
    ASSERT_TRUE(
        implementation->supports(node, graph, execution_context).supported);
    auto executable = implementation->compile({{0}, {0}, {1}}, context);
    ASSERT_TRUE(executable.ok());
    ASSERT_EQ(executable.get()->reshape(), Error::Ok);
    ASSERT_EQ(executable.get()->bind(), Error::Ok);
    ASSERT_EQ(executable.get()->run(execution_context), Error::Ok);
    EXPECT_FLOAT_EQ(5.0f, *static_cast<const float*>(output.buffer.data));
  }
}

TEST_F(ProviderTest, ETBoxesIntegerLiteralsForOptionalFloatParameters) {
  gelu_graph(3);
  auto& node = graph.nodes[0];
  node.target = "torch.ops.aten.logit.default";
  GuardedBuffer input(12), output(12);
  buffers = {input.buffer, output.buffer};
  const std::array<float, 3> source{0.25f, 0.5f, 0.75f};
  const std::array<float, 3> expected{-std::log(3.0f), 0, std::log(3.0f)};
  std::memcpy(input.buffer.data, source.data(), sizeof(source));
  PreparationContext context{
      graph, buffers, allocator, private_bytes, execution_context};
  auto provider = create_et_provider();
  auto* implementation = provider->implementations()[0];
  const std::array<ptn::Argument, 3> epsilons{
      ptn::IntArg{0}, ptn::FloatArg{0.0}, ptn::NoneArg{}};
  for (const auto& epsilon : epsilons) {
    SCOPED_TRACE(static_cast<int>(epsilon.kind()));
    node.inputs = {{"self", ptn::TensorArg{0}}, {"eps", epsilon}};
    ASSERT_TRUE(
        implementation->supports(node, graph, execution_context).supported);
    auto executable = implementation->compile({{0}, {0}, {1}}, context);
    ASSERT_TRUE(executable.ok());
    ASSERT_EQ(executable.get()->reshape(), Error::Ok);
    ASSERT_EQ(executable.get()->bind(), Error::Ok);
    ASSERT_EQ(executable.get()->run(execution_context), Error::Ok);
    const auto* result = static_cast<const float*>(output.buffer.data);
    for (size_t index = 0; index < expected.size(); ++index) {
      EXPECT_NEAR(expected[index], result[index], 1e-6f);
    }
  }
}

TEST_F(ProviderTest, ETRejectsUnbindableArgumentsBeforeCompilation) {
  gelu_graph(4);
  auto& node = graph.nodes[0];
  node.target = "torch.ops.aten.add.Tensor";
  node.inputs = {
      {"self", ptn::TensorArg{0}},
      {"other", ptn::TensorArg{0}},
      {"alpha", ptn::IntArg{1}}};
  PreparationContext context{
      graph, buffers, allocator, private_bytes, execution_context};
  auto provider = create_et_provider();
  auto* implementation = provider->implementations()[0];
  ASSERT_TRUE(
      implementation->supports(node, graph, execution_context).supported);
  const std::vector<ptn::Argument> unsupported{
      ptn::BoolListArg{{true, false}},
      ptn::GraphArg{"subgraph", 0},
      ptn::IntArg{1, 0},
      ptn::FloatArg{1.0, 0},
      ptn::BoolArg{true, 0},
      ptn::IntListArg{{2, 2}, {0, ptn::kInvalid}}};
  for (const auto& arg : unsupported) {
    node.inputs[2].arg = arg;
    EXPECT_FALSE(
        implementation->supports(node, graph, execution_context).supported);
    EXPECT_EQ(
        implementation->compile({{0}, {0}, {1}}, context).error(),
        Error::NotSupported);
  }
}

TEST_F(ProviderTest, ETRejectsIncompleteTensorListsBeforeCompilation) {
  gelu_graph(4);
  graph.values[1] = ptn::Value("first", ptn::ScalarType::Float, {2});
  graph.values.emplace_back(
      "second", ptn::ScalarType::Float, std::vector<int64_t>{2});
  auto& node = graph.nodes[0];
  node.target = "torch.ops.aten.split_with_sizes_copy.default";
  node.inputs = {
      {"self", ptn::TensorArg{0}},
      {"split_sizes", ptn::IntListArg{{2, 2}, {}}},
      {"dim", ptn::IntArg{0}}};
  node.outputs = {{ptn::OutputValueKind::TensorList, ptn::kInvalid, {1, 2}}};
  PreparationContext context{
      graph, buffers, allocator, private_bytes, execution_context};
  auto provider = create_et_provider();
  auto* implementation = provider->implementations()[0];
  ASSERT_TRUE(
      implementation->supports(node, graph, execution_context).supported);
  const std::vector<ptn::Output> unsupported{
      {ptn::OutputValueKind::TensorList, ptn::kInvalid, {}},
      {ptn::OutputValueKind::TensorList, ptn::kInvalid, {1, ptn::kInvalid}},
      {ptn::OutputValueKind::TensorList, ptn::kInvalid, {1, 3}},
      {ptn::OutputValueKind::Tensor, 1, {}}};
  for (const auto& output_value : unsupported) {
    node.outputs[0] = output_value;
    EXPECT_FALSE(
        implementation->supports(node, graph, execution_context).supported);
    EXPECT_EQ(
        implementation->compile({{0}, {0}, {1, 2}}, context).error(),
        Error::NotSupported);
  }
}

TEST_F(ProviderTest, ETConcatenatesTensorList) {
  graph.values.emplace_back(
      "a", ptn::ScalarType::Float, std::vector<int64_t>{2});
  graph.values.emplace_back(
      "b", ptn::ScalarType::Float, std::vector<int64_t>{3});
  graph.values.emplace_back(
      "out", ptn::ScalarType::Float, std::vector<int64_t>{5});
  graph.values[0].role = ptn::ValueRole::UserInput;
  graph.values[1].role = ptn::ValueRole::UserInput;
  for (const auto& value : graph.values) {
    const auto& shape = value.tensor_meta().sizes;
    values.emplace_back(
        factory.zeros(std::vector<int32_t>(shape.begin(), shape.end())));
    auto buffer = allocate_buffer(allocator, tensor_bytes(value).get());
    ASSERT_TRUE(buffer.ok());
    buffers.push_back(buffer.get());
    values.back().toTensor().unsafeGetTensorImpl()->set_data(buffer->data);
  }
  ptn::Node node;
  node.name = "cat";
  node.target = "torch.ops.aten.cat.default";
  node.inputs = {
      {"tensors", ptn::TensorListArg{{0, 1}}}, {"dim", ptn::IntArg{0}}};
  node.outputs = {{ptn::OutputValueKind::Tensor, 2, {}}};
  graph.nodes.push_back(node);
  std::fill_n(static_cast<float*>(buffers[0].data), 2, 1);
  std::fill_n(static_cast<float*>(buffers[1].data), 3, 2);
  PreparationContext context{
      graph, buffers, allocator, private_bytes, execution_context};
  auto provider = create_et_provider();
  auto executable =
      provider->implementations()[0]->compile({{0}, {0, 1}, {2}}, context);
  ASSERT_TRUE(executable.ok());
  ASSERT_EQ(provider->finish(), Error::Ok);
  ASSERT_EQ(executable.get()->reshape(), Error::Ok);
  ASSERT_EQ(executable.get()->bind(), Error::Ok);
  ExecutionContext execution = execution_context;
  ASSERT_EQ(executable.get()->run(execution), Error::Ok);
  auto* result = static_cast<float*>(buffers[2].data);
  EXPECT_EQ(
      (std::vector<float>{1, 1, 2, 2, 2}),
      std::vector<float>(result, result + 5));
}

TEST_F(ProviderTest, ETSparseTensorIdsAndRepeatedReferencesRebind) {
  for (size_t id = 0; id < 1024; ++id) {
    graph.values.emplace_back(
        std::to_string(id), ptn::ScalarType::Float, std::vector<int64_t>{2});
  }
  const ptn::ValueId input_id = 1023;
  const ptn::ValueId output_id = 7;
  graph.values[input_id].role = ptn::ValueRole::UserInput;
  graph.values[output_id] = ptn::Value("output", ptn::ScalarType::Float, {4});
  buffers.resize(graph.values.size());
  GuardedBuffer first(8), second(8), output(16);
  const std::array<float, 2> first_values{1.25f, -2.0f};
  const std::array<float, 2> second_values{3.0f, 0.5f};
  std::memcpy(first.buffer.data, first_values.data(), 8);
  std::memcpy(second.buffer.data, second_values.data(), 8);
  first.buffer.writable_bytes = 0;
  second.buffer.writable_bytes = 0;
  buffers[input_id] = first.buffer;
  buffers[output_id] = output.buffer;
  ptn::Node node;
  node.name = "cat";
  node.target = "torch.ops.aten.cat.default";
  node.inputs = {
      {"tensors", ptn::TensorListArg{{input_id, input_id}}},
      {"dim", ptn::IntArg{0}}};
  node.outputs = {{ptn::OutputValueKind::Tensor, output_id, {}}};
  graph.nodes.push_back(std::move(node));
  PreparationContext context{
      graph, buffers, allocator, private_bytes, execution_context};
  auto provider = create_et_provider();
  auto executable = provider->implementations()[0]->compile(
      {{0}, {input_id}, {output_id}}, context);
  ASSERT_TRUE(executable.ok());
  ASSERT_EQ(provider->finish(), Error::Ok);
  ASSERT_EQ(executable.get()->reshape(), Error::Ok);
  ASSERT_EQ(executable.get()->bind(), Error::Ok);
  ASSERT_EQ(executable.get()->run(execution_context), Error::Ok);
  const auto* result = static_cast<const float*>(output.buffer.data);
  EXPECT_EQ(
      (std::vector<float>{1.25f, -2.0f, 1.25f, -2.0f}),
      std::vector<float>(result, result + 4));

  buffers[input_id] = second.buffer;
  buffers[output_id].writable_bytes = 0;
  EXPECT_EQ(executable.get()->bind(), Error::InvalidArgument);
  EXPECT_EQ(executable.get()->run(execution_context), Error::InvalidState);
  buffers[output_id] = output.buffer;
  ASSERT_EQ(executable.get()->bind(), Error::Ok);
  ASSERT_EQ(executable.get()->run(execution_context), Error::Ok);
  EXPECT_EQ(
      (std::vector<float>{3.0f, 0.5f, 3.0f, 0.5f}),
      std::vector<float>(result, result + 4));
}

TEST_F(ProviderTest, ETSplitsIntoTensorListOutput) {
  graph.values.emplace_back(
      "input", ptn::ScalarType::Float, std::vector<int64_t>{5});
  graph.values.emplace_back(
      "first", ptn::ScalarType::Float, std::vector<int64_t>{2});
  graph.values.emplace_back(
      "second", ptn::ScalarType::Float, std::vector<int64_t>{3});
  graph.values[0].role = ptn::ValueRole::UserInput;
  for (const auto& value : graph.values) {
    const auto& shape = value.tensor_meta().sizes;
    values.emplace_back(
        factory.zeros(std::vector<int32_t>(shape.begin(), shape.end())));
    auto buffer = allocate_buffer(allocator, tensor_bytes(value).get());
    ASSERT_TRUE(buffer.ok());
    buffers.push_back(buffer.get());
    values.back().toTensor().unsafeGetTensorImpl()->set_data(buffer->data);
  }
  ptn::Node node;
  node.name = "split";
  node.target = "torch.ops.aten.split_with_sizes_copy.default";
  node.inputs = {
      {"self", ptn::TensorArg{0}},
      {"split_sizes", ptn::IntListArg{{2, 3}, {}}},
      {"dim", ptn::IntArg{0}}};
  node.outputs = {{ptn::OutputValueKind::TensorList, ptn::kInvalid, {1, 2}}};
  graph.nodes.push_back(node);
  for (size_t i = 0; i < 5; ++i) {
    static_cast<float*>(buffers[0].data)[i] = static_cast<float>(i);
  }
  PreparationContext context{
      graph, buffers, allocator, private_bytes, execution_context};
  auto provider = create_et_provider();
  auto* implementation = provider->implementations()[0];
  ASSERT_TRUE(
      implementation->supports(node, graph, execution_context).supported);
  auto executable = implementation->compile({{0}, {0}, {1, 2}}, context);
  ASSERT_TRUE(executable.ok());
  ASSERT_EQ(provider->finish(), Error::Ok);
  ASSERT_EQ(executable.get()->reshape(), Error::Ok);
  ASSERT_EQ(executable.get()->bind(), Error::Ok);
  ExecutionContext execution = execution_context;
  ASSERT_EQ(executable.get()->run(execution), Error::Ok);
  EXPECT_FLOAT_EQ(static_cast<float*>(buffers[1].data)[0], 0);
  EXPECT_FLOAT_EQ(static_cast<float*>(buffers[1].data)[1], 1);
  EXPECT_FLOAT_EQ(static_cast<float*>(buffers[2].data)[0], 2);
  EXPECT_FLOAT_EQ(static_cast<float*>(buffers[2].data)[2], 4);
}

TEST_F(ProviderTest, ETClonesWithMemoryFormat) {
  gelu_graph(4);
  auto& node = graph.node(0);
  node.target = "torch.ops.aten.clone.default";
  node.inputs = {
      {"self", ptn::TensorArg{0}},
      {"memory_format", ptn::StringArg{"torch.contiguous_format"}}};
  GuardedBuffer input(16), output(16);
  bind_value(0, input.buffer);
  bind_value(1, output.buffer);
  for (size_t i = 0; i < 4; ++i) {
    static_cast<float*>(input.buffer.data)[i] = static_cast<float>(i) + 0.5f;
  }
  PreparationContext context{
      graph, buffers, allocator, private_bytes, execution_context};
  auto provider = create_et_provider();
  auto* implementation = provider->implementations()[0];
  ASSERT_TRUE(
      implementation->supports(node, graph, execution_context).supported);
  auto executable = implementation->compile({{0}, {0}, {1}}, context);
  ASSERT_TRUE(executable.ok());
  ASSERT_EQ(provider->finish(), Error::Ok);
  ASSERT_EQ(executable.get()->reshape(), Error::Ok);
  ASSERT_EQ(executable.get()->bind(), Error::Ok);
  ExecutionContext execution = execution_context;
  ASSERT_EQ(executable.get()->run(execution), Error::Ok);
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(
        static_cast<float*>(output.buffer.data)[i],
        static_cast<float>(i) + 0.5f);
  }
  node.inputs[1].arg = ptn::StringArg{"torch.channels_last"};
  EXPECT_FALSE(
      implementation->supports(node, graph, execution_context).supported);
}
// cppcheck-suppress-end syntaxError
} // namespace
} // namespace executorch::backends::cpu
