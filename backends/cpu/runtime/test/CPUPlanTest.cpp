// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/cpu/runtime/CPUPlan.h>
#include <executorch/backends/cpu/runtime/providers/reference/ReferenceProvider.h>
#include <executorch/backends/cpu/runtime/providers/xnnpack/XNNPACKProvider.h>
#include <executorch/devtools/etdump/etdump_flatcc.h>
#include <executorch/devtools/etdump/etdump_schema_flatcc_reader.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace executorch::backends::cpu {
namespace {
using namespace executorch::runtime;

struct TraceRun {
  std::vector<std::string> delegates;
  std::vector<std::string> metadata;
  std::vector<std::string> kernels;
};

std::vector<TraceRun> read_trace(etdump::ETDumpGen& tracer) {
  const auto result = tracer.get_etdump_data();
  std::unique_ptr<void, decltype(&free)> data(result.buf, &free);
  std::vector<TraceRun> runs;
  if (!data) {
    return runs;
  }
  size_t size = 0;
  const auto run_data =
      etdump_ETDump_run_data(etdump_ETDump_as_root_with_identifier(
          flatbuffers_read_size_prefix(result.buf, &size),
          etdump_ETDump_file_identifier));
  for (size_t run = 0; run < etdump_RunData_vec_len(run_data); ++run) {
    auto& trace = runs.emplace_back();
    const auto events =
        etdump_RunData_events(etdump_RunData_vec_at(run_data, run));
    for (size_t index = 0; index < etdump_Event_vec_len(events); ++index) {
      const auto event =
          etdump_Event_profile_event(etdump_Event_vec_at(events, index));
      EXPECT_LE(
          etdump_ProfileEvent_start_time(event),
          etdump_ProfileEvent_end_time(event));
      if (const auto name = etdump_ProfileEvent_delegate_debug_id_str(event)) {
        trace.delegates.emplace_back(name);
        const auto bytes = etdump_ProfileEvent_delegate_debug_metadata(event);
        trace.metadata.emplace_back(
            reinterpret_cast<const char*>(bytes),
            flatbuffers_uint8_vec_len(bytes));
      } else {
        trace.kernels.emplace_back(etdump_ProfileEvent_name(event));
      }
    }
  }
  return runs;
}

class CPUPlanTest : public ::testing::Test {
 protected:
  void value(const char* name, bool input = false) {
    graph.values.emplace_back(
        name, ptn::ScalarType::Float, std::vector<int64_t>{11});
    if (input) {
      graph.input_ids.push_back(
          static_cast<ptn::ValueId>(graph.values.size() - 1));
      graph.values.back().role = ptn::ValueRole::UserInput;
    }
  }
  void add(const char* name, ptn::ValueId a, ptn::ValueId b, ptn::ValueId out) {
    ptn::Node node;
    node.name = name;
    node.target = "torch.ops.aten.add.Tensor";
    node.inputs = {
        {"self", ptn::TensorArg{a}},
        {"other", ptn::TensorArg{b}},
        {"alpha", ptn::IntArg{1}}};
    node.outputs = {{ptn::OutputValueKind::Tensor, out, {}}};
    graph.nodes.push_back(std::move(node));
  }
  void gelu(const char* name, ptn::ValueId in, ptn::ValueId out) {
    ptn::Node node;
    node.name = name;
    node.target = "torch.ops.aten.gelu.default";
    node.inputs = {
        {"self", ptn::TensorArg{in}}, {"approximate", ptn::StringArg{"tanh"}}};
    node.outputs = {{ptn::OutputValueKind::Tensor, out, {}}};
    graph.nodes.push_back(std::move(node));
  }
  void sum_gelu_out() {
    value("x", true);
    value("y", true);
    value("sum");
    value("gelu");
    value("out");
    add("sum", 0, 1, 2);
    gelu("gelu", 2, 3);
    add("out", 2, 3, 4);
    finish({4});
  }
  void finish(std::vector<ptn::ValueId> outputs) {
    graph.output_ids = std::move(outputs);
    graph.initialize_schedule();
    graph.rebuild_def_use();
    buffers.resize(graph.values.size());
  }
  static float gelu(float x) {
    return x * 0.5f *
        (1 + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
  }
  template <size_t Capacity>
  BackendInitContext context_for(BackendOptions<Capacity>& options) {
    const auto values = options.view();
    return BackendInitContext(
        &allocator, nullptr, nullptr, nullptr, {values.data(), values.size()});
  }
  alignas(128) std::array<uint8_t, 65536> storage{};
  MemoryAllocator allocator{
      static_cast<uint32_t>(storage.size()),
      storage.data()};
  ptn::Graph graph;
  std::vector<Buffer> buffers;
  ExecutionContext execution;
  const std::array<ProviderFactory, 2> factories{
      create_xnnpack_provider,
      create_reference_provider};
  RuntimeConfiguration configuration{
      {factories.data(), factories.size()},
      {},
      false};
};

// Cppcheck cannot expand GTest fixture macros without build headers.
// cppcheck-suppress-begin syntaxError
class XNNLayoutTest : public CPUPlanTest {
 protected:
  ptn::ValueId tensor(
      std::vector<int64_t> shape,
      ptn::ValueRole role = ptn::ValueRole::Intermediate) {
    const auto id = static_cast<ptn::ValueId>(graph.values.size());
    graph.values.emplace_back(
        std::to_string(id), ptn::ScalarType::Float, std::move(shape));
    graph.values.back().role = role;
    if (role == ptn::ValueRole::UserInput) {
      graph.input_ids.push_back(id);
    }
    return id;
  }

  ptn::ValueId operation(
      const char* target,
      std::vector<ptn::Argument> arguments,
      std::vector<int64_t> shape) {
    const auto output = tensor(std::move(shape));
    ptn::Node node;
    node.name = "layout_" + std::to_string(graph.nodes.size());
    node.target = target;
    for (auto& argument : arguments) {
      node.inputs.push_back({"", std::move(argument)});
    }
    node.outputs = {{ptn::OutputValueKind::Tensor, output, {}}};
    graph.nodes.push_back(std::move(node));
    return output;
  }

  ptn::ValueId convolution(
      ptn::ValueId input,
      ptn::ValueId weight,
      ptn::ValueId bias,
      int64_t groups = 1) {
    auto shape = graph.value(input).tensor_meta().sizes;
    const auto filter = graph.value(weight).tensor_meta().sizes;
    shape[1] = filter[0];
    return operation(
        "torch.ops.aten.convolution.default",
        {ptn::TensorArg{input},
         ptn::TensorArg{weight},
         bias == ptn::kInvalid ? ptn::Argument(ptn::NoneArg{})
                               : ptn::Argument(ptn::TensorArg{bias}),
         ptn::IntListArg{{1, 1}, {}},
         ptn::IntListArg{{filter[2] / 2, filter[3] / 2}, {}},
         ptn::IntListArg{{1, 1}, {}},
         ptn::BoolArg{false},
         ptn::IntListArg{{0, 0}, {}},
         ptn::IntArg{groups}},
        shape);
  }

  ptn::ValueId activation(const char* target, ptn::ValueId input) {
    return operation(
        target,
        {ptn::TensorArg{input}},
        graph.value(input).tensor_meta().sizes);
  }

  ptn::ValueId binary(
      const char* target,
      ptn::ValueId lhs,
      ptn::ValueId rhs,
      const std::vector<int64_t>& shape) {
    std::vector<ptn::Argument> arguments{
        ptn::TensorArg{lhs}, ptn::TensorArg{rhs}};
    if (std::string_view(target) == "torch.ops.aten.add.Tensor") {
      arguments.emplace_back(ptn::IntArg{1});
    }
    return operation(target, std::move(arguments), shape);
  }

  void check(
      std::vector<ptn::ValueId> outputs,
      const std::vector<std::string>& providers = {"XNNPACK"}) {
    finish(std::move(outputs));
    alignas(128) std::array<uint8_t, 256 * 1024> memory{};
    MemoryAllocator test_allocator(memory.size(), memory.data());
    std::vector<Buffer> reference_buffers(graph.values.size());
    for (size_t id = 0; id < graph.values.size(); ++id) {
      const auto role = graph.values[id].role;
      if (role != ptn::ValueRole::Parameter &&
          role != ptn::ValueRole::ConstantTensor &&
          role != ptn::ValueRole::Buffer) {
        continue;
      }
      const auto bytes = tensor_bytes(graph.values[id]).get();
      auto constant = allocate_buffer(test_allocator, bytes);
      auto reference_constant = allocate_buffer(test_allocator, bytes);
      ASSERT_TRUE(constant.ok());
      ASSERT_TRUE(reference_constant.ok());
      buffers[id] = constant.get();
      reference_buffers[id] = reference_constant.get();
      auto* data = static_cast<float*>(constant->data);
      for (size_t index = 0; index < graph.values[id].tensor_meta().numel();
           ++index) {
        data[index] = (float((index * 7 + id) % 17) - 8) / 32;
      }
      std::memcpy(reference_constant->data, constant->data, bytes);
    }
    CPUPlan plan(graph, buffers, configuration, execution);
    auto reference_configuration = configuration;
    reference_configuration.preferences = {{"Reference", ""}};
    reference_configuration.force = true;
    CPUPlan reference(
        graph, reference_buffers, reference_configuration, ExecutionContext{});
    ASSERT_EQ(reference.select(), Error::Ok);
    ASSERT_EQ(plan.select(), Error::Ok);
    ASSERT_EQ(plan.steps().size(), providers.size());
    for (size_t index = 0; index < providers.size(); ++index) {
      EXPECT_EQ(plan.steps()[index].provider->name(), providers[index]);
    }
    for (auto id : preparation_only_constants) {
      EXPECT_FALSE(plan.can_release_constant(id));
    }
    ASSERT_EQ(plan.prepare(test_allocator), Error::Ok);
    ASSERT_EQ(reference.prepare(test_allocator), Error::Ok);
    for (auto id : runtime_constants) {
      EXPECT_FALSE(plan.can_release_constant(id)) << id;
    }
    for (auto id : preparation_only_constants) {
      ASSERT_TRUE(plan.can_release_constant(id)) << id;
      std::memset(buffers[id].data, 0xFF, buffers[id].readable_bytes);
      buffers[id] = {};
    }
    auto binding_ids = graph.input_ids;
    for (auto id : graph.output_ids) {
      const auto role = graph.value(id).role;
      if (role != ptn::ValueRole::Parameter &&
          role != ptn::ValueRole::ConstantTensor &&
          role != ptn::ValueRole::Buffer) {
        binding_ids.push_back(id);
      }
    }
    std::vector<Buffer> originals;
    std::vector<Buffer> replacements;
    for (auto id : binding_ids) {
      originals.push_back(buffers[id]);
      auto replacement =
          allocate_buffer(test_allocator, tensor_bytes(graph.value(id)).get());
      ASSERT_TRUE(replacement.ok());
      replacements.push_back(replacement.get());
    }
    for (int run = 0; run < 3; ++run) {
      for (size_t binding = 0; binding < binding_ids.size(); ++binding) {
        buffers[binding_ids[binding]] =
            run == 1 ? replacements[binding] : originals[binding];
      }
      for (size_t input = 0; input < graph.input_ids.size(); ++input) {
        const auto id = graph.input_ids[input];
        const auto count = graph.value(id).tensor_meta().numel();
        for (size_t index = 0; index < count; ++index) {
          const float value =
              (float((index * 13 + input) % 31) - 15) / 8 + float(run) / 4;
          static_cast<float*>(buffers[id].data)[index] = value;
          static_cast<float*>(reference_buffers[id].data)[index] = value;
        }
      }
      ASSERT_EQ(plan.execute(execution), Error::Ok);
      ASSERT_EQ(reference.execute(ExecutionContext{}), Error::Ok);
      for (auto id : graph.output_ids) {
        SCOPED_TRACE(id);
        EXPECT_TRUE(graph.value(id).tensor_meta().is_contiguous());
        const auto* actual = static_cast<const float*>(buffers[id].data);
        const auto* expected =
            static_cast<const float*>(reference_buffers[id].data);
        for (size_t index = 0; index < graph.value(id).tensor_meta().numel();
             ++index) {
          EXPECT_NEAR(
              actual[index],
              expected[index],
              2e-5f * std::max(1.0f, std::abs(expected[index])))
              << index;
        }
      }
    }
    for (auto id : runtime_constants) {
      const bool required = std::any_of(
          plan.steps().begin(), plan.steps().end(), [id](const auto& step) {
            return std::any_of(
                step.storage.values.begin(),
                step.storage.values.end(),
                [id](const auto& value) {
                  return value.id == id && !value.preparation_only;
                });
          });
      if (required) {
        const auto original = buffers[id];
        buffers[id] = {};
        EXPECT_EQ(plan.execute(execution), Error::InvalidArgument) << id;
        buffers[id] = original;
      }
    }
  }

  std::vector<ptn::ValueId> preparation_only_constants;
  std::vector<ptn::ValueId> runtime_constants;
};

TEST_F(XNNLayoutTest, ConvolutionPackedConstantsSurvivePoisoningAndRebinding) {
  struct Case {
    int64_t channels, output_channels, groups, kernel;
  };
  const std::array<Case, 7> cases{
      {{3, 3, 1, 3},
       {3, 3, 1, 1},
       {3, 3, 3, 3},
       {3, 3, 3, 1},
       {1, 1, 1, 3},
       {4, 4, 2, 3},
       {3, 6, 3, 3}}};
  for (const auto& [channels, output_channels, groups, kernel] : cases) {
    for (const bool with_bias : {false, true}) {
      SCOPED_TRACE(::testing::PrintToString(std::make_tuple(
          channels, output_channels, groups, kernel, with_bias)));
      graph = {};
      buffers.clear();
      const auto input = tensor({2, channels, 4, 5}, ptn::ValueRole::UserInput);
      const auto weight = tensor(
          {output_channels, channels / groups, kernel, kernel},
          ptn::ValueRole::Parameter);
      const auto bias = with_bias
          ? tensor({output_channels}, ptn::ValueRole::Buffer)
          : ptn::kInvalid;
      preparation_only_constants = {weight};
      if (with_bias) {
        preparation_only_constants.push_back(bias);
      }
      check({convolution(input, weight, bias, groups)});
    }
  }
}

TEST_F(XNNLayoutTest, ConvolutionSharesPackedWeightsAndBiasAcrossRegions) {
  for (const bool depthwise : {false, true}) {
    for (const int64_t kernel : {1, 3}) {
      SCOPED_TRACE(::testing::PrintToString(std::make_pair(depthwise, kernel)));
      graph = {};
      buffers.clear();
      const auto input = tensor({2, 3, 4, 5}, ptn::ValueRole::UserInput);
      const auto weight = tensor(
          {3, depthwise ? 1 : 3, kernel, kernel},
          ptn::ValueRole::ConstantTensor);
      const auto bias = tensor({3}, ptn::ValueRole::Parameter);
      const auto first = convolution(input, weight, bias, depthwise ? 3 : 1);
      const auto middle = operation(
          "torch.ops.aten.gelu.default",
          {ptn::TensorArg{first}, ptn::StringArg{"tanh"}},
          {2, 3, 4, 5});
      preparation_only_constants = {weight, bias};
      check(
          {first, convolution(middle, weight, bias, depthwise ? 3 : 1)},
          {"XNNPACK", "Reference", "XNNPACK"});
    }
  }
}

TEST_F(XNNLayoutTest, ConvolutionAndLinearSharePackedBias) {
  const auto input = tensor({2, 3, 4, 5}, ptn::ValueRole::UserInput);
  const auto weight = tensor({3, 3, 3, 3}, ptn::ValueRole::Parameter);
  const auto bias = tensor({3}, ptn::ValueRole::ConstantTensor);
  const auto first = convolution(input, weight, bias);
  const auto linear_weight = tensor({3, 5}, ptn::ValueRole::Parameter);
  const auto output = operation(
      "torch.ops.aten.linear.default",
      {ptn::TensorArg{first},
       ptn::TensorArg{linear_weight},
       ptn::TensorArg{bias}},
      {2, 3, 4, 3});
  preparation_only_constants = {weight, bias, linear_weight};
  check({first, output});
}

TEST_F(XNNLayoutTest, ConvolutionRetainsMixedPackedAndUnpackedConsumers) {
  for (const std::string use :
       {"unary",
        "nhwc_binary",
        "linear_input",
        "conv_input",
        "bias_binary",
        "bias_linear_input",
        "reference",
        "output"}) {
    SCOPED_TRACE(use);
    graph = {};
    buffers.clear();
    const std::vector<int64_t> shape{3, 3, 3, 3};
    const auto input = tensor(shape, ptn::ValueRole::UserInput);
    const auto weight = tensor(shape, ptn::ValueRole::Parameter);
    const auto bias = tensor({3}, ptn::ValueRole::ConstantTensor);
    const auto first = convolution(input, weight, bias);
    std::vector<ptn::ValueId> outputs{first};
    std::vector<std::string> providers{"XNNPACK"};
    preparation_only_constants = {bias};
    runtime_constants = {weight};
    if (use == "unary") {
      outputs.push_back(activation("torch.ops.aten.relu.default", weight));
      providers.emplace_back("XNNPACK");
    } else if (use == "nhwc_binary") {
      outputs.push_back(
          binary("torch.ops.aten.add.Tensor", first, weight, shape));
    } else if (use == "linear_input") {
      const auto linear_weight = tensor({3, 3}, ptn::ValueRole::Parameter);
      preparation_only_constants.push_back(linear_weight);
      outputs.push_back(operation(
          "torch.ops.aten.linear.default",
          {ptn::TensorArg{weight},
           ptn::TensorArg{linear_weight},
           ptn::TensorArg{bias}},
          shape));
      providers.emplace_back("XNNPACK");
    } else if (use == "conv_input") {
      outputs.push_back(convolution(weight, weight, bias));
      providers.emplace_back("XNNPACK");
    } else if (use == "bias_binary") {
      outputs.push_back(
          binary("torch.ops.aten.add.Tensor", first, bias, shape));
      preparation_only_constants = {weight};
      runtime_constants = {bias};
    } else if (use == "bias_linear_input") {
      const auto linear_weight = tensor({3, 3}, ptn::ValueRole::Parameter);
      outputs.push_back(operation(
          "torch.ops.aten.linear.default",
          {ptn::TensorArg{bias},
           ptn::TensorArg{linear_weight},
           ptn::TensorArg{bias}},
          {3}));
      providers.emplace_back("XNNPACK");
      preparation_only_constants = {weight, linear_weight};
      runtime_constants = {bias};
    } else if (use == "reference") {
      for (auto id : {weight, bias}) {
        outputs.push_back(operation(
            "torch.ops.aten.gelu.default",
            {ptn::TensorArg{id}, ptn::StringArg{"tanh"}},
            graph.value(id).tensor_meta().sizes));
        providers.emplace_back("Reference");
      }
      preparation_only_constants.clear();
      runtime_constants = {weight, bias};
    } else {
      outputs.push_back(weight);
      outputs.push_back(bias);
      preparation_only_constants.clear();
      runtime_constants = {weight, bias};
    }
    check(outputs, providers);
  }
}

TEST_F(XNNLayoutTest, DepthwiseRetainsWeightUsedByPrivateBroadcastTranspose) {
  const auto input = tensor({3, 3, 3, 3}, ptn::ValueRole::UserInput);
  const auto weight = tensor({3, 1, 3, 3}, ptn::ValueRole::Parameter);
  const auto bias = tensor({3}, ptn::ValueRole::ConstantTensor);
  const auto first = convolution(input, weight, bias, 3);
  const auto output =
      binary("torch.ops.aten.mul.Tensor", first, weight, {3, 3, 3, 3});
  const auto last = convolution(output, weight, bias, 3);
  preparation_only_constants = {bias};
  runtime_constants = {weight};
  check({first, last});
}

TEST_F(XNNLayoutTest, DepthwiseRetainsSharedBiasAndConstantOutputs) {
  for (const bool expose_constants : {false, true}) {
    SCOPED_TRACE(expose_constants);
    graph = {};
    buffers.clear();
    const std::vector<int64_t> shape{2, 3, 4, 3};
    const auto input = tensor(shape, ptn::ValueRole::UserInput);
    const auto weight = tensor({3, 1, 3, 3}, ptn::ValueRole::Parameter);
    const auto bias = tensor({3}, ptn::ValueRole::ConstantTensor);
    const auto first = convolution(input, weight, bias, 3);
    if (expose_constants) {
      preparation_only_constants.clear();
      runtime_constants = {weight, bias};
      check({first, weight, bias});
    } else {
      preparation_only_constants = {weight};
      runtime_constants = {bias};
      check({binary("torch.ops.aten.add.Tensor", first, bias, shape)});
    }
  }
}

TEST_F(XNNLayoutTest, BmmReleasesOnlyAConstantRightHandSide) {
  const std::vector<int64_t> shape{2, 3, 3};
  for (const std::string use : {"rhs", "lhs", "both"}) {
    SCOPED_TRACE(use);
    graph = {};
    buffers.clear();
    const auto input = tensor(shape, ptn::ValueRole::UserInput);
    const auto constant = tensor(shape, ptn::ValueRole::Parameter);
    const auto lhs = use == "rhs" ? input : constant;
    const auto rhs = use == "lhs" ? input : constant;
    auto output = operation(
        "torch.ops.aten.bmm.default",
        {ptn::TensorArg{lhs}, ptn::TensorArg{rhs}},
        shape);
    if (use == "both") {
      output = binary("torch.ops.aten.add.Tensor", output, input, shape);
    }
    preparation_only_constants.clear();
    runtime_constants.clear();
    if (use == "rhs") {
      preparation_only_constants.push_back(constant);
    } else {
      runtime_constants.push_back(constant);
    }
    check({output});
  }
}

TEST_F(CPUPlanTest, MaxPool2dPreservesReferenceFallbackAndSkipsUnusedIndices) {
  graph.values.emplace_back(
      "input", ptn::ScalarType::Float, std::vector<int64_t>{1, 1, 4, 4});
  graph.values[0].role = ptn::ValueRole::UserInput;
  graph.values.emplace_back(
      "output", ptn::ScalarType::Float, std::vector<int64_t>{1, 1, 2, 2});
  graph.values.emplace_back(
      "indices", ptn::ScalarType::Long, std::vector<int64_t>{1, 1, 2, 2});
  ptn::Node node;
  node.name = "pool";
  node.target = "torch.ops.aten.max_pool2d_with_indices.default";
  node.inputs = {
      {"self", ptn::TensorArg{0}},
      {"kernel_size", ptn::IntListArg{{2, 2}, {}}},
      {"stride", ptn::IntListArg{{2, 2}, {}}},
      {"padding", ptn::IntListArg{{0, 0}, {}}},
      {"dilation", ptn::IntListArg{{1, 1}, {}}},
      {"ceil_mode", ptn::BoolArg{false}}};
  node.outputs = {
      {ptn::OutputValueKind::Tensor, 1}, {ptn::OutputValueKind::Tensor, 2}};
  graph.nodes.push_back(std::move(node));
  graph.input_ids = {0};
  finish({1});
  for (const bool reference_only : {false, true}) {
    if (reference_only) {
      configuration.providers = {factories.data() + 1, 1};
    }
    buffers.assign(graph.values.size(), Buffer{});
    CPUPlan plan(graph, buffers, configuration, execution);
    ASSERT_EQ(plan.select(), Error::Ok);
    ASSERT_EQ(plan.steps().size(), 1);
    EXPECT_EQ(
        plan.steps()[0].provider->name(),
        reference_only ? "Reference" : "XNNPACK");
    ASSERT_EQ(plan.prepare(allocator), Error::Ok);
    EXPECT_EQ(buffers[2].data != nullptr, reference_only);
    for (int i = 0; i < 16; ++i) {
      static_cast<float*>(buffers[0].data)[i] = i;
    }
    ASSERT_EQ(plan.execute(execution), Error::Ok);
    const std::array<float, 4> expected{5, 7, 13, 15};
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_FLOAT_EQ(static_cast<float*>(buffers[1].data)[i], expected[i]);
    }
  }
}

TEST_F(
    CPUPlanTest,
    MixedRegionsPreserveFanoutOutputsAndReuseArenaAcrossInvocations) {
  value("x", true);
  value("y", true);
  for (const auto* name : {"sum", "gelu", "skip", "second_gelu", "out"}) {
    value(name);
  }
  add("sum", 0, 1, 2);
  gelu("gelu", 2, 3);
  add("skip", 2, 3, 4);
  gelu("second_gelu", 4, 5);
  add("out", 3, 5, 6);
  finish({2, 6});
  CPUPlan plan(graph, buffers, configuration, execution);
  ASSERT_EQ(plan.select(), Error::Ok);
  ASSERT_EQ(plan.steps().size(), 5);
  EXPECT_EQ(plan.steps()[0].provider->name(), "XNNPACK");
  EXPECT_EQ(plan.steps()[1].provider->name(), "Reference");
  EXPECT_EQ(plan.steps()[2].region.inputs, (std::vector<ptn::ValueId>{2, 3}));
  ASSERT_EQ(plan.prepare(allocator), Error::Ok);
  EXPECT_LT(plan.arena_bytes(), 7 * 128);
  EXPECT_NE(buffers[0].data, buffers[2].data);
  EXPECT_NE(buffers[1].data, buffers[2].data);
  EXPECT_NE(buffers[3].data, buffers[4].data);
  EXPECT_NE(buffers[4].data, buffers[5].data);
  EXPECT_NE(buffers[2].data, buffers[6].data);
  for (float input : {1.0f, -0.25f, 2.0f}) {
    std::fill_n(static_cast<float*>(buffers[0].data), 11, input);
    std::fill_n(static_cast<float*>(buffers[1].data), 11, input);
    ASSERT_EQ(plan.execute(execution), Error::Ok);
    EXPECT_FLOAT_EQ(static_cast<float*>(buffers[2].data)[10], 2 * input);
    EXPECT_NEAR(
        static_cast<float*>(buffers[6].data)[10],
        gelu(2 * input) + gelu(2 * input + gelu(2 * input)),
        1e-5);
  }
  const float before = static_cast<float*>(buffers[6].data)[0];
  buffers[5].readable_bytes = 0;
  EXPECT_EQ(plan.execute(execution), Error::InvalidArgument);
  EXPECT_FLOAT_EQ(static_cast<float*>(buffers[6].data)[0], before);
}

TEST_F(CPUPlanTest, ConnectedRegionsExposeIntermediateModelOutputs) {
  value("x", true);
  value("y", true);
  value("sum");
  value("out");
  add("sum", 0, 1, 2);
  add("out", 2, 1, 3);
  finish({2, 3});
  CPUPlan plan(graph, buffers, configuration, execution);
  ASSERT_EQ(plan.select(), Error::Ok);
  ASSERT_EQ(plan.steps().size(), 1);
  EXPECT_EQ(plan.steps()[0].region.outputs, (std::vector<ptn::ValueId>{2, 3}));
  ASSERT_EQ(plan.prepare(allocator), Error::Ok);
  std::fill_n(static_cast<float*>(buffers[0].data), 11, 1);
  std::fill_n(static_cast<float*>(buffers[1].data), 11, 2);
  ASSERT_EQ(plan.execute(execution), Error::Ok);
  EXPECT_FLOAT_EQ(static_cast<float*>(buffers[2].data)[0], 3);
  EXPECT_FLOAT_EQ(static_cast<float*>(buffers[3].data)[0], 5);
}

TEST_F(CPUPlanTest, ActivationsStayInXNNPACKRegionAndPreserveFanout) {
  value("x", true);
  value("y", true);
  for (const auto* name : {"sum", "relu", "sigmoid", "out"}) {
    value(name);
  }
  add("sum", 0, 1, 2);
  for (const auto* target :
       {"torch.ops.aten.relu.default", "torch.ops.aten.sigmoid.default"}) {
    const ptn::ValueId input = graph.nodes.size() + 1;
    ptn::Node node;
    node.name = target;
    node.target = target;
    node.inputs = {{"self", ptn::TensorArg{input}}};
    node.outputs = {{ptn::OutputValueKind::Tensor, input + 1}};
    graph.nodes.push_back(std::move(node));
  }
  add("out", 2, 4, 5);
  finish({2, 3, 5});
  CPUPlan plan(graph, buffers, configuration, execution);
  ASSERT_EQ(plan.select(), Error::Ok);
  ASSERT_EQ(plan.steps().size(), 1);
  EXPECT_EQ(plan.steps()[0].provider->name(), "XNNPACK");
  EXPECT_EQ(plan.steps()[0].region.nodes, (std::vector<KernelId>{0, 1, 2, 3}));
  ASSERT_EQ(plan.prepare(allocator), Error::Ok);
  for (float shift : {-1.0f, 1.0f}) {
    for (size_t index = 0; index < 11; ++index) {
      static_cast<float*>(buffers[0].data)[index] = float(index) - 5;
      static_cast<float*>(buffers[1].data)[index] = shift;
    }
    ASSERT_EQ(plan.execute(execution), Error::Ok);
    for (size_t index = 0; index < 11; ++index) {
      const float sum = float(index) - 5 + shift;
      const float relu = std::max(0.0f, sum);
      EXPECT_FLOAT_EQ(static_cast<float*>(buffers[2].data)[index], sum);
      EXPECT_FLOAT_EQ(static_cast<float*>(buffers[3].data)[index], relu);
      EXPECT_NEAR(
          static_cast<float*>(buffers[5].data)[index],
          sum + 1.0f / (1.0f + std::exp(-relu)),
          1e-6);
    }
  }
}

TEST_F(CPUPlanTest, BatchedMatrixMultiplyUsesXNNPACKForDynamicOperands) {
  graph.values.emplace_back(
      "lhs", ptn::ScalarType::Float, std::vector<int64_t>{2, 2, 3});
  graph.values.emplace_back(
      "rhs", ptn::ScalarType::Float, std::vector<int64_t>{2, 3, 2});
  graph.values.emplace_back(
      "output", ptn::ScalarType::Float, std::vector<int64_t>{2, 2, 2});
  graph.values[0].role = ptn::ValueRole::UserInput;
  graph.values[1].role = ptn::ValueRole::UserInput;
  graph.input_ids = {0, 1};
  ptn::Node bmm;
  bmm.name = "bmm";
  bmm.target = "torch.ops.aten.bmm.default";
  bmm.inputs = {{"self", ptn::TensorArg{0}}, {"mat2", ptn::TensorArg{1}}};
  bmm.outputs = {{ptn::OutputValueKind::Tensor, 2, {}}};
  graph.nodes.push_back(std::move(bmm));
  finish({2});

  CPUPlan plan(graph, buffers, configuration, execution);
  ASSERT_EQ(plan.select(), Error::Ok);
  ASSERT_EQ(plan.steps().size(), 1);
  EXPECT_EQ(plan.steps()[0].provider->name(), "XNNPACK");
  ASSERT_EQ(plan.prepare(allocator), Error::Ok);
  const std::array<float, 12> lhs{1, 2, 3, 4, 5, 6, 1, 0, 2, 0, 1, 3};
  const std::array<float, 12> rhs{1, 2, 3, 4, 5, 6, 2, 0, 0, 3, 4, 5};
  std::copy(lhs.begin(), lhs.end(), static_cast<float*>(buffers[0].data));
  std::copy(rhs.begin(), rhs.end(), static_cast<float*>(buffers[1].data));
  ASSERT_EQ(plan.execute(execution), Error::Ok);
  const auto* result = static_cast<const float*>(buffers[2].data);
  EXPECT_EQ(
      (std::vector<float>(result, result + 8)),
      (std::vector<float>{22, 28, 49, 64, 10, 10, 12, 18}));
}

TEST_F(CPUPlanTest, LastDimensionSoftmaxUsesXNNPACKForDynamicInput) {
  graph.values.emplace_back(
      "input", ptn::ScalarType::Float, std::vector<int64_t>{2, 2, 3});
  graph.values.back().role = ptn::ValueRole::UserInput;
  graph.values.emplace_back(
      "output", ptn::ScalarType::Float, std::vector<int64_t>{2, 2, 3});
  graph.input_ids = {0};
  ptn::Node softmax;
  softmax.name = "softmax";
  softmax.target = "torch.ops.aten._softmax.default";
  softmax.inputs = {
      {"self", ptn::TensorArg{0}},
      {"dim", ptn::IntArg{-1}},
      {"half_to_float", ptn::BoolArg{false}}};
  softmax.outputs = {{ptn::OutputValueKind::Tensor, 1}};
  graph.nodes.push_back(std::move(softmax));
  finish({1});

  CPUPlan plan(graph, buffers, configuration, execution);
  ASSERT_EQ(plan.select(), Error::Ok);
  ASSERT_EQ(plan.steps().size(), 1);
  EXPECT_EQ(plan.steps()[0].provider->name(), "XNNPACK");
  ASSERT_EQ(plan.prepare(allocator), Error::Ok);
  const std::array<float, 12> input{
      0, 0, 0, 0, 1, 2, 1000, 1000, 1000, -1000, -999, -998};
  std::copy(input.begin(), input.end(), static_cast<float*>(buffers[0].data));
  ASSERT_EQ(plan.execute(execution), Error::Ok);
  const float denominator = 1 + std::exp(1.0f) + std::exp(2.0f);
  const std::array<float, 12> expected{
      1.0f / 3,
      1.0f / 3,
      1.0f / 3,
      1 / denominator,
      std::exp(1.0f) / denominator,
      std::exp(2.0f) / denominator,
      1.0f / 3,
      1.0f / 3,
      1.0f / 3,
      1 / denominator,
      std::exp(1.0f) / denominator,
      std::exp(2.0f) / denominator};
  const auto* result = static_cast<const float*>(buffers[1].data);
  for (size_t index = 0; index < expected.size(); ++index) {
    EXPECT_NEAR(result[index], expected[index], 1e-5) << index;
  }
}

TEST_F(CPUPlanTest, RetainsConstantsWithRuntimeConsumersOrGraphOutputs) {
  for (const std::string use : {"add", "gelu", "output"}) {
    SCOPED_TRACE(use);
    graph = {};
    buffers.clear();
    for (const auto* name : {"input", "weight", "bias", "linear", "out"}) {
      graph.values.emplace_back(
          name, ptn::ScalarType::Float, std::vector<int64_t>{1, 3});
    }
    graph.values[0].role = ptn::ValueRole::UserInput;
    graph.values[1] = ptn::Value("weight", ptn::ScalarType::Float, {3, 3});
    graph.values[1].role = ptn::ValueRole::Parameter;
    graph.values[2] = ptn::Value("bias", ptn::ScalarType::Float, {3});
    graph.values[2].role = ptn::ValueRole::ConstantTensor;
    graph.input_ids = {0};
    ptn::Node linear;
    linear.name = "linear";
    linear.target = "torch.ops.aten.linear.default";
    linear.inputs = {
        {"input", ptn::TensorArg{0}},
        {"weight", ptn::TensorArg{1}},
        {"bias", ptn::TensorArg{2}}};
    linear.outputs = {{ptn::OutputValueKind::Tensor, 3}};
    graph.nodes.push_back(std::move(linear));
    if (use == "add") {
      add("out", 3, 2, 4);
      finish({4});
    } else if (use == "gelu") {
      graph.values[4] = ptn::Value("out", ptn::ScalarType::Float, {3});
      gelu("out", 2, 4);
      finish({3, 4});
    } else {
      finish({3, 1, 2});
    }
    for (auto id : {1, 2}) {
      auto constant =
          allocate_buffer(allocator, tensor_bytes(graph.value(id)).get());
      ASSERT_TRUE(constant.ok());
      buffers[id] = constant.get();
      std::fill_n(
          static_cast<float*>(buffers[id].data),
          id == 1 ? 9 : 3,
          id == 1 ? 2.0f : 3.0f);
    }
    CPUPlan plan(graph, buffers, configuration, execution);
    ASSERT_EQ(plan.select(), Error::Ok);
    ASSERT_EQ(plan.prepare(allocator), Error::Ok);
    EXPECT_EQ(plan.steps().size(), use == "gelu" ? 2 : 1);
    EXPECT_FALSE(plan.can_release_constant(2));
    EXPECT_EQ(plan.can_release_constant(1), use != "output");
    if (plan.can_release_constant(1)) {
      std::memset(buffers[1].data, 0, buffers[1].readable_bytes);
      buffers[1] = {};
    }
    for (float input : {1.0f, -1.0f}) {
      std::fill_n(static_cast<float*>(buffers[0].data), 3, input);
      ASSERT_EQ(plan.execute(execution), Error::Ok);
      const auto* result =
          static_cast<const float*>(buffers[use == "add" ? 4 : 3].data);
      for (size_t index = 0; index < 3; ++index) {
        EXPECT_FLOAT_EQ(result[index], input * 6 + (use == "add" ? 6 : 3));
        if (use == "gelu") {
          EXPECT_NEAR(
              static_cast<const float*>(buffers[4].data)[index], gelu(3), 1e-5);
        }
      }
    }
    buffers[2].readable_bytes = 0;
    if (use != "output") {
      EXPECT_EQ(plan.execute(execution), Error::InvalidArgument);
    }
  }
}

TEST_F(CPUPlanTest, TracesEveryStepAndXNNPACKOperatorOnEachExecution) {
#ifndef ET_EVENT_TRACER_ENABLED
  GTEST_SKIP() << "Requires -c executorch.event_tracer_enabled=true";
#endif
  sum_gelu_out();
  etdump::ETDumpGen tracer;
  execution.event_tracer = &tracer;
  CPUPlan plan(graph, buffers, configuration, execution);
  ASSERT_EQ(plan.select(), Error::Ok);
  ASSERT_EQ(plan.prepare(allocator), Error::Ok);
  for (int run = 0; run < 2; ++run) {
    tracer.create_event_block("forward");
    ASSERT_EQ(plan.execute(execution), Error::Ok);
  }

  const auto runs = read_trace(tracer);
  ASSERT_EQ(runs.size(), 2);
  const auto& names = runs[0].delegates;
  EXPECT_EQ(
      runs[0].kernels, (std::vector<std::string>{"native_call_gelu.out"}));
  ASSERT_EQ(names.size(), 5);
  EXPECT_EQ(names[0], "XNNPACK/subgraph:sum");
  EXPECT_EQ(names[2], "Reference/registry:gelu");
  EXPECT_EQ(names[3], "XNNPACK/subgraph:out");
  const auto suffix = names[1].rfind(" #1");
  ASSERT_EQ(suffix, names[1].size() - 3) << names[1];
  EXPECT_EQ(names[4], names[1].substr(0, suffix) + " #2");
#ifdef ENABLE_CPU_XNNPACK_OPERATOR_SHAPES
  const std::string add_shapes = R"({"inputs":[[11],[11]],"outputs":[[11]]})";
#else
  const std::string add_shapes;
#endif
  EXPECT_EQ(runs[0].metadata[1], add_shapes);
  EXPECT_EQ(runs[0].metadata[4], add_shapes);
  EXPECT_EQ(runs[1].delegates, names);
  EXPECT_EQ(runs[1].metadata, runs[0].metadata);
}

TEST_F(CPUPlanTest, MethodOnlyProfilingTracesStepsWithoutXNNPACKOperators) {
#ifndef ET_EVENT_TRACER_ENABLED
  GTEST_SKIP() << "Requires -c executorch.event_tracer_enabled=true";
#endif
  sum_gelu_out();
  etdump::ETDumpGen tracer;
  tracer.set_event_tracer_profiling_level(
      EventTracerProfilingLevel::kProfileMethodOnly);
  execution.event_tracer = &tracer;
  CPUPlan plan(graph, buffers, configuration, execution);
  ASSERT_EQ(plan.select(), Error::Ok);
  ASSERT_EQ(plan.prepare(allocator), Error::Ok);
  tracer.create_event_block("forward");
  ASSERT_EQ(plan.execute(execution), Error::Ok);

  const auto runs = read_trace(tracer);
  ASSERT_EQ(runs.size(), 1);
  EXPECT_EQ(
      runs[0].delegates,
      (std::vector<std::string>{
          "XNNPACK/subgraph:sum",
          "Reference/registry:gelu",
          "XNNPACK/subgraph:out"}));
  EXPECT_TRUE(runs[0].kernels.empty());
}

TEST_F(CPUPlanTest, XNNPACKRunSubmitsTraceLeftOpenByPreviousRun) {
#ifndef ET_EVENT_TRACER_ENABLED
  GTEST_SKIP() << "Requires -c executorch.event_tracer_enabled=true";
#endif
  value("x", true);
  value("y", true);
  value("sum");
  add("sum", 0, 1, 2);
  finish({2});
  etdump::ETDumpGen tracer;
  execution.event_tracer = &tracer;
  CPUPlan plan(graph, buffers, configuration, execution);
  ASSERT_EQ(plan.select(), Error::Ok);
  ASSERT_EQ(plan.prepare(allocator), Error::Ok);
  tracer.create_event_block("forward");
  ASSERT_EQ(plan.execute(execution), Error::Ok);
  auto& executable = *plan.steps()[0].executable;
  ASSERT_EQ(executable.run(execution), Error::Ok);
  ASSERT_EQ(executable.run(execution), Error::Ok);
  executable.submit_trace();

  const auto runs = read_trace(tracer);
  ASSERT_EQ(runs.size(), 1);
  const auto& names = runs[0].delegates;
  ASSERT_EQ(names.size(), 4);
  EXPECT_EQ(names[0], "XNNPACK/subgraph:sum");
  EXPECT_EQ(names[2], names[1]);
  EXPECT_EQ(names[3], names[1]);
}

TEST_F(CPUPlanTest, StepTraceNameCountsTheRegionsRemainingNodes) {
#ifndef ET_EVENT_TRACER_ENABLED
  GTEST_SKIP() << "Requires -c executorch.event_tracer_enabled=true";
#endif
  value("x", true);
  value("y", true);
  value("sum");
  value("out");
  add("sum", 0, 1, 2);
  add("out", 2, 1, 3);
  finish({3});
  CPUPlan plan(graph, buffers, configuration, execution);
  ASSERT_EQ(plan.select(), Error::Ok);
  ASSERT_EQ(plan.steps().size(), 1);
  EXPECT_EQ(plan.steps()[0].trace_name, "XNNPACK/subgraph:sum+1");
}

TEST_F(XNNLayoutTest, TracesEachXNNPACKOperatorWithItsOwnShapes) {
#ifndef ET_EVENT_TRACER_ENABLED
  GTEST_SKIP() << "Requires -c executorch.event_tracer_enabled=true";
#endif
  const auto x = tensor({2, 3}, ptn::ValueRole::UserInput);
  const auto y = tensor({1, 3}, ptn::ValueRole::UserInput);
  const auto sum = binary("torch.ops.aten.add.Tensor", x, y, {2, 3});
  // XNNPACK merges the reshapes into one operator, so events follow XNNPACK's
  // operators rather than the region's nodes.
  const auto flat = operation(
      "torch.ops.aten.view_copy.default",
      {ptn::TensorArg{sum}, ptn::IntListArg{{6}, {}}},
      {6});
  const auto tall = operation(
      "torch.ops.aten.view_copy.default",
      {ptn::TensorArg{flat}, ptn::IntListArg{{3, 2}, {}}},
      {3, 2});
  const auto out = operation(
      "torch.ops.aten.permute_copy.default",
      {ptn::TensorArg{tall}, ptn::IntListArg{{1, 0}, {}}},
      {2, 3});
  finish({out});
  etdump::ETDumpGen tracer;
  execution.event_tracer = &tracer;
  CPUPlan plan(graph, buffers, configuration, execution);
  ASSERT_EQ(plan.select(), Error::Ok);
  ASSERT_EQ(plan.steps().size(), 1);
  ASSERT_EQ(plan.prepare(allocator), Error::Ok);
  tracer.create_event_block("forward");
  ASSERT_EQ(plan.execute(execution), Error::Ok);

  const auto runs = read_trace(tracer);
  ASSERT_EQ(runs.size(), 1);
  const auto& names = runs[0].delegates;
  ASSERT_EQ(names.size(), 4);
  EXPECT_EQ(names[0], "XNNPACK/subgraph:layout_0+3");
  EXPECT_EQ(names[1].rfind("Add", 0), 0) << names[1];
  EXPECT_EQ(names[2].rfind("Static Reshape", 0), 0) << names[2];
  EXPECT_EQ(names[3].rfind("Transpose", 0), 0) << names[3];
#ifdef ENABLE_CPU_XNNPACK_OPERATOR_SHAPES
  const std::vector<std::string> metadata{
      "",
      R"({"inputs":[[2,3],[1,3]],"outputs":[[2,3]]})",
      R"({"inputs":[[2,3]],"outputs":[[3,2]]})",
      R"({"inputs":[[3,2]],"outputs":[[2,3]]})"};
#else
  const std::vector<std::string> metadata(4);
#endif
  EXPECT_EQ(runs[0].metadata, metadata);
}

TEST_F(
    CPUPlanTest,
    DisconnectedBranchesRemainSeparateAndRejectNonTopologicalSchedule) {
  value("x", true);
  value("y", true);
  value("a");
  value("b");
  value("out");
  add("a", 0, 1, 2);
  add("b", 0, 1, 3);
  add("out", 2, 3, 4);
  finish({4});
  CPUPlan plan(graph, buffers, configuration, execution);
  ASSERT_EQ(plan.select(), Error::Ok);
  ASSERT_EQ(plan.steps().size(), 2);
  EXPECT_EQ(plan.steps()[0].region.nodes, (std::vector<KernelId>{0}));
  EXPECT_EQ(plan.steps()[1].region.nodes, (std::vector<KernelId>{1, 2}));
  graph.schedule = {2, 0, 1};
  CPUPlan invalid(graph, buffers, configuration, execution);
  EXPECT_EQ(invalid.select(), Error::InvalidProgram);
}

TEST_F(CPUPlanTest, PreferenceForceAndExecutionSignatureAreChecked) {
  value("x", true);
  value("y", true);
  value("out");
  add("out", 0, 1, 2);
  finish({2});
  configuration.preferences = {{"Reference", ""}};
  configuration.force = true;
  CPUPlan plan(graph, buffers, configuration, execution);
  ASSERT_EQ(plan.select(), Error::Ok);
  EXPECT_EQ(plan.steps()[0].provider->name(), "Reference");
  ASSERT_EQ(plan.prepare(allocator), Error::Ok);
  auto changed = execution;
  changed.threads = 2;
  EXPECT_EQ(plan.execute(changed), Error::NotSupported);
  configuration.preferences = {{"unlinked", ""}};
  CPUPlan missing(graph, buffers, configuration, execution);
  EXPECT_EQ(missing.select(), Error::InvalidArgument);
  configuration.preferences = {{"XNNPACK", ""}};
  graph.nodes[0].inputs[2].arg = ptn::IntArg{2};
  CPUPlan unmatched(graph, buffers, configuration, execution);
  EXPECT_EQ(unmatched.select(), Error::NotSupported);
}

class SelectionImplementation final : public KernelImplementation {
 public:
  SelectionImplementation(const char* name, const char* target, int priority)
      : name_(name), target_(target), priority_(priority) {}
  std::string_view name() const override {
    return name_;
  }
  int priority() const override {
    return priority_;
  }
  Support supports(
      const Kernel& kernel,
      const ptn::Graph&,
      const ExecutionContext&) const override {
    return {
        target_.empty() || kernel.target == target_, "test selection route"};
  }
  Result<std::unique_ptr<Executable>> compile(
      const KernelRegion&,
      PreparationContext&) override {
    return Error::NotSupported;
  }

 private:
  std::string_view name_;
  std::string_view target_;
  int priority_;
};

class SelectionProvider final : public KernelProvider {
 public:
  std::string_view name() const override {
    return "selection_test";
  }
  std::vector<KernelImplementation*> implementations() override {
    return {&general_, &add_};
  }

 private:
  SelectionImplementation general_{"general", "", 1000};
  SelectionImplementation add_{"add", "torch.ops.aten.add.Tensor", -1000};
};

std::unique_ptr<KernelProvider> create_selection_provider() {
  return std::make_unique<SelectionProvider>();
}

class TiedProvider final : public KernelProvider {
 public:
  std::string_view name() const override {
    return "tied_test";
  }
  std::vector<KernelImplementation*> implementations() override {
    return {&first_, &second_};
  }

 private:
  SelectionImplementation first_{"first", "", 5};
  SelectionImplementation second_{"second", "", 5};
};

std::unique_ptr<KernelProvider> create_tied_provider() {
  return std::make_unique<TiedProvider>();
}

TEST_F(CPUPlanTest, EqualEligibleCandidatesRequireAPreference) {
  value("x", true);
  value("out");
  gelu("out", 0, 1);
  finish({1});
  const std::array<ProviderFactory, 1> providers{create_tied_provider};
  configuration.providers = {providers.data(), providers.size()};
  CPUPlan tied(graph, buffers, configuration, execution);
  EXPECT_EQ(tied.select(), Error::InvalidArgument);
  configuration.preferences = {{"tied_test", "second"}};
  CPUPlan preferred(graph, buffers, configuration, execution);
  ASSERT_EQ(preferred.select(), Error::Ok);
  EXPECT_EQ(preferred.steps()[0].implementation->name(), "second");
}

TEST_F(CPUPlanTest, SpecificImplementationsPrecedeAProviderWideFallback) {
  value("x", true);
  value("y", true);
  for (const auto* name : {"sum", "exact", "approximate"}) {
    value(name);
  }
  add("sum", 0, 1, 2);
  gelu("exact", 2, 3);
  graph.nodes.back().inputs[1].arg = ptn::StringArg{"none"};
  gelu("approximate", 3, 4);
  finish({4});
  const std::array<ProviderFactory, 3> providers{
      create_selection_provider,
      create_xnnpack_provider,
      create_reference_provider};
  configuration.providers = {providers.data(), providers.size()};
  configuration.preferences = {
      {"selection_test", "add"}, {"XNNPACK", "subgraph"}, {"Reference", ""}};
  configuration.force = true;
  CPUPlan plan(graph, buffers, configuration, execution);
  ASSERT_EQ(plan.select(), Error::Ok);
  ASSERT_EQ(plan.steps().size(), 3);
  EXPECT_EQ(plan.steps()[0].implementation->name(), "add");
  EXPECT_EQ(plan.steps()[1].provider->name(), "XNNPACK");
  EXPECT_EQ(plan.steps()[2].provider->name(), "Reference");
}

TEST_F(CPUPlanTest, UnbindableReferenceNodesFallBackDuringSelection) {
  value("x", true);
  value("out");
  add("out", 0, 0, 1);
  finish({1});
  const std::array<ProviderFactory, 2> providers{
      create_reference_provider, create_selection_provider};
  configuration.providers = {providers.data(), providers.size()};
  configuration.preferences = {{"Reference", ""}};
  CPUPlan supported(graph, buffers, configuration, execution);
  ASSERT_EQ(supported.select(), Error::Ok);
  ASSERT_EQ(supported.steps().size(), 1);
  EXPECT_EQ(supported.steps()[0].provider->name(), "Reference");

  auto& node = graph.nodes[0];
  node.inputs[2].arg = ptn::FloatArg{1.0, 0};
  CPUPlan dynamic_argument(graph, buffers, configuration, execution);
  ASSERT_EQ(dynamic_argument.select(), Error::Ok);
  ASSERT_EQ(dynamic_argument.steps().size(), 1);
  EXPECT_EQ(dynamic_argument.steps()[0].provider->name(), "selection_test");

  graph.values[1] = ptn::Value("out", ptn::ScalarType::Float, {5});
  node.target = "torch.ops.aten.split_with_sizes_copy.default";
  node.inputs = {
      {"self", ptn::TensorArg{0}},
      {"split_sizes", ptn::IntListArg{{5, 6}, {}}},
      {"dim", ptn::IntArg{0}}};
  node.outputs = {
      {ptn::OutputValueKind::TensorList, ptn::kInvalid, {1, ptn::kInvalid}}};
  graph.rebuild_def_use();
  CPUPlan sparse_outputs(graph, buffers, configuration, execution);
  ASSERT_EQ(sparse_outputs.select(), Error::Ok);
  ASSERT_EQ(sparse_outputs.steps().size(), 1);
  EXPECT_EQ(sparse_outputs.steps()[0].provider->name(), "selection_test");
}

TEST_F(CPUPlanTest, FirstMatchingPreferenceWinsBeforeNumericPriority) {
  value("x", true);
  value("y", true);
  value("out");
  add("out", 0, 1, 2);
  finish({2});
  const std::array<ProviderFactory, 2> providers{
      create_selection_provider, create_reference_provider};
  configuration.providers = {providers.data(), providers.size()};
  configuration.preferences = {{"Reference", ""}, {"selection_test", "add"}};
  CPUPlan reference_first(graph, buffers, configuration, execution);
  ASSERT_EQ(reference_first.select(), Error::Ok);
  EXPECT_EQ(reference_first.steps()[0].provider->name(), "Reference");
  configuration.preferences = {{"selection_test", "add"}, {"Reference", ""}};
  CPUPlan specific_first(graph, buffers, configuration, execution);
  ASSERT_EQ(specific_first.select(), Error::Ok);
  EXPECT_EQ(specific_first.steps()[0].implementation->name(), "add");
  configuration.preferences = {
      {"selection_test", ""}, {"selection_test", "add"}};
  CPUPlan wildcard_first(graph, buffers, configuration, execution);
  ASSERT_EQ(wildcard_first.select(), Error::Ok);
  EXPECT_EQ(wildcard_first.steps()[0].implementation->name(), "general");
  configuration.preferences = {
      {"selection_test", "add"}, {"selection_test", ""}};
  CPUPlan wildcard_last(graph, buffers, configuration, execution);
  ASSERT_EQ(wildcard_last.select(), Error::Ok);
  EXPECT_EQ(wildcard_last.steps()[0].implementation->name(), "add");
}

TEST_F(CPUPlanTest, UnmatchedPreferencesFallBackAndInvalidNamesAreRejected) {
  value("x", true);
  value("out");
  gelu("out", 0, 1);
  finish({1});
  const std::array<ProviderFactory, 2> providers{
      create_selection_provider, create_reference_provider};
  configuration.providers = {providers.data(), providers.size()};
  configuration.preferences = {{"selection_test", "add"}};
  CPUPlan fallback(graph, buffers, configuration, execution);
  ASSERT_EQ(fallback.select(), Error::Ok);
  EXPECT_EQ(fallback.steps()[0].implementation->name(), "general");
  configuration.force = true;
  CPUPlan forced(graph, buffers, configuration, execution);
  EXPECT_EQ(forced.select(), Error::NotSupported);
  configuration.preferences.push_back({"Reference", ""});
  CPUPlan later_match(graph, buffers, configuration, execution);
  ASSERT_EQ(later_match.select(), Error::Ok);
  EXPECT_EQ(later_match.steps()[0].provider->name(), "Reference");
  configuration.preferences.push_back({"selection_test", "missing"});
  CPUPlan unknown(graph, buffers, configuration, execution);
  EXPECT_EQ(unknown.select(), Error::InvalidArgument);
  configuration.preferences = {{"", "add"}};
  CPUPlan empty_provider(graph, buffers, configuration, execution);
  EXPECT_EQ(empty_provider.select(), Error::InvalidArgument);
  configuration.preferences.clear();
  CPUPlan empty_forced(graph, buffers, configuration, execution);
  EXPECT_EQ(empty_forced.select(), Error::InvalidArgument);
}

TEST_F(CPUPlanTest, IndexedLoadPreferencesReplaceDefaultsAndOwnTheirStrings) {
  value("x", true);
  value("y", true);
  value("out");
  add("out", 0, 1, 2);
  finish({2});
  configuration.preferences = {{"XNNPACK", ""}};
  BackendOptions<5> options;
  ASSERT_EQ(options.set_option("preference_count", 2), Error::Ok);
  ASSERT_EQ(options.set_option("preferred_provider_0", "Reference"), Error::Ok);
  ASSERT_EQ(options.set_option("preferred_provider_1", "XNNPACK"), Error::Ok);
  ASSERT_EQ(
      options.set_option("preferred_implementation_1", "subgraph"), Error::Ok);
  ASSERT_EQ(options.set_option("force", true), Error::Ok);
  auto context = context_for(options);
  ASSERT_EQ(configuration.apply_options(context), Error::Ok);
  ASSERT_EQ(options.set_option("preferred_provider_0", "unlinked"), Error::Ok);
  CPUPlan plan(graph, buffers, configuration, execution);
  ASSERT_EQ(plan.select(), Error::Ok);
  EXPECT_EQ(plan.steps()[0].provider->name(), "Reference");
  ASSERT_EQ(plan.prepare(allocator), Error::Ok);
  std::fill_n(static_cast<float*>(buffers[0].data), 11, 1);
  std::fill_n(static_cast<float*>(buffers[1].data), 11, 2);
  ASSERT_EQ(plan.execute(execution), Error::Ok);
  EXPECT_FLOAT_EQ(static_cast<float*>(buffers[2].data)[0], 3);
  BackendOptions<2> clear;
  ASSERT_EQ(clear.set_option("preference_count", 0), Error::Ok);
  ASSERT_EQ(clear.set_option("force", false), Error::Ok);
  auto clear_context = context_for(clear);
  ASSERT_EQ(configuration.apply_options(clear_context), Error::Ok);
  CPUPlan baseline(graph, buffers, configuration, execution);
  ASSERT_EQ(baseline.select(), Error::Ok);
  EXPECT_EQ(baseline.steps()[0].provider->name(), "XNNPACK");
}

TEST_F(CPUPlanTest, SinglePairLoadOptionsRemainAShorthand) {
  value("x", true);
  value("y", true);
  value("out");
  add("out", 0, 1, 2);
  finish({2});
  configuration.preferences = {{"XNNPACK", ""}, {"Reference", ""}};
  BackendInitContext absent(&allocator);
  ASSERT_EQ(configuration.apply_options(absent), Error::Ok);
  CPUPlan defaults(graph, buffers, configuration, execution);
  ASSERT_EQ(defaults.select(), Error::Ok);
  EXPECT_EQ(defaults.steps()[0].provider->name(), "XNNPACK");
  BackendOptions<1> options;
  ASSERT_EQ(options.set_option("preferred_provider", "Reference"), Error::Ok);
  auto context = context_for(options);
  ASSERT_EQ(configuration.apply_options(context), Error::Ok);
  CPUPlan loaded(graph, buffers, configuration, execution);
  ASSERT_EQ(loaded.select(), Error::Ok);
  EXPECT_EQ(loaded.steps()[0].provider->name(), "Reference");
  ASSERT_EQ(configuration.preferences.size(), 1);
}

TEST_F(CPUPlanTest, LoadPreferencesRejectMalformedListsAndMixedForms) {
  BackendOptions<4> options;
  ASSERT_EQ(options.set_option("preference_count", "1"), Error::Ok);
  auto context = context_for(options);
  EXPECT_EQ(configuration.apply_options(context), Error::InvalidArgument);
  ASSERT_EQ(options.set_option("preference_count", -1), Error::Ok);
  EXPECT_EQ(configuration.apply_options(context), Error::InvalidArgument);
  ASSERT_EQ(options.set_option("preference_count", 1), Error::Ok);
  EXPECT_EQ(configuration.apply_options(context), Error::InvalidArgument);
  ASSERT_EQ(options.set_option("preferred_provider_0", ""), Error::Ok);
  context = context_for(options);
  EXPECT_EQ(configuration.apply_options(context), Error::InvalidArgument);
  ASSERT_EQ(options.set_option("preferred_provider_0", "Reference"), Error::Ok);
  ASSERT_EQ(options.set_option("preferred_implementation_0", true), Error::Ok);
  context = context_for(options);
  EXPECT_EQ(configuration.apply_options(context), Error::InvalidArgument);
  ASSERT_EQ(options.set_option("preferred_implementation_0", ""), Error::Ok);
  ASSERT_EQ(options.set_option("preferred_provider", "XNNPACK"), Error::Ok);
  context = context_for(options);
  EXPECT_EQ(configuration.apply_options(context), Error::InvalidArgument);
}

struct ProbeState {
  int enumerations = 0;
  int compiled = 0;
  int destroyed = 0;
  bool fail_second = false;
  std::vector<void*> scratch;
};

class ProbeExecutable final : public Executable {
 public:
  explicit ProbeExecutable(std::shared_ptr<ProbeState> state)
      : state_(std::move(state)) {}
  ~ProbeExecutable() override {
    ++state_->destroyed;
  }
  Error reshape() override {
    return Error::Ok;
  }
  Error bind(const Buffer& scratch) override {
    state_->scratch.push_back(scratch.data);
    return scratch.accepts({640, 512, 128}) ? Error::Ok
                                            : Error::InvalidArgument;
  }
  Error run(const ExecutionContext&) override {
    return Error::Ok;
  }

 private:
  std::shared_ptr<ProbeState> state_;
};

class ProbeProvider final : public KernelProvider, public KernelImplementation {
 public:
  std::string_view name() const override {
    return "probe";
  }
  std::vector<KernelImplementation*> implementations() override {
    ++state_->enumerations;
    return {this};
  }
  int priority() const override {
    return 100;
  }
  Support supports(const Kernel&, const ptn::Graph&, const ExecutionContext&)
      const override {
    return {true, "test route"};
  }
  Result<StorageRequirements> requirements(
      const KernelRegion& region,
      const ptn::Graph& graph,
      const ExecutionContext& execution) const override {
    auto storage = KernelImplementation::requirements(region, graph, execution);
    if (!storage.ok()) {
      return storage.error();
    }
    for (auto& value : storage->values) {
      value.buffer.alignment = 128;
      value.buffer.readable_bytes += 128;
    }
    storage->scratch = {640, 512, 128};
    return storage;
  }
  Result<std::unique_ptr<Executable>> compile(
      const KernelRegion&,
      PreparationContext&) override {
    ++state_->compiled;
    if (state_->fail_second && state_->compiled == 2) {
      return Error::MemoryAllocationFailed;
    }
    return std::unique_ptr<Executable>(new ProbeExecutable(state_));
  }
  std::shared_ptr<ProbeState> state() const {
    return state_;
  }

 private:
  std::shared_ptr<ProbeState> state_ = std::make_shared<ProbeState>();
};
std::unique_ptr<KernelProvider> create_probe() {
  return std::make_unique<ProbeProvider>();
}

TEST_F(CPUPlanTest, DeclaredStoragePrecedesCompilationAndScratchIsShared) {
  value("x", true);
  value("y", true);
  value("a");
  value("b");
  add("a", 0, 1, 2);
  add("b", 2, 1, 3);
  finish({3});
  const ProviderFactory factory = create_probe;
  CPUPlan plan(graph, buffers, {{&factory, 1}, {}, false}, execution);
  ASSERT_EQ(plan.select(), Error::Ok);
  const auto probe =
      static_cast<ProbeProvider*>(plan.steps().at(0).provider)->state();
  EXPECT_EQ(probe->enumerations, 1);
  EXPECT_EQ(probe->compiled, 0);
  ASSERT_EQ(plan.prepare(allocator), Error::Ok);
  EXPECT_EQ(plan.scratch_bytes(), 640);
  for (const auto& buffer : buffers) {
    EXPECT_EQ(reinterpret_cast<uintptr_t>(buffer.data) % 128, 0);
    EXPECT_GE(buffer.readable_bytes, 44 + 64 + 128);
  }
  ASSERT_EQ(plan.execute(execution), Error::Ok);
  ASSERT_EQ(probe->scratch.size(), 2);
  EXPECT_EQ(probe->scratch[0], probe->scratch[1]);
}

TEST_F(CPUPlanTest, PreparationFailureCleansUpAndCannotExecute) {
  std::shared_ptr<ProbeState> probe;
  value("x", true);
  value("y", true);
  value("a");
  value("b");
  add("a", 0, 1, 2);
  add("b", 2, 1, 3);
  finish({3});
  const ProviderFactory factory = create_probe;
  {
    CPUPlan plan(graph, buffers, {{&factory, 1}, {}, false}, execution);
    ASSERT_EQ(plan.select(), Error::Ok);
    probe = static_cast<ProbeProvider*>(plan.steps().at(0).provider)->state();
    probe->fail_second = true;
    EXPECT_EQ(plan.prepare(allocator), Error::MemoryAllocationFailed);
    EXPECT_EQ(plan.execute(execution), Error::InvalidState);
  }
  EXPECT_EQ(probe->destroyed, 1);
  buffers.assign(graph.values.size(), {});
  CPUPlan plan(graph, buffers, {{&factory, 1}, {}, false}, execution);
  ASSERT_EQ(plan.select(), Error::Ok);
  probe = static_cast<ProbeProvider*>(plan.steps().at(0).provider)->state();
  MemoryAllocator empty(0, nullptr);
  EXPECT_EQ(plan.prepare(empty), Error::MemoryAllocationFailed);
  EXPECT_EQ(probe->compiled, 0);
  graph.values[1].role = ptn::ValueRole::Parameter;
  buffers[1] = {storage.data(), 44 + kReadableTail, 0, 128};
  CPUPlan short_constant(graph, buffers, {{&factory, 1}, {}, false}, execution);
  ASSERT_EQ(short_constant.select(), Error::Ok);
  probe = static_cast<ProbeProvider*>(short_constant.steps().at(0).provider)
              ->state();
  EXPECT_EQ(short_constant.prepare(allocator), Error::InvalidArgument);
  EXPECT_EQ(probe->compiled, 0);
}

TEST_F(CPUPlanTest, TensorListOutputsPlanAgainstReferenceRegistry) {
  value("x", true);
  value("first");
  value("second");
  ptn::Node split;
  split.name = "split";
  split.target = "torch.ops.aten.split_with_sizes_copy.default";
  split.inputs = {
      {"self", ptn::TensorArg{0}},
      {"split_sizes", ptn::IntListArg{{5, 6}, {}}},
      {"dim", ptn::IntArg{0}}};
  split.outputs = {{ptn::OutputValueKind::TensorList, ptn::kInvalid, {1, 2}}};
  graph.nodes.push_back(std::move(split));
  finish({1, 2});
  CPUPlan plan(graph, buffers, configuration, execution);
  ASSERT_EQ(plan.select(), Error::Ok);
  ASSERT_EQ(plan.steps().size(), 1);
  EXPECT_EQ(plan.steps()[0].provider->name(), "Reference");
  EXPECT_EQ(plan.steps()[0].implementation->name(), "registry");
}
// cppcheck-suppress-end syntaxError
} // namespace
} // namespace executorch::backends::cpu
