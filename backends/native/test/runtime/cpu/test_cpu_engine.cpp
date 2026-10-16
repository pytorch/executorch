// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/cpu/CPUEngine.h>

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>

#include <executorch/backends/cpu/runtime/providers/reference/ReferenceProvider.h>
#include <executorch/backends/native/runtime/Program.h>
#include <executorch/backends/native/runtime/api/Model.h>
#include <executorch/backends/native/runtime/api/Session.h>
#include <executorch/backends/native/runtime/deserialize/Package.h>
#include <executorch/backends/native/runtime/deserialize/test/PackageTestData.h>
#include <executorch/backends/native/runtime/native_graph_generated.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>

namespace ptn {
namespace {
namespace fbs = ::native_backend;
namespace cpu = executorch::backends::cpu;
template <typename T>
using Offset = flatbuffers::Offset<T>;

class TestGraph {
 public:
  flatbuffers::FlatBufferBuilder builder;
  std::vector<Offset<fbs::Node>> nodes;
  std::vector<Offset<fbs::TensorValue>> values;
  std::vector<Offset<fbs::NamedTensorRef>> constants;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::vector<Offset<fbs::OutputSpec>> output_specs;

  Offset<fbs::TensorMeta> meta(
      fbs::ScalarType dtype = fbs::ScalarType::FLOAT,
      std::vector<int64_t> sizes = {4},
      std::vector<int32_t> order = {},
      bool dynamic = false) {
    std::vector<Offset<fbs::Dim>> dims;
    for (const auto size : sizes) {
      dims.push_back(fbs::CreateDim(builder, dynamic ? 0 : size, size));
    }
    return fbs::CreateTensorMeta(
        builder,
        dtype,
        builder.CreateVector(dims),
        builder.CreateVector(order));
  }

  void tensor(const char* name, Offset<fbs::TensorMeta> metadata) {
    values.push_back(fbs::CreateTensorValueDirect(builder, name, metadata));
  }

  void input(const char* name, Offset<fbs::TensorMeta> metadata) {
    tensor(name, metadata);
    const std::vector<Offset<fbs::Output>> out{
        fbs::CreateOutputDirect(builder, name)};
    nodes.push_back(fbs::CreateNodeDirect(
        builder, name, fbs::OpKind::PLACEHOLDER, "", nullptr, &out));
    inputs.emplace_back(name);
  }

  void constant(
      const char* name,
      const char* key,
      Offset<fbs::TensorMeta> metadata) {
    input(name, metadata);
    inputs.pop_back();
    constants.push_back(
        fbs::CreateNamedTensorRefDirect(builder, name, key, metadata));
  }

  Offset<fbs::NamedArgument>
  tensor_arg(const char* parameter, const char* name, bool mutated = false) {
    return fbs::CreateNamedArgumentDirect(
        builder,
        parameter,
        fbs::CreateArgument(
            builder,
            fbs::ArgumentValue::TensorArg,
            fbs::CreateTensorArgDirect(builder, name).Union()),
        mutated);
  }

  void add(
      const char* name,
      const char* a,
      const char* b,
      Offset<fbs::TensorMeta> metadata,
      bool mutated = false) {
    tensor(name, metadata);
    const std::vector<Offset<fbs::NamedArgument>> args{
        tensor_arg("self", a, mutated),
        tensor_arg("other", b),
        fbs::CreateNamedArgumentDirect(
            builder,
            "alpha",
            fbs::CreateArgument(
                builder,
                fbs::ArgumentValue::IntArg,
                fbs::CreateIntArg(builder, 1).Union()))};
    const std::vector<Offset<fbs::Output>> out{
        fbs::CreateOutputDirect(builder, name)};
    nodes.push_back(fbs::CreateNodeDirect(
        builder,
        name,
        fbs::OpKind::CALL_FUNCTION,
        "torch.ops.aten.add.Tensor",
        &args,
        &out));
  }

  void unary(
      const char* name,
      const char* input,
      const char* target,
      Offset<fbs::TensorMeta> metadata,
      const char* alias = nullptr,
      bool mutated = false) {
    tensor(name, metadata);
    std::vector<Offset<fbs::NamedArgument>> args{
        tensor_arg("self", input, mutated)};
    if (std::string(target) == "torch.ops.aten.gelu.default" ||
        std::string(target) == "torch.ops.aten.gelu_.default") {
      args.push_back(fbs::CreateNamedArgumentDirect(
          builder,
          "approximate",
          fbs::CreateArgument(
              builder,
              fbs::ArgumentValue::StringArg,
              fbs::CreateStringArgDirect(builder, "tanh").Union())));
    }
    const std::vector<Offset<fbs::Output>> out{
        fbs::CreateOutputDirect(builder, name, alias)};
    nodes.push_back(fbs::CreateNodeDirect(
        builder, name, fbs::OpKind::CALL_FUNCTION, target, &args, &out));
  }

  void transpose(const char* name, const char* input) {
    tensor(name, meta(fbs::ScalarType::FLOAT, {2, 2}, {1, 0}));
    const std::vector<int64_t> dims{1, 0};
    const std::vector<Offset<fbs::NamedArgument>> args{
        tensor_arg("self", input),
        fbs::CreateNamedArgumentDirect(
            builder,
            "dims",
            fbs::CreateArgument(
                builder,
                fbs::ArgumentValue::IntListArg,
                fbs::CreateIntListArgDirect(builder, &dims).Union()))};
    const std::vector<Offset<fbs::Output>> out{
        fbs::CreateOutputDirect(builder, name, input)};
    nodes.push_back(fbs::CreateNodeDirect(
        builder,
        name,
        fbs::OpKind::CALL_FUNCTION,
        "torch.ops.aten.permute.default",
        &args,
        &out));
  }

  void strided(
      const char* name,
      const char* input,
      std::vector<int64_t> sizes,
      std::vector<int64_t> strides,
      int64_t offset = 0) {
    tensor(name, meta(fbs::ScalarType::FLOAT, sizes));
    const auto int_list = [&](const char* parameter, const auto& values) {
      return fbs::CreateNamedArgumentDirect(
          builder,
          parameter,
          fbs::CreateArgument(
              builder,
              fbs::ArgumentValue::IntListArg,
              fbs::CreateIntListArgDirect(builder, &values).Union()));
    };
    const std::vector<Offset<fbs::NamedArgument>> args{
        tensor_arg("self", input),
        int_list("size", sizes),
        int_list("stride", strides),
        fbs::CreateNamedArgumentDirect(
            builder,
            "storage_offset",
            fbs::CreateArgument(
                builder,
                fbs::ArgumentValue::IntArg,
                fbs::CreateIntArg(builder, offset).Union()))};
    const std::vector<Offset<fbs::Output>> out{
        fbs::CreateOutputDirect(builder, name, input)};
    nodes.push_back(fbs::CreateNodeDirect(
        builder,
        name,
        fbs::OpKind::CALL_FUNCTION,
        "torch.ops.aten.as_strided.default",
        &args,
        &out));
  }

  void output(
      const char* name,
      fbs::OutputKind kind = fbs::OutputKind::USER_OUTPUT,
      const char* target = nullptr) {
    outputs.emplace_back(name);
    output_specs.push_back(
        fbs::CreateOutputSpecDirect(builder, name, kind, target));
  }

  std::vector<uint8_t> program(
      std::vector<std::string> method_names = {"forward"}) {
    std::vector<Offset<fbs::NamedArgument>> args;
    for (const auto& name : outputs) {
      args.push_back(tensor_arg("", name.c_str()));
    }
    nodes.push_back(fbs::CreateNodeDirect(
        builder, "return", fbs::OpKind::OUTPUT, "", &args));
    const auto graph = fbs::CreateGraph(
        builder,
        builder.CreateVector(nodes),
        builder.CreateVectorOfStrings(inputs),
        builder.CreateVectorOfStrings(outputs),
        builder.CreateVector(values));
    std::vector<Offset<fbs::Method>> methods;
    for (const auto& name : method_names) {
      methods.push_back(fbs::CreateMethod(
          builder,
          builder.CreateString(name),
          graph,
          builder.CreateVector(constants),
          builder.CreateVector(output_specs)));
    }
    fbs::FinishProgramBuffer(
        builder,
        fbs::CreateProgram(
            builder,
            builder.CreateString("1.0"),
            builder.CreateVector(methods)));
    return {
        builder.GetBufferPointer(),
        builder.GetBufferPointer() + builder.GetSize()};
  }
};

std::vector<uint8_t> package_bytes(
    const std::vector<uint8_t>& program,
    std::vector<testing::StoredMember> members = {}) {
  members.push_back({kProgramEntry, program});
  return testing::make_zip(members);
}

struct CompiledProgram {
  std::unique_ptr<CPUEngineHost> host;
  std::shared_ptr<const Program> program;
  std::shared_ptr<const Package> package;
  std::unique_ptr<CPUEngineContext> context;
  std::unique_ptr<EngineExecutable> executable;
};

CompiledProgram compile(
    const std::vector<uint8_t>& bytes,
    cpu::RuntimeConfiguration configuration = {},
    std::vector<testing::StoredMember> members = {}) {
  CompiledProgram result;
  result.host = CPUEngineHost::create(std::move(configuration));
  result.program = std::make_shared<const Program>(
      Program::load(bytes.data(), bytes.size()));
  result.package = std::make_shared<const Package>(Package::load(
      OwnedBytes::from_vector(package_bytes(bytes, std::move(members)))));
  result.context =
      result.host->create_cpu_context(result.program, result.package);
  result.executable = result.context->compile("forward");
  return result;
}

std::vector<uint8_t> add_program(
    fbs::ScalarType dtype = fbs::ScalarType::FLOAT) {
  TestGraph graph;
  const auto meta = graph.meta(dtype);
  graph.input("left", meta);
  graph.input("right", meta);
  graph.add("sum", "left", "right", meta);
  graph.output("sum");
  return graph.program();
}

std::vector<float> output(EngineExecutable& executable, size_t index = 0) {
  std::vector<float> values(4);
  executable.get_output(index, values.data(), values.size(), ScalarType::Float);
  return values;
}

// cppcheck-suppress-begin syntaxError
TEST(CPUEngineTest, RunsMixedProvidersAndRetainsInputsAcrossRepeatedExecution) {
  TestGraph graph;
  const auto meta = graph.meta();
  graph.input("x", meta);
  graph.input("y", meta);
  graph.add("sum", "x", "y", meta);
  graph.unary("activation", "sum", "torch.ops.aten.gelu.default", meta);
  graph.add("result", "sum", "activation", meta);
  graph.output("sum");
  graph.output("result");
  auto compiled = compile(graph.program());
  auto& executable = *compiled.executable;
  const std::array<float, 4> x{-2, -1, 1, 2};
  const std::array<float, 4> y{-1, 1, 1, 1};
  executable.set_input(0, x.data(), x.size(), ScalarType::Float);
  executable.set_input(1, y.data(), y.size(), ScalarType::Float);
  for (int repeat = 0; repeat < 3; ++repeat) {
    executable.execute();
    EXPECT_EQ(output(executable), (std::vector<float>{-3, 0, 2, 3}));
    const auto actual = output(executable, 1);
    for (size_t i = 0; i < actual.size(); ++i) {
      const float sum = x[i] + y[i];
      const float gelu = 0.5f * sum *
          (1 + std::tanh(0.7978845608f * (sum + 0.044715f * sum * sum * sum)));
      EXPECT_NEAR(actual[i], sum + gelu, 1e-5);
    }
  }
  executable.set_input(0, y.data(), y.size(), ScalarType::Float);
  executable.execute();
  EXPECT_EQ(output(executable), (std::vector<float>{-2, 2, 2, 2}));
}

TEST(CPUEngineTest, ValidatesHostIoAndRequiresSuccessfulExecution) {
  auto compiled = compile(add_program());
  auto& executable = *compiled.executable;
  std::array<float, 4> values{1, 2, 3, 4};
  EXPECT_EQ(executable.num_inputs(), 2);
  EXPECT_EQ(executable.num_outputs(), 1);
  EXPECT_EQ(executable.input_sizes(0), (std::vector<int64_t>{4}));
  EXPECT_EQ(executable.output_sizes(0), (std::vector<int64_t>{4}));
  EXPECT_THROW(executable.execute(), std::runtime_error);
  EXPECT_THROW(output(executable), std::runtime_error);
  EXPECT_THROW(
      executable.set_input(2, values.data(), 4, ScalarType::Float),
      std::runtime_error);
  EXPECT_THROW(
      executable.set_input(0, values.data(), 3, ScalarType::Float),
      std::runtime_error);
  EXPECT_THROW(
      executable.set_input(0, nullptr, 4, ScalarType::Float),
      std::runtime_error);
  EXPECT_THROW(executable.resize_input(0, {2}), std::runtime_error);
  executable.set_input(0, values.data(), 4, ScalarType::Float);
  EXPECT_THROW(executable.execute(), std::runtime_error);
  executable.set_input(1, values.data(), 4, ScalarType::Float);
  executable.execute();
  EXPECT_EQ(output(executable), (std::vector<float>{2, 4, 6, 8}));
  EXPECT_THROW(
      executable.get_output(1, values.data(), 4, ScalarType::Float),
      std::runtime_error);
  EXPECT_THROW(
      executable.get_output(0, values.data(), 3, ScalarType::Float),
      std::runtime_error);
  EXPECT_THROW(
      executable.get_output(0, nullptr, 4, ScalarType::Float),
      std::runtime_error);
}

TEST(CPUEngineTest, PreservesInt64AndChecksNarrowing) {
  auto compiled = compile(add_program(fbs::ScalarType::LONG));
  auto& executable = *compiled.executable;
  const std::array<int64_t, 4> values{
      int64_t{1} << 40, -(int64_t{1} << 40), 3, -4};
  executable.set_input(0, values.data(), 4, ScalarType::Long);
  executable.set_input(1, values.data(), 4, ScalarType::Long);
  executable.execute();
  std::array<int64_t, 4> actual{};
  executable.get_output(0, actual.data(), 4, ScalarType::Long);
  EXPECT_EQ(executable.input_dtype(0), ScalarType::Long);
  EXPECT_EQ(executable.output_dtype(0), ScalarType::Long);
  EXPECT_EQ(
      actual,
      (std::array<int64_t, 4>{int64_t{1} << 41, -(int64_t{1} << 41), 6, -8}));
  std::array<int32_t, 4> narrow{};
  EXPECT_THROW(
      executable.get_output(0, narrow.data(), 4, ScalarType::Int),
      std::runtime_error);
  const std::array<double, 4> overflow{std::ldexp(1.0, 63), 0, 0, 0};
  EXPECT_THROW(
      executable.set_input(0, overflow.data(), 4, ScalarType::Double),
      std::runtime_error);
  EXPECT_THROW(executable.execute(), std::runtime_error);
}

TEST(CPUEngineTest, ConvertsUnalignedHostInputsAndOutputs) {
  auto compiled = compile(add_program());
  const std::array<double, 4> values{1.25, -2.5, 3.75, 0};
  std::array<uint8_t, sizeof(values) + 1> bytes{};
  std::memcpy(bytes.data() + 1, values.data(), sizeof(values));
  for (size_t index = 0; index < 2; ++index) {
    compiled.executable->set_input(
        index, bytes.data() + 1, 4, ScalarType::Double);
  }
  compiled.executable->execute();
  compiled.executable->get_output(0, bytes.data() + 1, 4, ScalarType::Double);
  std::array<double, 4> actual{};
  std::memcpy(actual.data(), bytes.data() + 1, sizeof(actual));
  EXPECT_EQ(actual, (std::array<double, 4>{2.5, -5, 7.5, 0}));
}

TEST(CPUEngineTest, ConvertsHalfAndBFloat16HostTensors) {
  const std::array<executorch::aten::Half, 4> half{1.25f, -2.5f, 3.75f, 0};
  const std::array<executorch::aten::BFloat16, 4> bfloat{
      1.25f, -2.5f, 3.75f, 0};
  auto compiled = compile(add_program());
  compiled.executable->set_input(0, half.data(), 4, ScalarType::Half);
  compiled.executable->set_input(1, bfloat.data(), 4, ScalarType::BFloat16);
  compiled.executable->execute();
  std::array<executorch::aten::Half, 4> half_output{};
  std::array<executorch::aten::BFloat16, 4> bfloat_output{};
  compiled.executable->get_output(0, half_output.data(), 4, ScalarType::Half);
  compiled.executable->get_output(
      0, bfloat_output.data(), 4, ScalarType::BFloat16);
  const std::vector<float> expected{2.5f, -5, 7.5f, 0};
  EXPECT_EQ(
      (std::vector<float>{half_output.begin(), half_output.end()}), expected);
  EXPECT_EQ(
      (std::vector<float>{bfloat_output.begin(), bfloat_output.end()}),
      expected);

  auto integer = compile(add_program(fbs::ScalarType::LONG));
  integer.executable->set_input(0, half.data(), 4, ScalarType::Half);
  integer.executable->set_input(1, bfloat.data(), 4, ScalarType::BFloat16);
  integer.executable->execute();
  std::array<int64_t, 4> actual{};
  integer.executable->get_output(0, actual.data(), 4, ScalarType::Long);
  EXPECT_EQ(actual, (std::array<int64_t, 4>{2, -4, 6, 0}));
}

TEST(CPUEngineTest, CopiesProviderConfigurationAndHonorsPreferences) {
  std::array<cpu::ProviderFactory, 1> providers{cpu::create_reference_provider};
  cpu::RuntimeConfiguration configuration{
      {providers.data(), providers.size()},
      {{"Reference", "registry"}},
      true};
  auto compiled = compile(add_program(), configuration);
  providers[0] = nullptr;
  auto second = compiled.context->compile("forward");
  const std::array<float, 4> values{1, 2, 3, 4};
  second->set_input(0, values.data(), 4, ScalarType::Float);
  second->set_input(1, values.data(), 4, ScalarType::Float);
  second->execute();
  EXPECT_EQ(output(*second), (std::vector<float>{2, 4, 6, 8}));
  configuration.providers = {};
  configuration.preferences = {{"missing", ""}};
  EXPECT_THROW(compile(add_program(), configuration), std::runtime_error);
}

TEST(CPUEngineTest, SharesPackageConstantsAcrossMethodsAndIsolatesContexts) {
  TestGraph graph;
  const auto meta = graph.meta();
  graph.input("input", meta);
  graph.constant("weight", "shared_alias", meta);
  graph.add("sum", "input", "weight", meta);
  graph.output("sum");
  graph.output("weight");
  const auto bytes = graph.program({"forward", "other"});
  const std::string header =
      R"({"shared":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}})";
  auto weights = testing::make_safetensors(header, 16);
  const std::array<float, 4> values{10, 20, 30, 40};
  std::memcpy(
      weights.data() + 8 + header.size(), values.data(), sizeof(values));
  const std::string aliases = R"({"shared_alias":"shared"})";
  auto compiled = compile(
      bytes,
      {},
      {{kSafeTensorsEntry, weights},
       {kAliasesEntry, {aliases.begin(), aliases.end()}}});
  EXPECT_EQ(compiled.executable->num_inputs(), 1);
  EXPECT_EQ(compiled.context->unique_constant_bytes(), 16);
  auto second = compiled.context->compile("other");
  EXPECT_EQ(compiled.context->materialized_constant_bytes(), 16);
  compiled.executable.reset();
  const std::array<float, 4> input{1, 2, 3, 4};
  second->set_input(0, input.data(), 4, ScalarType::Float);
  second->execute();
  EXPECT_EQ(output(*second), (std::vector<float>{11, 22, 33, 44}));
  EXPECT_EQ(output(*second, 1), (std::vector<float>{10, 20, 30, 40}));
  auto isolated =
      compiled.host->create_cpu_context(compiled.program, compiled.package);
  EXPECT_EQ(isolated->unique_constant_bytes(), 0);
  auto third = isolated->compile("other");
  third->set_input(0, values.data(), 4, ScalarType::Float);
  third->execute();
  EXPECT_EQ(output(*third), (std::vector<float>{20, 40, 60, 80}));
  EXPECT_EQ(output(*second), (std::vector<float>{11, 22, 33, 44}));
}

TEST(CPUEngineTest, SupportsEmptyAndPassthroughGraphs) {
  for (const int64_t size : {0, 4}) {
    TestGraph graph;
    graph.input("input", graph.meta(fbs::ScalarType::FLOAT, {size}));
    graph.output("input");
    auto compiled = compile(graph.program());
    std::vector<float> values(static_cast<size_t>(size), 7);
    compiled.executable->set_input(
        0, size ? values.data() : nullptr, size, ScalarType::Float);
    compiled.executable->execute();
    std::vector<float> actual(size);
    compiled.executable->get_output(
        0, size ? actual.data() : nullptr, size, ScalarType::Float);
    EXPECT_EQ(actual, values);
  }
}

TEST(CPUEngineTest, RejectsUnsupportedSemanticsDuringCompilation) {
  for (const std::string scenario :
       {"alias",
        "mutation_output",
        "mutation_input",
        "layout",
        "dynamic",
        "operator"}) {
    SCOPED_TRACE(scenario);
    TestGraph graph;
    const auto meta = graph.meta(
        fbs::ScalarType::FLOAT,
        {2, 2},
        scenario == "layout" ? std::vector<int32_t>{1, 0}
                             : std::vector<int32_t>{},
        scenario == "dynamic");
    graph.input("x", meta);
    if (scenario == "alias") {
      graph.unary("result", "x", "torch.ops.aten.alias.default", meta, "x");
    } else if (scenario == "operator") {
      graph.unary("result", "x", "test.unsupported", meta);
    } else {
      graph.add("result", "x", "x", meta, scenario == "mutation_input");
    }
    graph.output(
        "result",
        scenario == "mutation_output" ? fbs::OutputKind::USER_INPUT_MUTATION
                                      : fbs::OutputKind::USER_OUTPUT,
        scenario == "mutation_output" ? "x" : nullptr);
    EXPECT_THROW(compile(graph.program()), std::runtime_error);
  }
}

TEST(CPUEngineTest, MaterializesNativePermuteAliases) {
  TestGraph graph;
  const auto meta = graph.meta(fbs::ScalarType::FLOAT, {2, 2});
  graph.input("x", meta);
  graph.transpose("view", "x");
  graph.add("result", "view", "view", meta);
  graph.output("result");
  auto compiled = compile(graph.program());
  const std::array<float, 4> input{1, 2, 3, 4};
  compiled.executable->set_input(
      0, input.data(), input.size(), ScalarType::Float);
  for (int repeat = 0; repeat < 2; ++repeat) {
    compiled.executable->execute();
    EXPECT_EQ(output(*compiled.executable), (std::vector<float>{2, 6, 4, 8}));
  }
}

TEST(CPUEngineTest, MaterializesContiguousNativeStridedReshape) {
  TestGraph graph;
  graph.input("x", graph.meta(fbs::ScalarType::FLOAT, {2, 2}));
  graph.strided("view", "x", {1, 4, 1, 1}, {4, 1, 4, 4});
  graph.output("view");
  auto compiled = compile(graph.program());
  const std::array<float, 4> input{1, 2, 3, 4};
  compiled.executable->set_input(
      0, input.data(), input.size(), ScalarType::Float);
  compiled.executable->execute();
  EXPECT_EQ(output(*compiled.executable), (std::vector<float>{1, 2, 3, 4}));
}

TEST(CPUEngineTest, RejectsNativeStridedViewsThatAreNotContiguousReshapes) {
  for (const std::string scenario : {"transpose", "offset", "alias"}) {
    SCOPED_TRACE(scenario);
    TestGraph graph;
    graph.input("x", graph.meta(fbs::ScalarType::FLOAT, {2, 2}));
    if (scenario == "alias") {
      graph.transpose("base", "x");
    }
    graph.strided(
        "view",
        scenario == "alias" ? "base" : "x",
        {2, 2},
        scenario == "transpose" ? std::vector<int64_t>{1, 2}
                                : std::vector<int64_t>{2, 1},
        scenario == "offset" ? 1 : 0);
    graph.output("view");
    EXPECT_THROW(compile(graph.program()), std::runtime_error);
  }
}

TEST(CPUEngineTest, PreservesReadsOfBaseAcrossSuccessiveNativeMutations) {
  TestGraph graph;
  const auto meta = graph.meta();
  graph.input("x", meta);
  graph.add("sum", "x", "x", meta);
  graph.unary("relu", "sum", "torch.ops.aten.relu_.default", meta, "sum", true);
  graph.unary(
      "sigmoid", "sum", "torch.ops.aten.sigmoid_.default", meta, "sum", true);
  graph.output("sum");
  auto compiled = compile(graph.program());
  const std::array<float, 4> input{-2, 0, 2, 4};
  compiled.executable->set_input(
      0, input.data(), input.size(), ScalarType::Float);
  for (int repeat = 0; repeat < 2; ++repeat) {
    compiled.executable->execute();
    const auto actual = output(*compiled.executable);
    for (size_t index = 0; index < actual.size(); ++index) {
      EXPECT_NEAR(
          actual[index],
          1 / (1 + std::exp(-std::max(0.0f, 2 * input[index]))),
          1e-6);
    }
  }
}

TEST(CPUEngineTest, RejectsNativeMutationWithLiveTransposedAlias) {
  TestGraph graph;
  const auto meta = graph.meta(fbs::ScalarType::FLOAT, {2, 2});
  graph.input("x", meta);
  graph.add("sum", "x", "x", meta);
  graph.transpose("view", "sum");
  graph.unary("relu", "sum", "torch.ops.aten.relu_.default", meta, "sum", true);
  graph.add("result", "view", "view", meta);
  graph.output("result");
  EXPECT_THROW(compile(graph.program()), std::runtime_error);
}

TEST(CPUEngineTest, RejectsNativeMutationOfUserInput) {
  TestGraph graph;
  const auto meta = graph.meta();
  graph.input("x", meta);
  graph.unary("relu", "x", "torch.ops.aten.relu_.default", meta, "x", true);
  graph.output("relu");
  EXPECT_THROW(compile(graph.program()), std::runtime_error);
}

TEST(CPUEngineTest, RunsThroughNativeModelSessionApi) {
  const auto model = Model::load_bytes(package_bytes(add_program()));
  cpu::RuntimeConfiguration configuration;
  auto host = CPUEngineHost::create(configuration);
  auto session = model.create_session(*host);
  const std::array<int64_t, 1> sizes{4};
  const std::array<int64_t, 1> strides{1};
  const std::array<float, 4> values{2, 4, 6, 8};
  std::array<float, 4> actual{};
  const std::array<ConstTensorView, 2> inputs{
      ConstTensorView{
          ScalarType::Float, sizes, strides, values.data(), sizeof(values)},
      ConstTensorView{
          ScalarType::Float, sizes, strides, values.data(), sizeof(values)}};
  std::array<MutableTensorView, 1> outputs{MutableTensorView{
      ScalarType::Float, sizes, strides, actual.data(), sizeof(actual)}};
  session.run("forward", inputs, outputs);
  EXPECT_EQ(actual, (std::array<float, 4>{4, 8, 12, 16}));
  session.release("forward");
  EXPECT_FALSE(session.is_prepared("forward"));
  session.run("forward", inputs, outputs);
  EXPECT_EQ(actual, (std::array<float, 4>{4, 8, 12, 16}));
}
// cppcheck-suppress-end syntaxError

} // namespace
} // namespace ptn
