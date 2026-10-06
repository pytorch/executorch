// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/vulkan/VulkanEngine.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>

#include <executorch/backends/native/runtime/Program.h>
#include <executorch/backends/native/runtime/deserialize/OwnedBytes.h>
#include <executorch/backends/native/runtime/deserialize/Package.h>
#include <executorch/backends/native/runtime/deserialize/test/PackageTestData.h>
#include <executorch/backends/native/runtime/engine/Engine.h>
#include <executorch/backends/native/runtime/native_graph_generated.h>

namespace ptn {
namespace {

namespace fbs = ::native_backend;

flatbuffers::Offset<
    flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>>
create_strings(
    flatbuffers::FlatBufferBuilder& builder,
    const std::vector<std::string>& values) {
  std::vector<flatbuffers::Offset<flatbuffers::String>> strings;
  strings.reserve(values.size());
  for (const std::string& value : values) {
    strings.push_back(builder.CreateString(value));
  }
  return builder.CreateVector(strings);
}

flatbuffers::Offset<fbs::TensorMeta> create_tensor_meta(
    flatbuffers::FlatBufferBuilder& builder,
    fbs::ScalarType dtype) {
  const std::vector<flatbuffers::Offset<fbs::Dim>> sizes = {
      fbs::CreateDim(builder, /*min=*/4, /*max=*/4)};
  return fbs::CreateTensorMeta(builder, dtype, builder.CreateVector(sizes));
}

flatbuffers::Offset<fbs::Argument> create_tensor_arg(
    flatbuffers::FlatBufferBuilder& builder,
    const char* value_name) {
  const auto value = fbs::CreateTensorArgDirect(builder, value_name);
  return fbs::CreateArgument(
      builder, fbs::ArgumentValue::TensorArg, value.Union());
}

flatbuffers::Offset<fbs::Argument> create_int_arg(
    flatbuffers::FlatBufferBuilder& builder,
    int64_t value) {
  const auto argument = fbs::CreateIntArg(builder, value);
  return fbs::CreateArgument(
      builder, fbs::ArgumentValue::IntArg, argument.Union());
}

std::vector<uint8_t> finish_program(
    flatbuffers::FlatBufferBuilder& builder,
    flatbuffers::Offset<fbs::Graph> graph,
    const std::vector<flatbuffers::Offset<fbs::OutputSpec>>& output_specs) {
  const auto method = fbs::CreateMethod(
      builder,
      builder.CreateString("forward"),
      graph,
      /*constants=*/0,
      builder.CreateVector(output_specs));
  const auto program = fbs::CreateProgram(
      builder,
      builder.CreateString("1.0"),
      builder.CreateVector(
          std::vector<flatbuffers::Offset<fbs::Method>>{method}));
  fbs::FinishProgramBuffer(builder, program);
  return {
      builder.GetBufferPointer(),
      builder.GetBufferPointer() + builder.GetSize()};
}

std::vector<uint8_t> make_add_program(fbs::ScalarType dtype) {
  flatbuffers::FlatBufferBuilder builder;
  const auto meta = create_tensor_meta(builder, dtype);
  const std::vector<flatbuffers::Offset<fbs::TensorValue>> tensor_values = {
      fbs::CreateTensorValueDirect(builder, "left", meta),
      fbs::CreateTensorValueDirect(builder, "right", meta),
      fbs::CreateTensorValueDirect(builder, "sum", meta),
  };

  const std::vector<flatbuffers::Offset<fbs::Output>> left_outputs = {
      fbs::CreateOutputDirect(builder, "left")};
  const std::vector<flatbuffers::Offset<fbs::Output>> right_outputs = {
      fbs::CreateOutputDirect(builder, "right")};
  const std::vector<flatbuffers::Offset<fbs::Output>> sum_outputs = {
      fbs::CreateOutputDirect(builder, "sum")};
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> add_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "self", create_tensor_arg(builder, "left")),
      fbs::CreateNamedArgumentDirect(
          builder, "other", create_tensor_arg(builder, "right")),
      fbs::CreateNamedArgumentDirect(
          builder, "alpha", create_int_arg(builder, 1)),
  };
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> output_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "", create_tensor_arg(builder, "sum"))};
  const std::vector<flatbuffers::Offset<fbs::Node>> nodes = {
      fbs::CreateNodeDirect(
          builder,
          "left",
          fbs::OpKind::PLACEHOLDER,
          "",
          nullptr,
          &left_outputs),
      fbs::CreateNodeDirect(
          builder,
          "right",
          fbs::OpKind::PLACEHOLDER,
          "",
          nullptr,
          &right_outputs),
      fbs::CreateNodeDirect(
          builder,
          "sum",
          fbs::OpKind::CALL_FUNCTION,
          "torch.ops.aten.add.Tensor",
          &add_inputs,
          &sum_outputs),
      fbs::CreateNodeDirect(
          builder, "output", fbs::OpKind::OUTPUT, "", &output_inputs),
  };
  const auto graph = fbs::CreateGraph(
      builder,
      builder.CreateVector(nodes),
      create_strings(builder, {"left", "right"}),
      create_strings(builder, {"sum"}),
      builder.CreateVector(tensor_values));
  return finish_program(
      builder, graph, {fbs::CreateOutputSpecDirect(builder, "sum")});
}

std::vector<uint8_t> make_alias_program() {
  flatbuffers::FlatBufferBuilder builder;
  const auto meta = create_tensor_meta(builder, fbs::ScalarType::FLOAT);
  const std::vector<flatbuffers::Offset<fbs::TensorValue>> tensor_values = {
      fbs::CreateTensorValueDirect(builder, "input", meta),
      fbs::CreateTensorValueDirect(builder, "alias", meta),
  };
  const std::vector<flatbuffers::Offset<fbs::Output>> input_outputs = {
      fbs::CreateOutputDirect(builder, "input")};
  const std::vector<flatbuffers::Offset<fbs::Output>> alias_outputs = {
      fbs::CreateOutputDirect(builder, "alias", "input")};
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> alias_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "self", create_tensor_arg(builder, "input"))};
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> output_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "", create_tensor_arg(builder, "alias"))};
  const std::vector<flatbuffers::Offset<fbs::Node>> nodes = {
      fbs::CreateNodeDirect(
          builder,
          "input",
          fbs::OpKind::PLACEHOLDER,
          "",
          nullptr,
          &input_outputs),
      fbs::CreateNodeDirect(
          builder,
          "alias",
          fbs::OpKind::CALL_FUNCTION,
          "torch.ops.aten.alias.default",
          &alias_inputs,
          &alias_outputs),
      fbs::CreateNodeDirect(
          builder, "output", fbs::OpKind::OUTPUT, "", &output_inputs),
  };
  const auto graph = fbs::CreateGraph(
      builder,
      builder.CreateVector(nodes),
      create_strings(builder, {"input"}),
      create_strings(builder, {"alias"}),
      builder.CreateVector(tensor_values));
  return finish_program(
      builder, graph, {fbs::CreateOutputSpecDirect(builder, "alias")});
}

std::vector<uint8_t> make_mutation_program() {
  flatbuffers::FlatBufferBuilder builder;
  const auto meta = create_tensor_meta(builder, fbs::ScalarType::FLOAT);
  const std::vector<flatbuffers::Offset<fbs::TensorValue>> tensor_values = {
      fbs::CreateTensorValueDirect(builder, "input", meta)};
  const std::vector<flatbuffers::Offset<fbs::Output>> input_outputs = {
      fbs::CreateOutputDirect(builder, "input")};
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> output_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "", create_tensor_arg(builder, "input"))};
  const std::vector<flatbuffers::Offset<fbs::Node>> nodes = {
      fbs::CreateNodeDirect(
          builder,
          "input",
          fbs::OpKind::PLACEHOLDER,
          "",
          nullptr,
          &input_outputs),
      fbs::CreateNodeDirect(
          builder, "output", fbs::OpKind::OUTPUT, "", &output_inputs),
  };
  const auto graph = fbs::CreateGraph(
      builder,
      builder.CreateVector(nodes),
      create_strings(builder, {"input"}),
      create_strings(builder, {"input"}),
      builder.CreateVector(tensor_values));
  return finish_program(
      builder,
      graph,
      {fbs::CreateOutputSpecDirect(
          builder, "input", fbs::OutputKind::USER_INPUT_MUTATION, "input")});
}

struct CompiledProgram {
  std::unique_ptr<VulkanEngineHost> host;
  std::shared_ptr<const Program> program;
  std::shared_ptr<const Package> package;
  std::unique_ptr<VulkanEngineContext> context;
  std::unique_ptr<EngineExecutable> executable;
};

CompiledProgram compile_program(const std::vector<uint8_t>& program_bytes) {
  CompiledProgram result;
  result.host = VulkanEngineHost::create();
  result.program = std::make_shared<const Program>(
      Program::load(program_bytes.data(), program_bytes.size()));
  result.package =
      std::make_shared<const Package>(Package::load(OwnedBytes::from_vector(
          testing::make_zip({{kProgramEntry, program_bytes}}))));
  result.context =
      result.host->create_vulkan_context(result.program, result.package);
  result.executable = result.context->compile("forward");
  return result;
}

// cppcheck-suppress-begin syntaxError
TEST(VulkanEngineTest, ExecutesStaticAddAndValidatesIoElementCounts) {
  CompiledProgram compiled =
      compile_program(make_add_program(fbs::ScalarType::FLOAT));
  const std::vector<float> left = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> right = {10.0f, 20.0f, 30.0f, 40.0f};

  EXPECT_THROW(
      compiled.executable->set_input(
          0, left.data(), left.size() - 1, ScalarType::Float),
      std::runtime_error);
  EXPECT_THROW(
      compiled.executable->set_input(
          0, nullptr, left.size(), ScalarType::Float),
      std::runtime_error);
  compiled.executable->set_input(
      0, left.data(), left.size(), ScalarType::Float);
  compiled.executable->set_input(
      1, right.data(), right.size(), ScalarType::Float);
  compiled.executable->execute();

  std::vector<float> output(4);
  EXPECT_THROW(
      compiled.executable->get_output(
          0, output.data(), output.size() - 1, ScalarType::Float),
      std::runtime_error);
  EXPECT_THROW(
      compiled.executable->get_output(
          0, nullptr, output.size(), ScalarType::Float),
      std::runtime_error);
  compiled.executable->get_output(
      0, output.data(), output.size(), ScalarType::Float);
  EXPECT_EQ(output, (std::vector<float>{11.0f, 22.0f, 33.0f, 44.0f}));
}

TEST(VulkanEngineTest, PreservesLogicalLongDtypeAndChecksNarrowing) {
  CompiledProgram compiled =
      compile_program(make_add_program(fbs::ScalarType::LONG));
  EXPECT_EQ(compiled.executable->input_dtype(0), ScalarType::Long);
  EXPECT_EQ(compiled.executable->output_dtype(0), ScalarType::Long);

  const std::vector<int64_t> out_of_range = {
      std::numeric_limits<int32_t>::max(),
      static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1,
      0,
      1,
  };
  EXPECT_THROW(
      compiled.executable->set_input(
          0, out_of_range.data(), out_of_range.size(), ScalarType::Long),
      std::runtime_error);
}

TEST(VulkanEngineTest, RejectsAliasesUntilAliasLoweringIsIntroduced) {
  EXPECT_THROW(compile_program(make_alias_program()), std::runtime_error);
}

TEST(VulkanEngineTest, RejectsMutationOutputsUntilStateSupportIsIntroduced) {
  EXPECT_THROW(compile_program(make_mutation_program()), std::runtime_error);
}
// cppcheck-suppress-end syntaxError

} // namespace
} // namespace ptn
