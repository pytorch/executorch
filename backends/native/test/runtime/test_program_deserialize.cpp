// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/Program.h>

#include <cstdint>
#include <string>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>

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

flatbuffers::Offset<fbs::Graph> create_graph(
    flatbuffers::FlatBufferBuilder& builder,
    const std::vector<flatbuffers::Offset<fbs::Node>>& nodes = {},
    const std::vector<std::string>& inputs = {},
    const std::vector<std::string>& outputs = {},
    const std::vector<flatbuffers::Offset<fbs::TensorValue>>& tensor_values =
        {}) {
  return fbs::CreateGraph(
      builder,
      builder.CreateVector(nodes),
      inputs.empty() ? 0 : create_strings(builder, inputs),
      outputs.empty() ? 0 : create_strings(builder, outputs),
      tensor_values.empty() ? 0 : builder.CreateVector(tensor_values));
}

flatbuffers::Offset<fbs::Method> create_method(
    flatbuffers::FlatBufferBuilder& builder,
    const std::string& name,
    flatbuffers::Offset<fbs::Graph> graph,
    const std::vector<flatbuffers::Offset<fbs::OutputSpec>>& output_specs = {},
    const std::vector<flatbuffers::Offset<fbs::NamedTensorRef>>& constants = {},
    const std::vector<flatbuffers::Offset<fbs::MutableBufferSpec>>&
        mutable_buffers = {}) {
  return fbs::CreateMethod(
      builder,
      builder.CreateString(name),
      graph,
      constants.empty() ? 0 : builder.CreateVector(constants),
      output_specs.empty() ? 0 : builder.CreateVector(output_specs),
      mutable_buffers.empty() ? 0 : builder.CreateVector(mutable_buffers));
}

std::vector<uint8_t> finish_program(
    flatbuffers::FlatBufferBuilder& builder,
    const std::vector<flatbuffers::Offset<fbs::Method>>& methods) {
  const auto program = fbs::CreateProgram(
      builder, builder.CreateString("1"), builder.CreateVector(methods));
  fbs::FinishProgramBuffer(builder, program);
  return {
      builder.GetBufferPointer(),
      builder.GetBufferPointer() + builder.GetSize()};
}

Program load_program(const std::vector<uint8_t>& bytes) {
  return Program::load(bytes.data(), bytes.size());
}

// cppcheck-suppress-begin syntaxError
TEST(ProgramTest, LoadRejectsEmptyMethodName) {
  flatbuffers::FlatBufferBuilder builder;
  const auto graph = create_graph(builder);
  const auto bytes =
      finish_program(builder, {create_method(builder, "", graph)});

  EXPECT_THROW(load_program(bytes), std::runtime_error);
}

TEST(ProgramTest, LoadRejectsDuplicateMethodNames) {
  flatbuffers::FlatBufferBuilder builder;
  const auto graph = create_graph(builder);
  const auto first = create_method(builder, "forward", graph);
  const auto second = create_method(builder, "forward", graph);
  const auto bytes = finish_program(builder, {first, second});

  EXPECT_THROW(load_program(bytes), std::runtime_error);
}

TEST(ProgramTest, GetMethodRejectsMismatchedOutputSpecs) {
  flatbuffers::FlatBufferBuilder builder;
  const auto graph = create_graph(builder, {}, {}, {"output"});
  const auto first = fbs::CreateOutputSpecDirect(builder, "output");
  const auto second = fbs::CreateOutputSpecDirect(builder, "extra");
  const auto method = create_method(builder, "forward", graph, {first, second});
  const auto bytes = finish_program(builder, {method});
  const Program program = load_program(bytes);

  EXPECT_THROW(program.get_method("forward"), std::runtime_error);
}

TEST(ProgramTest, GetMethodPreservesAliasWhenTargetIsCreatedOnDemand) {
  flatbuffers::FlatBufferBuilder builder;
  const auto output = fbs::CreateOutputDirect(builder, "view", "input");
  const std::vector<flatbuffers::Offset<fbs::Output>> outputs = {output};
  const auto node = fbs::CreateNodeDirect(
      builder,
      "view",
      fbs::OpKind::CALL_FUNCTION,
      "aten.view",
      nullptr,
      &outputs);
  const auto graph = create_graph(builder, {node}, {"input"}, {"view"});
  const auto bytes =
      finish_program(builder, {create_method(builder, "forward", graph)});
  const Program program = load_program(bytes);

  const Graph& loaded = program.get_method("forward").graph;
  ASSERT_EQ(loaded.input_ids.size(), 1);
  ASSERT_EQ(loaded.output_ids.size(), 1);
  EXPECT_EQ(loaded.value(loaded.output_ids[0]).alias_id, loaded.input_ids[0]);
}

TEST(ProgramTest, GetMethodRejectsSelfAlias) {
  flatbuffers::FlatBufferBuilder builder;
  const auto output = fbs::CreateOutputDirect(builder, "view", "view");
  const std::vector<flatbuffers::Offset<fbs::Output>> outputs = {output};
  const auto node = fbs::CreateNodeDirect(
      builder,
      "view",
      fbs::OpKind::CALL_FUNCTION,
      "aten.view",
      nullptr,
      &outputs);
  const auto graph = create_graph(builder, {node}, {}, {"view"});
  const auto bytes =
      finish_program(builder, {create_method(builder, "forward", graph)});
  const Program program = load_program(bytes);

  EXPECT_THROW(program.get_method("forward"), std::runtime_error);
}

TEST(ProgramTest, GetMethodRejectsDynamicTensorExtent) {
  flatbuffers::FlatBufferBuilder builder;
  const std::vector<flatbuffers::Offset<fbs::Dim>> sizes = {
      fbs::CreateDim(builder, 2, 16)};
  const auto meta =
      fbs::CreateTensorMetaDirect(builder, fbs::ScalarType::FLOAT, &sizes);
  const auto tensor = fbs::CreateTensorValueDirect(builder, "input", meta);
  const auto graph = create_graph(builder, {}, {"input"}, {}, {tensor});
  const auto bytes =
      finish_program(builder, {create_method(builder, "forward", graph)});
  const Program program = load_program(bytes);

  EXPECT_THROW(program.get_method("forward"), std::runtime_error);
}

TEST(ProgramTest, GetMethodRejectsUnknownEnumValues) {
  {
    flatbuffers::FlatBufferBuilder builder;
    const auto node = fbs::CreateNodeDirect(
        builder, "node", static_cast<fbs::OpKind>(127), "unknown");
    const auto graph = create_graph(builder, {node});
    const auto bytes =
        finish_program(builder, {create_method(builder, "forward", graph)});
    const Program program = load_program(bytes);
    EXPECT_THROW(program.get_method("forward"), std::runtime_error);
  }
  {
    flatbuffers::FlatBufferBuilder builder;
    const auto output = fbs::CreateOutput(
        builder,
        builder.CreateString("output"),
        0,
        static_cast<fbs::OutputValueKind>(127));
    const std::vector<flatbuffers::Offset<fbs::Output>> outputs = {output};
    const auto node = fbs::CreateNodeDirect(
        builder,
        "node",
        fbs::OpKind::CALL_FUNCTION,
        "unknown",
        nullptr,
        &outputs);
    const auto graph = create_graph(builder, {node}, {}, {"output"});
    const auto bytes =
        finish_program(builder, {create_method(builder, "forward", graph)});
    const Program program = load_program(bytes);
    EXPECT_THROW(program.get_method("forward"), std::runtime_error);
  }
  {
    flatbuffers::FlatBufferBuilder builder;
    const auto meta = fbs::CreateTensorMeta(builder);
    const auto constant = fbs::CreateNamedTensorRefDirect(
        builder, "input", "weight", meta, static_cast<fbs::InputKind>(127));
    const auto graph = create_graph(builder, {}, {"input"});
    const auto method =
        create_method(builder, "forward", graph, {}, {constant});
    const auto bytes = finish_program(builder, {method});
    const Program program = load_program(bytes);
    EXPECT_THROW(program.get_method("forward"), std::runtime_error);
  }
  {
    flatbuffers::FlatBufferBuilder builder;
    const auto argument =
        fbs::CreateArgument(builder, static_cast<fbs::ArgumentValue>(127), 0);
    const auto named_argument =
        fbs::CreateNamedArgumentDirect(builder, "input", argument);
    const std::vector<flatbuffers::Offset<fbs::NamedArgument>> inputs = {
        named_argument};
    const auto node = fbs::CreateNodeDirect(
        builder, "node", fbs::OpKind::CALL_FUNCTION, "unknown", &inputs);
    const auto graph = create_graph(builder, {node});
    const auto bytes =
        finish_program(builder, {create_method(builder, "forward", graph)});
    const Program program = load_program(bytes);
    EXPECT_THROW(program.get_method("forward"), std::runtime_error);
  }
  {
    flatbuffers::FlatBufferBuilder builder;
    const auto output_spec = fbs::CreateOutputSpec(
        builder,
        builder.CreateString("output"),
        static_cast<fbs::OutputKind>(127));
    const auto graph = create_graph(builder, {}, {}, {"output"});
    const auto method = create_method(builder, "forward", graph, {output_spec});
    const auto bytes = finish_program(builder, {method});
    const Program program = load_program(bytes);
    EXPECT_THROW(program.get_method("forward"), std::runtime_error);
  }
}

TEST(ProgramTest, GetMethodRejectsUnresolvedDataBinding) {
  flatbuffers::FlatBufferBuilder builder;
  const auto meta = fbs::CreateTensorMeta(builder);
  const auto constant = fbs::CreateNamedTensorRefDirect(
      builder, "missing", "weight", meta, fbs::InputKind::PARAMETER);
  const auto graph = create_graph(builder);
  const auto method = create_method(builder, "forward", graph, {}, {constant});
  const auto bytes = finish_program(builder, {method});
  const Program program = load_program(bytes);

  EXPECT_THROW(program.get_method("forward"), std::runtime_error);
}

TEST(ProgramTest, GetMethodRejectsUnresolvedMutationTarget) {
  {
    flatbuffers::FlatBufferBuilder builder;
    const auto output_spec = fbs::CreateOutputSpecDirect(
        builder, "output", fbs::OutputKind::BUFFER_MUTATION, "missing");
    const auto graph = create_graph(builder, {}, {}, {"output"});
    const auto method = create_method(builder, "forward", graph, {output_spec});
    const auto bytes = finish_program(builder, {method});
    const Program program = load_program(bytes);
    EXPECT_THROW(program.get_method("forward"), std::runtime_error);
  }
  {
    flatbuffers::FlatBufferBuilder builder;
    const auto output_spec = fbs::CreateOutputSpecDirect(
        builder, "output", fbs::OutputKind::USER_INPUT_MUTATION, "missing");
    const auto graph = create_graph(builder, {}, {}, {"output"});
    const auto method = create_method(builder, "forward", graph, {output_spec});
    const auto bytes = finish_program(builder, {method});
    const Program program = load_program(bytes);
    EXPECT_THROW(program.get_method("forward"), std::runtime_error);
  }
}

TEST(ProgramTest, GetMethodRejectsDuplicateDataBinding) {
  flatbuffers::FlatBufferBuilder builder;
  const auto meta = fbs::CreateTensorMeta(builder);
  const auto constant = fbs::CreateNamedTensorRefDirect(
      builder, "state", "state", meta, fbs::InputKind::BUFFER);
  const auto mutable_buffer =
      fbs::CreateMutableBufferSpecDirect(builder, "state", "state");
  const auto graph = create_graph(builder, {}, {"state"});
  const auto method = create_method(
      builder, "forward", graph, {}, {constant}, {mutable_buffer});
  const auto bytes = finish_program(builder, {method});
  const Program program = load_program(bytes);

  EXPECT_THROW(program.get_method("forward"), std::runtime_error);
}
// cppcheck-suppress-end syntaxError

} // namespace
} // namespace ptn
