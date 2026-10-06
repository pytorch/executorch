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
#include <utility>
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
    fbs::ScalarType dtype,
    const std::vector<int64_t>& shape = {4}) {
  std::vector<flatbuffers::Offset<fbs::Dim>> sizes;
  for (const int64_t size : shape) {
    sizes.push_back(fbs::CreateDim(builder, /*min=*/size, /*max=*/size));
  }
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

std::vector<uint8_t> make_add_program(
    fbs::ScalarType dtype,
    bool include_mutation_output = false) {
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
  std::vector<flatbuffers::Offset<fbs::NamedArgument>> output_inputs;
  std::vector<std::string> graph_outputs;
  std::vector<flatbuffers::Offset<fbs::OutputSpec>> output_specs;
  if (include_mutation_output) {
    output_inputs.push_back(fbs::CreateNamedArgumentDirect(
        builder, "", create_tensor_arg(builder, "left")));
    graph_outputs.push_back("left");
    output_specs.push_back(fbs::CreateOutputSpecDirect(
        builder, "left", fbs::OutputKind::USER_INPUT_MUTATION, "left"));
  }
  output_inputs.push_back(fbs::CreateNamedArgumentDirect(
      builder, "", create_tensor_arg(builder, "sum")));
  graph_outputs.push_back("sum");
  output_specs.push_back(fbs::CreateOutputSpecDirect(builder, "sum"));
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
      create_strings(builder, graph_outputs),
      builder.CreateVector(tensor_values));
  return finish_program(builder, graph, output_specs);
}

// With `alias_first`, the alias value is serialized ahead of its source.
std::vector<uint8_t> make_alias_program(
    fbs::ScalarType alias_dtype = fbs::ScalarType::FLOAT,
    bool alias_first = false) {
  flatbuffers::FlatBufferBuilder builder;
  const auto meta = create_tensor_meta(builder, fbs::ScalarType::FLOAT);
  const auto alias_meta = create_tensor_meta(builder, alias_dtype);
  std::vector<flatbuffers::Offset<fbs::TensorValue>> tensor_values = {
      fbs::CreateTensorValueDirect(builder, "input", meta),
      fbs::CreateTensorValueDirect(builder, "alias", alias_meta),
  };
  if (alias_first) {
    std::swap(tensor_values[0], tensor_values[1]);
  }
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

struct IntOperand {
  const char* name;
  std::vector<int64_t> values;
  bool is_list = false;
};

// output = target(input, operands...) over a [2, 2] input. A `list_output` is
// a one-element Tensor[] return; otherwise `output` aliases `input`.
std::vector<uint8_t> make_unary_program(
    const char* target,
    const std::vector<int64_t>& output_shape,
    const std::vector<IntOperand>& operands,
    bool list_output) {
  flatbuffers::FlatBufferBuilder builder;
  const std::vector<flatbuffers::Offset<fbs::TensorValue>> tensor_values = {
      fbs::CreateTensorValueDirect(
          builder,
          "input",
          create_tensor_meta(builder, fbs::ScalarType::FLOAT, {2, 2})),
      fbs::CreateTensorValueDirect(
          builder,
          "output",
          create_tensor_meta(builder, fbs::ScalarType::FLOAT, output_shape)),
  };
  const std::vector<flatbuffers::Offset<fbs::Output>> input_outputs = {
      fbs::CreateOutputDirect(builder, "input")};
  const std::vector<flatbuffers::Offset<flatbuffers::String>> list_names = {
      builder.CreateString("output")};
  const std::vector<flatbuffers::Offset<fbs::Output>> op_outputs = {
      list_output ? fbs::CreateOutputDirect(
                        builder,
                        "",
                        nullptr,
                        fbs::OutputValueKind::TENSOR_LIST,
                        &list_names)
                  : fbs::CreateOutputDirect(builder, "output", "input")};
  std::vector<flatbuffers::Offset<fbs::NamedArgument>> op_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "self", create_tensor_arg(builder, "input"))};
  for (const IntOperand& operand : operands) {
    const auto arg = operand.is_list
        ? fbs::CreateArgument(
              builder,
              fbs::ArgumentValue::IntListArg,
              fbs::CreateIntListArgDirect(builder, &operand.values).Union())
        : create_int_arg(builder, operand.values.at(0));
    op_inputs.push_back(
        fbs::CreateNamedArgumentDirect(builder, operand.name, arg));
  }
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> output_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "", create_tensor_arg(builder, "output"))};
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
          "op",
          fbs::OpKind::CALL_FUNCTION,
          target,
          &op_inputs,
          &op_outputs),
      fbs::CreateNodeDirect(
          builder, "output", fbs::OpKind::OUTPUT, "", &output_inputs),
  };
  const auto graph = fbs::CreateGraph(
      builder,
      builder.CreateVector(nodes),
      create_strings(builder, {"input"}),
      create_strings(builder, {"output"}),
      builder.CreateVector(tensor_values));
  return finish_program(
      builder, graph, {fbs::CreateOutputSpecDirect(builder, "output")});
}

std::vector<uint8_t> make_select_program(int64_t index) {
  return make_unary_program(
      "torch.ops.aten.select.int",
      {2},
      {{"dim", {0}}, {"index", {index}}},
      /*list_output=*/false);
}

// One method per (name, fqn) pair, each computing x + state over a
// zero-initialized mutable buffer bound to `fqn`. With `write_back_sum`, the
// sum is a mutation output for the buffer instead of a user output. With
// `return_state`, the unchanged buffer is also a mutation output ahead of sum.
std::vector<uint8_t> make_state_program(
    const std::vector<std::pair<std::string, std::string>>& methods,
    bool write_back_sum = false,
    bool return_state = false) {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<fbs::Method>> method_offsets;
  for (const auto& [name, fqn] : methods) {
    const auto meta = create_tensor_meta(builder, fbs::ScalarType::FLOAT);
    const std::vector<flatbuffers::Offset<fbs::TensorValue>> tensor_values = {
        fbs::CreateTensorValueDirect(builder, "x", meta),
        fbs::CreateTensorValueDirect(builder, "state", meta),
        fbs::CreateTensorValueDirect(builder, "sum", meta),
    };
    const std::vector<flatbuffers::Offset<fbs::Output>> x_outputs = {
        fbs::CreateOutputDirect(builder, "x")};
    const std::vector<flatbuffers::Offset<fbs::Output>> state_outputs = {
        fbs::CreateOutputDirect(builder, "state")};
    const std::vector<flatbuffers::Offset<fbs::Output>> sum_outputs = {
        fbs::CreateOutputDirect(builder, "sum")};
    const std::vector<flatbuffers::Offset<fbs::NamedArgument>> add_inputs = {
        fbs::CreateNamedArgumentDirect(
            builder, "self", create_tensor_arg(builder, "x")),
        fbs::CreateNamedArgumentDirect(
            builder, "other", create_tensor_arg(builder, "state")),
        fbs::CreateNamedArgumentDirect(
            builder, "alpha", create_int_arg(builder, 1)),
    };
    std::vector<flatbuffers::Offset<fbs::NamedArgument>> output_inputs;
    std::vector<std::string> graph_outputs;
    std::vector<flatbuffers::Offset<fbs::OutputSpec>> output_specs;
    if (return_state) {
      output_inputs.push_back(fbs::CreateNamedArgumentDirect(
          builder, "", create_tensor_arg(builder, "state")));
      graph_outputs.push_back("state");
      output_specs.push_back(fbs::CreateOutputSpecDirect(
          builder, "state", fbs::OutputKind::BUFFER_MUTATION, fqn.c_str()));
    }
    output_inputs.push_back(fbs::CreateNamedArgumentDirect(
        builder, "", create_tensor_arg(builder, "sum")));
    graph_outputs.push_back("sum");
    output_specs.push_back(
        write_back_sum
            ? fbs::CreateOutputSpecDirect(
                  builder, "sum", fbs::OutputKind::BUFFER_MUTATION, fqn.c_str())
            : fbs::CreateOutputSpecDirect(builder, "sum"));
    const std::vector<flatbuffers::Offset<fbs::Node>> nodes = {
        fbs::CreateNodeDirect(
            builder, "x", fbs::OpKind::PLACEHOLDER, "", nullptr, &x_outputs),
        fbs::CreateNodeDirect(
            builder,
            "state",
            fbs::OpKind::PLACEHOLDER,
            "",
            nullptr,
            &state_outputs),
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
        create_strings(builder, {"x"}),
        create_strings(builder, graph_outputs),
        builder.CreateVector(tensor_values));
    const std::vector<flatbuffers::Offset<fbs::MutableBufferSpec>>
        mutable_buffers = {
            fbs::CreateMutableBufferSpecDirect(builder, "state", fqn.c_str())};
    method_offsets.push_back(fbs::CreateMethod(
        builder,
        builder.CreateString(name),
        graph,
        0,
        builder.CreateVector(output_specs),
        builder.CreateVector(mutable_buffers)));
  }
  const auto program = fbs::CreateProgram(
      builder,
      builder.CreateString("1.0"),
      builder.CreateVector(method_offsets));
  fbs::FinishProgramBuffer(builder, program);
  return {
      builder.GetBufferPointer(),
      builder.GetBufferPointer() + builder.GetSize()};
}

struct CompiledProgram {
  std::unique_ptr<VulkanEngineHost> host;
  std::shared_ptr<const Program> program;
  std::shared_ptr<const Package> package;
  std::unique_ptr<VulkanEngineContext> context;
  std::unique_ptr<EngineExecutable> executable;
};

CompiledProgram compile_program(
    const std::vector<uint8_t>& program_bytes,
    const std::string& method_name = "forward") {
  CompiledProgram result;
  result.host = VulkanEngineHost::create();
  result.program = std::make_shared<const Program>(
      Program::load(program_bytes.data(), program_bytes.size()));
  result.package =
      std::make_shared<const Package>(Package::load(OwnedBytes::from_vector(
          testing::make_zip({{kProgramEntry, program_bytes}}))));
  result.context =
      result.host->create_vulkan_context(result.program, result.package);
  result.executable = result.context->compile(method_name);
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

TEST(VulkanEngineTest, ExposesOnlyUserOutputsWhenMutationOutputsArePresent) {
  CompiledProgram compiled =
      compile_program(make_state_program({{"forward", "state"}}, false, true));
  EXPECT_EQ(compiled.executable->num_outputs(), 1);
  EXPECT_EQ(compiled.executable->output_sizes(0), (std::vector<int64_t>{4}));
  EXPECT_THROW(compiled.executable->output_sizes(1), std::out_of_range);

  const std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
  compiled.executable->set_input(0, x.data(), x.size(), ScalarType::Float);
  compiled.executable->execute();

  std::vector<float> output(4);
  compiled.executable->get_output(
      0, output.data(), output.size(), ScalarType::Float);
  EXPECT_EQ(output, x);
  EXPECT_THROW(
      compiled.executable->get_output(
          1, output.data(), output.size(), ScalarType::Float),
      std::out_of_range);
}

TEST(VulkanEngineTest, RejectsUserInputMutation) {
  try {
    compile_program(make_add_program(fbs::ScalarType::FLOAT, true));
    FAIL() << "expected the user input mutation to be rejected";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(
        std::string(e.what()).find("mutation of user input 'left'"),
        std::string::npos)
        << e.what();
  }
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

TEST(VulkanEngineTest, ExecutesMetadataOnlyAliasAfterAliasLowering) {
  CompiledProgram compiled = compile_program(make_alias_program());
  const std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
  compiled.executable->set_input(
      0, input.data(), input.size(), ScalarType::Float);
  compiled.executable->execute();

  std::vector<float> output(4);
  compiled.executable->get_output(
      0, output.data(), output.size(), ScalarType::Float);
  EXPECT_EQ(output, input);
}

TEST(VulkanEngineTest, ExecutesAliasSerializedBeforeItsSource) {
  CompiledProgram compiled = compile_program(
      make_alias_program(fbs::ScalarType::FLOAT, /*alias_first=*/true));
  const std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
  compiled.executable->set_input(
      0, input.data(), input.size(), ScalarType::Float);
  compiled.executable->execute();

  std::vector<float> output(4);
  compiled.executable->get_output(
      0, output.data(), output.size(), ScalarType::Float);
  EXPECT_EQ(output, input);
}

TEST(VulkanEngineTest, ExecutesSelectViewOnlyAtIndexZero) {
  CompiledProgram compiled = compile_program(make_select_program(0));
  const std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
  compiled.executable->set_input(
      0, input.data(), input.size(), ScalarType::Float);
  compiled.executable->execute();
  std::vector<float> output(2);
  compiled.executable->get_output(
      0, output.data(), output.size(), ScalarType::Float);
  EXPECT_EQ(output, (std::vector<float>{1.0f, 2.0f}));

  try {
    compile_program(make_select_program(1));
    FAIL() << "expected the offset select view to be rejected";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(
        std::string(e.what()).find("select view at a nonzero index"),
        std::string::npos)
        << e.what();
  }
}

TEST(VulkanEngineTest, PassesOneElementTensorListOutputAsList) {
  CompiledProgram compiled = compile_program(make_unary_program(
      "torch.ops.aten.split_with_sizes_copy.default",
      {2, 2},
      {{"split_sizes", {2}, /*is_list=*/true}, {"dim", {0}}},
      /*list_output=*/true));
  const std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
  compiled.executable->set_input(
      0, input.data(), input.size(), ScalarType::Float);
  compiled.executable->execute();

  std::vector<float> output(4);
  compiled.executable->get_output(
      0, output.data(), output.size(), ScalarType::Float);
  EXPECT_EQ(output, input);
}

TEST(VulkanEngineTest, RejectsAliasWithDifferentDtypeFromSource) {
  try {
    compile_program(make_alias_program(fbs::ScalarType::INT));
    FAIL() << "expected the alias dtype mismatch to be rejected";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(
        std::string(e.what()).find("alias must match its source dtype"),
        std::string::npos)
        << e.what();
  }
}

TEST(VulkanEngineTest, RejectsMutableStateHeldByAnotherLiveExecutable) {
  CompiledProgram compiled = compile_program(
      make_state_program({{"prefill", "cache"}, {"decode", "cache"}}),
      "prefill");
  EXPECT_THROW(compiled.context->compile("decode"), std::runtime_error);
  EXPECT_THROW(compiled.context->compile("prefill"), std::runtime_error);

  compiled.executable.reset();
  EXPECT_NE(compiled.context->compile("decode"), nullptr);
}

TEST(VulkanEngineTest, CompilesMethodsWithPrivateMutableState) {
  CompiledProgram compiled = compile_program(
      make_state_program({{"prefill", "prefill_cache"}, {"decode", "cache"}}),
      "prefill");
  const std::unique_ptr<EngineExecutable> decode =
      compiled.context->compile("decode");

  const std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
  decode->set_input(0, x.data(), x.size(), ScalarType::Float);
  decode->execute();
  std::vector<float> output(4);
  decode->get_output(0, output.data(), output.size(), ScalarType::Float);
  EXPECT_EQ(output, x);
}

TEST(VulkanEngineTest, RejectsMutationOutputNotWrittenInPlace) {
  try {
    compile_program(make_state_program({{"forward", "state"}}, true));
    FAIL() << "expected the out-of-place buffer mutation to be rejected";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(
        std::string(e.what()).find("is not written in place to 'state'"),
        std::string::npos)
        << e.what();
  }
}
// cppcheck-suppress-end syntaxError

} // namespace
} // namespace ptn
