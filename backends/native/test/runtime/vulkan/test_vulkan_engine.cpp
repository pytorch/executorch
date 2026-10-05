// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/vulkan/VulkanEngine.h>

#include <cmath>
#include <cstdint>
#include <cstring>
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
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& lower_bounds = {},
    const std::vector<int32_t>& dim_order = {}) {
  std::vector<flatbuffers::Offset<fbs::Dim>> sizes;
  sizes.reserve(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    const int64_t lower = lower_bounds.empty() ? shape[i] : lower_bounds.at(i);
    sizes.push_back(fbs::CreateDim(builder, lower, shape[i]));
  }
  return fbs::CreateTensorMeta(
      builder,
      dtype,
      builder.CreateVector(sizes),
      builder.CreateVector(dim_order));
}

flatbuffers::Offset<fbs::TensorMeta> create_tensor_meta(
    flatbuffers::FlatBufferBuilder& builder,
    fbs::ScalarType dtype) {
  return create_tensor_meta(builder, dtype, {4});
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

flatbuffers::Offset<fbs::Argument> create_int_list_arg(
    flatbuffers::FlatBufferBuilder& builder,
    const std::vector<int64_t>& values) {
  const auto argument = fbs::CreateIntListArgDirect(builder, &values);
  return fbs::CreateArgument(
      builder, fbs::ArgumentValue::IntListArg, argument.Union());
}

flatbuffers::Offset<fbs::Argument> create_bool_arg(
    flatbuffers::FlatBufferBuilder& builder,
    bool value) {
  const auto argument = fbs::CreateBoolArg(builder, value);
  return fbs::CreateArgument(
      builder, fbs::ArgumentValue::BoolArg, argument.Union());
}

flatbuffers::Offset<fbs::Argument> create_none_arg(
    flatbuffers::FlatBufferBuilder& builder) {
  const auto argument = fbs::CreateNoneArg(builder);
  return fbs::CreateArgument(
      builder, fbs::ArgumentValue::NoneArg, argument.Union());
}

std::vector<uint8_t> finish_program(
    flatbuffers::FlatBufferBuilder& builder,
    flatbuffers::Offset<fbs::Graph> graph,
    const std::vector<flatbuffers::Offset<fbs::OutputSpec>>& output_specs,
    const std::vector<flatbuffers::Offset<fbs::NamedTensorRef>>& constants =
        {}) {
  const auto method = fbs::CreateMethod(
      builder,
      builder.CreateString("forward"),
      graph,
      constants.empty() ? 0 : builder.CreateVector(constants),
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
    const std::vector<int64_t>& shape = {4},
    const std::vector<int64_t>& lower_bounds = {},
    bool include_mutation_output = false,
    const std::vector<int64_t>& unused_shape = {}) {
  flatbuffers::FlatBufferBuilder builder;
  const auto meta = create_tensor_meta(builder, dtype, shape, lower_bounds);
  std::vector<flatbuffers::Offset<fbs::TensorValue>> tensor_values = {
      fbs::CreateTensorValueDirect(builder, "left", meta),
      fbs::CreateTensorValueDirect(builder, "right", meta),
      fbs::CreateTensorValueDirect(builder, "sum", meta),
  };
  if (!unused_shape.empty()) {
    tensor_values.push_back(fbs::CreateTensorValueDirect(
        builder, "unused", create_tensor_meta(builder, dtype, unused_shape)));
  }

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
    const std::vector<int64_t>& lower_bounds = {},
    bool alias_first = false) {
  flatbuffers::FlatBufferBuilder builder;
  const auto meta =
      create_tensor_meta(builder, fbs::ScalarType::FLOAT, {4}, lower_bounds);
  const auto alias_meta =
      create_tensor_meta(builder, alias_dtype, {4}, lower_bounds);
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

// relu_ mutates `input`, or with `from_sum` the intermediate input + input.
// With `output_mutated`, the mutated tensor is also a graph output.
std::vector<uint8_t> make_inplace_relu_program(
    bool from_sum = false,
    bool output_mutated = false) {
  flatbuffers::FlatBufferBuilder builder;
  const auto meta = create_tensor_meta(builder, fbs::ScalarType::FLOAT);
  const char* mutated = from_sum ? "sum" : "input";
  std::vector<flatbuffers::Offset<fbs::TensorValue>> tensor_values = {
      fbs::CreateTensorValueDirect(builder, "input", meta),
      fbs::CreateTensorValueDirect(builder, "relu", meta),
  };
  const std::vector<flatbuffers::Offset<fbs::Output>> input_outputs = {
      fbs::CreateOutputDirect(builder, "input")};
  const std::vector<flatbuffers::Offset<fbs::Output>> sum_outputs = {
      fbs::CreateOutputDirect(builder, "sum")};
  const std::vector<flatbuffers::Offset<fbs::Output>> relu_outputs = {
      fbs::CreateOutputDirect(builder, "relu", mutated)};
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> add_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "self", create_tensor_arg(builder, "input")),
      fbs::CreateNamedArgumentDirect(
          builder, "other", create_tensor_arg(builder, "input")),
      fbs::CreateNamedArgumentDirect(
          builder, "alpha", create_int_arg(builder, 1)),
  };
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> relu_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "self", create_tensor_arg(builder, mutated), true)};
  std::vector<flatbuffers::Offset<fbs::NamedArgument>> output_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "", create_tensor_arg(builder, "relu"))};
  std::vector<std::string> graph_outputs = {"relu"};
  std::vector<flatbuffers::Offset<fbs::OutputSpec>> output_specs = {
      fbs::CreateOutputSpecDirect(builder, "relu")};
  if (output_mutated) {
    output_inputs.push_back(fbs::CreateNamedArgumentDirect(
        builder, "", create_tensor_arg(builder, mutated)));
    graph_outputs.emplace_back(mutated);
    output_specs.push_back(fbs::CreateOutputSpecDirect(builder, mutated));
  }
  std::vector<flatbuffers::Offset<fbs::Node>> nodes = {fbs::CreateNodeDirect(
      builder, "input", fbs::OpKind::PLACEHOLDER, "", nullptr, &input_outputs)};
  if (from_sum) {
    tensor_values.push_back(fbs::CreateTensorValueDirect(builder, "sum", meta));
    nodes.push_back(fbs::CreateNodeDirect(
        builder,
        "sum",
        fbs::OpKind::CALL_FUNCTION,
        "torch.ops.aten.add.Tensor",
        &add_inputs,
        &sum_outputs));
  }
  nodes.push_back(fbs::CreateNodeDirect(
      builder,
      "relu",
      fbs::OpKind::CALL_FUNCTION,
      "torch.ops.aten.relu_.default",
      &relu_inputs,
      &relu_outputs));
  nodes.push_back(fbs::CreateNodeDirect(
      builder, "output", fbs::OpKind::OUTPUT, "", &output_inputs));
  const auto graph = fbs::CreateGraph(
      builder,
      builder.CreateVector(nodes),
      create_strings(builder, {"input"}),
      create_strings(builder, graph_outputs),
      builder.CreateVector(tensor_values));
  return finish_program(builder, graph, output_specs);
}

std::vector<uint8_t> make_mean_view_mm_program() {
  flatbuffers::FlatBufferBuilder builder;
  const auto input_meta =
      create_tensor_meta(builder, fbs::ScalarType::FLOAT, {1, 2, 2, 2});
  const auto mean_meta =
      create_tensor_meta(builder, fbs::ScalarType::FLOAT, {1, 2, 1, 1});
  const auto matrix_meta =
      create_tensor_meta(builder, fbs::ScalarType::FLOAT, {1, 2});
  const auto weight_meta =
      create_tensor_meta(builder, fbs::ScalarType::FLOAT, {2, 2});
  const std::vector<flatbuffers::Offset<fbs::TensorValue>> tensor_values = {
      fbs::CreateTensorValueDirect(builder, "input", input_meta),
      fbs::CreateTensorValueDirect(builder, "mean", mean_meta),
      fbs::CreateTensorValueDirect(builder, "matrix", matrix_meta),
      fbs::CreateTensorValueDirect(builder, "weight", weight_meta),
      fbs::CreateTensorValueDirect(builder, "product", matrix_meta),
  };

  const std::vector<flatbuffers::Offset<fbs::Output>> input_outputs = {
      fbs::CreateOutputDirect(builder, "input")};
  const std::vector<flatbuffers::Offset<fbs::Output>> weight_outputs = {
      fbs::CreateOutputDirect(builder, "weight")};
  const std::vector<flatbuffers::Offset<fbs::Output>> mean_outputs = {
      fbs::CreateOutputDirect(builder, "mean")};
  const std::vector<flatbuffers::Offset<fbs::Output>> view_outputs = {
      fbs::CreateOutputDirect(builder, "matrix", "mean")};
  const std::vector<flatbuffers::Offset<fbs::Output>> product_outputs = {
      fbs::CreateOutputDirect(builder, "product")};
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> mean_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "self", create_tensor_arg(builder, "input")),
      fbs::CreateNamedArgumentDirect(
          builder, "dim", create_int_list_arg(builder, {-1, -2})),
      fbs::CreateNamedArgumentDirect(
          builder, "keepdim", create_bool_arg(builder, true)),
      fbs::CreateNamedArgumentDirect(
          builder, "dtype", create_none_arg(builder)),
  };
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> view_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "self", create_tensor_arg(builder, "mean")),
      fbs::CreateNamedArgumentDirect(
          builder, "size", create_int_list_arg(builder, {1, 2})),
  };
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> mm_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "self", create_tensor_arg(builder, "matrix")),
      fbs::CreateNamedArgumentDirect(
          builder, "mat2", create_tensor_arg(builder, "weight")),
  };
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> output_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "", create_tensor_arg(builder, "product"))};
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
          "weight",
          fbs::OpKind::PLACEHOLDER,
          "",
          nullptr,
          &weight_outputs),
      fbs::CreateNodeDirect(
          builder,
          "mean",
          fbs::OpKind::CALL_FUNCTION,
          "torch.ops.aten.mean.dim",
          &mean_inputs,
          &mean_outputs),
      fbs::CreateNodeDirect(
          builder,
          "view",
          fbs::OpKind::CALL_FUNCTION,
          "torch.ops.aten.view.default",
          &view_inputs,
          &view_outputs),
      fbs::CreateNodeDirect(
          builder,
          "mm",
          fbs::OpKind::CALL_FUNCTION,
          "torch.ops.aten.mm.default",
          &mm_inputs,
          &product_outputs),
      fbs::CreateNodeDirect(
          builder, "output", fbs::OpKind::OUTPUT, "", &output_inputs),
  };
  const auto graph = fbs::CreateGraph(
      builder,
      builder.CreateVector(nodes),
      create_strings(builder, {"input", "weight"}),
      create_strings(builder, {"product"}),
      builder.CreateVector(tensor_values));
  return finish_program(
      builder, graph, {fbs::CreateOutputSpecDirect(builder, "product")});
}

std::vector<uint8_t> make_relu_program(const std::vector<int32_t>& dim_order) {
  flatbuffers::FlatBufferBuilder builder;
  const auto meta = create_tensor_meta(
      builder, fbs::ScalarType::FLOAT, {2, 2}, /*lower_bounds=*/{}, dim_order);
  const std::vector<flatbuffers::Offset<fbs::TensorValue>> tensor_values = {
      fbs::CreateTensorValueDirect(builder, "input", meta),
      fbs::CreateTensorValueDirect(builder, "relu", meta),
  };
  const std::vector<flatbuffers::Offset<fbs::Output>> input_outputs = {
      fbs::CreateOutputDirect(builder, "input")};
  const std::vector<flatbuffers::Offset<fbs::Output>> relu_outputs = {
      fbs::CreateOutputDirect(builder, "relu")};
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> relu_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "self", create_tensor_arg(builder, "input"))};
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> output_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "", create_tensor_arg(builder, "relu"))};
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
          "relu",
          fbs::OpKind::CALL_FUNCTION,
          "torch.ops.aten.relu.default",
          &relu_inputs,
          &relu_outputs),
      fbs::CreateNodeDirect(
          builder, "output", fbs::OpKind::OUTPUT, "", &output_inputs),
  };
  const auto graph = fbs::CreateGraph(
      builder,
      builder.CreateVector(nodes),
      create_strings(builder, {"input"}),
      create_strings(builder, {"relu"}),
      builder.CreateVector(tensor_values));
  return finish_program(
      builder, graph, {fbs::CreateOutputSpecDirect(builder, "relu")});
}

std::vector<uint8_t> make_dynamic_view_mm_program() {
  flatbuffers::FlatBufferBuilder builder;
  const auto input_meta =
      create_tensor_meta(builder, fbs::ScalarType::FLOAT, {4, 2}, {1, 2});
  const auto weight_meta =
      create_tensor_meta(builder, fbs::ScalarType::FLOAT, {2, 1});
  const auto output_meta =
      create_tensor_meta(builder, fbs::ScalarType::FLOAT, {4, 1}, {1, 1});
  const std::vector<flatbuffers::Offset<fbs::TensorValue>> tensor_values = {
      fbs::CreateTensorValueDirect(builder, "input", input_meta),
      fbs::CreateTensorValueDirect(builder, "matrix", input_meta),
      fbs::CreateTensorValueDirect(builder, "weight", weight_meta),
      fbs::CreateTensorValueDirect(builder, "product", output_meta),
  };

  const std::vector<flatbuffers::Offset<fbs::Output>> input_outputs = {
      fbs::CreateOutputDirect(builder, "input")};
  const std::vector<flatbuffers::Offset<fbs::Output>> weight_outputs = {
      fbs::CreateOutputDirect(builder, "weight")};
  const std::vector<flatbuffers::Offset<fbs::Output>> view_outputs = {
      fbs::CreateOutputDirect(builder, "matrix", "input")};
  const std::vector<flatbuffers::Offset<fbs::Output>> product_outputs = {
      fbs::CreateOutputDirect(builder, "product")};
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> view_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "self", create_tensor_arg(builder, "input")),
      fbs::CreateNamedArgumentDirect(
          builder, "size", create_int_list_arg(builder, {-1, 2})),
  };
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> mm_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "self", create_tensor_arg(builder, "matrix")),
      fbs::CreateNamedArgumentDirect(
          builder, "mat2", create_tensor_arg(builder, "weight")),
  };
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> output_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "", create_tensor_arg(builder, "product"))};
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
          "weight",
          fbs::OpKind::PLACEHOLDER,
          "",
          nullptr,
          &weight_outputs),
      fbs::CreateNodeDirect(
          builder,
          "view",
          fbs::OpKind::CALL_FUNCTION,
          "torch.ops.aten.view.default",
          &view_inputs,
          &view_outputs),
      fbs::CreateNodeDirect(
          builder,
          "mm",
          fbs::OpKind::CALL_FUNCTION,
          "torch.ops.aten.mm.default",
          &mm_inputs,
          &product_outputs),
      fbs::CreateNodeDirect(
          builder, "output", fbs::OpKind::OUTPUT, "", &output_inputs),
  };
  const auto graph = fbs::CreateGraph(
      builder,
      builder.CreateVector(nodes),
      create_strings(builder, {"input", "weight"}),
      create_strings(builder, {"product"}),
      builder.CreateVector(tensor_values));
  return finish_program(
      builder, graph, {fbs::CreateOutputSpecDirect(builder, "product")});
}

// index.Tensor keeps its operands in buffer storage, so `rows` is a zero-copy
// buffer view of `input`, and gathering along dim 1 sizes the output by it.
std::vector<uint8_t> make_dynamic_buffer_view_index_program() {
  flatbuffers::FlatBufferBuilder builder;
  const auto input_meta =
      create_tensor_meta(builder, fbs::ScalarType::FLOAT, {4, 2}, {1, 2});
  const auto rows_meta =
      create_tensor_meta(builder, fbs::ScalarType::FLOAT, {2, 4}, {1, 4});
  const auto index_meta =
      create_tensor_meta(builder, fbs::ScalarType::LONG, {2});
  const auto gathered_meta =
      create_tensor_meta(builder, fbs::ScalarType::FLOAT, {2, 2}, {1, 2});
  const std::vector<flatbuffers::Offset<fbs::TensorValue>> tensor_values = {
      fbs::CreateTensorValueDirect(builder, "input", input_meta),
      fbs::CreateTensorValueDirect(builder, "index", index_meta),
      fbs::CreateTensorValueDirect(builder, "rows", rows_meta),
      fbs::CreateTensorValueDirect(builder, "gathered", gathered_meta),
  };

  const std::vector<flatbuffers::Offset<fbs::Output>> input_outputs = {
      fbs::CreateOutputDirect(builder, "input")};
  const std::vector<flatbuffers::Offset<fbs::Output>> index_outputs = {
      fbs::CreateOutputDirect(builder, "index")};
  const std::vector<flatbuffers::Offset<fbs::Output>> view_outputs = {
      fbs::CreateOutputDirect(builder, "rows", "input")};
  const std::vector<flatbuffers::Offset<fbs::Output>> gathered_outputs = {
      fbs::CreateOutputDirect(builder, "gathered")};
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> view_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "self", create_tensor_arg(builder, "input")),
      fbs::CreateNamedArgumentDirect(
          builder, "size", create_int_list_arg(builder, {-1, 4})),
  };
  const auto indices = fbs::CreateOptionalTensorListArg(
      builder,
      create_strings(builder, {"", "index"}),
      builder.CreateVector(std::vector<uint8_t>{0, 1}));
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> index_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "self", create_tensor_arg(builder, "rows")),
      fbs::CreateNamedArgumentDirect(
          builder,
          "indices",
          fbs::CreateArgument(
              builder,
              fbs::ArgumentValue::OptionalTensorListArg,
              indices.Union())),
  };
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> output_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "", create_tensor_arg(builder, "gathered"))};
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
          "index",
          fbs::OpKind::PLACEHOLDER,
          "",
          nullptr,
          &index_outputs),
      fbs::CreateNodeDirect(
          builder,
          "view",
          fbs::OpKind::CALL_FUNCTION,
          "torch.ops.aten.view.default",
          &view_inputs,
          &view_outputs),
      fbs::CreateNodeDirect(
          builder,
          "gather",
          fbs::OpKind::CALL_FUNCTION,
          "torch.ops.aten.index.Tensor",
          &index_inputs,
          &gathered_outputs),
      fbs::CreateNodeDirect(
          builder, "output", fbs::OpKind::OUTPUT, "", &output_inputs),
  };
  const auto graph = fbs::CreateGraph(
      builder,
      builder.CreateVector(nodes),
      create_strings(builder, {"input", "index"}),
      create_strings(builder, {"gathered"}),
      builder.CreateVector(tensor_values));
  return finish_program(
      builder, graph, {fbs::CreateOutputSpecDirect(builder, "gathered")});
}

// Grouped-query causal attention of a query with a dynamic sequence length,
// starting at the position equal to that length, over caches of kSdpaCache.
constexpr int64_t kSdpaSeq = 4;
constexpr int64_t kSdpaCache = 8;
constexpr int64_t kSdpaHeads = 2;
constexpr int64_t kSdpaKvHeads = 1;
constexpr int64_t kSdpaDim = 8;

std::vector<uint8_t> make_dynamic_sdpa_program() {
  flatbuffers::FlatBufferBuilder builder;
  const auto query_meta = create_tensor_meta(
      builder,
      fbs::ScalarType::FLOAT,
      {1, kSdpaSeq, kSdpaHeads, kSdpaDim},
      {1, 1, kSdpaHeads, kSdpaDim});
  const auto cache_meta = create_tensor_meta(
      builder, fbs::ScalarType::FLOAT, {1, kSdpaCache, kSdpaKvHeads, kSdpaDim});
  const auto flat_meta = create_tensor_meta(
      builder,
      fbs::ScalarType::FLOAT,
      {1, kSdpaSeq, kSdpaHeads * kSdpaDim},
      {1, 1, kSdpaHeads * kSdpaDim});
  const std::vector<flatbuffers::Offset<fbs::TensorValue>> tensor_values = {
      fbs::CreateTensorValueDirect(builder, "query", query_meta),
      fbs::CreateTensorValueDirect(builder, "key", cache_meta),
      fbs::CreateTensorValueDirect(builder, "value", cache_meta),
      fbs::CreateTensorValueDirect(builder, "attention", query_meta),
      fbs::CreateTensorValueDirect(builder, "flat", flat_meta),
  };

  const std::vector<flatbuffers::Offset<fbs::Output>> query_outputs = {
      fbs::CreateOutputDirect(builder, "query")};
  const std::vector<flatbuffers::Offset<fbs::Output>> key_outputs = {
      fbs::CreateOutputDirect(builder, "key")};
  const std::vector<flatbuffers::Offset<fbs::Output>> value_outputs = {
      fbs::CreateOutputDirect(builder, "value")};
  const std::vector<flatbuffers::Offset<fbs::Output>> pos_outputs = {
      fbs::CreateOutputDirect(
          builder, "pos", nullptr, fbs::OutputValueKind::INT)};
  const std::vector<flatbuffers::Offset<fbs::Output>> attention_outputs = {
      fbs::CreateOutputDirect(builder, "attention")};
  const std::vector<flatbuffers::Offset<fbs::Output>> view_outputs = {
      fbs::CreateOutputDirect(builder, "flat", "attention")};
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> sym_size_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "self", create_tensor_arg(builder, "query")),
      fbs::CreateNamedArgumentDirect(
          builder, "dim", create_int_arg(builder, 1)),
  };
  const auto start_pos = fbs::CreateIntArgDirect(builder, 0, "pos");
  const auto dropout = fbs::CreateFloatArg(builder, 0.0);
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> sdpa_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "query", create_tensor_arg(builder, "query")),
      fbs::CreateNamedArgumentDirect(
          builder, "key", create_tensor_arg(builder, "key")),
      fbs::CreateNamedArgumentDirect(
          builder, "value", create_tensor_arg(builder, "value")),
      fbs::CreateNamedArgumentDirect(
          builder,
          "start_pos",
          fbs::CreateArgument(
              builder, fbs::ArgumentValue::IntArg, start_pos.Union())),
      fbs::CreateNamedArgumentDirect(
          builder, "attn_mask", create_none_arg(builder)),
      fbs::CreateNamedArgumentDirect(
          builder,
          "drpout_p",
          fbs::CreateArgument(
              builder, fbs::ArgumentValue::FloatArg, dropout.Union())),
      fbs::CreateNamedArgumentDirect(
          builder, "is_causal", create_bool_arg(builder, true)),
      fbs::CreateNamedArgumentDirect(
          builder, "scale", create_none_arg(builder)),
  };
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> view_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "self", create_tensor_arg(builder, "attention")),
      fbs::CreateNamedArgumentDirect(
          builder,
          "size",
          create_int_list_arg(builder, {1, -1, kSdpaHeads * kSdpaDim})),
  };
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> output_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "", create_tensor_arg(builder, "flat"))};
  const std::vector<flatbuffers::Offset<fbs::Node>> nodes = {
      fbs::CreateNodeDirect(
          builder,
          "query",
          fbs::OpKind::PLACEHOLDER,
          "",
          nullptr,
          &query_outputs),
      fbs::CreateNodeDirect(
          builder, "key", fbs::OpKind::PLACEHOLDER, "", nullptr, &key_outputs),
      fbs::CreateNodeDirect(
          builder,
          "value",
          fbs::OpKind::PLACEHOLDER,
          "",
          nullptr,
          &value_outputs),
      fbs::CreateNodeDirect(
          builder,
          "pos",
          fbs::OpKind::CALL_FUNCTION,
          "torch.ops.aten.sym_size.int",
          &sym_size_inputs,
          &pos_outputs),
      fbs::CreateNodeDirect(
          builder,
          "sdpa",
          fbs::OpKind::CALL_FUNCTION,
          "torch.ops.llama.custom_sdpa.default",
          &sdpa_inputs,
          &attention_outputs),
      fbs::CreateNodeDirect(
          builder,
          "view",
          fbs::OpKind::CALL_FUNCTION,
          "torch.ops.aten.view.default",
          &view_inputs,
          &view_outputs),
      fbs::CreateNodeDirect(
          builder, "output", fbs::OpKind::OUTPUT, "", &output_inputs),
  };
  const auto graph = fbs::CreateGraph(
      builder,
      builder.CreateVector(nodes),
      create_strings(builder, {"query", "key", "value"}),
      create_strings(builder, {"flat"}),
      builder.CreateVector(tensor_values));
  return finish_program(
      builder, graph, {fbs::CreateOutputSpecDirect(builder, "flat")});
}

float sdpa_input_value(size_t index, int64_t salt) {
  return 0.125f * static_cast<float>((static_cast<int64_t>(index) * salt) % 9) -
      0.5f;
}

// Values of the int4 [4, 8] weight in groups of 4 used by the program below.
constexpr int64_t kQ4Out = 4;
constexpr int64_t kQ4In = 8;
constexpr int64_t kQ4GroupSize = 4;

int8_t q4_weight(int64_t n, int64_t k) {
  return static_cast<int8_t>((n * kQ4In + k * 3) % 16 - 8);
}

float q4_scale(int64_t n, int64_t group) {
  return 0.5f + 0.25f * static_cast<float>(n) +
      0.125f * static_cast<float>(group);
}

// out = linear(input, weight), where the weight is a symmetric int4 constant
// read directly by aten.linear, with no dequantize node.
std::vector<uint8_t> make_direct_q4_linear_program() {
  flatbuffers::FlatBufferBuilder builder;
  const auto input_meta =
      create_tensor_meta(builder, fbs::ScalarType::FLOAT, {1, kQ4In});
  const auto output_meta =
      create_tensor_meta(builder, fbs::ScalarType::FLOAT, {1, kQ4Out});
  const auto affine = fbs::CreateAffineGroupDirect(
      builder,
      "weight_scales",
      fbs::ScalarType::FLOAT,
      /*quant_min=*/-8,
      /*quant_max=*/7,
      kQ4GroupSize);
  const auto quant = fbs::CreateQuantSpec(
      builder, fbs::QuantScheme::AffineGroup, affine.Union());
  const std::vector<flatbuffers::Offset<fbs::Dim>> weight_sizes = {
      fbs::CreateDim(builder, kQ4Out, kQ4Out),
      fbs::CreateDim(builder, kQ4In, kQ4In)};
  const auto weight_meta = fbs::CreateTensorMeta(
      builder,
      fbs::ScalarType::BYTE,
      builder.CreateVector(weight_sizes),
      /*dim_order=*/0,
      quant);
  const auto scales_meta = create_tensor_meta(
      builder, fbs::ScalarType::FLOAT, {kQ4Out, kQ4In / kQ4GroupSize});
  const std::vector<flatbuffers::Offset<fbs::TensorValue>> tensor_values = {
      fbs::CreateTensorValueDirect(builder, "input", input_meta),
      fbs::CreateTensorValueDirect(builder, "product", output_meta),
  };

  const std::vector<flatbuffers::Offset<fbs::Output>> input_outputs = {
      fbs::CreateOutputDirect(builder, "input")};
  const std::vector<flatbuffers::Offset<fbs::Output>> weight_outputs = {
      fbs::CreateOutputDirect(builder, "weight")};
  const std::vector<flatbuffers::Offset<fbs::Output>> scales_outputs = {
      fbs::CreateOutputDirect(builder, "weight_scales")};
  const std::vector<flatbuffers::Offset<fbs::Output>> product_outputs = {
      fbs::CreateOutputDirect(builder, "product")};
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> linear_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "input", create_tensor_arg(builder, "input")),
      fbs::CreateNamedArgumentDirect(
          builder, "weight", create_tensor_arg(builder, "weight")),
      fbs::CreateNamedArgumentDirect(builder, "bias", create_none_arg(builder)),
  };
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> output_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "", create_tensor_arg(builder, "product"))};
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
          "weight",
          fbs::OpKind::PLACEHOLDER,
          "",
          nullptr,
          &weight_outputs),
      fbs::CreateNodeDirect(
          builder,
          "weight_scales",
          fbs::OpKind::PLACEHOLDER,
          "",
          nullptr,
          &scales_outputs),
      fbs::CreateNodeDirect(
          builder,
          "linear",
          fbs::OpKind::CALL_FUNCTION,
          "torch.ops.aten.linear.default",
          &linear_inputs,
          &product_outputs),
      fbs::CreateNodeDirect(
          builder, "output", fbs::OpKind::OUTPUT, "", &output_inputs),
  };
  const auto graph = fbs::CreateGraph(
      builder,
      builder.CreateVector(nodes),
      create_strings(builder, {"input"}),
      create_strings(builder, {"product"}),
      builder.CreateVector(tensor_values));
  const std::vector<flatbuffers::Offset<fbs::NamedTensorRef>> constants = {
      fbs::CreateNamedTensorRefDirect(
          builder, "weight", "weight", weight_meta, fbs::InputKind::PARAMETER),
      fbs::CreateNamedTensorRefDirect(
          builder,
          "weight_scales",
          "weight_scales",
          scales_meta,
          fbs::InputKind::PARAMETER),
  };
  return finish_program(
      builder,
      graph,
      {fbs::CreateOutputSpecDirect(builder, "product")},
      constants);
}

// The stored constants of make_direct_q4_linear_program: the weight packed two
// values per byte (even k low, odd k high, offset by 8) and fp32 scales.
std::vector<uint8_t> make_direct_q4_linear_tensors() {
  const std::string header =
      R"({"weight":{"dtype":"U8","shape":[4,4],"data_offsets":[0,16]},)"
      R"("weight_scales":{"dtype":"F32","shape":[4,2],"data_offsets":[16,48]}})";
  std::vector<uint8_t> tensors = testing::make_safetensors(header, 48);
  uint8_t* data = tensors.data() + sizeof(uint64_t) + header.size();
  for (int64_t n = 0; n < kQ4Out; ++n) {
    for (int64_t k = 0; k < kQ4In; k += 2) {
      data[n * kQ4In / 2 + k / 2] = static_cast<uint8_t>(
          ((q4_weight(n, k + 1) + 8) << 4) | (q4_weight(n, k) + 8));
    }
  }
  for (int64_t n = 0; n < kQ4Out; ++n) {
    for (int64_t g = 0; g < kQ4In / kQ4GroupSize; ++g) {
      const float scale = q4_scale(n, g);
      std::memcpy(
          data + 16 + (n * (kQ4In / kQ4GroupSize) + g) * sizeof(float),
          &scale,
          sizeof(scale));
    }
  }
  return tensors;
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
    std::vector<uint8_t> tensors = {},
    const std::string& method_name = "forward") {
  std::vector<testing::StoredMember> members = {{kProgramEntry, program_bytes}};
  if (!tensors.empty()) {
    members.push_back({kSafeTensorsEntry, std::move(tensors)});
  }
  CompiledProgram result;
  result.host = VulkanEngineHost::create();
  result.program = std::make_shared<const Program>(
      Program::load(program_bytes.data(), program_bytes.size()));
  result.package = std::make_shared<const Package>(
      Package::load(OwnedBytes::from_vector(testing::make_zip(members))));
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
    compile_program(make_add_program(fbs::ScalarType::FLOAT, {4}, {}, true));
    FAIL() << "expected the user input mutation to be rejected";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(
        std::string(e.what()).find("mutation of user input 'left'"),
        std::string::npos)
        << e.what();
  }
}

TEST(VulkanEngineTest, DoesNotMaterializeUnusedTensors) {
  CompiledProgram compiled = compile_program(make_add_program(
      fbs::ScalarType::FLOAT, {4}, {}, false, {1 << 18, 1 << 19}));
  const std::vector<float> left = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> right = {10.0f, 20.0f, 30.0f, 40.0f};
  compiled.executable->set_input(
      0, left.data(), left.size(), ScalarType::Float);
  compiled.executable->set_input(
      1, right.data(), right.size(), ScalarType::Float);
  compiled.executable->execute();

  std::vector<float> output(4);
  compiled.executable->get_output(
      0, output.data(), output.size(), ScalarType::Float);
  EXPECT_EQ(output, (std::vector<float>{11.0f, 22.0f, 33.0f, 44.0f}));
}

TEST(VulkanEngineTest, ResizesInputsWithinSerializedBounds) {
  CompiledProgram compiled =
      compile_program(make_add_program(fbs::ScalarType::FLOAT, {4}, {1}));

  EXPECT_THROW(compiled.executable->resize_input(0, {0}), std::runtime_error);
  EXPECT_THROW(compiled.executable->resize_input(0, {5}), std::runtime_error);
  compiled.executable->resize_input(0, {2});
  compiled.executable->resize_input(1, {2});
  EXPECT_EQ(compiled.executable->input_sizes(0), (std::vector<int64_t>{2}));

  const std::vector<float> left = {1.0f, 2.0f};
  const std::vector<float> right = {10.0f, 20.0f};
  compiled.executable->set_input(
      0, left.data(), left.size(), ScalarType::Float);
  compiled.executable->set_input(
      1, right.data(), right.size(), ScalarType::Float);
  compiled.executable->execute();

  EXPECT_EQ(compiled.executable->output_sizes(0), (std::vector<int64_t>{2}));
  std::vector<float> output(2);
  compiled.executable->get_output(
      0, output.data(), output.size(), ScalarType::Float);
  EXPECT_EQ(output, (std::vector<float>{11.0f, 22.0f}));
}

TEST(VulkanEngineTest, RejectsNonContiguousGraphInputsAndOutputs) {
  EXPECT_NO_THROW(compile_program(make_relu_program({0, 1})));
  EXPECT_THROW(compile_program(make_relu_program({1, 0})), std::runtime_error);
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
      make_alias_program(fbs::ScalarType::FLOAT, {}, /*alias_first=*/true));
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

TEST(VulkanEngineTest, RejectsNonViewAliasOfDynamicSource) {
  try {
    compile_program(make_alias_program(fbs::ScalarType::FLOAT, {1}));
    FAIL() << "expected the dynamic alias to be rejected";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(
        std::string(e.what()).find("only aten.view can alias"),
        std::string::npos)
        << e.what();
  }
}

TEST(VulkanEngineTest, RejectsMutableStateHeldByAnotherLiveExecutable) {
  CompiledProgram compiled = compile_program(
      make_state_program({{"prefill", "cache"}, {"decode", "cache"}}),
      {},
      "prefill");
  EXPECT_THROW(compiled.context->compile("decode"), std::runtime_error);
  EXPECT_THROW(compiled.context->compile("prefill"), std::runtime_error);

  compiled.executable.reset();
  EXPECT_NE(compiled.context->compile("decode"), nullptr);
}

TEST(VulkanEngineTest, CompilesMethodsWithPrivateMutableState) {
  CompiledProgram compiled = compile_program(
      make_state_program({{"prefill", "prefill_cache"}, {"decode", "cache"}}),
      {},
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

TEST(VulkanEngineTest, ExecutesSerializedInPlaceRelu) {
  CompiledProgram compiled = compile_program(make_inplace_relu_program());
  const std::vector<float> input = {-2.0f, -0.5f, 0.0f, 3.0f};
  compiled.executable->set_input(
      0, input.data(), input.size(), ScalarType::Float);
  compiled.executable->execute();

  std::vector<float> output(4);
  compiled.executable->get_output(
      0, output.data(), output.size(), ScalarType::Float);
  EXPECT_EQ(output, (std::vector<float>{0.0f, 0.0f, 0.0f, 3.0f}));
}

TEST(VulkanEngineTest, ExecutesInPlaceReluOfIntermediate) {
  CompiledProgram compiled =
      compile_program(make_inplace_relu_program(/*from_sum=*/true));
  const std::vector<float> input = {-2.0f, -0.5f, 0.0f, 3.0f};
  compiled.executable->set_input(
      0, input.data(), input.size(), ScalarType::Float);
  compiled.executable->execute();

  std::vector<float> output(4);
  compiled.executable->get_output(
      0, output.data(), output.size(), ScalarType::Float);
  EXPECT_EQ(output, (std::vector<float>{0.0f, 0.0f, 0.0f, 6.0f}));
}

TEST(VulkanEngineTest, RejectsInPlaceReluWhoseInputIsReadElsewhere) {
  try {
    compile_program(make_inplace_relu_program(
        /*from_sum=*/true, /*output_mutated=*/true));
    FAIL() << "expected the observed in-place relu to be rejected";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(
        std::string(e.what()).find(
            "in-place update of 'sum' is read elsewhere"),
        std::string::npos)
        << e.what();
  }
}

TEST(VulkanEngineTest, ExecutesMeanBeforeWidthPackedView) {
  CompiledProgram compiled = compile_program(make_mean_view_mm_program());
  const std::vector<float> input = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  const std::vector<float> weight = {1.0f, 0.0f, 0.0f, 1.0f};
  compiled.executable->set_input(
      0, input.data(), input.size(), ScalarType::Float);
  compiled.executable->set_input(
      1, weight.data(), weight.size(), ScalarType::Float);
  compiled.executable->execute();

  std::vector<float> output(2);
  compiled.executable->get_output(
      0, output.data(), output.size(), ScalarType::Float);
  EXPECT_EQ(output, (std::vector<float>{2.5f, 6.5f}));
}

TEST(VulkanEngineTest, ResizesWidthPackedViewOfWidthPackedInput) {
  CompiledProgram compiled = compile_program(make_dynamic_view_mm_program());
  compiled.executable->resize_input(0, {2, 2});

  const std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> weight = {10.0f, 1.0f};
  compiled.executable->set_input(
      0, input.data(), input.size(), ScalarType::Float);
  compiled.executable->set_input(
      1, weight.data(), weight.size(), ScalarType::Float);
  compiled.executable->execute();

  EXPECT_EQ(compiled.executable->output_sizes(0), (std::vector<int64_t>{2, 1}));
  std::vector<float> output(2);
  compiled.executable->get_output(
      0, output.data(), output.size(), ScalarType::Float);
  EXPECT_EQ(output, (std::vector<float>{12.0f, 34.0f}));
}

TEST(VulkanEngineTest, ResizesBufferViewWithItsSource) {
  CompiledProgram compiled =
      compile_program(make_dynamic_buffer_view_index_program());
  compiled.executable->resize_input(0, {2, 2});

  const std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<int64_t> index = {3, 0};
  compiled.executable->set_input(
      0, input.data(), input.size(), ScalarType::Float);
  compiled.executable->set_input(
      1, index.data(), index.size(), ScalarType::Long);
  compiled.executable->execute();

  EXPECT_EQ(compiled.executable->output_sizes(0), (std::vector<int64_t>{1, 2}));
  std::vector<float> output(2);
  compiled.executable->get_output(
      0, output.data(), output.size(), ScalarType::Float);
  EXPECT_EQ(output, (std::vector<float>{4.0f, 1.0f}));
}

TEST(VulkanEngineTest, RejectsBufferViewResizeThatDoesNotDivide) {
  CompiledProgram compiled =
      compile_program(make_dynamic_buffer_view_index_program());
  compiled.executable->resize_input(0, {3, 2});
  EXPECT_THROW(compiled.executable->execute(), std::runtime_error);
}

TEST(VulkanEngineTest, ExecutesCausalSdpaOnResizedQuery) {
  CompiledProgram compiled = compile_program(make_dynamic_sdpa_program());
  constexpr int64_t kSeq = 3;
  compiled.executable->resize_input(0, {1, kSeq, kSdpaHeads, kSdpaDim});

  std::vector<float> query(kSeq * kSdpaHeads * kSdpaDim);
  std::vector<float> key(kSdpaCache * kSdpaKvHeads * kSdpaDim);
  std::vector<float> value(key.size());
  for (size_t i = 0; i < query.size(); ++i) {
    query[i] = sdpa_input_value(i, 5);
  }
  for (size_t i = 0; i < key.size(); ++i) {
    key[i] = sdpa_input_value(i, 7);
    value[i] = sdpa_input_value(i, 4);
  }
  compiled.executable->set_input(
      0, query.data(), query.size(), ScalarType::Float);
  compiled.executable->set_input(1, key.data(), key.size(), ScalarType::Float);
  compiled.executable->set_input(
      2, value.data(), value.size(), ScalarType::Float);
  compiled.executable->execute();

  EXPECT_EQ(
      compiled.executable->output_sizes(0),
      (std::vector<int64_t>{1, kSeq, kSdpaHeads * kSdpaDim}));
  std::vector<float> output(query.size());
  compiled.executable->get_output(
      0, output.data(), output.size(), ScalarType::Float);

  const float scale = 1.0f / std::sqrt(static_cast<float>(kSdpaDim));
  for (int64_t s = 0; s < kSeq; ++s) {
    const int64_t context = kSeq + s + 1;
    for (int64_t h = 0; h < kSdpaHeads; ++h) {
      const int64_t kv_head = h / (kSdpaHeads / kSdpaKvHeads);
      const float* q = &query[(s * kSdpaHeads + h) * kSdpaDim];
      std::vector<float> weights(context);
      float total = 0.0f;
      for (int64_t c = 0; c < context; ++c) {
        const float* k = &key[(c * kSdpaKvHeads + kv_head) * kSdpaDim];
        float dot = 0.0f;
        for (int64_t d = 0; d < kSdpaDim; ++d) {
          dot += q[d] * k[d];
        }
        weights[c] = std::exp(dot * scale);
        total += weights[c];
      }
      for (int64_t d = 0; d < kSdpaDim; ++d) {
        float expected = 0.0f;
        for (int64_t c = 0; c < context; ++c) {
          expected += weights[c] / total *
              value[(c * kSdpaKvHeads + kv_head) * kSdpaDim + d];
        }
        EXPECT_NEAR(
            output[(s * kSdpaHeads + h) * kSdpaDim + d], expected, 1e-4f)
            << "position " << s << " head " << h << " dim " << d;
      }
    }
  }
}

TEST(VulkanEngineTest, ExecutesLinearOnDirectlyQuantizedWeight) {
  CompiledProgram compiled = compile_program(
      make_direct_q4_linear_program(), make_direct_q4_linear_tensors());
  std::vector<float> input(kQ4In);
  for (int64_t k = 0; k < kQ4In; ++k) {
    input[k] = 0.5f * static_cast<float>(k) - 1.5f;
  }
  compiled.executable->set_input(
      0, input.data(), input.size(), ScalarType::Float);
  compiled.executable->execute();

  std::vector<float> output(kQ4Out);
  compiled.executable->get_output(
      0, output.data(), output.size(), ScalarType::Float);
  for (int64_t n = 0; n < kQ4Out; ++n) {
    float expected = 0.0f;
    for (int64_t k = 0; k < kQ4In; ++k) {
      expected += input[k] * static_cast<float>(q4_weight(n, k)) *
          q4_scale(n, k / kQ4GroupSize);
    }
    EXPECT_NEAR(output[n], expected, 1e-3f) << "channel " << n;
  }
}
// cppcheck-suppress-end syntaxError

} // namespace
} // namespace ptn
