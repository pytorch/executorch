// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/vulkan/VulkanEngine.h>

#include <algorithm>
#include <array>
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

// output = target(input, operands...) over a [2, 2] input by default. A
// `list_output` is a one-element Tensor[] return; otherwise `output` aliases
// `input`.
std::vector<uint8_t> make_unary_program(
    const char* target,
    const std::vector<int64_t>& output_shape,
    const std::vector<IntOperand>& operands,
    bool list_output,
    const std::vector<int64_t>& input_shape = {2, 2},
    const std::vector<int64_t>& input_lower_bounds = {},
    const std::vector<int64_t>& output_lower_bounds = {}) {
  flatbuffers::FlatBufferBuilder builder;
  const std::vector<flatbuffers::Offset<fbs::TensorValue>> tensor_values = {
      fbs::CreateTensorValueDirect(
          builder,
          "input",
          create_tensor_meta(
              builder,
              fbs::ScalarType::FLOAT,
              input_shape,
              input_lower_bounds)),
      fbs::CreateTensorValueDirect(
          builder,
          "output",
          create_tensor_meta(
              builder,
              fbs::ScalarType::FLOAT,
              output_shape,
              output_lower_bounds)),
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

std::vector<uint8_t> make_dynamic_view_mm_program(
    bool static_reshape = false,
    bool permute = false) {
  flatbuffers::FlatBufferBuilder builder;
  const auto input_meta = static_reshape
      ? create_tensor_meta(builder, fbs::ScalarType::FLOAT, {2, 4})
      : create_tensor_meta(builder, fbs::ScalarType::FLOAT, {4, 2}, {1, 2});
  const auto matrix_meta = static_reshape
      ? create_tensor_meta(
            builder,
            fbs::ScalarType::FLOAT,
            {4, 2},
            {},
            permute ? std::vector<int32_t>{1, 0} : std::vector<int32_t>{})
      : input_meta;
  const auto weight_meta =
      create_tensor_meta(builder, fbs::ScalarType::FLOAT, {2, 1});
  const auto output_meta =
      create_tensor_meta(builder, fbs::ScalarType::FLOAT, {4, 1}, {1, 1});
  const std::vector<flatbuffers::Offset<fbs::TensorValue>> tensor_values = {
      fbs::CreateTensorValueDirect(builder, "input", input_meta),
      fbs::CreateTensorValueDirect(builder, "matrix", matrix_meta),
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
          builder,
          permute ? "dims" : "size",
          create_int_list_arg(
              builder,
              permute ? std::vector<int64_t>{1, 0}
                      : std::vector<int64_t>{-1, 2})),
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
          permute ? "torch.ops.aten.permute.default"
                  : "torch.ops.aten.view.default",
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
std::vector<uint8_t> make_dynamic_buffer_view_index_program(
    bool repeat_index = false,
    bool add_output = false) {
  flatbuffers::FlatBufferBuilder builder;
  const auto input_meta =
      create_tensor_meta(builder, fbs::ScalarType::FLOAT, {4, 2}, {1, 2});
  const auto rows_meta =
      create_tensor_meta(builder, fbs::ScalarType::FLOAT, {2, 4}, {1, 4});
  const auto index_meta = create_tensor_meta(
      builder, fbs::ScalarType::LONG, {repeat_index ? 1 : 2});
  const auto gathered_meta =
      create_tensor_meta(builder, fbs::ScalarType::FLOAT, {2, 2}, {1, 2});
  std::vector<flatbuffers::Offset<fbs::TensorValue>> tensor_values = {
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
      create_strings(builder, {"", repeat_index ? "repeated" : "index"}),
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
  const char* output_name = add_output ? "sum" : "gathered";
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> output_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "", create_tensor_arg(builder, output_name))};
  std::vector<flatbuffers::Offset<fbs::Node>> nodes = {
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
  if (repeat_index) {
    tensor_values.push_back(
        fbs::CreateTensorValueDirect(
            builder,
            "repeated",
            create_tensor_meta(builder, fbs::ScalarType::LONG, {2})));
    const std::vector<flatbuffers::Offset<fbs::NamedArgument>> repeat_inputs = {
        fbs::CreateNamedArgumentDirect(
            builder, "self", create_tensor_arg(builder, "index")),
        fbs::CreateNamedArgumentDirect(
            builder, "repeats", create_int_list_arg(builder, {2}))};
    const std::vector<flatbuffers::Offset<fbs::Output>> repeat_outputs = {
        fbs::CreateOutputDirect(builder, "repeated")};
    nodes.insert(
        nodes.begin() + 2,
        fbs::CreateNodeDirect(
            builder,
            "repeat",
            fbs::OpKind::CALL_FUNCTION,
            "torch.ops.aten.repeat.default",
            &repeat_inputs,
            &repeat_outputs));
  }
  if (add_output) {
    tensor_values.push_back(
        fbs::CreateTensorValueDirect(builder, "sum", gathered_meta));
    const std::vector<flatbuffers::Offset<fbs::NamedArgument>> add_inputs = {
        fbs::CreateNamedArgumentDirect(
            builder, "self", create_tensor_arg(builder, "gathered")),
        fbs::CreateNamedArgumentDirect(
            builder, "other", create_tensor_arg(builder, "gathered")),
        fbs::CreateNamedArgumentDirect(
            builder, "alpha", create_int_arg(builder, 1))};
    const std::vector<flatbuffers::Offset<fbs::Output>> add_outputs = {
        fbs::CreateOutputDirect(builder, "sum")};
    nodes.insert(
        nodes.end() - 1,
        fbs::CreateNodeDirect(
            builder,
            "add",
            fbs::OpKind::CALL_FUNCTION,
            "torch.ops.aten.add.Tensor",
            &add_inputs,
            &add_outputs));
  }
  const auto graph = fbs::CreateGraph(
      builder,
      builder.CreateVector(nodes),
      create_strings(builder, {"input", "index"}),
      create_strings(builder, {output_name}),
      builder.CreateVector(tensor_values));
  return finish_program(
      builder, graph, {fbs::CreateOutputSpecDirect(builder, output_name)});
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

std::vector<uint8_t> make_cache_free_sdpa_program(
    fbs::ScalarType mask_dtype,
    int64_t keys,
    bool index_output) {
  flatbuffers::FlatBufferBuilder builder;
  const auto query_meta =
      create_tensor_meta(builder, fbs::ScalarType::FLOAT, {1, 1, 2, 4});
  const auto kv_meta =
      create_tensor_meta(builder, fbs::ScalarType::FLOAT, {1, 1, keys, 4});
  const auto mask_meta =
      create_tensor_meta(builder, mask_dtype, {1, 1, 2, keys});
  std::vector<flatbuffers::Offset<fbs::TensorValue>> tensor_values = {
      fbs::CreateTensorValueDirect(builder, "query", query_meta),
      fbs::CreateTensorValueDirect(builder, "key", kv_meta),
      fbs::CreateTensorValueDirect(builder, "value", kv_meta),
      fbs::CreateTensorValueDirect(builder, "mask", mask_meta),
      fbs::CreateTensorValueDirect(builder, "attention", query_meta),
  };
  std::vector<std::string> input_names = {"query", "key", "value", "mask"};
  if (index_output) {
    input_names.emplace_back("index");
    tensor_values.push_back(
        fbs::CreateTensorValueDirect(
            builder,
            "index",
            create_tensor_meta(builder, fbs::ScalarType::LONG, {1})));
    tensor_values.push_back(
        fbs::CreateTensorValueDirect(builder, "gathered", query_meta));
  }
  std::vector<flatbuffers::Offset<fbs::Node>> nodes;
  for (const auto& name : input_names) {
    const std::vector<flatbuffers::Offset<fbs::Output>> outputs = {
        fbs::CreateOutputDirect(builder, name.c_str())};
    nodes.push_back(
        fbs::CreateNodeDirect(
            builder,
            name.c_str(),
            fbs::OpKind::PLACEHOLDER,
            "",
            nullptr,
            &outputs));
  }
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> sdpa_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "query", create_tensor_arg(builder, "query")),
      fbs::CreateNamedArgumentDirect(
          builder, "key", create_tensor_arg(builder, "key")),
      fbs::CreateNamedArgumentDirect(
          builder, "value", create_tensor_arg(builder, "value")),
      fbs::CreateNamedArgumentDirect(
          builder, "attn_mask", create_tensor_arg(builder, "mask")),
  };
  const std::vector<flatbuffers::Offset<fbs::Output>> sdpa_outputs = {
      fbs::CreateOutputDirect(builder, "attention")};
  nodes.push_back(
      fbs::CreateNodeDirect(
          builder,
          "sdpa",
          fbs::OpKind::CALL_FUNCTION,
          "torch.ops.aten.scaled_dot_product_attention.default",
          &sdpa_inputs,
          &sdpa_outputs));
  if (index_output) {
    const auto indices = fbs::CreateOptionalTensorListArg(
        builder,
        create_strings(builder, {"index"}),
        builder.CreateVector(std::vector<uint8_t>{1}));
    const std::vector<flatbuffers::Offset<fbs::NamedArgument>> index_inputs = {
        fbs::CreateNamedArgumentDirect(
            builder, "self", create_tensor_arg(builder, "attention")),
        fbs::CreateNamedArgumentDirect(
            builder,
            "indices",
            fbs::CreateArgument(
                builder,
                fbs::ArgumentValue::OptionalTensorListArg,
                indices.Union())),
    };
    const std::vector<flatbuffers::Offset<fbs::Output>> outputs = {
        fbs::CreateOutputDirect(builder, "gathered")};
    nodes.push_back(
        fbs::CreateNodeDirect(
            builder,
            "gather",
            fbs::OpKind::CALL_FUNCTION,
            "torch.ops.aten.index.Tensor",
            &index_inputs,
            &outputs));
  }
  const char* output_name = index_output ? "gathered" : "attention";
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> output_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "", create_tensor_arg(builder, output_name))};
  nodes.push_back(
      fbs::CreateNodeDirect(
          builder, "output", fbs::OpKind::OUTPUT, "", &output_inputs));
  const auto graph = fbs::CreateGraph(
      builder,
      builder.CreateVector(nodes),
      create_strings(builder, input_names),
      create_strings(builder, {output_name}),
      builder.CreateVector(tensor_values));
  return finish_program(
      builder, graph, {fbs::CreateOutputSpecDirect(builder, output_name)});
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

// A transposed [1, 2, 2, 2] -> [1, 8, 3, 3] convolution with a 2x2 kernel and
// no bias, so the engine sizes the bias from the transposed weight layout
// [in_channels, out_channels / groups, kh, kw].
constexpr int64_t kConvIn = 2;
constexpr int64_t kConvOut = 8;
constexpr int64_t kConvKernel = 2;
constexpr int64_t kConvInSize = 2;
constexpr int64_t kConvOutSize = kConvInSize + kConvKernel - 1;

float conv_weight(int64_t ci, int64_t co, int64_t ky, int64_t kx) {
  return 0.25f * static_cast<float>((ci * 5 + co * 3 + ky * 2 + kx) % 7) -
      0.75f;
}

std::vector<uint8_t> make_transposed_conv_program() {
  flatbuffers::FlatBufferBuilder builder;
  const std::vector<flatbuffers::Offset<fbs::TensorValue>> tensor_values = {
      fbs::CreateTensorValueDirect(
          builder,
          "input",
          create_tensor_meta(
              builder,
              fbs::ScalarType::FLOAT,
              {1, kConvIn, kConvInSize, kConvInSize})),
      fbs::CreateTensorValueDirect(
          builder,
          "conv",
          create_tensor_meta(
              builder,
              fbs::ScalarType::FLOAT,
              {1, kConvOut, kConvOutSize, kConvOutSize})),
  };
  const std::vector<flatbuffers::Offset<fbs::Output>> input_outputs = {
      fbs::CreateOutputDirect(builder, "input")};
  const std::vector<flatbuffers::Offset<fbs::Output>> weight_outputs = {
      fbs::CreateOutputDirect(builder, "weight")};
  const std::vector<flatbuffers::Offset<fbs::Output>> conv_outputs = {
      fbs::CreateOutputDirect(builder, "conv")};
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> conv_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "input", create_tensor_arg(builder, "input")),
      fbs::CreateNamedArgumentDirect(
          builder, "weight", create_tensor_arg(builder, "weight")),
      fbs::CreateNamedArgumentDirect(builder, "bias", create_none_arg(builder)),
      fbs::CreateNamedArgumentDirect(
          builder, "stride", create_int_list_arg(builder, {1, 1})),
      fbs::CreateNamedArgumentDirect(
          builder, "padding", create_int_list_arg(builder, {0, 0})),
      fbs::CreateNamedArgumentDirect(
          builder, "dilation", create_int_list_arg(builder, {1, 1})),
      fbs::CreateNamedArgumentDirect(
          builder, "transposed", create_bool_arg(builder, true)),
      fbs::CreateNamedArgumentDirect(
          builder, "output_padding", create_int_list_arg(builder, {0, 0})),
      fbs::CreateNamedArgumentDirect(
          builder, "groups", create_int_arg(builder, 1)),
  };
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> output_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "", create_tensor_arg(builder, "conv"))};
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
          "conv",
          fbs::OpKind::CALL_FUNCTION,
          "torch.ops.aten.convolution.default",
          &conv_inputs,
          &conv_outputs),
      fbs::CreateNodeDirect(
          builder, "output", fbs::OpKind::OUTPUT, "", &output_inputs),
  };
  const auto graph = fbs::CreateGraph(
      builder,
      builder.CreateVector(nodes),
      create_strings(builder, {"input"}),
      create_strings(builder, {"conv"}),
      builder.CreateVector(tensor_values));
  const std::vector<flatbuffers::Offset<fbs::NamedTensorRef>> constants = {
      fbs::CreateNamedTensorRefDirect(
          builder,
          "weight",
          "weight",
          create_tensor_meta(
              builder,
              fbs::ScalarType::FLOAT,
              {kConvIn, kConvOut, kConvKernel, kConvKernel}),
          fbs::InputKind::PARAMETER),
  };
  return finish_program(
      builder,
      graph,
      {fbs::CreateOutputSpecDirect(builder, "conv")},
      constants);
}

std::vector<uint8_t> make_transposed_conv_tensors() {
  constexpr int64_t kNumel = kConvIn * kConvOut * kConvKernel * kConvKernel;
  const std::string header =
      R"({"weight":{"dtype":"F32","shape":[2,8,2,2],"data_offsets":[0,256]}})";
  std::vector<uint8_t> tensors =
      testing::make_safetensors(header, kNumel * sizeof(float));
  uint8_t* data = tensors.data() + sizeof(uint64_t) + header.size();
  size_t offset = 0;
  for (int64_t ci = 0; ci < kConvIn; ++ci) {
    for (int64_t co = 0; co < kConvOut; ++co) {
      for (int64_t ky = 0; ky < kConvKernel; ++ky) {
        for (int64_t kx = 0; kx < kConvKernel; ++kx) {
          const float weight = conv_weight(ci, co, ky, kx);
          std::memcpy(data + offset, &weight, sizeof(weight));
          offset += sizeof(weight);
        }
      }
    }
  }
  return tensors;
}

// relu over a [1, 4] uint8 user input quantized per group, with its scale
// bound as a stored constant.
std::vector<uint8_t> make_quantized_input_program() {
  flatbuffers::FlatBufferBuilder builder;
  const auto affine = fbs::CreateAffineGroupDirect(
      builder,
      "input_scales",
      fbs::ScalarType::FLOAT,
      /*quant_min=*/0,
      /*quant_max=*/255,
      /*group_size=*/4);
  const auto quant = fbs::CreateQuantSpec(
      builder, fbs::QuantScheme::AffineGroup, affine.Union());
  const std::vector<flatbuffers::Offset<fbs::Dim>> input_sizes = {
      fbs::CreateDim(builder, 1, 1), fbs::CreateDim(builder, 4, 4)};
  const auto input_meta = fbs::CreateTensorMeta(
      builder,
      fbs::ScalarType::BYTE,
      builder.CreateVector(input_sizes),
      /*dim_order=*/0,
      quant);
  const std::vector<flatbuffers::Offset<fbs::TensorValue>> tensor_values = {
      fbs::CreateTensorValueDirect(builder, "input", input_meta),
      fbs::CreateTensorValueDirect(
          builder,
          "relu",
          create_tensor_meta(builder, fbs::ScalarType::FLOAT, {1, 4})),
  };
  const std::vector<flatbuffers::Offset<fbs::Output>> input_outputs = {
      fbs::CreateOutputDirect(builder, "input")};
  const std::vector<flatbuffers::Offset<fbs::Output>> scales_outputs = {
      fbs::CreateOutputDirect(builder, "input_scales")};
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
          "input_scales",
          fbs::OpKind::PLACEHOLDER,
          "",
          nullptr,
          &scales_outputs),
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
  const std::vector<flatbuffers::Offset<fbs::NamedTensorRef>> constants = {
      fbs::CreateNamedTensorRefDirect(
          builder,
          "input_scales",
          "input_scales",
          create_tensor_meta(builder, fbs::ScalarType::FLOAT, {1, 1}),
          fbs::InputKind::CONSTANT_TENSOR),
  };
  return finish_program(
      builder,
      graph,
      {fbs::CreateOutputSpecDirect(builder, "relu")},
      constants);
}

std::vector<uint8_t> make_quantized_input_tensors() {
  const std::string header =
      R"({"input_scales":{"dtype":"F32","shape":[1,1],"data_offsets":[0,4]}})";
  std::vector<uint8_t> tensors = testing::make_safetensors(header, 4);
  const float scale = 0.5f;
  std::memcpy(
      tensors.data() + sizeof(uint64_t) + header.size(), &scale, sizeof(scale));
  return tensors;
}

constexpr int64_t kHfTokens = 4;
constexpr int64_t kHfHeads = 4;
constexpr int64_t kHfKvHeads = 2;
constexpr int64_t kHfDim = 16;
constexpr int64_t kHfCache = 32;

// One HF static-cache attention block as torch.export emits it: rope on q and
// k, index_put_ into zero-initialized KV caches at cache_position, and SDPA
// under the causal mask le(arange(0, cache) + kv_offset, cache_position).
// kv_offset is a stored buffer, as in HF's StaticCache.
std::vector<uint8_t> make_hf_attention_program(bool use_rope = true) {
  flatbuffers::FlatBufferBuilder builder;
  using fbs::ScalarType;
  std::vector<flatbuffers::Offset<fbs::TensorValue>> tensor_values;
  std::vector<flatbuffers::Offset<fbs::Node>> nodes;
  const auto tensor = [&](const char* name,
                          ScalarType dtype,
                          const std::vector<int64_t>& shape) {
    tensor_values.push_back(fbs::CreateTensorValueDirect(
        builder, name, create_tensor_meta(builder, dtype, shape)));
  };
  const auto placeholder = [&](const char* name) {
    const std::vector<flatbuffers::Offset<fbs::Output>> outputs = {
        fbs::CreateOutputDirect(builder, name)};
    nodes.push_back(fbs::CreateNodeDirect(
        builder, name, fbs::OpKind::PLACEHOLDER, "", nullptr, &outputs));
  };
  const auto call =
      [&](const char* name,
          const char* target,
          const std::vector<flatbuffers::Offset<fbs::NamedArgument>>& inputs,
          const char* alias_of = nullptr) {
        const std::vector<flatbuffers::Offset<fbs::Output>> outputs = {
            fbs::CreateOutputDirect(builder, name, alias_of)};
        nodes.push_back(fbs::CreateNodeDirect(
            builder,
            name,
            fbs::OpKind::CALL_FUNCTION,
            target,
            &inputs,
            &outputs));
      };
  const auto arg = [&](const char* name,
                       flatbuffers::Offset<fbs::Argument> value,
                       bool mutated = false) {
    return fbs::CreateNamedArgumentDirect(builder, name, value, mutated);
  };
  const auto float_arg = [&](double value) {
    return fbs::CreateArgument(
        builder,
        fbs::ArgumentValue::FloatArg,
        fbs::CreateFloatArg(builder, value).Union());
  };
  const auto cache_indices = [&]() {
    const std::vector<uint8_t> has_value = {0, 0, 1};
    return fbs::CreateArgument(
        builder,
        fbs::ArgumentValue::OptionalTensorListArg,
        fbs::CreateOptionalTensorListArg(
            builder,
            create_strings(builder, {"", "", "cache_position"}),
            builder.CreateVector(has_value))
            .Union());
  };

  const std::vector<int64_t> q_shape = {1, kHfTokens, kHfHeads, kHfDim};
  const std::vector<int64_t> kv_shape = {1, kHfTokens, kHfKvHeads, kHfDim};
  const std::vector<int64_t> q_heads = {1, kHfHeads, kHfTokens, kHfDim};
  const std::vector<int64_t> kv_heads = {1, kHfKvHeads, kHfTokens, kHfDim};
  const std::vector<int64_t> cache_shape = {1, kHfKvHeads, kHfCache, kHfDim};
  const std::vector<int64_t> mask_shape = {1, 1, kHfTokens, kHfCache};
  tensor("q", ScalarType::FLOAT, q_shape);
  tensor("k", ScalarType::FLOAT, kv_shape);
  tensor("v", ScalarType::FLOAT, kv_shape);
  tensor("cache_position", ScalarType::LONG, {kHfTokens});
  tensor("inv_freq", ScalarType::FLOAT, {kHfDim / 2});
  tensor("key_cache", ScalarType::FLOAT, cache_shape);
  tensor("value_cache", ScalarType::FLOAT, cache_shape);
  tensor("kv_offset", ScalarType::LONG, {1});
  tensor("position_ids", ScalarType::LONG, {1, kHfTokens});
  tensor("q_perm", ScalarType::FLOAT, q_heads);
  tensor("q_rope", ScalarType::FLOAT, q_heads);
  tensor("k_perm", ScalarType::FLOAT, kv_heads);
  tensor("k_rope", ScalarType::FLOAT, kv_heads);
  tensor("v_perm", ScalarType::FLOAT, kv_heads);
  tensor("k_all", ScalarType::FLOAT, cache_shape);
  tensor("v_all", ScalarType::FLOAT, cache_shape);
  tensor("kv_arange", ScalarType::LONG, {kHfCache});
  tensor("kv_index", ScalarType::LONG, {kHfCache});
  tensor("query_index", ScalarType::LONG, {kHfTokens, 1});
  tensor("le", ScalarType::BOOL, {kHfTokens, kHfCache});
  tensor("mask_0", ScalarType::BOOL, {1, kHfTokens, kHfCache});
  tensor("mask_1", ScalarType::BOOL, mask_shape);
  tensor("mask_2", ScalarType::BOOL, mask_shape);
  tensor("mask", ScalarType::BOOL, mask_shape);
  tensor("attention", ScalarType::FLOAT, q_heads);
  tensor("out", ScalarType::FLOAT, q_shape);

  for (const char* name :
       {"q",
        "k",
        "v",
        "cache_position",
        "inv_freq",
        "key_cache",
        "value_cache",
        "kv_offset"}) {
    placeholder(name);
  }
  call(
      "position_ids",
      "torch.ops.aten.unsqueeze_copy.default",
      {arg("self", create_tensor_arg(builder, "cache_position")),
       arg("dim", create_int_arg(builder, 0))});
  const auto permute = [&](const char* name, const char* input) {
    call(
        name,
        "torch.ops.aten.permute_copy.default",
        {arg("self", create_tensor_arg(builder, input)),
         arg("dims", create_int_list_arg(builder, {0, 2, 1, 3}))});
  };
  const auto rope = [&](const char* name, const char* input) {
    call(
        name,
        "torch.ops.native.rope.default",
        {arg("input", create_tensor_arg(builder, input)),
         arg("position_ids", create_tensor_arg(builder, "position_ids")),
         arg("inv_freq", create_tensor_arg(builder, "inv_freq")),
         arg("interleaved", create_bool_arg(builder, false)),
         arg("attention_scale", float_arg(1.0))});
  };
  const auto cache_write =
      [&](const char* name, const char* cache, const char* values) {
        call(
            name,
            "torch.ops.aten.index_put_.default",
            {arg("self", create_tensor_arg(builder, cache), true),
             arg("indices", cache_indices()),
             arg("values", create_tensor_arg(builder, values)),
             arg("accumulate", create_bool_arg(builder, false))},
            cache);
      };
  permute("q_perm", "q");
  permute("k_perm", "k");
  if (use_rope) {
    rope("q_rope", "q_perm");
    rope("k_rope", "k_perm");
  }
  permute("v_perm", "v");
  cache_write("k_all", "key_cache", use_rope ? "k_rope" : "k_perm");
  cache_write("v_all", "value_cache", "v_perm");
  call(
      "kv_arange",
      "torch.ops.aten.arange.start_step",
      {arg("start", create_int_arg(builder, 0)),
       arg("end", create_int_arg(builder, kHfCache))});
  call(
      "kv_index",
      "torch.ops.aten.add.Tensor",
      {arg("self", create_tensor_arg(builder, "kv_arange")),
       arg("other", create_tensor_arg(builder, "kv_offset"))});
  call(
      "query_index",
      "torch.ops.aten.view_copy.default",
      {arg("self", create_tensor_arg(builder, "cache_position")),
       arg("size", create_int_list_arg(builder, {-1, 1}))});
  call(
      "le",
      "torch.ops.aten.le.Tensor",
      {arg("self", create_tensor_arg(builder, "kv_index")),
       arg("other", create_tensor_arg(builder, "query_index"))});
  call(
      "mask_0",
      "torch.ops.aten.unsqueeze_copy.default",
      {arg("self", create_tensor_arg(builder, "le")),
       arg("dim", create_int_arg(builder, 0))});
  call(
      "mask_1",
      "torch.ops.aten.unsqueeze_copy.default",
      {arg("self", create_tensor_arg(builder, "mask_0")),
       arg("dim", create_int_arg(builder, 1))});
  call(
      "mask_2",
      "torch.ops.aten.expand_copy.default",
      {arg("self", create_tensor_arg(builder, "mask_1")),
       arg("size", create_int_list_arg(builder, {1, -1, -1, -1}))});
  call(
      "mask",
      "torch.ops.aten.alias_copy.default",
      {arg("self", create_tensor_arg(builder, "mask_2"))});
  call(
      "attention",
      "torch.ops.aten.scaled_dot_product_attention.default",
      {arg("query", create_tensor_arg(builder, use_rope ? "q_rope" : "q_perm")),
       arg("key", create_tensor_arg(builder, "k_all")),
       arg("value", create_tensor_arg(builder, "v_all")),
       arg("attn_mask", create_tensor_arg(builder, "mask")),
       arg("dropout_p", float_arg(0.0)),
       arg("is_causal", create_bool_arg(builder, false)),
       arg("scale", use_rope ? create_none_arg(builder) : float_arg(1.0)),
       arg("enable_gqa", create_bool_arg(builder, true))});
  permute("out", "attention");
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> output_inputs = {
      arg("", create_tensor_arg(builder, "k_all")),
      arg("", create_tensor_arg(builder, "v_all")),
      arg("", create_tensor_arg(builder, "out"))};
  nodes.push_back(fbs::CreateNodeDirect(
      builder, "output", fbs::OpKind::OUTPUT, "", &output_inputs));

  const auto graph = fbs::CreateGraph(
      builder,
      builder.CreateVector(nodes),
      create_strings(builder, {"q", "k", "v", "cache_position"}),
      create_strings(builder, {"k_all", "v_all", "out"}),
      builder.CreateVector(tensor_values));
  const std::vector<flatbuffers::Offset<fbs::NamedTensorRef>> constants = {
      fbs::CreateNamedTensorRefDirect(
          builder,
          "inv_freq",
          "inv_freq",
          create_tensor_meta(builder, ScalarType::FLOAT, {kHfDim / 2}),
          fbs::InputKind::CONSTANT_TENSOR),
      fbs::CreateNamedTensorRefDirect(
          builder,
          "kv_offset",
          "kv_offset",
          create_tensor_meta(builder, ScalarType::LONG, {1}),
          fbs::InputKind::BUFFER),
  };
  const std::vector<flatbuffers::Offset<fbs::OutputSpec>> output_specs = {
      fbs::CreateOutputSpecDirect(
          builder, "k_all", fbs::OutputKind::BUFFER_MUTATION, "key_cache"),
      fbs::CreateOutputSpecDirect(
          builder, "v_all", fbs::OutputKind::BUFFER_MUTATION, "value_cache"),
      fbs::CreateOutputSpecDirect(builder, "out"),
  };
  const std::vector<flatbuffers::Offset<fbs::MutableBufferSpec>>
      mutable_buffers = {
          fbs::CreateMutableBufferSpecDirect(builder, "key_cache", "key_cache"),
          fbs::CreateMutableBufferSpecDirect(
              builder, "value_cache", "value_cache")};
  const auto method = fbs::CreateMethod(
      builder,
      builder.CreateString("forward"),
      graph,
      builder.CreateVector(constants),
      builder.CreateVector(output_specs),
      builder.CreateVector(mutable_buffers));
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

std::vector<uint8_t> make_hf_attention_tensors(int64_t kv_offset) {
  const std::string header =
      R"({"inv_freq":{"dtype":"F32","shape":[8],"data_offsets":[0,32]},)"
      R"("kv_offset":{"dtype":"I64","shape":[1],"data_offsets":[32,40]}})";
  std::vector<uint8_t> tensors = testing::make_safetensors(header, 40);
  uint8_t* data = tensors.data() + sizeof(uint64_t) + header.size();
  for (int64_t i = 0; i < kHfDim / 2; ++i) {
    const float inv_freq = std::pow(10000.0f, -2.0f * i / kHfDim);
    std::memcpy(data + i * sizeof(float), &inv_freq, sizeof(inv_freq));
  }
  std::memcpy(data + 32, &kv_offset, sizeof(kv_offset));
  return tensors;
}

constexpr int64_t kQ4KOut = 2;
constexpr int64_t kQ4KIn = 256;
constexpr size_t kQ4KBlockBytes = 144;

// out = linear(input, weight), where the [2, 256] weight is stored `codec`
// blocks read directly by aten.linear.
std::vector<uint8_t> make_gguf_q4_k_linear_program(
    const char* codec = "gguf:q4_k") {
  flatbuffers::FlatBufferBuilder builder;
  const auto input_meta =
      create_tensor_meta(builder, fbs::ScalarType::FLOAT, {1, kQ4KIn});
  const auto output_meta =
      create_tensor_meta(builder, fbs::ScalarType::FLOAT, {1, kQ4KOut});
  const auto packed = fbs::CreatePackedQuantDirect(builder, codec);
  const auto quant = fbs::CreateQuantSpec(
      builder, fbs::QuantScheme::PackedQuant, packed.Union());
  const std::vector<flatbuffers::Offset<fbs::Dim>> weight_sizes = {
      fbs::CreateDim(builder, kQ4KOut, kQ4KOut),
      fbs::CreateDim(builder, kQ4KIn, kQ4KIn)};
  const auto weight_meta = fbs::CreateTensorMeta(
      builder,
      fbs::ScalarType::BYTE,
      builder.CreateVector(weight_sizes),
      /*dim_order=*/0,
      quant);
  const std::vector<flatbuffers::Offset<fbs::TensorValue>> tensor_values = {
      fbs::CreateTensorValueDirect(builder, "input", input_meta),
      fbs::CreateTensorValueDirect(builder, "product", output_meta),
  };
  const std::vector<flatbuffers::Offset<fbs::Output>> input_outputs = {
      fbs::CreateOutputDirect(builder, "input")};
  const std::vector<flatbuffers::Offset<fbs::Output>> weight_outputs = {
      fbs::CreateOutputDirect(builder, "weight")};
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
  };
  return finish_program(
      builder,
      graph,
      {fbs::CreateOutputSpecDirect(builder, "product")},
      constants);
}

// Each 32-byte group of the test blocks shifts its low and high nibbles by
// different steps, so a decoder that reads the wrong group gets a different
// dot product.
uint8_t q4_k_nibble(uint8_t base, int group, bool high) {
  return static_cast<uint8_t>((base + (high ? 5 : 3) * group) & 0xF);
}

// One Q4_K block per row: d = 1, dmin = 0, every sub-block scale 1 and min 0,
// so each weight is its nibble. Each 32-byte group feeds 32 weights from its
// low nibbles, then 32 from its high nibbles.
std::vector<uint8_t> make_gguf_q4_k_linear_tensors(
    const std::array<std::array<uint8_t, 2>, kQ4KOut>& nibbles) {
  const std::string header =
      R"({"weight":{"dtype":"U8","shape":[288],"data_offsets":[0,288]}})";
  std::vector<uint8_t> tensors =
      testing::make_safetensors(header, kQ4KOut * kQ4KBlockBytes);
  uint8_t* data = tensors.data() + sizeof(uint64_t) + header.size();
  for (int64_t n = 0; n < kQ4KOut; ++n) {
    uint8_t* block = data + n * kQ4KBlockBytes;
    block[1] = 0x3C;
    for (int j = 0; j < 4; ++j) {
      block[4 + j] = 1;
      block[12 + j] = 1;
    }
    for (int group = 0; group < 4; ++group) {
      std::fill_n(
          block + 16 + 32 * group,
          32,
          static_cast<uint8_t>(
              q4_k_nibble(nibbles[n][0], group, false) |
              (q4_k_nibble(nibbles[n][1], group, true) << 4)));
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

TEST(VulkanEngineTest, ExecutesAliasOfDynamicSourceAsCopy) {
  CompiledProgram compiled =
      compile_program(make_alias_program(fbs::ScalarType::FLOAT, {1}));
  compiled.executable->resize_input(0, {3});
  const std::vector<float> input = {1.0f, -2.0f, 3.5f};
  compiled.executable->set_input(
      0, input.data(), input.size(), ScalarType::Float);
  compiled.executable->execute();

  EXPECT_EQ(compiled.executable->output_sizes(0), (std::vector<int64_t>{3}));
  std::vector<float> output(3);
  compiled.executable->get_output(
      0, output.data(), output.size(), ScalarType::Float);
  EXPECT_EQ(output, input);
}

TEST(VulkanEngineTest, ExecutesUnsqueezeOfDynamicSourceToRankFive) {
  // Rank-5 tensors only fit buffers, and ET-VK's unsqueeze_copy reads and
  // writes one storage type.
  CompiledProgram compiled = compile_program(make_unary_program(
      "torch.ops.aten.unsqueeze.default",
      {1, 3, 1, 1, 2},
      {{"dim", {0}}},
      /*list_output=*/false,
      /*input_shape=*/{3, 1, 1, 2},
      /*input_lower_bounds=*/{1, 1, 1, 2},
      /*output_lower_bounds=*/{1, 1, 1, 1, 2}));
  compiled.executable->resize_input(0, {2, 1, 1, 2});
  const std::vector<float> input = {1.0f, -2.0f, 3.5f, 4.0f};
  compiled.executable->set_input(
      0, input.data(), input.size(), ScalarType::Float);
  compiled.executable->execute();

  EXPECT_EQ(
      compiled.executable->output_sizes(0),
      (std::vector<int64_t>{1, 2, 1, 1, 2}));
  std::vector<float> output(4);
  compiled.executable->get_output(
      0, output.data(), output.size(), ScalarType::Float);
  EXPECT_EQ(output, input);
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

TEST(VulkanEngineTest, MaterializesStaticWidthPackedReshape) {
  CompiledProgram compiled =
      compile_program(make_dynamic_view_mm_program(true));
  const std::vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8};
  const std::vector<float> weight = {10, 1};
  compiled.executable->set_input(
      0, input.data(), input.size(), ScalarType::Float);
  compiled.executable->set_input(
      1, weight.data(), weight.size(), ScalarType::Float);
  compiled.executable->execute();

  std::vector<float> output(4);
  compiled.executable->get_output(
      0, output.data(), output.size(), ScalarType::Float);
  EXPECT_EQ(output, (std::vector<float>{12, 34, 56, 78}));
}

TEST(VulkanEngineTest, MaterializesStaticWidthPackedPermute) {
  CompiledProgram compiled =
      compile_program(make_dynamic_view_mm_program(true, true));
  const std::vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8};
  const std::vector<float> weight = {10, 1};
  compiled.executable->set_input(
      0, input.data(), input.size(), ScalarType::Float);
  compiled.executable->set_input(
      1, weight.data(), weight.size(), ScalarType::Float);
  compiled.executable->execute();

  std::vector<float> output(4);
  compiled.executable->get_output(
      0, output.data(), output.size(), ScalarType::Float);
  EXPECT_EQ(output, (std::vector<float>{15, 26, 37, 48}));
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

TEST(VulkanEngineTest, ConvertsRepeatedIndicesToBufferStorage) {
  CompiledProgram compiled =
      compile_program(make_dynamic_buffer_view_index_program(true));
  compiled.executable->resize_input(0, {2, 2});
  const std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
  compiled.executable->set_input(
      0, input.data(), input.size(), ScalarType::Float);

  for (const int64_t index : {3, 1}) {
    compiled.executable->set_input(1, &index, 1, ScalarType::Long);
    compiled.executable->execute();
    EXPECT_EQ(
        compiled.executable->output_sizes(0), (std::vector<int64_t>{1, 2}));
    std::vector<float> output(2);
    compiled.executable->get_output(
        0, output.data(), output.size(), ScalarType::Float);
    EXPECT_EQ(output, (std::vector<float>(2, input[index])));
  }
}

TEST(VulkanEngineTest, AddsBufferValuesIntoTextureOutput) {
  CompiledProgram compiled =
      compile_program(make_dynamic_buffer_view_index_program(false, true));
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
  EXPECT_EQ(output, (std::vector<float>{8.0f, 2.0f}));
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

TEST(VulkanEngineTest, ExecutesLinearOnGgufQ4KWeight) {
  constexpr std::array<std::array<uint8_t, 2>, kQ4KOut> kNibbles = {
      {{2, 1}, {3, 1}}};
  CompiledProgram compiled = compile_program(
      make_gguf_q4_k_linear_program(), make_gguf_q4_k_linear_tensors(kNibbles));
  EXPECT_EQ(
      compiled.context->unique_constant_bytes(), kQ4KOut * kQ4KBlockBytes);
  EXPECT_EQ(
      compiled.context->materialized_constant_bytes(),
      kQ4KOut * kQ4KBlockBytes);
  // +1 for weights from low nibbles, -1 for those from high nibbles.
  std::vector<float> input(kQ4KIn);
  for (int64_t k = 0; k < kQ4KIn; ++k) {
    input[k] = k % 64 < 32 ? 1.0f : -1.0f;
  }
  compiled.executable->set_input(
      0, input.data(), input.size(), ScalarType::Float);
  compiled.executable->execute();

  std::vector<float> output(kQ4KOut);
  compiled.executable->get_output(
      0, output.data(), output.size(), ScalarType::Float);
  // Every group contributes 32 low-nibble weights minus 32 high-nibble ones.
  std::vector<float> expected(kQ4KOut, 0.0f);
  for (int64_t n = 0; n < kQ4KOut; ++n) {
    for (int group = 0; group < 4; ++group) {
      expected[n] += 32.0f *
          (q4_k_nibble(kNibbles[n][0], group, false) -
           q4_k_nibble(kNibbles[n][1], group, true));
    }
  }
  EXPECT_EQ(output, expected);
}

TEST(VulkanEngineTest, RejectsGgufQ4KWeightOfWrongSize) {
  const std::string header =
      R"({"weight":{"dtype":"U8","shape":[287],"data_offsets":[0,287]}})";
  try {
    compile_program(
        make_gguf_q4_k_linear_program(),
        testing::make_safetensors(header, 287));
    FAIL() << "expected the truncated Q4_K weight to be rejected";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(
        std::string(e.what()).find("does not hold its declared shape"),
        std::string::npos)
        << e.what();
  }
}

TEST(VulkanEngineTest, KeepsNonKQuantPackedConstantsQuantized) {
  try {
    compile_program(
        make_gguf_q4_k_linear_program("gguf:q8_0"),
        make_gguf_q4_k_linear_tensors({{{2, 1}, {3, 1}}}));
    FAIL() << "expected the gguf:q8_0 weight to stay quantized";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("does not lower"), std::string::npos)
        << e.what();
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

TEST(VulkanEngineTest, ExecutesTransposedConvolutionWithoutBias) {
  CompiledProgram compiled = compile_program(
      make_transposed_conv_program(), make_transposed_conv_tensors());
  std::vector<float> input(kConvIn * kConvInSize * kConvInSize);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = 0.5f * static_cast<float>(i) - 1.0f;
  }
  compiled.executable->set_input(
      0, input.data(), input.size(), ScalarType::Float);
  compiled.executable->execute();

  std::vector<float> output(kConvOut * kConvOutSize * kConvOutSize);
  compiled.executable->get_output(
      0, output.data(), output.size(), ScalarType::Float);
  for (int64_t co = 0; co < kConvOut; ++co) {
    for (int64_t y = 0; y < kConvOutSize; ++y) {
      for (int64_t x = 0; x < kConvOutSize; ++x) {
        float expected = 0.0f;
        for (int64_t ci = 0; ci < kConvIn; ++ci) {
          for (int64_t ky = 0; ky < kConvKernel; ++ky) {
            for (int64_t kx = 0; kx < kConvKernel; ++kx) {
              const int64_t iy = y - ky;
              const int64_t ix = x - kx;
              if (iy >= 0 && iy < kConvInSize && ix >= 0 && ix < kConvInSize) {
                expected += input[(ci * kConvInSize + iy) * kConvInSize + ix] *
                    conv_weight(ci, co, ky, kx);
              }
            }
          }
        }
        EXPECT_NEAR(
            output[(co * kConvOutSize + y) * kConvOutSize + x], expected, 1e-4f)
            << "channel " << co << " at (" << y << ", " << x << ")";
      }
    }
  }
}

TEST(VulkanEngineTest, RejectsQuantizedGraphInput) {
  try {
    compile_program(
        make_quantized_input_program(), make_quantized_input_tensors());
    FAIL() << "expected the quantized graph input to be rejected";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(
        std::string(e.what()).find(
            "quantized graph inputs and outputs are not supported"),
        std::string::npos)
        << e.what();
  }
}

TEST(VulkanEngineTest, CacheFreeSdpaPreservesMasksWithTextureAndBufferOutputs) {
  for (const auto dtype : {fbs::ScalarType::BOOL, fbs::ScalarType::FLOAT}) {
    SCOPED_TRACE(static_cast<int>(dtype));
    for (const bool index_output : {false, true}) {
      SCOPED_TRACE(index_output);
      for (const int64_t keys : {1, 8, 9}) {
        SCOPED_TRACE(keys);
        CompiledProgram compiled = compile_program(
            make_cache_free_sdpa_program(dtype, keys, index_output));
        std::vector<float> query(8, 0.0f);
        std::vector<float> key(keys * 4, 0.0f);
        std::vector<float> value(keys * 4);
        for (size_t i = 0; i < value.size(); ++i) {
          value[i] = static_cast<float>(i + 1);
        }
        compiled.executable->set_input(
            0, query.data(), query.size(), ScalarType::Float);
        compiled.executable->set_input(
            1, key.data(), key.size(), ScalarType::Float);
        compiled.executable->set_input(
            2, value.data(), value.size(), ScalarType::Float);
        if (index_output) {
          const int64_t index = 0;
          compiled.executable->set_input(4, &index, 1, ScalarType::Long);
        }
        // Exercise repeated execution with fully masked, partially masked,
        // and unmasked rows, including both complete and partial texels.
        for (const int pattern : {0, 1, 2}) {
          SCOPED_TRACE(pattern);
          std::vector<uint8_t> keep(2 * keys);
          std::vector<float> bias(2 * keys);
          std::vector<float> expected(8, 0.0f);
          for (int64_t row = 0; row < 2; ++row) {
            int count = 0;
            for (int64_t k = 0; k < keys; ++k) {
              const bool attend =
                  pattern == 2 || (pattern == 1 && row == 0 && k % 2 == 0);
              keep[row * keys + k] = attend;
              bias[row * keys + k] =
                  attend ? 0.0f : -std::numeric_limits<float>::infinity();
              if (attend) {
                ++count;
                for (int d = 0; d < 4; ++d) {
                  expected[row * 4 + d] += value[k * 4 + d];
                }
              }
            }
            if (count > 0) {
              for (int d = 0; d < 4; ++d) {
                expected[row * 4 + d] /= count;
              }
            }
          }
          if (dtype == fbs::ScalarType::BOOL) {
            compiled.executable->set_input(
                3, keep.data(), keep.size(), ScalarType::Bool);
          } else {
            compiled.executable->set_input(
                3, bias.data(), bias.size(), ScalarType::Float);
          }
          compiled.executable->execute();
          std::vector<float> output(8);
          compiled.executable->get_output(
              0, output.data(), output.size(), ScalarType::Float);
          for (size_t i = 0; i < output.size(); ++i) {
            EXPECT_NEAR(output[i], expected[i], 1e-4f) << i;
          }
        }
      }
    }
  }
}

TEST(VulkanEngineTest, RejectsHFAttentionWithNonzeroKvOffset) {
  try {
    compile_program(make_hf_attention_program(), make_hf_attention_tensors(1));
    FAIL() << "expected the nonzero kv_offset to be rejected";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(
        std::string(e.what()).find("constant 'kv_offset' must all be zero"),
        std::string::npos)
        << e.what();
  }
}

TEST(VulkanEngineTest, ScalesLearnedPositionAttentionWithDirectQueryInput) {
  CompiledProgram compiled = compile_program(
      make_hf_attention_program(false), make_hf_attention_tensors(0));
  const std::vector<int64_t> positions = {0, 1, 2, 3};
  std::vector<float> q(kHfTokens * kHfHeads * kHfDim);
  std::vector<float> k(kHfTokens * kHfKvHeads * kHfDim);
  std::vector<float> v(k.size());
  for (size_t i = 0; i < q.size(); ++i) {
    q[i] = sdpa_input_value(i, 5);
  }
  for (size_t i = 0; i < k.size(); ++i) {
    k[i] = sdpa_input_value(i, 7);
    v[i] = sdpa_input_value(i, 2);
  }
  compiled.executable->set_input(0, q.data(), q.size(), ScalarType::Float);
  compiled.executable->set_input(1, k.data(), k.size(), ScalarType::Float);
  compiled.executable->set_input(2, v.data(), v.size(), ScalarType::Float);
  compiled.executable->set_input(
      3, positions.data(), positions.size(), ScalarType::Long);
  compiled.executable->execute();
  std::vector<float> output(q.size());
  compiled.executable->get_output(
      0, output.data(), output.size(), ScalarType::Float);

  for (int64_t t = 0; t < kHfTokens; ++t) {
    for (int64_t h = 0; h < kHfHeads; ++h) {
      const int64_t kv_head = h / (kHfHeads / kHfKvHeads);
      std::vector<float> weights(t + 1, 0.0f);
      for (int64_t s = 0; s <= t; ++s) {
        for (int64_t d = 0; d < kHfDim; ++d) {
          weights[s] += q[(t * kHfHeads + h) * kHfDim + d] *
              k[(s * kHfKvHeads + kv_head) * kHfDim + d];
        }
      }
      const float max_score = *std::max_element(weights.begin(), weights.end());
      float sum = 0.0f;
      for (float& weight : weights) {
        weight = std::exp(weight - max_score);
        sum += weight;
      }
      for (int64_t d = 0; d < kHfDim; ++d) {
        float expected = 0.0f;
        for (int64_t s = 0; s <= t; ++s) {
          expected +=
              weights[s] / sum * v[(s * kHfKvHeads + kv_head) * kHfDim + d];
        }
        EXPECT_NEAR(output[(t * kHfHeads + h) * kHfDim + d], expected, 1e-4f)
            << "token " << t << ", head " << h << ", dimension " << d;
      }
    }
  }
}

TEST(VulkanEngineTest, CountsHFAttentionCacheRowsOnlyAfterExecute) {
  CompiledProgram compiled = compile_program(
      make_hf_attention_program(), make_hf_attention_tensors(0));
  const std::vector<int64_t> prefill = {0, 1, 2, 3};
  const std::vector<int64_t> next = {4, 5, 6, 7};
  std::vector<float> q(kHfTokens * kHfHeads * kHfDim);
  std::vector<float> kv(kHfTokens * kHfKvHeads * kHfDim);
  for (size_t i = 0; i < q.size(); ++i) {
    q[i] = sdpa_input_value(i, 5);
  }
  for (size_t i = 0; i < kv.size(); ++i) {
    kv[i] = sdpa_input_value(i, 7);
  }

  // Setting a position input without executing writes no cache rows.
  compiled.executable->set_input(
      3, prefill.data(), prefill.size(), ScalarType::Long);
  EXPECT_THROW(
      compiled.executable->set_input(
          3, next.data(), next.size(), ScalarType::Long),
      std::runtime_error);

  compiled.executable->set_input(
      3, prefill.data(), prefill.size(), ScalarType::Long);
  compiled.executable->set_input(0, q.data(), q.size(), ScalarType::Float);
  compiled.executable->set_input(1, kv.data(), kv.size(), ScalarType::Float);
  compiled.executable->set_input(2, kv.data(), kv.size(), ScalarType::Float);
  compiled.executable->execute();
  EXPECT_NO_THROW(compiled.executable->set_input(
      3, next.data(), next.size(), ScalarType::Long));
}
// cppcheck-suppress-end syntaxError

} // namespace
} // namespace ptn
