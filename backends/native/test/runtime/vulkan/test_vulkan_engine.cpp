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
#include <numeric>
#include <string>
#include <tuple>
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
#include <executorch/backends/vulkan/runtime/vk_api/Runtime.h>

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

// relu of a view of `input`, by default a static [1, 2, 3, 4] view of a
// [1, 6, 4] input. Nothing requires width packing, so both keep the default
// channels-packed layout.
struct ViewRelu {
  std::vector<int64_t> input_sizes = {1, 6, 4};
  std::vector<int64_t> view_sizes = {1, 2, 3, 4};
  std::vector<int64_t> size_arg = {1, 2, 3, 4};
  // Nonempty for a dynamically shaped input and view.
  std::vector<int64_t> input_lower_bounds;
  std::vector<int64_t> view_lower_bounds;
};

std::vector<uint8_t> make_view_relu_program(const ViewRelu& shape = {}) {
  flatbuffers::FlatBufferBuilder builder;
  const auto input_meta = create_tensor_meta(
      builder,
      fbs::ScalarType::FLOAT,
      shape.input_sizes,
      shape.input_lower_bounds);
  const auto view_meta = create_tensor_meta(
      builder,
      fbs::ScalarType::FLOAT,
      shape.view_sizes,
      shape.view_lower_bounds);
  const std::vector<flatbuffers::Offset<fbs::TensorValue>> tensor_values = {
      fbs::CreateTensorValueDirect(builder, "input", input_meta),
      fbs::CreateTensorValueDirect(builder, "view", view_meta),
      fbs::CreateTensorValueDirect(builder, "relu", view_meta),
  };
  const std::vector<flatbuffers::Offset<fbs::Output>> input_outputs = {
      fbs::CreateOutputDirect(builder, "input")};
  const std::vector<flatbuffers::Offset<fbs::Output>> view_outputs = {
      fbs::CreateOutputDirect(builder, "view", "input")};
  const std::vector<flatbuffers::Offset<fbs::Output>> relu_outputs = {
      fbs::CreateOutputDirect(builder, "relu")};
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> view_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "self", create_tensor_arg(builder, "input")),
      fbs::CreateNamedArgumentDirect(
          builder, "size", create_int_list_arg(builder, shape.size_arg)),
  };
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> relu_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "self", create_tensor_arg(builder, "view"))};
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
          "view",
          fbs::OpKind::CALL_FUNCTION,
          "torch.ops.aten.view.default",
          &view_inputs,
          &view_outputs),
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

// permute_copy of a [1, 1, rows, 4] input to [1, rows, 1, 4], both
// channels-packed.
std::vector<uint8_t> make_permute_copy_program(int64_t rows) {
  flatbuffers::FlatBufferBuilder builder;
  const std::vector<flatbuffers::Offset<fbs::TensorValue>> tensor_values = {
      fbs::CreateTensorValueDirect(
          builder,
          "input",
          create_tensor_meta(builder, fbs::ScalarType::FLOAT, {1, 1, rows, 4})),
      fbs::CreateTensorValueDirect(
          builder,
          "permuted",
          create_tensor_meta(builder, fbs::ScalarType::FLOAT, {1, rows, 1, 4})),
  };
  const std::vector<flatbuffers::Offset<fbs::Output>> input_outputs = {
      fbs::CreateOutputDirect(builder, "input")};
  const std::vector<flatbuffers::Offset<fbs::Output>> permute_outputs = {
      fbs::CreateOutputDirect(builder, "permuted")};
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> permute_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "self", create_tensor_arg(builder, "input")),
      fbs::CreateNamedArgumentDirect(
          builder, "dims", create_int_list_arg(builder, {0, 2, 1, 3})),
  };
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> output_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "", create_tensor_arg(builder, "permuted"))};
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
          "permute",
          fbs::OpKind::CALL_FUNCTION,
          "torch.ops.aten.permute_copy.default",
          &permute_inputs,
          &permute_outputs),
      fbs::CreateNodeDirect(
          builder, "output", fbs::OpKind::OUTPUT, "", &output_inputs),
  };
  const auto graph = fbs::CreateGraph(
      builder,
      builder.CreateVector(nodes),
      create_strings(builder, {"input"}),
      create_strings(builder, {"permuted"}),
      builder.CreateVector(tensor_values));
  return finish_program(
      builder, graph, {fbs::CreateOutputSpecDirect(builder, "permuted")});
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

// The 2D mean over dims 1 and 3 needs a height-packed input, so the permute's
// output is height-packed while its channels-packed source is not.
std::vector<uint8_t> make_permute_mean_program() {
  flatbuffers::FlatBufferBuilder builder;
  const std::vector<flatbuffers::Offset<fbs::TensorValue>> tensor_values = {
      fbs::CreateTensorValueDirect(
          builder,
          "input",
          create_tensor_meta(builder, fbs::ScalarType::FLOAT, {1, 2, 3, 4})),
      fbs::CreateTensorValueDirect(
          builder,
          "permuted",
          create_tensor_meta(
              builder, fbs::ScalarType::FLOAT, {1, 2, 4, 3}, {}, {0, 1, 3, 2})),
      fbs::CreateTensorValueDirect(
          builder,
          "mean",
          create_tensor_meta(builder, fbs::ScalarType::FLOAT, {1, 1, 4, 1})),
  };
  const std::vector<flatbuffers::Offset<fbs::Output>> input_outputs = {
      fbs::CreateOutputDirect(builder, "input")};
  const std::vector<flatbuffers::Offset<fbs::Output>> permute_outputs = {
      fbs::CreateOutputDirect(builder, "permuted", "input")};
  const std::vector<flatbuffers::Offset<fbs::Output>> mean_outputs = {
      fbs::CreateOutputDirect(builder, "mean")};
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> permute_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "self", create_tensor_arg(builder, "input")),
      fbs::CreateNamedArgumentDirect(
          builder, "dims", create_int_list_arg(builder, {0, 1, 3, 2})),
  };
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> mean_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "self", create_tensor_arg(builder, "permuted")),
      fbs::CreateNamedArgumentDirect(
          builder, "dim", create_int_list_arg(builder, {1, 3})),
      fbs::CreateNamedArgumentDirect(
          builder, "keepdim", create_bool_arg(builder, true)),
      fbs::CreateNamedArgumentDirect(
          builder, "dtype", create_none_arg(builder)),
  };
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> output_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "", create_tensor_arg(builder, "mean"))};
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
          "permute",
          fbs::OpKind::CALL_FUNCTION,
          "torch.ops.aten.permute.default",
          &permute_inputs,
          &permute_outputs),
      fbs::CreateNodeDirect(
          builder,
          "mean",
          fbs::OpKind::CALL_FUNCTION,
          "torch.ops.aten.mean.dim",
          &mean_inputs,
          &mean_outputs),
      fbs::CreateNodeDirect(
          builder, "output", fbs::OpKind::OUTPUT, "", &output_inputs),
  };
  const auto graph = fbs::CreateGraph(
      builder,
      builder.CreateVector(nodes),
      create_strings(builder, {"input"}),
      create_strings(builder, {"mean"}),
      builder.CreateVector(tensor_values));
  return finish_program(
      builder, graph, {fbs::CreateOutputSpecDirect(builder, "mean")});
}

// A view `matrix` of `state`, produced by `target` with `arg` = `values`.
struct StateView {
  const char* target = "torch.ops.aten.view.default";
  const char* arg = "size";
  std::vector<int64_t> values;
  std::vector<int64_t> state_sizes = {2, 4};
  std::vector<int64_t> matrix_sizes = {4, 2};
  std::vector<int32_t> matrix_dim_order;
  // Nonempty for a dynamically shaped state and view.
  std::vector<int64_t> state_lower_bounds;
  std::vector<int64_t> matrix_lower_bounds;
};

// update_cache overwrites the input `state` with the input `value`, and mm then
// reads a width-packed texture view of it.
std::vector<uint8_t> make_state_view_mm_program(const StateView& view) {
  flatbuffers::FlatBufferBuilder builder;
  const auto state_meta = create_tensor_meta(
      builder,
      fbs::ScalarType::FLOAT,
      view.state_sizes,
      view.state_lower_bounds);
  const std::vector<int64_t> product_lower_bounds =
      view.matrix_lower_bounds.empty()
      ? std::vector<int64_t>{}
      : std::vector<int64_t>{view.matrix_lower_bounds[0], 1};
  const std::vector<flatbuffers::Offset<fbs::TensorValue>> tensor_values = {
      fbs::CreateTensorValueDirect(builder, "state", state_meta),
      fbs::CreateTensorValueDirect(builder, "value", state_meta),
      fbs::CreateTensorValueDirect(
          builder,
          "updated",
          create_tensor_meta(builder, fbs::ScalarType::FLOAT, {1})),
      fbs::CreateTensorValueDirect(
          builder,
          "matrix",
          create_tensor_meta(
              builder,
              fbs::ScalarType::FLOAT,
              view.matrix_sizes,
              view.matrix_lower_bounds,
              view.matrix_dim_order)),
      fbs::CreateTensorValueDirect(
          builder,
          "weight",
          create_tensor_meta(
              builder, fbs::ScalarType::FLOAT, {view.matrix_sizes[1], 1})),
      fbs::CreateTensorValueDirect(
          builder,
          "product",
          create_tensor_meta(
              builder,
              fbs::ScalarType::FLOAT,
              {view.matrix_sizes[0], 1},
              product_lower_bounds)),
  };
  const auto outputs = [&](const char* name, const char* alias_of = nullptr) {
    return std::vector<flatbuffers::Offset<fbs::Output>>{
        fbs::CreateOutputDirect(builder, name, alias_of)};
  };
  const auto state_outputs = outputs("state");
  const auto value_outputs = outputs("value");
  const auto weight_outputs = outputs("weight");
  const auto update_outputs = outputs("updated");
  const auto view_outputs = outputs("matrix", "state");
  const auto mm_outputs = outputs("product");
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> update_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "value", create_tensor_arg(builder, "value")),
      fbs::CreateNamedArgumentDirect(
          builder, "cache", create_tensor_arg(builder, "state"), true),
      fbs::CreateNamedArgumentDirect(
          builder, "start_pos", create_int_arg(builder, 0)),
  };
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> view_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "self", create_tensor_arg(builder, "state")),
      fbs::CreateNamedArgumentDirect(
          builder, view.arg, create_int_list_arg(builder, view.values)),
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
  const auto placeholder = [&](const char* name, const auto& node_outputs) {
    return fbs::CreateNodeDirect(
        builder, name, fbs::OpKind::PLACEHOLDER, "", nullptr, &node_outputs);
  };
  const auto call = [&](const char* name,
                        const char* target,
                        const auto& node_inputs,
                        const auto& node_outputs) {
    return fbs::CreateNodeDirect(
        builder,
        name,
        fbs::OpKind::CALL_FUNCTION,
        target,
        &node_inputs,
        &node_outputs);
  };
  const std::vector<flatbuffers::Offset<fbs::Node>> nodes = {
      placeholder("state", state_outputs),
      placeholder("value", value_outputs),
      placeholder("weight", weight_outputs),
      call(
          "updated",
          "torch.ops.llama.update_cache.default",
          update_inputs,
          update_outputs),
      call("view", view.target, view_inputs, view_outputs),
      call("mm", "torch.ops.aten.mm.default", mm_inputs, mm_outputs),
      fbs::CreateNodeDirect(
          builder, "output", fbs::OpKind::OUTPUT, "", &output_inputs),
  };
  const auto graph = fbs::CreateGraph(
      builder,
      builder.CreateVector(nodes),
      create_strings(builder, {"state", "value", "weight"}),
      create_strings(builder, {"product"}),
      builder.CreateVector(tensor_values));
  return finish_program(
      builder, graph, {fbs::CreateOutputSpecDirect(builder, "product")});
}

// index.Tensor keeps its operands in buffer storage; the reshape must preserve
// logical element order even when the source's physical rows contain padding.
std::vector<uint8_t> make_dynamic_buffer_view_index_program(
    bool dynamic = true,
    bool aligned_width = false,
    const std::string& view_target = "torch.ops.aten.view.default") {
  const bool unsqueeze = view_target == "torch.ops.aten.unsqueeze.default";
  const bool squeeze = view_target == "torch.ops.aten.squeeze.dim" ||
      view_target == "torch.ops.aten.squeeze.dims";
  flatbuffers::FlatBufferBuilder builder;
  const auto input_meta = create_tensor_meta(
      builder,
      fbs::ScalarType::FLOAT,
      squeeze ? std::vector<int64_t>{2, 4, 1}
              : (aligned_width || unsqueeze ? std::vector<int64_t>{2, 4}
                                            : std::vector<int64_t>{4, 2}),
      !dynamic ? std::vector<int64_t>{}
               : (squeeze ? std::vector<int64_t>{1, 4, 1}
                          : std::vector<int64_t>{1, 2}));
  const auto rows_meta = create_tensor_meta(
      builder,
      fbs::ScalarType::FLOAT,
      unsqueeze ? std::vector<int64_t>{2, 4, 1} : std::vector<int64_t>{2, 4},
      dynamic ? std::vector<int64_t>{1, 4} : std::vector<int64_t>{});
  const auto index_meta =
      create_tensor_meta(builder, fbs::ScalarType::LONG, {2});
  const auto gathered_meta = create_tensor_meta(
      builder,
      fbs::ScalarType::FLOAT,
      unsqueeze ? std::vector<int64_t>{2, 2, 1} : std::vector<int64_t>{2, 2},
      dynamic ? std::vector<int64_t>{1, 2} : std::vector<int64_t>{});
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
          builder,
          unsqueeze || squeeze ? "dim" : "size",
          unsqueeze || view_target == "torch.ops.aten.squeeze.dim"
              ? create_int_arg(builder, -1)
              : create_int_list_arg(
                    builder,
                    squeeze ? std::vector<int64_t>{-1}
                            : std::vector<int64_t>{-1, 4})),
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
          view_target.c_str(),
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

// update_cache writes a [1, 4, 1, 2] input that index.Tensor then reads through
// a [2, 4] view, whose width padding differs from the input's.
std::vector<uint8_t> make_mutated_buffer_view_program() {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<fbs::TensorValue>> tensor_values;
  std::vector<flatbuffers::Offset<fbs::Node>> nodes;
  const auto tensor = [&](const char* name,
                          fbs::ScalarType dtype,
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

  tensor("cache", fbs::ScalarType::FLOAT, {1, 4, 1, 2});
  tensor("value", fbs::ScalarType::FLOAT, {1, 1, 1, 2});
  tensor("index", fbs::ScalarType::LONG, {2});
  tensor("updated", fbs::ScalarType::FLOAT, {1});
  tensor("rows", fbs::ScalarType::FLOAT, {2, 4});
  tensor("gathered", fbs::ScalarType::FLOAT, {2, 2});
  placeholder("cache");
  placeholder("value");
  placeholder("index");
  call(
      "updated",
      "torch.ops.llama.update_cache.default",
      {arg("value", create_tensor_arg(builder, "value")),
       arg("cache", create_tensor_arg(builder, "cache"), true),
       arg("start_pos", create_int_arg(builder, 0))});
  call(
      "rows",
      "torch.ops.aten.view.default",
      {arg("self", create_tensor_arg(builder, "cache")),
       arg("size", create_int_list_arg(builder, {2, 4}))},
      "cache");
  const auto indices = fbs::CreateOptionalTensorListArg(
      builder,
      create_strings(builder, {"", "index"}),
      builder.CreateVector(std::vector<uint8_t>{0, 1}));
  call(
      "gathered",
      "torch.ops.aten.index.Tensor",
      {arg("self", create_tensor_arg(builder, "rows")),
       arg("indices",
           fbs::CreateArgument(
               builder,
               fbs::ArgumentValue::OptionalTensorListArg,
               indices.Union()))});
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> output_inputs = {
      arg("", create_tensor_arg(builder, "gathered"))};
  nodes.push_back(fbs::CreateNodeDirect(
      builder, "output", fbs::OpKind::OUTPUT, "", &output_inputs));
  const auto graph = fbs::CreateGraph(
      builder,
      builder.CreateVector(nodes),
      create_strings(builder, {"cache", "value", "index"}),
      create_strings(builder, {"gathered"}),
      builder.CreateVector(tensor_values));
  return finish_program(
      builder, graph, {fbs::CreateOutputSpecDirect(builder, "gathered")});
}

// index.Tensor gathers columns of a static [rows, cols] buffer input through a
// permute or transpose view of it.
std::vector<uint8_t> make_static_buffer_transpose_index_program(
    int64_t rows,
    int64_t cols,
    const std::string& view_target) {
  flatbuffers::FlatBufferBuilder builder;
  const auto input_meta =
      create_tensor_meta(builder, fbs::ScalarType::FLOAT, {rows, cols});
  const auto transposed_meta = create_tensor_meta(
      builder, fbs::ScalarType::FLOAT, {cols, rows}, {}, {1, 0});
  const auto index_meta =
      create_tensor_meta(builder, fbs::ScalarType::LONG, {2});
  const auto gathered_meta =
      create_tensor_meta(builder, fbs::ScalarType::FLOAT, {cols, 2});
  const std::vector<flatbuffers::Offset<fbs::TensorValue>> tensor_values = {
      fbs::CreateTensorValueDirect(builder, "input", input_meta),
      fbs::CreateTensorValueDirect(builder, "index", index_meta),
      fbs::CreateTensorValueDirect(builder, "transposed", transposed_meta),
      fbs::CreateTensorValueDirect(builder, "gathered", gathered_meta),
  };

  const std::vector<flatbuffers::Offset<fbs::Output>> input_outputs = {
      fbs::CreateOutputDirect(builder, "input")};
  const std::vector<flatbuffers::Offset<fbs::Output>> index_outputs = {
      fbs::CreateOutputDirect(builder, "index")};
  const std::vector<flatbuffers::Offset<fbs::Output>> view_outputs = {
      fbs::CreateOutputDirect(builder, "transposed", "input")};
  const std::vector<flatbuffers::Offset<fbs::Output>> gathered_outputs = {
      fbs::CreateOutputDirect(builder, "gathered")};
  std::vector<flatbuffers::Offset<fbs::NamedArgument>> view_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "self", create_tensor_arg(builder, "input"))};
  if (view_target == "torch.ops.aten.permute.default") {
    view_inputs.push_back(fbs::CreateNamedArgumentDirect(
        builder, "dims", create_int_list_arg(builder, {1, 0})));
  } else {
    view_inputs.push_back(fbs::CreateNamedArgumentDirect(
        builder, "dim0", create_int_arg(builder, 0)));
    view_inputs.push_back(fbs::CreateNamedArgumentDirect(
        builder, "dim1", create_int_arg(builder, -1)));
  }
  const auto indices = fbs::CreateOptionalTensorListArg(
      builder,
      create_strings(builder, {"", "index"}),
      builder.CreateVector(std::vector<uint8_t>{0, 1}));
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> index_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "self", create_tensor_arg(builder, "transposed")),
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
          view_target.c_str(),
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

// Cache-free SDPA over query [B, H, S, D] and key/value [B, H, L, D]. An
// identity index on dimension 0 after SDPA places its output in buffer storage;
// one before it places key and value there while the output stays in texture.
struct CacheFreeSdpaCase {
  int64_t batch = 2;
  int64_t heads = 2;
  int64_t queries = 3;
  int64_t keys = 9;
  // When nonzero, key, value and mask are dynamic along the key axis up to
  // max_keys, and run resized to `keys`.
  int64_t max_keys = 0;
  int64_t dim = 8;
  // Of query, key, value and output.
  fbs::ScalarType dtype = fbs::ScalarType::FLOAT;
  fbs::ScalarType mask_dtype = fbs::ScalarType::BOOL;
  std::vector<int64_t> mask_sizes;
  bool constant_mask = false;
  bool index_output = false;
  bool index_key_value = false;
  // Key and value arrive as [B, L, H * D] projections, viewed as [B, L, H, D]
  // and permuted to [B, H, L, D].
  bool permute_key_value = false;

  bool indexed() const {
    return index_output || index_key_value;
  }
};

int64_t numel_of(const std::vector<int64_t>& sizes) {
  int64_t numel = 1;
  for (const int64_t size : sizes) {
    numel *= size;
  }
  return numel;
}

// Masks key positions pseudo-randomly, and masks a whole query row when the
// mask has a query axis.
bool cache_free_sdpa_keeps(const CacheFreeSdpaCase& c, int64_t i, int salt) {
  const bool has_query_axis =
      c.mask_sizes.size() >= 2 && c.mask_sizes[c.mask_sizes.size() - 2] > 1;
  return !(has_query_axis && i < c.keys) && (i * salt + salt) % 5 != 0;
}

// The additive bias the mask stands for, one entry per mask element.
std::vector<float> cache_free_sdpa_bias(const CacheFreeSdpaCase& c, int salt) {
  std::vector<float> bias(numel_of(c.mask_sizes));
  for (int64_t i = 0; i < static_cast<int64_t>(bias.size()); ++i) {
    const bool keep = cache_free_sdpa_keeps(c, i, salt);
    if (!keep) {
      bias[i] = -std::numeric_limits<float>::infinity();
    } else if (c.mask_dtype != fbs::ScalarType::BOOL) {
      bias[i] = 0.25f * static_cast<float>((i * 3) % 5) - 0.5f;
    }
  }
  return bias;
}

std::vector<uint8_t> cache_free_sdpa_keep(
    const CacheFreeSdpaCase& c,
    int salt) {
  std::vector<uint8_t> keep(numel_of(c.mask_sizes));
  for (int64_t i = 0; i < static_cast<int64_t>(keep.size()); ++i) {
    keep[i] = cache_free_sdpa_keeps(c, i, salt);
  }
  return keep;
}

std::vector<uint8_t> make_cache_free_sdpa_program(const CacheFreeSdpaCase& c) {
  flatbuffers::FlatBufferBuilder builder;
  const int64_t max_keys = c.max_keys == 0 ? c.keys : c.max_keys;
  const auto key_axis_meta = [&](fbs::ScalarType dtype,
                                 const std::vector<int64_t>& sizes,
                                 size_t key_axis,
                                 const std::vector<int32_t>& dim_order = {}) {
    std::vector<int64_t> lower_bounds;
    if (c.max_keys != 0) {
      lower_bounds = sizes;
      lower_bounds.at(key_axis) = 1;
    }
    return create_tensor_meta(builder, dtype, sizes, lower_bounds, dim_order);
  };
  const auto query_meta = create_tensor_meta(
      builder, c.dtype, {c.batch, c.heads, c.queries, c.dim});
  const auto kv_meta = key_axis_meta(
      c.dtype,
      {c.batch, c.heads, max_keys, c.dim},
      2,
      c.permute_key_value ? std::vector<int32_t>{0, 2, 1, 3}
                          : std::vector<int32_t>{});
  std::vector<int64_t> mask_sizes = c.mask_sizes;
  mask_sizes.back() = max_keys;
  const auto mask_meta =
      key_axis_meta(c.mask_dtype, mask_sizes, mask_sizes.size() - 1);
  std::vector<flatbuffers::Offset<fbs::TensorValue>> tensor_values = {
      fbs::CreateTensorValueDirect(builder, "query", query_meta),
      fbs::CreateTensorValueDirect(builder, "key", kv_meta),
      fbs::CreateTensorValueDirect(builder, "value", kv_meta),
      fbs::CreateTensorValueDirect(builder, "attention", query_meta),
  };
  std::vector<std::string> input_names = {"query"};
  if (c.index_key_value || c.permute_key_value) {
    const auto src_meta = c.permute_key_value
        ? create_tensor_meta(
              builder, c.dtype, {c.batch, c.keys, c.heads * c.dim})
        : kv_meta;
    input_names.insert(input_names.end(), {"key_src", "value_src"});
    tensor_values.push_back(
        fbs::CreateTensorValueDirect(builder, "key_src", src_meta));
    tensor_values.push_back(
        fbs::CreateTensorValueDirect(builder, "value_src", src_meta));
  } else {
    input_names.insert(input_names.end(), {"key", "value"});
  }
  if (!c.constant_mask) {
    input_names.emplace_back("mask");
    tensor_values.push_back(
        fbs::CreateTensorValueDirect(builder, "mask", mask_meta));
  }
  if (c.indexed()) {
    input_names.emplace_back("index");
    tensor_values.push_back(fbs::CreateTensorValueDirect(
        builder,
        "index",
        create_tensor_meta(builder, fbs::ScalarType::LONG, {c.batch})));
  }
  if (c.index_output) {
    tensor_values.push_back(
        fbs::CreateTensorValueDirect(builder, "gathered", query_meta));
  }
  if (c.permute_key_value) {
    const auto tokens_meta =
        create_tensor_meta(builder, c.dtype, {c.batch, c.keys, c.heads, c.dim});
    tensor_values.push_back(
        fbs::CreateTensorValueDirect(builder, "key_tokens", tokens_meta));
    tensor_values.push_back(
        fbs::CreateTensorValueDirect(builder, "value_tokens", tokens_meta));
  }

  std::vector<flatbuffers::Offset<fbs::Node>> nodes;
  const auto add_placeholder = [&](const std::string& name) {
    const std::vector<flatbuffers::Offset<fbs::Output>> outputs = {
        fbs::CreateOutputDirect(builder, name.c_str())};
    nodes.push_back(fbs::CreateNodeDirect(
        builder,
        name.c_str(),
        fbs::OpKind::PLACEHOLDER,
        "",
        nullptr,
        &outputs));
  };
  for (const auto& name : input_names) {
    add_placeholder(name);
  }
  if (c.constant_mask) {
    add_placeholder("mask");
  }
  const auto add_index = [&](const char* self, const char* output) {
    const auto indices = fbs::CreateOptionalTensorListArg(
        builder,
        create_strings(builder, {"index"}),
        builder.CreateVector(std::vector<uint8_t>{1}));
    const std::vector<flatbuffers::Offset<fbs::NamedArgument>> inputs = {
        fbs::CreateNamedArgumentDirect(
            builder, "self", create_tensor_arg(builder, self)),
        fbs::CreateNamedArgumentDirect(
            builder,
            "indices",
            fbs::CreateArgument(
                builder,
                fbs::ArgumentValue::OptionalTensorListArg,
                indices.Union())),
    };
    const std::vector<flatbuffers::Offset<fbs::Output>> outputs = {
        fbs::CreateOutputDirect(builder, output)};
    nodes.push_back(fbs::CreateNodeDirect(
        builder,
        output,
        fbs::OpKind::CALL_FUNCTION,
        "torch.ops.aten.index.Tensor",
        &inputs,
        &outputs));
  };
  if (c.index_key_value) {
    add_index("key_src", "key");
    add_index("value_src", "value");
  }
  // A view op whose output aliases `self`.
  const auto add_view = [&](const char* target,
                            const char* self,
                            const char* output,
                            const char* arg,
                            const std::vector<int64_t>& values) {
    const std::vector<flatbuffers::Offset<fbs::NamedArgument>> inputs = {
        fbs::CreateNamedArgumentDirect(
            builder, "self", create_tensor_arg(builder, self)),
        fbs::CreateNamedArgumentDirect(
            builder, arg, create_int_list_arg(builder, values)),
    };
    const std::vector<flatbuffers::Offset<fbs::Output>> outputs = {
        fbs::CreateOutputDirect(builder, output, self)};
    nodes.push_back(fbs::CreateNodeDirect(
        builder,
        output,
        fbs::OpKind::CALL_FUNCTION,
        target,
        &inputs,
        &outputs));
  };
  if (c.permute_key_value) {
    for (const auto& [src, tokens, heads_major] :
         {std::tuple{"key_src", "key_tokens", "key"},
          std::tuple{"value_src", "value_tokens", "value"}}) {
      add_view(
          "torch.ops.aten.view.default",
          src,
          tokens,
          "size",
          {c.batch, c.keys, c.heads, c.dim});
      add_view(
          "torch.ops.aten.permute.default",
          tokens,
          heads_major,
          "dims",
          {0, 2, 1, 3});
    }
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
  nodes.push_back(fbs::CreateNodeDirect(
      builder,
      "sdpa",
      fbs::OpKind::CALL_FUNCTION,
      "torch.ops.aten.scaled_dot_product_attention.default",
      &sdpa_inputs,
      &sdpa_outputs));
  if (c.index_output) {
    add_index("attention", "gathered");
  }
  const char* output_name = c.index_output ? "gathered" : "attention";
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> output_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "", create_tensor_arg(builder, output_name))};
  nodes.push_back(fbs::CreateNodeDirect(
      builder, "output", fbs::OpKind::OUTPUT, "", &output_inputs));
  const auto graph = fbs::CreateGraph(
      builder,
      builder.CreateVector(nodes),
      create_strings(builder, input_names),
      create_strings(builder, {output_name}),
      builder.CreateVector(tensor_values));
  std::vector<flatbuffers::Offset<fbs::NamedTensorRef>> constants;
  if (c.constant_mask) {
    constants.push_back(fbs::CreateNamedTensorRefDirect(
        builder, "mask", "mask", mask_meta, fbs::InputKind::CONSTANT_TENSOR));
  }
  return finish_program(
      builder,
      graph,
      {fbs::CreateOutputSpecDirect(builder, output_name)},
      constants);
}

// The stored mask of make_cache_free_sdpa_program when it is a constant.
std::vector<uint8_t> make_cache_free_sdpa_tensors(
    const CacheFreeSdpaCase& c,
    int salt) {
  std::vector<uint8_t> payload;
  if (c.mask_dtype == fbs::ScalarType::BOOL) {
    payload = cache_free_sdpa_keep(c, salt);
  } else {
    const std::vector<float> bias = cache_free_sdpa_bias(c, salt);
    payload.resize(bias.size() * sizeof(float));
    std::memcpy(payload.data(), bias.data(), payload.size());
  }
  std::string shape;
  for (const int64_t size : c.mask_sizes) {
    shape += (shape.empty() ? "" : ",") + std::to_string(size);
  }
  const std::string header = std::string(R"({"mask":{"dtype":")") +
      (c.mask_dtype == fbs::ScalarType::BOOL ? "BOOL" : "F32") +
      R"(","shape":[)" + shape + R"(],"data_offsets":[0,)" +
      std::to_string(payload.size()) + "]}}";
  std::vector<uint8_t> tensors =
      testing::make_safetensors(header, payload.size());
  std::memcpy(
      tensors.data() + sizeof(uint64_t) + header.size(),
      payload.data(),
      payload.size());
  return tensors;
}

// softmax(q k^T / sqrt(D) + bias) v, with rows masked entirely set to zero as
// in ATen. `bias` broadcasts against [B, H, S, L] from the right.
std::vector<float> cache_free_sdpa_reference(
    const CacheFreeSdpaCase& c,
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value,
    const std::vector<float>& bias) {
  const std::array<int64_t, 4> score_sizes{c.batch, c.heads, c.queries, c.keys};
  const size_t offset = score_sizes.size() - c.mask_sizes.size();
  const float scale = 1.0f / std::sqrt(static_cast<float>(c.dim));
  std::vector<float> output(query.size(), 0.0f);
  for (int64_t b = 0; b < c.batch; ++b) {
    for (int64_t h = 0; h < c.heads; ++h) {
      const int64_t bh = b * c.heads + h;
      for (int64_t s = 0; s < c.queries; ++s) {
        const float* q = &query[(bh * c.queries + s) * c.dim];
        std::vector<float> scores(c.keys);
        float max_score = -std::numeric_limits<float>::infinity();
        for (int64_t l = 0; l < c.keys; ++l) {
          const std::array<int64_t, 4> coords{b, h, s, l};
          int64_t mask_index = 0;
          for (size_t i = 0; i < c.mask_sizes.size(); ++i) {
            mask_index = mask_index * c.mask_sizes[i] +
                (c.mask_sizes[i] == 1 ? 0 : coords[offset + i]);
          }
          const float* k = &key[(bh * c.keys + l) * c.dim];
          float dot = 0.0f;
          for (int64_t d = 0; d < c.dim; ++d) {
            dot += q[d] * k[d];
          }
          scores[l] = dot * scale + bias[mask_index];
          max_score = std::max(max_score, scores[l]);
        }
        if (std::isinf(max_score)) {
          continue;
        }
        float total = 0.0f;
        for (float& score : scores) {
          score = std::exp(score - max_score);
          total += score;
        }
        float* out = &output[(bh * c.queries + s) * c.dim];
        for (int64_t l = 0; l < c.keys; ++l) {
          for (int64_t d = 0; d < c.dim; ++d) {
            out[d] += scores[l] / total * value[(bh * c.keys + l) * c.dim + d];
          }
        }
      }
    }
  }
  return output;
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

// sum = input + _to_copy(offsets, float), where `offsets` is a stored int64
// constant.
std::vector<uint8_t> make_stored_long_to_copy_program() {
  flatbuffers::FlatBufferBuilder builder;
  const auto float_meta =
      create_tensor_meta(builder, fbs::ScalarType::FLOAT, {4});
  const auto long_meta =
      create_tensor_meta(builder, fbs::ScalarType::LONG, {4});
  const std::vector<flatbuffers::Offset<fbs::TensorValue>> tensor_values = {
      fbs::CreateTensorValueDirect(builder, "input", float_meta),
      fbs::CreateTensorValueDirect(builder, "offsets", long_meta),
      fbs::CreateTensorValueDirect(builder, "converted", float_meta),
      fbs::CreateTensorValueDirect(builder, "sum", float_meta),
  };
  const std::vector<flatbuffers::Offset<fbs::Output>> input_outputs = {
      fbs::CreateOutputDirect(builder, "input")};
  const std::vector<flatbuffers::Offset<fbs::Output>> offsets_outputs = {
      fbs::CreateOutputDirect(builder, "offsets")};
  const std::vector<flatbuffers::Offset<fbs::Output>> converted_outputs = {
      fbs::CreateOutputDirect(builder, "converted")};
  const std::vector<flatbuffers::Offset<fbs::Output>> sum_outputs = {
      fbs::CreateOutputDirect(builder, "sum")};
  const auto dtype = fbs::CreateScalarTypeArg(builder, fbs::ScalarType::FLOAT);
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> to_copy_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "self", create_tensor_arg(builder, "offsets")),
      fbs::CreateNamedArgumentDirect(
          builder,
          "dtype",
          fbs::CreateArgument(
              builder, fbs::ArgumentValue::ScalarTypeArg, dtype.Union())),
      fbs::CreateNamedArgumentDirect(
          builder, "layout", create_none_arg(builder)),
      fbs::CreateNamedArgumentDirect(
          builder, "device", create_none_arg(builder)),
      fbs::CreateNamedArgumentDirect(
          builder, "pin_memory", create_none_arg(builder)),
      fbs::CreateNamedArgumentDirect(
          builder, "non_blocking", create_bool_arg(builder, false)),
      fbs::CreateNamedArgumentDirect(
          builder, "memory_format", create_none_arg(builder)),
  };
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> add_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "self", create_tensor_arg(builder, "input")),
      fbs::CreateNamedArgumentDirect(
          builder, "other", create_tensor_arg(builder, "converted")),
      fbs::CreateNamedArgumentDirect(
          builder, "alpha", create_int_arg(builder, 1)),
  };
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> output_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "", create_tensor_arg(builder, "sum"))};
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
          "offsets",
          fbs::OpKind::PLACEHOLDER,
          "",
          nullptr,
          &offsets_outputs),
      fbs::CreateNodeDirect(
          builder,
          "converted",
          fbs::OpKind::CALL_FUNCTION,
          "torch.ops.aten._to_copy.default",
          &to_copy_inputs,
          &converted_outputs),
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
      create_strings(builder, {"input"}),
      create_strings(builder, {"sum"}),
      builder.CreateVector(tensor_values));
  const std::vector<flatbuffers::Offset<fbs::NamedTensorRef>> constants = {
      fbs::CreateNamedTensorRefDirect(
          builder,
          "offsets",
          "offsets",
          long_meta,
          fbs::InputKind::CONSTANT_TENSOR),
  };
  return finish_program(
      builder, graph, {fbs::CreateOutputSpecDirect(builder, "sum")}, constants);
}

// sum = input + op(bias), where `bias` is a stored constant and op is clone,
// or with `clamp` a clamp to [-1, 50].
std::vector<uint8_t> make_stored_operand_program(bool clamp) {
  flatbuffers::FlatBufferBuilder builder;
  const auto meta = create_tensor_meta(builder, fbs::ScalarType::FLOAT, {4});
  const std::vector<flatbuffers::Offset<fbs::TensorValue>> tensor_values = {
      fbs::CreateTensorValueDirect(builder, "input", meta),
      fbs::CreateTensorValueDirect(builder, "bias", meta),
      fbs::CreateTensorValueDirect(builder, "operand", meta),
      fbs::CreateTensorValueDirect(builder, "sum", meta),
  };
  std::vector<flatbuffers::Offset<fbs::Node>> nodes;
  for (const char* name : {"input", "bias"}) {
    const std::vector<flatbuffers::Offset<fbs::Output>> outputs = {
        fbs::CreateOutputDirect(builder, name)};
    nodes.push_back(fbs::CreateNodeDirect(
        builder, name, fbs::OpKind::PLACEHOLDER, "", nullptr, &outputs));
  }
  const auto float_arg = [&](double value) {
    return fbs::CreateArgument(
        builder,
        fbs::ArgumentValue::FloatArg,
        fbs::CreateFloatArg(builder, value).Union());
  };
  std::vector<flatbuffers::Offset<fbs::NamedArgument>> op_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "self", create_tensor_arg(builder, "bias"))};
  if (clamp) {
    op_inputs.push_back(
        fbs::CreateNamedArgumentDirect(builder, "min", float_arg(-1.0)));
    op_inputs.push_back(
        fbs::CreateNamedArgumentDirect(builder, "max", float_arg(50.0)));
  } else {
    op_inputs.push_back(fbs::CreateNamedArgumentDirect(
        builder, "memory_format", create_none_arg(builder)));
  }
  const std::vector<flatbuffers::Offset<fbs::Output>> operand_outputs = {
      fbs::CreateOutputDirect(builder, "operand")};
  nodes.push_back(fbs::CreateNodeDirect(
      builder,
      "operand",
      fbs::OpKind::CALL_FUNCTION,
      clamp ? "torch.ops.aten.clamp.default" : "torch.ops.aten.clone.default",
      &op_inputs,
      &operand_outputs));
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> add_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "self", create_tensor_arg(builder, "input")),
      fbs::CreateNamedArgumentDirect(
          builder, "other", create_tensor_arg(builder, "operand")),
      fbs::CreateNamedArgumentDirect(
          builder, "alpha", create_int_arg(builder, 1)),
  };
  const std::vector<flatbuffers::Offset<fbs::Output>> sum_outputs = {
      fbs::CreateOutputDirect(builder, "sum")};
  nodes.push_back(fbs::CreateNodeDirect(
      builder,
      "sum",
      fbs::OpKind::CALL_FUNCTION,
      "torch.ops.aten.add.Tensor",
      &add_inputs,
      &sum_outputs));
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> output_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "", create_tensor_arg(builder, "sum"))};
  nodes.push_back(fbs::CreateNodeDirect(
      builder, "output", fbs::OpKind::OUTPUT, "", &output_inputs));
  const auto graph = fbs::CreateGraph(
      builder,
      builder.CreateVector(nodes),
      create_strings(builder, {"input"}),
      create_strings(builder, {"sum"}),
      builder.CreateVector(tensor_values));
  const std::vector<flatbuffers::Offset<fbs::NamedTensorRef>> constants = {
      fbs::CreateNamedTensorRefDirect(
          builder, "bias", "bias", meta, fbs::InputKind::PARAMETER),
  };
  return finish_program(
      builder, graph, {fbs::CreateOutputSpecDirect(builder, "sum")}, constants);
}

// result = _softmax(index(input, batch), -1) over a [2, 3, 8] input. aten.index
// keeps its operands in buffer storage, so softmax reads a buffer into a
// texture output.
std::vector<uint8_t> make_buffered_softmax_program() {
  flatbuffers::FlatBufferBuilder builder;
  const auto float_meta =
      create_tensor_meta(builder, fbs::ScalarType::FLOAT, {2, 3, 8});
  const std::vector<flatbuffers::Offset<fbs::TensorValue>> tensor_values = {
      fbs::CreateTensorValueDirect(builder, "input", float_meta),
      fbs::CreateTensorValueDirect(
          builder,
          "batch",
          create_tensor_meta(builder, fbs::ScalarType::LONG, {2})),
      fbs::CreateTensorValueDirect(builder, "gathered", float_meta),
      fbs::CreateTensorValueDirect(builder, "result", float_meta),
  };
  std::vector<flatbuffers::Offset<fbs::Node>> nodes;
  for (const char* name : {"input", "batch"}) {
    const std::vector<flatbuffers::Offset<fbs::Output>> outputs = {
        fbs::CreateOutputDirect(builder, name)};
    nodes.push_back(fbs::CreateNodeDirect(
        builder, name, fbs::OpKind::PLACEHOLDER, "", nullptr, &outputs));
  }
  const auto add_call =
      [&](const char* name,
          const char* target,
          const std::vector<flatbuffers::Offset<fbs::NamedArgument>>& inputs) {
        const std::vector<flatbuffers::Offset<fbs::Output>> outputs = {
            fbs::CreateOutputDirect(builder, name)};
        nodes.push_back(fbs::CreateNodeDirect(
            builder,
            name,
            fbs::OpKind::CALL_FUNCTION,
            target,
            &inputs,
            &outputs));
      };
  const auto indices = fbs::CreateOptionalTensorListArg(
      builder,
      create_strings(builder, {"batch"}),
      builder.CreateVector(std::vector<uint8_t>{1}));
  add_call(
      "gathered",
      "torch.ops.aten.index.Tensor",
      {
          fbs::CreateNamedArgumentDirect(
              builder, "self", create_tensor_arg(builder, "input")),
          fbs::CreateNamedArgumentDirect(
              builder,
              "indices",
              fbs::CreateArgument(
                  builder,
                  fbs::ArgumentValue::OptionalTensorListArg,
                  indices.Union())),
      });
  add_call(
      "result",
      "torch.ops.aten._softmax.default",
      {
          fbs::CreateNamedArgumentDirect(
              builder, "self", create_tensor_arg(builder, "gathered")),
          fbs::CreateNamedArgumentDirect(
              builder, "dim", create_int_arg(builder, -1)),
          fbs::CreateNamedArgumentDirect(
              builder, "half_to_float", create_bool_arg(builder, false)),
      });
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> output_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "", create_tensor_arg(builder, "result"))};
  nodes.push_back(fbs::CreateNodeDirect(
      builder, "output", fbs::OpKind::OUTPUT, "", &output_inputs));
  const auto graph = fbs::CreateGraph(
      builder,
      builder.CreateVector(nodes),
      create_strings(builder, {"input", "batch"}),
      create_strings(builder, {"result"}),
      builder.CreateVector(tensor_values));
  return finish_program(
      builder, graph, {fbs::CreateOutputSpecDirect(builder, "result")});
}

// result = _softmax(linear(input, weight), -1) over a [2, 3, 8] input and a
// stored [8, 8] weight. The linear output is a width-packed texture, while the
// softmax output is planned channels-packed.
std::vector<uint8_t> make_linear_softmax_program() {
  flatbuffers::FlatBufferBuilder builder;
  const auto float_meta =
      create_tensor_meta(builder, fbs::ScalarType::FLOAT, {2, 3, 8});
  const auto weight_meta =
      create_tensor_meta(builder, fbs::ScalarType::FLOAT, {8, 8});
  const std::vector<flatbuffers::Offset<fbs::TensorValue>> tensor_values = {
      fbs::CreateTensorValueDirect(builder, "input", float_meta),
      fbs::CreateTensorValueDirect(builder, "product", float_meta),
      fbs::CreateTensorValueDirect(builder, "result", float_meta),
  };
  std::vector<flatbuffers::Offset<fbs::Node>> nodes;
  for (const char* name : {"input", "weight"}) {
    const std::vector<flatbuffers::Offset<fbs::Output>> outputs = {
        fbs::CreateOutputDirect(builder, name)};
    nodes.push_back(fbs::CreateNodeDirect(
        builder, name, fbs::OpKind::PLACEHOLDER, "", nullptr, &outputs));
  }
  const auto add_call =
      [&](const char* name,
          const char* target,
          const std::vector<flatbuffers::Offset<fbs::NamedArgument>>& inputs) {
        const std::vector<flatbuffers::Offset<fbs::Output>> outputs = {
            fbs::CreateOutputDirect(builder, name)};
        nodes.push_back(fbs::CreateNodeDirect(
            builder,
            name,
            fbs::OpKind::CALL_FUNCTION,
            target,
            &inputs,
            &outputs));
      };
  add_call(
      "product",
      "torch.ops.aten.linear.default",
      {
          fbs::CreateNamedArgumentDirect(
              builder, "input", create_tensor_arg(builder, "input")),
          fbs::CreateNamedArgumentDirect(
              builder, "weight", create_tensor_arg(builder, "weight")),
          fbs::CreateNamedArgumentDirect(
              builder, "bias", create_none_arg(builder)),
      });
  add_call(
      "result",
      "torch.ops.aten._softmax.default",
      {
          fbs::CreateNamedArgumentDirect(
              builder, "self", create_tensor_arg(builder, "product")),
          fbs::CreateNamedArgumentDirect(
              builder, "dim", create_int_arg(builder, -1)),
          fbs::CreateNamedArgumentDirect(
              builder, "half_to_float", create_bool_arg(builder, false)),
      });
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> output_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "", create_tensor_arg(builder, "result"))};
  nodes.push_back(fbs::CreateNodeDirect(
      builder, "output", fbs::OpKind::OUTPUT, "", &output_inputs));
  const auto graph = fbs::CreateGraph(
      builder,
      builder.CreateVector(nodes),
      create_strings(builder, {"input"}),
      create_strings(builder, {"result"}),
      builder.CreateVector(tensor_values));
  const std::vector<flatbuffers::Offset<fbs::NamedTensorRef>> constants = {
      fbs::CreateNamedTensorRefDirect(
          builder, "weight", "weight", weight_meta, fbs::InputKind::PARAMETER),
  };
  return finish_program(
      builder,
      graph,
      {fbs::CreateOutputSpecDirect(builder, "result")},
      constants);
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
std::vector<uint8_t> make_hf_attention_program() {
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
  rope("q_rope", "q_perm");
  permute("k_perm", "k");
  rope("k_rope", "k_perm");
  permute("v_perm", "v");
  cache_write("k_all", "key_cache", "k_rope");
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
      {arg("query", create_tensor_arg(builder, "q_rope")),
       arg("key", create_tensor_arg(builder, "k_all")),
       arg("value", create_tensor_arg(builder, "v_all")),
       arg("attn_mask", create_tensor_arg(builder, "mask")),
       arg("dropout_p", float_arg(0.0)),
       arg("is_causal", create_bool_arg(builder, false)),
       arg("scale", create_none_arg(builder)),
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

// The select changes the texels' extents, so it is copied rather than aliased
// at an offset.
TEST(VulkanEngineTest, ExecutesTextureSelectAtAnyIndex) {
  const std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
  for (const int64_t index : {0, 1}) {
    SCOPED_TRACE(index);
    CompiledProgram compiled = compile_program(make_select_program(index));
    compiled.executable->set_input(
        0, input.data(), input.size(), ScalarType::Float);
    compiled.executable->execute();
    std::vector<float> output(2);
    compiled.executable->get_output(
        0, output.data(), output.size(), ScalarType::Float);
    EXPECT_EQ(
        output, (std::vector<float>{input[2 * index], input[2 * index + 1]}));
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
      compile_program(make_dynamic_view_mm_program(/*static_reshape=*/true));
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

TEST(VulkanEngineTest, MaterializesStaticChannelsPackedReshape) {
  CompiledProgram compiled = compile_program(make_view_relu_program());
  std::vector<float> input(24);
  std::iota(input.begin(), input.end(), -8.0f);
  compiled.executable->set_input(
      0, input.data(), input.size(), ScalarType::Float);
  compiled.executable->execute();

  std::vector<float> output(input.size());
  compiled.executable->get_output(
      0, output.data(), output.size(), ScalarType::Float);
  std::vector<float> expected(input.size());
  std::ranges::transform(
      input, expected.begin(), [](float x) { return std::max(x, 0.0f); });
  EXPECT_EQ(output, expected);
}

// [S, 4, 4] viewed as [1, 4, S, 4] keeps its texels only at S = 4, the upper
// bound.
TEST(VulkanEngineTest, MaterializesDynamicChannelsPackedReshape) {
  CompiledProgram compiled = compile_program(make_view_relu_program(
      {.input_sizes = {4, 4, 4},
       .view_sizes = {1, 4, 4, 4},
       .size_arg = {1, 4, -1, 4},
       .input_lower_bounds = {1, 4, 4},
       .view_lower_bounds = {1, 4, 1, 4}}));
  compiled.executable->resize_input(0, {2, 4, 4});
  std::vector<float> input(32);
  std::iota(input.begin(), input.end(), -8.0f);
  compiled.executable->set_input(
      0, input.data(), input.size(), ScalarType::Float);
  compiled.executable->execute();

  std::vector<float> output(input.size());
  compiled.executable->get_output(
      0, output.data(), output.size(), ScalarType::Float);
  std::vector<float> expected(input.size());
  std::ranges::transform(
      input, expected.begin(), [](float x) { return std::max(x, 0.0f); });
  EXPECT_EQ(output, expected);
}

TEST(VulkanEngineTest, MaterializesStaticWidthPackedPermute) {
  CompiledProgram compiled = compile_program(make_dynamic_view_mm_program(
      /*static_reshape=*/true, /*permute=*/true));
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

TEST(VulkanEngineTest, MaterializesPermuteWhoseLayoutDiffersFromItsSource) {
  CompiledProgram compiled = compile_program(make_permute_mean_program());
  std::vector<float> input(24);
  std::iota(input.begin(), input.end(), 0.0f);
  compiled.executable->set_input(
      0, input.data(), input.size(), ScalarType::Float);
  compiled.executable->execute();

  // Each output averages input[0, c, h, w] = 12c + 4h + w over c and h.
  std::vector<float> output(4);
  compiled.executable->get_output(
      0, output.data(), output.size(), ScalarType::Float);
  EXPECT_EQ(output, (std::vector<float>{10, 11, 12, 13}));
}

// The input exceeds the texture limit and so is a buffer, but the output fits
// in a texture, and permute reads its input in the output's storage.
TEST(VulkanEngineTest, RejectsRepackBeyondTextureLimit) {
  const int64_t texture_limit =
      vkcompute::vkapi::runtime()->get_adapter_p()->max_texture3d_dim();
  try {
    compile_program(make_permute_copy_program(texture_limit + 1));
    FAIL() << "expected the oversized texture repack to be rejected";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(
        std::string(e.what()).find(
            "repack exceeds the device's 3D texture limit"),
        std::string::npos)
        << e.what();
  }
}

// Views of mutated state stay aliases, so one that would read other texels
// than its source's fails the load.
TEST(VulkanEngineTest, RejectsTextureViewsThatMisreadMutatedState) {
  for (const auto& [name, view] :
       std::vector<std::pair<std::string, StateView>>{
           {"permute",
            {.target = "torch.ops.aten.permute.default",
             .arg = "dims",
             .values = {1, 0},
             .matrix_dim_order = {1, 0}}},
           {"reshape", {.values = {4, 2}}},
           {"dynamic reshape",
            {.values = {-1, 2},
             .state_lower_bounds = {1, 4},
             .matrix_lower_bounds = {2, 2}}},
           // [S, 4] viewed as [4, S] keeps its texels only at S = 4.
           {"dynamic reshape that matches at its upper bounds",
            {.values = {4, -1},
             .state_sizes = {4, 4},
             .matrix_sizes = {4, 4},
             .state_lower_bounds = {1, 4},
             .matrix_lower_bounds = {4, 1}}},
           // [S, T] viewed as [T, S] keeps its texels only at S = T.
           {"reshape of two dynamic dims that matches at both bounds",
            {.values = {4, -1},
             .state_sizes = {4, 4},
             .matrix_sizes = {4, 4},
             .state_lower_bounds = {1, 1},
             .matrix_lower_bounds = {1, 1}}},
       }) {
    SCOPED_TRACE(name);
    try {
      compile_program(make_state_view_mm_program(view));
      FAIL() << "expected the view of mutated state to be rejected";
    } catch (const std::runtime_error& e) {
      EXPECT_NE(
          std::string(e.what()).find("texture view 'matrix' of a mutated"),
          std::string::npos)
          << e.what();
    }
  }
}

// [1, 1, S, 4] viewed as [S, 4] keeps its source's texels at every S, so it may
// alias the mutated state.
TEST(VulkanEngineTest, KeepsTextureViewThatKeepsMutatedStateTexels) {
  for (const bool dynamic : {false, true}) {
    SCOPED_TRACE(dynamic ? "dynamic" : "static");
    CompiledProgram compiled = compile_program(make_state_view_mm_program(
        {.values = {-1, 4},
         .state_sizes = {1, 1, 4, 4},
         .matrix_sizes = {4, 4},
         .state_lower_bounds = dynamic ? std::vector<int64_t>{1, 1, 1, 4}
                                       : std::vector<int64_t>{},
         .matrix_lower_bounds =
             dynamic ? std::vector<int64_t>{1, 4} : std::vector<int64_t>{}}));
    std::vector<float> value = {1, 2, 3, 4, 5, 6, 7, 8};
    if (dynamic) {
      compiled.executable->resize_input(0, {1, 1, 2, 4});
      compiled.executable->resize_input(1, {1, 1, 2, 4});
    } else {
      value.resize(16, 0.0f);
    }
    const std::vector<float> state(value.size(), 100.0f);
    const std::vector<float> weight = {1, 10, 100, 1000};
    compiled.executable->set_input(
        0, state.data(), state.size(), ScalarType::Float);
    compiled.executable->set_input(
        1, value.data(), value.size(), ScalarType::Float);
    compiled.executable->set_input(
        2, weight.data(), weight.size(), ScalarType::Float);
    compiled.executable->execute();

    std::vector<float> output(value.size() / 4);
    compiled.executable->get_output(
        0, output.data(), output.size(), ScalarType::Float);
    std::vector<float> expected = {4321, 8765};
    expected.resize(output.size(), 0.0f);
    EXPECT_EQ(output, expected);
  }
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

TEST(VulkanEngineTest, ConvertsStoredLongConstant) {
  const std::string header =
      R"({"offsets":{"dtype":"I64","shape":[4],"data_offsets":[0,32]}})";
  std::vector<uint8_t> tensors = testing::make_safetensors(header, 32);
  const std::vector<int64_t> offsets = {3, -2, 7, 100};
  std::memcpy(
      tensors.data() + sizeof(uint64_t) + header.size(),
      offsets.data(),
      offsets.size() * sizeof(int64_t));
  CompiledProgram compiled =
      compile_program(make_stored_long_to_copy_program(), std::move(tensors));
  const std::vector<float> input = {0.5f, 1.5f, 2.5f, 3.5f};
  compiled.executable->set_input(
      0, input.data(), input.size(), ScalarType::Float);
  compiled.executable->execute();
  std::vector<float> output(4);
  compiled.executable->get_output(
      0, output.data(), output.size(), ScalarType::Float);
  EXPECT_EQ(output, (std::vector<float>{3.5f, -0.5f, 9.5f, 103.5f}));
}

TEST(VulkanEngineTest, ClonesAndClampsStoredConstant) {
  const std::string header =
      R"({"bias":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}})";
  const std::vector<float> bias = {3.0f, -2.0f, 7.0f, 100.0f};
  const std::vector<float> input = {0.5f, 1.5f, 2.5f, 3.5f};
  for (const bool clamp : {false, true}) {
    SCOPED_TRACE(clamp ? "clamp" : "clone");
    std::vector<uint8_t> tensors = testing::make_safetensors(header, 16);
    std::memcpy(
        tensors.data() + sizeof(uint64_t) + header.size(),
        bias.data(),
        bias.size() * sizeof(float));
    CompiledProgram compiled =
        compile_program(make_stored_operand_program(clamp), std::move(tensors));
    compiled.executable->set_input(
        0, input.data(), input.size(), ScalarType::Float);
    compiled.executable->execute();
    std::vector<float> output(4);
    compiled.executable->get_output(
        0, output.data(), output.size(), ScalarType::Float);
    std::vector<float> expected(4);
    for (size_t i = 0; i < expected.size(); ++i) {
      expected[i] =
          input[i] + (clamp ? std::clamp(bias[i], -1.0f, 50.0f) : bias[i]);
    }
    EXPECT_EQ(output, expected);
  }
}

TEST(VulkanEngineTest, ExecutesSoftmaxOnBufferPlannedInput) {
  std::vector<float> input(48);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<float>((i * 5) % 7) * 0.5f - 1.0f;
  }
  const std::vector<int64_t> batch = {1, 0};
  CompiledProgram compiled = compile_program(make_buffered_softmax_program());
  compiled.executable->set_input(
      0, input.data(), input.size(), ScalarType::Float);
  compiled.executable->set_input(
      1, batch.data(), batch.size(), ScalarType::Long);
  compiled.executable->execute();
  std::vector<float> output(48);
  compiled.executable->get_output(
      0, output.data(), output.size(), ScalarType::Float);
  for (size_t row = 0; row < 6; ++row) {
    const size_t source_row = (row + 3) % 6;
    double sum = 0.0;
    for (size_t i = 0; i < 8; ++i) {
      sum += std::exp(input[source_row * 8 + i]);
    }
    for (size_t i = 0; i < 8; ++i) {
      EXPECT_NEAR(
          output[row * 8 + i], std::exp(input[source_row * 8 + i]) / sum, 1e-6)
          << row << "," << i;
    }
  }
}

TEST(VulkanEngineTest, ExecutesSoftmaxOnWidthPackedTextureInput) {
  std::vector<float> weight(64, 0.0f);
  for (size_t o = 0; o < 8; ++o) {
    weight[o * 8 + o] = 0.5f;
    weight[o * 8 + (o + 3) % 8] = 0.25f;
  }
  const std::string header =
      R"({"weight":{"dtype":"F32","shape":[8,8],"data_offsets":[0,256]}})";
  std::vector<uint8_t> tensors = testing::make_safetensors(header, 256);
  std::memcpy(
      tensors.data() + sizeof(uint64_t) + header.size(), weight.data(), 256);
  CompiledProgram compiled =
      compile_program(make_linear_softmax_program(), std::move(tensors));
  std::vector<float> input(48);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<float>((i * 5) % 7) * 0.5f - 1.0f;
  }
  compiled.executable->set_input(
      0, input.data(), input.size(), ScalarType::Float);
  compiled.executable->execute();
  std::vector<float> output(48);
  compiled.executable->get_output(
      0, output.data(), output.size(), ScalarType::Float);
  for (size_t row = 0; row < 6; ++row) {
    std::vector<double> product(8);
    double sum = 0.0;
    for (size_t o = 0; o < 8; ++o) {
      product[o] =
          0.5 * input[row * 8 + o] + 0.25 * input[row * 8 + (o + 3) % 8];
      sum += std::exp(product[o]);
    }
    for (size_t o = 0; o < 8; ++o) {
      EXPECT_NEAR(output[row * 8 + o], std::exp(product[o]) / sum, 1e-6)
          << row << "," << o;
    }
  }
}

TEST(VulkanEngineTest, RejectsBufferViewResizeThatDoesNotDivide) {
  CompiledProgram compiled =
      compile_program(make_dynamic_buffer_view_index_program());
  compiled.executable->resize_input(0, {3, 2});
  EXPECT_THROW(compiled.executable->execute(), std::runtime_error);
}

TEST(VulkanEngineTest, MaterializesStaticPaddedBufferView) {
  CompiledProgram compiled =
      compile_program(make_dynamic_buffer_view_index_program(false));
  const std::vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8};
  const std::vector<int64_t> index = {3, 0};
  compiled.executable->set_input(
      0, input.data(), input.size(), ScalarType::Float);
  compiled.executable->set_input(
      1, index.data(), index.size(), ScalarType::Long);
  compiled.executable->execute();
  std::vector<float> output(4);
  compiled.executable->get_output(
      0, output.data(), output.size(), ScalarType::Float);
  EXPECT_EQ(output, (std::vector<float>{4, 1, 8, 5}));
}

TEST(VulkanEngineTest, MaterializesStaticRankChangingPaddedBufferViews) {
  for (const std::string& target :
       {"torch.ops.aten.unsqueeze.default",
        "torch.ops.aten.squeeze.dim",
        "torch.ops.aten.squeeze.dims"}) {
    SCOPED_TRACE(target);
    CompiledProgram compiled = compile_program(
        make_dynamic_buffer_view_index_program(false, true, target));
    const std::vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<int64_t> index = {3, 0};
    compiled.executable->set_input(
        0, input.data(), input.size(), ScalarType::Float);
    compiled.executable->set_input(
        1, index.data(), index.size(), ScalarType::Long);
    compiled.executable->execute();
    std::vector<float> output(4);
    compiled.executable->get_output(
        0, output.data(), output.size(), ScalarType::Float);
    EXPECT_EQ(output, (std::vector<float>{4, 1, 8, 5}));
  }
}

TEST(VulkanEngineTest, MaterializesStaticPaddedBufferTransposes) {
  // [2, 3] needs a larger padded buffer once transposed; in [4, 3] only the
  // padded strides disagree.
  for (const std::string& target :
       {"torch.ops.aten.permute.default", "torch.ops.aten.transpose.int"}) {
    for (const auto& [rows, cols] :
         std::vector<std::pair<int64_t, int64_t>>{{2, 3}, {4, 3}}) {
      SCOPED_TRACE(
          ::testing::Message() << target << " " << rows << "x" << cols);
      CompiledProgram compiled = compile_program(
          make_static_buffer_transpose_index_program(rows, cols, target));
      std::vector<float> input(rows * cols);
      std::iota(input.begin(), input.end(), 1.0f);
      const std::vector<int64_t> index = {1, 0};
      compiled.executable->set_input(
          0, input.data(), input.size(), ScalarType::Float);
      compiled.executable->set_input(
          1, index.data(), index.size(), ScalarType::Long);
      compiled.executable->execute();
      std::vector<float> output(cols * 2);
      compiled.executable->get_output(
          0, output.data(), output.size(), ScalarType::Float);
      std::vector<float> expected;
      for (int64_t col = 0; col < cols; ++col) {
        for (const int64_t row : index) {
          expected.push_back(input[row * cols + col]);
        }
      }
      EXPECT_EQ(output, expected);
    }
  }
}

TEST(VulkanEngineTest, ResizesMaterializedSqueezeDimView) {
  CompiledProgram compiled =
      compile_program(make_dynamic_buffer_view_index_program(
          true, false, "torch.ops.aten.squeeze.dim"));
  compiled.executable->resize_input(0, {1, 4, 1});
  const std::vector<float> input = {1, 2, 3, 4};
  const std::vector<int64_t> index = {3, 0};
  compiled.executable->set_input(
      0, input.data(), input.size(), ScalarType::Float);
  compiled.executable->set_input(
      1, index.data(), index.size(), ScalarType::Long);
  compiled.executable->execute();
  std::vector<float> output(2);
  compiled.executable->get_output(
      0, output.data(), output.size(), ScalarType::Float);
  EXPECT_EQ(output, (std::vector<float>{4, 1}));
}

TEST(VulkanEngineTest, RejectsMutatedPaddedBufferView) {
  try {
    compile_program(make_mutated_buffer_view_program());
    FAIL() << "expected the mutated padded buffer view to be rejected";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(
        std::string(e.what()).find(
            "buffer view 'rows' of a mutated tensor cannot alias"),
        std::string::npos)
        << e.what();
  }
}

TEST(VulkanEngineTest, MaterializesBufferViewAcrossPaddingBoundary) {
  CompiledProgram compiled =
      compile_program(make_dynamic_buffer_view_index_program(true, true));
  compiled.executable->resize_input(0, {2, 2});
  const std::vector<float> input = {1, 2, 3, 4};
  const std::vector<int64_t> index = {3, 0};
  compiled.executable->set_input(
      0, input.data(), input.size(), ScalarType::Float);
  compiled.executable->set_input(
      1, index.data(), index.size(), ScalarType::Long);
  compiled.executable->execute();
  std::vector<float> output(2);
  compiled.executable->get_output(
      0, output.data(), output.size(), ScalarType::Float);
  EXPECT_EQ(output, (std::vector<float>{4, 1}));
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

void expect_cache_free_sdpa_matches_reference(const CacheFreeSdpaCase& c) {
  constexpr int kConstantSalt = 3;
  CompiledProgram compiled = compile_program(
      make_cache_free_sdpa_program(c),
      c.constant_mask ? make_cache_free_sdpa_tensors(c, kConstantSalt)
                      : std::vector<uint8_t>{});
  std::vector<float> query(c.batch * c.heads * c.queries * c.dim);
  std::vector<float> key(c.batch * c.heads * c.keys * c.dim);
  std::vector<float> value(key.size());
  for (size_t i = 0; i < query.size(); ++i) {
    query[i] = sdpa_input_value(i, 5);
  }
  for (size_t i = 0; i < key.size(); ++i) {
    key[i] = sdpa_input_value(i, 7);
    value[i] = sdpa_input_value(i, 4);
  }
  // [B, H, L, D] to the [B, L, H, D] order of permuted inputs.
  const auto as_input = [&](const std::vector<float>& heads_major) {
    if (!c.permute_key_value) {
      return heads_major;
    }
    std::vector<float> token_major(heads_major.size());
    for (int64_t b = 0; b < c.batch; ++b) {
      for (int64_t h = 0; h < c.heads; ++h) {
        for (int64_t l = 0; l < c.keys; ++l) {
          std::copy_n(
              &heads_major[((b * c.heads + h) * c.keys + l) * c.dim],
              c.dim,
              &token_major[((b * c.keys + l) * c.heads + h) * c.dim]);
        }
      }
    }
    return token_major;
  };
  const std::vector<float> key_input = as_input(key);
  const std::vector<float> value_input = as_input(value);
  if (c.max_keys != 0) {
    compiled.executable->resize_input(1, {c.batch, c.heads, c.keys, c.dim});
    compiled.executable->resize_input(2, {c.batch, c.heads, c.keys, c.dim});
    if (!c.constant_mask) {
      compiled.executable->resize_input(3, c.mask_sizes);
    }
  }
  size_t input = 0;
  compiled.executable->set_input(
      input++, query.data(), query.size(), ScalarType::Float);
  compiled.executable->set_input(
      input++, key_input.data(), key_input.size(), ScalarType::Float);
  compiled.executable->set_input(
      input++, value_input.data(), value_input.size(), ScalarType::Float);
  const size_t mask_input = input;
  if (!c.constant_mask) {
    ++input;
  }
  if (c.indexed()) {
    std::vector<int64_t> index(c.batch);
    std::iota(index.begin(), index.end(), 0);
    compiled.executable->set_input(
        input, index.data(), index.size(), ScalarType::Long);
  }
  // Repeated execution with new masks, unless the mask is a constant.
  for (const int salt : c.constant_mask ? std::vector<int>{kConstantSalt}
                                        : std::vector<int>{3, 7}) {
    SCOPED_TRACE(salt);
    const std::vector<float> bias = cache_free_sdpa_bias(c, salt);
    if (!c.constant_mask && c.mask_dtype == fbs::ScalarType::BOOL) {
      const std::vector<uint8_t> keep = cache_free_sdpa_keep(c, salt);
      compiled.executable->set_input(
          mask_input, keep.data(), keep.size(), ScalarType::Bool);
    } else if (!c.constant_mask) {
      compiled.executable->set_input(
          mask_input, bias.data(), bias.size(), ScalarType::Float);
    }
    compiled.executable->execute();
    std::vector<float> output(query.size());
    compiled.executable->get_output(
        0, output.data(), output.size(), ScalarType::Float);
    const std::vector<float> expected =
        cache_free_sdpa_reference(c, query, key, value, bias);
    const float tolerance = c.dtype == fbs::ScalarType::HALF ? 1e-2f : 1e-4f;
    for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_NEAR(output[i], expected[i], tolerance) << i;
    }
  }
}

TEST(VulkanEngineTest, CacheFreeSdpaMatchesReference) {
  for (const int64_t keys : {1, 8, 9}) {
    for (const auto& [dtype, mask_sizes] :
         std::vector<std::pair<fbs::ScalarType, std::vector<int64_t>>>{
             {fbs::ScalarType::BOOL, {2, 2, 3, keys}},
             {fbs::ScalarType::FLOAT, {1, 1, 3, keys}},
             {fbs::ScalarType::FLOAT, {2, 2, 1, keys}},
             {fbs::ScalarType::FLOAT, {3, keys}},
             {fbs::ScalarType::BOOL, {keys}}}) {
      for (const auto& [index_output, index_key_value] :
           {std::pair{false, false},
            std::pair{true, false},
            std::pair{false, true}}) {
        SCOPED_TRACE(
            "keys " + std::to_string(keys) + " mask rank " +
            std::to_string(mask_sizes.size()) + " index_output " +
            std::to_string(index_output) + " index_key_value " +
            std::to_string(index_key_value));
        expect_cache_free_sdpa_matches_reference({
            .keys = keys,
            .mask_dtype = dtype,
            .mask_sizes = mask_sizes,
            .index_output = index_output,
            .index_key_value = index_key_value,
        });
      }
    }
  }
}

TEST(VulkanEngineTest, CacheFreeSdpaWithKeysExceedingTextureLimit) {
  auto* adapter = vkcompute::vkapi::runtime()->get_adapter_p();
  const int64_t texture_limit = adapter->max_texture3d_dim();
  CacheFreeSdpaCase c;
  c.batch = 1;
  c.heads = 1;
  c.queries = 1;
  c.keys = texture_limit + 1;
  c.mask_sizes = {c.keys};
  for (const auto dtype : {fbs::ScalarType::FLOAT, fbs::ScalarType::BOOL}) {
    if (dtype == fbs::ScalarType::BOOL &&
        !adapter->has_full_int8_buffers_support()) {
      continue;
    }
    for (const bool constant_mask : {false, true}) {
      SCOPED_TRACE(
          "dtype " + std::to_string(static_cast<int>(dtype)) +
          " constant_mask " + std::to_string(constant_mask));
      c.mask_dtype = dtype;
      c.constant_mask = constant_mask;
      expect_cache_free_sdpa_matches_reference(c);
    }
  }
}

TEST(VulkanEngineTest, CacheFreeSdpaReadsPermutedKeysAndValues) {
  const int64_t texture_limit =
      vkcompute::vkapi::runtime()->get_adapter_p()->max_texture3d_dim();
  // Past the texture limit, the views and permutes stay buffer views, the
  // permutes strided.
  for (const int64_t keys : {int64_t{9}, texture_limit + 1}) {
    SCOPED_TRACE(keys);
    expect_cache_free_sdpa_matches_reference({
        .queries = 1,
        .keys = keys,
        .mask_dtype = fbs::ScalarType::FLOAT,
        .mask_sizes = {keys},
        .permute_key_value = true,
    });
  }
}

// Buffer rows are padded to whole texels, like texture rows.
TEST(VulkanEngineTest, CacheFreeSdpaReadsHeadWidthNotMultipleOfFour) {
  for (const auto& [index_output, index_key_value] :
       {std::pair{false, false},
        std::pair{true, false},
        std::pair{false, true}}) {
    SCOPED_TRACE(
        "index_output " + std::to_string(index_output) + " index_key_value " +
        std::to_string(index_key_value));
    expect_cache_free_sdpa_matches_reference({
        .dim = 6,
        .mask_dtype = fbs::ScalarType::FLOAT,
        .mask_sizes = {1, 1, 3, 9},
        .index_output = index_output,
        .index_key_value = index_key_value,
    });
  }
}

TEST(VulkanEngineTest, CacheFreeSdpaRunsInHalf) {
  for (const auto mask_dtype : {fbs::ScalarType::BOOL, fbs::ScalarType::HALF}) {
    for (const bool index_output : {false, true}) {
      SCOPED_TRACE(
          "mask dtype " + std::to_string(static_cast<int>(mask_dtype)) +
          " index_output " + std::to_string(index_output));
      expect_cache_free_sdpa_matches_reference({
          .dtype = fbs::ScalarType::HALF,
          .mask_dtype = mask_dtype,
          .mask_sizes = {1, 1, 3, 9},
          .index_output = index_output,
      });
    }
  }
}

// The attention weights resize to whole texels per row.
TEST(VulkanEngineTest, CacheFreeSdpaResizesKeys) {
  for (const int64_t keys : {9, 1}) {
    for (const auto mask_dtype :
         {fbs::ScalarType::FLOAT, fbs::ScalarType::BOOL}) {
      for (const bool index_output : {false, true}) {
        SCOPED_TRACE(
            "keys " + std::to_string(keys) + " mask dtype " +
            std::to_string(static_cast<int>(mask_dtype)) + " index_output " +
            std::to_string(index_output));
        expect_cache_free_sdpa_matches_reference({
            .keys = keys,
            .max_keys = 12,
            .mask_dtype = mask_dtype,
            .mask_sizes = {1, 1, 3, keys},
            .index_output = index_output,
        });
      }
    }
  }
}

TEST(VulkanEngineTest, CacheFreeSdpaReadsConstantMasks) {
  for (const auto dtype : {fbs::ScalarType::BOOL, fbs::ScalarType::FLOAT}) {
    for (const bool index_output : {false, true}) {
      SCOPED_TRACE(
          "dtype " + std::to_string(static_cast<int>(dtype)) +
          " index_output " + std::to_string(index_output));
      expect_cache_free_sdpa_matches_reference({
          .mask_dtype = dtype,
          .mask_sizes = {1, 1, 3, 9},
          .constant_mask = true,
          .index_output = index_output,
      });
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
