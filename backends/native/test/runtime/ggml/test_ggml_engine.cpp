// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/engine/ggml/GgmlEngine.h>

#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <executorch/backends/native/runtime/Program.h>
#include <executorch/backends/native/runtime/deserialize/OwnedBytes.h>
#include <executorch/backends/native/runtime/deserialize/Package.h>
#include <executorch/backends/native/runtime/deserialize/test/PackageTestData.h>
#include <executorch/backends/native/runtime/engine/Engine.h>
#include <executorch/backends/native/runtime/native_graph_generated.h>
#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>

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
    flatbuffers::Offset<fbs::QuantSpec> quant = 0) {
  std::vector<flatbuffers::Offset<fbs::Dim>> sizes;
  sizes.reserve(shape.size());
  for (const int64_t size : shape) {
    sizes.push_back(fbs::CreateDim(builder, size, size));
  }
  return fbs::CreateTensorMeta(
      builder, dtype, builder.CreateVector(sizes), 0, quant);
}

flatbuffers::Offset<fbs::Argument> create_tensor_arg(
    flatbuffers::FlatBufferBuilder& builder,
    const char* name) {
  const auto value = fbs::CreateTensorArgDirect(builder, name);
  return fbs::CreateArgument(
      builder, fbs::ArgumentValue::TensorArg, value.Union());
}

flatbuffers::Offset<fbs::Argument> create_none_arg(
    flatbuffers::FlatBufferBuilder& builder) {
  const auto value = fbs::CreateNoneArg(builder);
  return fbs::CreateArgument(
      builder, fbs::ArgumentValue::NoneArg, value.Union());
}

constexpr size_t kQ4KBlockBytes = 144;

// A [1, 256] x [256] linear with gguf:q4_k weights, or with symmetric int4
// AffineGroup weights whose fp16 scales `weight_scale` cover groups of 32.
// `cache` is an unused mutable buffer.
std::vector<uint8_t> make_linear_program(
    const char* target = "torch.ops.aten.linear.default",
    bool with_cache = false,
    bool affine = false) {
  flatbuffers::FlatBufferBuilder builder;
  const auto quant = affine
      ? fbs::CreateQuantSpec(
            builder,
            fbs::QuantScheme::AffineGroup,
            fbs::CreateAffineGroupDirect(
                builder, "weight_scale", fbs::ScalarType::HALF, -8, 7, 32)
                .Union())
      : fbs::CreateQuantSpec(
            builder,
            fbs::QuantScheme::PackedQuant,
            fbs::CreatePackedQuantDirect(builder, "gguf:q4_k").Union());
  const auto input_meta =
      create_tensor_meta(builder, fbs::ScalarType::FLOAT, {1, 256});
  const auto weight_meta =
      create_tensor_meta(builder, fbs::ScalarType::BYTE, {1, 256}, quant);
  const auto output_meta =
      create_tensor_meta(builder, fbs::ScalarType::FLOAT, {1, 1});

  std::vector<flatbuffers::Offset<fbs::TensorValue>> values = {
      fbs::CreateTensorValueDirect(builder, "input", input_meta),
      fbs::CreateTensorValueDirect(builder, "weight", weight_meta),
      fbs::CreateTensorValueDirect(builder, "output", output_meta),
  };
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> linear_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "input", create_tensor_arg(builder, "input")),
      fbs::CreateNamedArgumentDirect(
          builder, "weight", create_tensor_arg(builder, "weight")),
      fbs::CreateNamedArgumentDirect(builder, "bias", create_none_arg(builder)),
  };
  const std::vector<flatbuffers::Offset<fbs::NamedArgument>> output_inputs = {
      fbs::CreateNamedArgumentDirect(
          builder, "", create_tensor_arg(builder, "output")),
  };
  const std::vector<flatbuffers::Offset<fbs::Output>> input_outputs = {
      fbs::CreateOutputDirect(builder, "input"),
  };
  const std::vector<flatbuffers::Offset<fbs::Output>> weight_outputs = {
      fbs::CreateOutputDirect(builder, "weight"),
  };
  const std::vector<flatbuffers::Offset<fbs::Output>> linear_outputs = {
      fbs::CreateOutputDirect(builder, "output"),
  };
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
          "weight",
          fbs::OpKind::PLACEHOLDER,
          "",
          nullptr,
          &weight_outputs),
      fbs::CreateNodeDirect(
          builder,
          "linear",
          fbs::OpKind::CALL_FUNCTION,
          target,
          &linear_inputs,
          &linear_outputs),
      fbs::CreateNodeDirect(
          builder, "output", fbs::OpKind::OUTPUT, "", &output_inputs),
  };
  std::vector<flatbuffers::Offset<fbs::MutableBufferSpec>> mutable_buffers;
  if (with_cache) {
    const std::vector<flatbuffers::Offset<fbs::Output>> cache_outputs = {
        fbs::CreateOutputDirect(builder, "cache"),
    };
    values.push_back(fbs::CreateTensorValueDirect(
        builder,
        "cache",
        create_tensor_meta(builder, fbs::ScalarType::FLOAT, {1, 256})));
    nodes.insert(
        nodes.begin(),
        fbs::CreateNodeDirect(
            builder,
            "cache",
            fbs::OpKind::PLACEHOLDER,
            "",
            nullptr,
            &cache_outputs));
    mutable_buffers.push_back(
        fbs::CreateMutableBufferSpecDirect(builder, "cache", "cache"));
  }
  std::vector<flatbuffers::Offset<fbs::NamedTensorRef>> bindings = {
      fbs::CreateNamedTensorRefDirect(
          builder, "weight", "weight", weight_meta, fbs::InputKind::PARAMETER),
  };
  if (affine) {
    const std::vector<flatbuffers::Offset<fbs::Output>> scale_outputs = {
        fbs::CreateOutputDirect(builder, "weight_scale"),
    };
    const auto scale_meta =
        create_tensor_meta(builder, fbs::ScalarType::HALF, {1, 8});
    values.push_back(
        fbs::CreateTensorValueDirect(builder, "weight_scale", scale_meta));
    nodes.insert(
        nodes.begin(),
        fbs::CreateNodeDirect(
            builder,
            "weight_scale",
            fbs::OpKind::PLACEHOLDER,
            "",
            nullptr,
            &scale_outputs));
    bindings.push_back(fbs::CreateNamedTensorRefDirect(
        builder,
        "weight_scale",
        "weight_scale",
        scale_meta,
        fbs::InputKind::PARAMETER));
  }
  const auto graph = fbs::CreateGraph(
      builder,
      builder.CreateVector(nodes),
      create_strings(builder, {"input"}),
      create_strings(builder, {"output"}),
      builder.CreateVector(values));
  const auto output_spec = fbs::CreateOutputSpecDirect(builder, "output");
  const auto method = fbs::CreateMethod(
      builder,
      builder.CreateString("forward"),
      graph,
      builder.CreateVector(bindings),
      builder.CreateVector(
          std::vector<flatbuffers::Offset<fbs::OutputSpec>>{output_spec}),
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

std::unique_ptr<GgmlEngineContext> make_context(
    GgmlEngineHost& host,
    const std::vector<uint8_t>& program_bytes) {
  // One block_q4_K: d = 1.0 (fp16), dmin = 0, every sub-block scale 1 and
  // min 0, and nibbles 1 (low) / 2 (high), so half the weights are 1 and
  // half are 2.
  std::vector<uint8_t> packed_weight(kQ4KBlockBytes, 0x21);
  std::fill_n(packed_weight.begin(), 16, uint8_t{0});
  packed_weight[1] = 0x3C;
  std::fill_n(packed_weight.begin() + 4, 4, uint8_t{1});
  std::fill_n(packed_weight.begin() + 12, 4, uint8_t{1});
  std::vector<uint8_t> safetensors = testing::make_safetensors(
      R"({"weight":{"dtype":"U8","shape":[144],"data_offsets":[0,144]}})",
      packed_weight.size());
  std::copy(
      packed_weight.begin(),
      packed_weight.end(),
      safetensors.end() - packed_weight.size());
  auto package = std::make_shared<const Package>(
      Package::load(OwnedBytes::from_vector(testing::make_zip({
          {kProgramEntry, program_bytes},
          {kSafeTensorsEntry, std::move(safetensors)},
      }))));
  auto program = std::make_shared<const Program>(
      Program::load(program_bytes.data(), program_bytes.size()));
  return host.create_ggml_context(std::move(program), std::move(package));
}

// A tensor (by name), an int, a float, an int list, or None.
using TestArg = std::
    variant<std::string, int64_t, double, std::vector<int64_t>, std::monostate>;

// A one-method program assembled value by value; every node's output value
// carries the node's name.
class TestProgram {
 public:
  void input(
      const std::string& name,
      fbs::ScalarType dtype,
      const std::vector<int64_t>& shape,
      const char* packed_codec = nullptr) {
    const auto quant = packed_codec == nullptr
        ? flatbuffers::Offset<fbs::QuantSpec>()
        : fbs::CreateQuantSpec(
              builder_,
              fbs::QuantScheme::PackedQuant,
              fbs::CreatePackedQuantDirect(builder_, packed_codec).Union());
    placeholder(name, create_tensor_meta(builder_, dtype, shape, quant));
    inputs_.push_back(name);
  }

  void buffer(
      const std::string& name,
      const std::vector<int64_t>& shape,
      fbs::ScalarType dtype = fbs::ScalarType::FLOAT) {
    placeholder(name, create_tensor_meta(builder_, dtype, shape));
    buffers_.push_back(fbs::CreateMutableBufferSpecDirect(
        builder_, name.c_str(), name.c_str()));
  }

  // Each execution ends by writing value `value` to buffer `name`.
  void mutate(const std::string& name, const std::string& value) {
    mutations_.emplace_back(name, value);
  }

  // make_context's packaged block_q4_K as a parameter, by default [1, 256].
  void q4_k_weight(
      const std::string& name,
      const std::vector<int64_t>& shape = {1, 256}) {
    const auto quant = fbs::CreateQuantSpec(
        builder_,
        fbs::QuantScheme::PackedQuant,
        fbs::CreatePackedQuantDirect(builder_, "gguf:q4_k").Union());
    const auto meta =
        create_tensor_meta(builder_, fbs::ScalarType::BYTE, shape, quant);
    placeholder(name, meta);
    bindings_.push_back(fbs::CreateNamedTensorRefDirect(
        builder_, name.c_str(), "weight", meta, fbs::InputKind::PARAMETER));
  }

  void call(
      const std::string& name,
      const char* target,
      const std::vector<std::pair<std::string, TestArg>>& args,
      fbs::ScalarType dtype,
      const std::vector<int64_t>& shape) {
    std::vector<flatbuffers::Offset<fbs::NamedArgument>> named;
    for (const auto& [arg_name, arg] : args) {
      named.push_back(fbs::CreateNamedArgumentDirect(
          builder_, arg_name.c_str(), create_arg(arg)));
    }
    values_.push_back(fbs::CreateTensorValueDirect(
        builder_, name.c_str(), create_tensor_meta(builder_, dtype, shape)));
    const std::vector<flatbuffers::Offset<fbs::Output>> outputs = {
        fbs::CreateOutputDirect(builder_, name.c_str())};
    nodes_.push_back(fbs::CreateNodeDirect(
        builder_,
        name.c_str(),
        fbs::OpKind::CALL_FUNCTION,
        target,
        &named,
        &outputs));
  }

  std::vector<uint8_t> finish(
      const std::string& output,
      fbs::OutputKind kind = fbs::OutputKind::USER_OUTPUT,
      const char* target = "") {
    const std::vector<flatbuffers::Offset<fbs::NamedArgument>> output_inputs = {
        fbs::CreateNamedArgumentDirect(
            builder_, "", create_tensor_arg(builder_, output.c_str()))};
    std::vector<std::string> outputs = {output};
    std::vector<flatbuffers::Offset<fbs::NamedArgument>> returned =
        output_inputs;
    for (const auto& [name, value] : mutations_) {
      outputs.push_back(value);
      returned.push_back(fbs::CreateNamedArgumentDirect(
          builder_, "", create_tensor_arg(builder_, value.c_str())));
    }
    nodes_.push_back(fbs::CreateNodeDirect(
        builder_, "output", fbs::OpKind::OUTPUT, "", &returned));
    const auto graph = fbs::CreateGraph(
        builder_,
        builder_.CreateVector(nodes_),
        create_strings(builder_, inputs_),
        create_strings(builder_, outputs),
        builder_.CreateVector(values_));
    const auto output_spec =
        fbs::CreateOutputSpecDirect(builder_, output.c_str(), kind, target);
    const auto method = fbs::CreateMethod(
        builder_,
        builder_.CreateString("forward"),
        graph,
        builder_.CreateVector(bindings_),
        builder_.CreateVector(output_specs(output_spec)),
        builder_.CreateVector(buffers_));
    const auto program = fbs::CreateProgram(
        builder_,
        builder_.CreateString("1.0"),
        builder_.CreateVector(
            std::vector<flatbuffers::Offset<fbs::Method>>{method}));
    fbs::FinishProgramBuffer(builder_, program);
    return {
        builder_.GetBufferPointer(),
        builder_.GetBufferPointer() + builder_.GetSize()};
  }

 private:
  void placeholder(
      const std::string& name,
      flatbuffers::Offset<fbs::TensorMeta> meta) {
    values_.push_back(
        fbs::CreateTensorValueDirect(builder_, name.c_str(), meta));
    const std::vector<flatbuffers::Offset<fbs::Output>> outputs = {
        fbs::CreateOutputDirect(builder_, name.c_str())};
    nodes_.push_back(fbs::CreateNodeDirect(
        builder_,
        name.c_str(),
        fbs::OpKind::PLACEHOLDER,
        "",
        nullptr,
        &outputs));
  }

  // `user_output`, then one BufferMutation spec per mutate().
  std::vector<flatbuffers::Offset<fbs::OutputSpec>> output_specs(
      flatbuffers::Offset<fbs::OutputSpec> user_output) {
    std::vector<flatbuffers::Offset<fbs::OutputSpec>> specs = {user_output};
    for (const auto& [name, value] : mutations_) {
      specs.push_back(fbs::CreateOutputSpecDirect(
          builder_,
          value.c_str(),
          fbs::OutputKind::BUFFER_MUTATION,
          name.c_str()));
    }
    return specs;
  }

  flatbuffers::Offset<fbs::Argument> create_arg(const TestArg& arg) {
    if (const auto* tensor = std::get_if<std::string>(&arg)) {
      return create_tensor_arg(builder_, tensor->c_str());
    }
    if (const auto* integer = std::get_if<int64_t>(&arg)) {
      return fbs::CreateArgument(
          builder_,
          fbs::ArgumentValue::IntArg,
          fbs::CreateIntArgDirect(builder_, *integer).Union());
    }
    if (std::holds_alternative<std::monostate>(arg)) {
      return create_none_arg(builder_);
    }
    if (const auto* list = std::get_if<std::vector<int64_t>>(&arg)) {
      return fbs::CreateArgument(
          builder_,
          fbs::ArgumentValue::IntListArg,
          fbs::CreateIntListArgDirect(builder_, list).Union());
    }
    return fbs::CreateArgument(
        builder_,
        fbs::ArgumentValue::FloatArg,
        fbs::CreateFloatArgDirect(builder_, std::get<double>(arg)).Union());
  }

  flatbuffers::FlatBufferBuilder builder_;
  std::vector<std::string> inputs_;
  std::vector<flatbuffers::Offset<fbs::Node>> nodes_;
  std::vector<flatbuffers::Offset<fbs::TensorValue>> values_;
  std::vector<flatbuffers::Offset<fbs::NamedTensorRef>> bindings_;
  std::vector<flatbuffers::Offset<fbs::MutableBufferSpec>> buffers_;
  std::vector<std::pair<std::string, std::string>> mutations_;
};

// old = cache * 1.0 (or a same-shape view_copy of cache if `view`), then rows
// [start_pos, start_pos + 1) of cache = value.
std::vector<uint8_t> make_cache_update_program(
    int64_t start_pos,
    bool view = false) {
  TestProgram program;
  program.input("value", fbs::ScalarType::FLOAT, {1, 1, 2, 4});
  program.buffer("cache", {1, 8, 2, 4});
  program.call(
      "old",
      view ? "torch.ops.aten.view_copy.default" : "torch.ops.aten.mul.Tensor",
      {{"self", std::string("cache")},
       {view ? "size" : "other",
        view ? TestArg(std::vector<int64_t>{1, 8, 2, 4}) : TestArg(1.0)}},
      fbs::ScalarType::FLOAT,
      {1, 8, 2, 4});
  program.call(
      "updated",
      "torch.ops.llama.update_cache.default",
      {{"value", std::string("value")},
       {"cache", std::string("cache")},
       {"start_pos", start_pos}},
      fbs::ScalarType::FLOAT,
      {1, 1, 2, 4});
  return program.finish("old");
}

// cppcheck-suppress syntaxError
TEST(GgmlEngineTest, ExecutesPackedQ4KLinear) {
  auto host = GgmlEngineHost::create();
  auto context = make_context(*host, make_linear_program());
  auto executable = context->compile("forward");

  EXPECT_EQ(context->materialized_constant_bytes(), kQ4KBlockBytes);
  ASSERT_EQ(executable->num_inputs(), 1);
  ASSERT_EQ(executable->num_outputs(), 1);
  const std::vector<float> input(256, 1.0f);
  executable->set_input(0, input.data(), input.size(), ScalarType::Float);
  executable->execute();
  float output = 0.0f;
  executable->get_output(0, &output, 1, ScalarType::Float);
  EXPECT_NEAR(output, 128.0f * 1.0f + 128.0f * 2.0f, 1e-2f);
}

TEST(GgmlEngineTest, RepeatedExecutionsReadFreshInputs) {
  for (const size_t max_cached_graphs : {size_t{0}, size_t{1}}) {
    SCOPED_TRACE(max_cached_graphs);
    auto host = GgmlEngineHost::create(
        GgmlEngineOptions{.max_cached_graphs = max_cached_graphs});
    auto context = make_context(*host, make_linear_program());
    auto executable = context->compile("forward");
    for (const float value : {1.0f, 2.0f, -1.0f}) {
      const std::vector<float> input(256, value);
      executable->set_input(0, input.data(), input.size(), ScalarType::Float);
      executable->execute();
      float output = 0.0f;
      executable->get_output(0, &output, 1, ScalarType::Float);
      EXPECT_NEAR(output, 384.0f * value, 1e-2f);
    }
  }
}

TEST(GgmlEngineTest, ReplayRecoversAfterRejectedIndices) {
  TestProgram program;
  program.q4_k_weight("weight");
  program.input("ids", fbs::ScalarType::LONG, {1, 2});
  program.call(
      "rows",
      "torch.ops.aten.embedding.default",
      {{"weight", std::string("weight")}, {"indices", std::string("ids")}},
      fbs::ScalarType::FLOAT,
      {1, 2, 256});
  auto host = GgmlEngineHost::create(GgmlEngineOptions{.max_cached_graphs = 1});
  auto context = make_context(*host, program.finish("rows"));
  auto executable = context->compile("forward");
  const std::vector<int64_t> good = {0, 0};
  const std::vector<int64_t> bad = {0, 1};
  std::vector<float> rows(512);
  for (const auto* ids : {&good, &bad, &good}) {
    executable->set_input(0, ids->data(), 2, ScalarType::Long);
    if (ids == &bad) {
      EXPECT_THROW(executable->execute(), std::out_of_range);
      continue;
    }
    executable->execute();
    executable->get_output(0, rows.data(), rows.size(), ScalarType::Float);
    EXPECT_EQ(std::accumulate(rows.begin(), rows.end(), 0.0f), 768.0f);
  }
}

TEST(GgmlEngineTest, RejectsMoreThan16CachedGraphs) {
  GgmlEngineOptions options;
  options.max_cached_graphs = 17;
  EXPECT_THROW(GgmlEngineHost::create(options), std::invalid_argument);
}

TEST(GgmlEngineTest, RejectsUnsupportedOpAtCompile) {
  auto host = GgmlEngineHost::create();
  auto context = make_context(
      *host, make_linear_program("torch.ops.aten.unsupported.default"));
  EXPECT_THROW(context->compile("forward"), std::runtime_error);
}

TEST(GgmlEngineTest, RejectsSecondExecutableBindingMutableBuffer) {
  auto host = GgmlEngineHost::create();
  auto context = make_context(
      *host,
      make_linear_program(
          "torch.ops.aten.linear.default", /*with_cache=*/true));
  auto executable = context->compile("forward");
  EXPECT_THROW(context->compile("forward"), std::runtime_error);
}

void expect_compile_error(
    const std::vector<uint8_t>& program,
    const std::string& message) {
  auto host = GgmlEngineHost::create();
  auto context = make_context(*host, program);
  try {
    context->compile("forward");
    FAIL() << "compile accepted the program";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find(message), std::string::npos)
        << error.what();
  }
}

TEST(GgmlEngineTest, RejectsUserInputMutation) {
  TestProgram program;
  program.input("x", fbs::ScalarType::FLOAT, {2, 3});
  program.call(
      "doubled",
      "torch.ops.aten.add.Tensor",
      {{"self", std::string("x")}, {"other", std::string("x")}},
      fbs::ScalarType::FLOAT,
      {2, 3});
  expect_compile_error(
      program.finish("doubled", fbs::OutputKind::USER_INPUT_MUTATION, "x"),
      "user input mutation");
}

TEST(GgmlEngineTest, RejectsCustomSdpaDropout) {
  TestProgram program;
  program.input("query", fbs::ScalarType::FLOAT, {1, 2, 2, 8});
  program.buffer("key", {1, 8, 2, 8});
  program.buffer("value", {1, 8, 2, 8});
  program.call(
      "attention",
      "torch.ops.llama.custom_sdpa.default",
      {{"query", std::string("query")},
       {"key", std::string("key")},
       {"value", std::string("value")},
       {"start_pos", int64_t{0}},
       {"dropout_p", 0.5}},
      fbs::ScalarType::FLOAT,
      {1, 2, 2, 8});
  expect_compile_error(program.finish("attention"), "dropout");
}

// conv2d of a [1, 2, 4, 4] input with [2, 2, 3, 3] weights.
std::vector<uint8_t> make_convolution_program(
    const std::vector<int64_t>& bias_shape,
    const std::vector<int64_t>& stride) {
  TestProgram program;
  program.input("x", fbs::ScalarType::FLOAT, {1, 2, 4, 4});
  program.input("w", fbs::ScalarType::FLOAT, {2, 2, 3, 3});
  program.input("b", fbs::ScalarType::FLOAT, bias_shape);
  program.call(
      "y",
      "torch.ops.aten.convolution.default",
      {{"input", std::string("x")},
       {"weight", std::string("w")},
       {"bias", std::string("b")},
       {"stride", stride},
       {"padding", std::vector<int64_t>{1, 1}},
       {"dilation", std::vector<int64_t>{1, 1}}},
      fbs::ScalarType::FLOAT,
      {1, 2, 4, 4});
  return program.finish("y");
}

TEST(GgmlEngineTest, RejectsMalformedConvolutionAndBatchNorm) {
  auto host = GgmlEngineHost::create();
  EXPECT_NO_THROW(make_context(*host, make_convolution_program({2}, {1, 1}))
                      ->compile("forward"));
  expect_compile_error(make_convolution_program({3}, {1, 1}), "bias");
  expect_compile_error(make_convolution_program({2}, {-1, 1}), "stride");

  TestProgram norm;
  norm.input("x", fbs::ScalarType::FLOAT, {1, 2, 4, 4});
  norm.input("mean", fbs::ScalarType::FLOAT, {3});
  norm.input("var", fbs::ScalarType::FLOAT, {2});
  norm.call(
      "y",
      "torch.ops.aten._native_batch_norm_legit_no_training.default",
      {{"input", std::string("x")},
       {"weight", std::monostate{}},
       {"bias", std::monostate{}},
       {"running_mean", std::string("mean")},
       {"running_var", std::string("var")}},
      fbs::ScalarType::FLOAT,
      {1, 2, 4, 4});
  expect_compile_error(norm.finish("y"), "one value per channel");
}

// m = 2 * x is a reusable attention mask; an in-place write to head 1 of m
// between two attention ops must reach the second one.
TEST(GgmlEngineTest, WriteToCachedMaskReachesLaterAttention) {
  TestProgram program;
  program.input("q", fbs::ScalarType::FLOAT, {1, 2, 3, 8});
  program.input("k", fbs::ScalarType::FLOAT, {1, 2, 8, 8});
  program.input("v", fbs::ScalarType::FLOAT, {1, 2, 8, 8});
  program.input("x", fbs::ScalarType::FLOAT, {1, 2, 3, 8});
  program.input("row", fbs::ScalarType::FLOAT, {1, 1, 3, 8});
  program.call(
      "m",
      "torch.ops.aten.mul.Tensor",
      {{"self", std::string("x")}, {"other", 2.0}},
      fbs::ScalarType::FLOAT,
      {1, 2, 3, 8});
  const std::vector<std::pair<std::string, TestArg>> attention = {
      {"query", std::string("q")},
      {"key", std::string("k")},
      {"value", std::string("v")},
      {"attn_mask", std::string("m")}};
  program.call(
      "before",
      "torch.ops.aten.scaled_dot_product_attention.default",
      attention,
      fbs::ScalarType::FLOAT,
      {1, 2, 3, 8});
  program.call(
      "written",
      "torch.ops.llama.update_cache.default",
      {{"value", std::string("row")},
       {"cache", std::string("m")},
       {"start_pos", int64_t{1}}},
      fbs::ScalarType::FLOAT,
      {1, 1, 3, 8});
  program.call(
      "after",
      "torch.ops.aten.scaled_dot_product_attention.default",
      attention,
      fbs::ScalarType::FLOAT,
      {1, 2, 3, 8});
  program.call(
      "difference",
      "torch.ops.aten.sub.Tensor",
      {{"self", std::string("after")}, {"other", std::string("before")}},
      fbs::ScalarType::FLOAT,
      {1, 2, 3, 8});
  auto host = GgmlEngineHost::create();
  auto context = make_context(*host, program.finish("difference"));
  auto executable = context->compile("forward");
  std::vector<float> q(48), kv(128), x(48, 0.0f);
  for (size_t i = 0; i < kv.size(); ++i) {
    kv[i] = static_cast<float>(i % 7) * 0.25f;
  }
  std::fill(q.begin(), q.end(), 0.5f);
  // Key 0 of head 1 becomes masked out for every query after the write.
  std::vector<float> row(24, 0.0f);
  for (size_t query = 0; query < 3; ++query) {
    row[query * 8] = -1e4f;
  }
  executable->set_input(0, q.data(), q.size(), ScalarType::Float);
  executable->set_input(1, kv.data(), kv.size(), ScalarType::Float);
  executable->set_input(2, kv.data(), kv.size(), ScalarType::Float);
  executable->set_input(3, x.data(), x.size(), ScalarType::Float);
  executable->set_input(4, row.data(), row.size(), ScalarType::Float);
  executable->execute();
  std::vector<float> difference(48);
  executable->get_output(
      0, difference.data(), difference.size(), ScalarType::Float);
  const auto max_abs = [&](size_t head) {
    float result = 0.0f;
    for (size_t i = head * 24; i < (head + 1) * 24; ++i) {
      result = std::max(result, std::abs(difference[i]));
    }
    return result;
  };
  EXPECT_EQ(max_abs(0), 0.0f);
  EXPECT_GT(max_abs(1), 1e-2f);
}

TEST(GgmlEngineTest, ReleasesMutableBufferWithItsExecutable) {
  auto host = GgmlEngineHost::create();
  auto context = make_context(
      *host,
      make_linear_program(
          "torch.ops.aten.linear.default", /*with_cache=*/true));
  auto executable = context->compile("forward");
  executable.reset();
  EXPECT_NO_THROW(context->compile("forward"));
}

TEST(GgmlEngineTest, ReadBeforeInPlaceWriteSeesOldValue) {
  for (const bool view : {false, true}) {
    SCOPED_TRACE(view);
    auto host = GgmlEngineHost::create();
    auto context = make_context(*host, make_cache_update_program(3, view));
    auto executable = context->compile("forward");
    const std::vector<float> value(8, 1.0f);
    std::vector<float> old(64);
    for (const float expected_sum : {0.0f, 8.0f}) {
      executable->set_input(0, value.data(), value.size(), ScalarType::Float);
      executable->execute();
      executable->get_output(0, old.data(), old.size(), ScalarType::Float);
      EXPECT_EQ(std::accumulate(old.begin(), old.end(), 0.0f), expected_sum);
    }
  }
}

// Each step returns prev + x, then sets prev to cur and cur to 2 * x, so prev
// must take cur's value from before the step, whether prev's new value is
// computed from cur or is cur itself.
TEST(GgmlEngineTest, WriteBacksReadBuffersBeforeAnyIsOverwritten) {
  for (const bool alias : {false, true}) {
    SCOPED_TRACE(alias);
    TestProgram program;
    program.input("x", fbs::ScalarType::FLOAT, {4});
    program.buffer("cur", {4});
    program.buffer("prev", {4});
    program.call(
        "out",
        "torch.ops.aten.add.Tensor",
        {{"self", std::string("prev")}, {"other", std::string("x")}},
        fbs::ScalarType::FLOAT,
        {4});
    program.call(
        "next_cur",
        "torch.ops.aten.mul.Tensor",
        {{"self", std::string("x")}, {"other", 2.0}},
        fbs::ScalarType::FLOAT,
        {4});
    program.mutate("cur", "next_cur");
    if (alias) {
      program.mutate("prev", "cur");
    } else {
      program.call(
          "next_prev",
          "torch.ops.aten.mul.Tensor",
          {{"self", std::string("cur")}, {"other", 1.0}},
          fbs::ScalarType::FLOAT,
          {4});
      program.mutate("prev", "next_prev");
    }
    auto host = GgmlEngineHost::create();
    auto context = make_context(*host, program.finish("out"));
    auto executable = context->compile("forward");
    const std::vector<float> x(4, 1.0f);
    std::vector<float> out(4);
    for (const float expected : {1.0f, 1.0f, 3.0f}) {
      executable->set_input(0, x.data(), x.size(), ScalarType::Float);
      executable->execute();
      executable->get_output(0, out.data(), out.size(), ScalarType::Float);
      EXPECT_EQ(out, std::vector<float>(4, expected));
    }
  }
}

// A *_copy view of a buffer, returned as an output, keeps the buffer's value
// from before the step's write-back.
TEST(GgmlEngineTest, CopyingViewsOfABufferKeepTheOldValue) {
  struct View {
    const char* target;
    std::pair<std::string, TestArg> arg;
    std::vector<int64_t> shape;
  };
  const std::vector<View> views = {
      {"torch.ops.aten.view_copy.default",
       {"size", std::vector<int64_t>{8}},
       {8}},
      {"torch.ops.aten.squeeze_copy.dims",
       {"dim", std::vector<int64_t>{0}},
       {8}},
      {"torch.ops.aten.unsqueeze_copy.default", {"dim", int64_t{0}}, {1, 1, 8}},
  };
  for (const View& view : views) {
    SCOPED_TRACE(view.target);
    TestProgram program;
    program.input("x", fbs::ScalarType::FLOAT, {1, 8});
    program.buffer("state", {1, 8});
    program.call(
        "old",
        view.target,
        {{"self", std::string("state")}, view.arg},
        fbs::ScalarType::FLOAT,
        view.shape);
    program.call(
        "next",
        "torch.ops.aten.add.Tensor",
        {{"self", std::string("state")}, {"other", std::string("x")}},
        fbs::ScalarType::FLOAT,
        {1, 8});
    program.mutate("state", "next");
    auto host = GgmlEngineHost::create();
    auto context = make_context(*host, program.finish("old"));
    auto executable = context->compile("forward");
    const std::vector<float> x(8, 1.0f);
    std::vector<float> old(8);
    for (const float expected_sum : {0.0f, 8.0f}) {
      executable->set_input(0, x.data(), x.size(), ScalarType::Float);
      executable->execute();
      executable->get_output(0, old.data(), old.size(), ScalarType::Float);
      EXPECT_EQ(std::accumulate(old.begin(), old.end(), 0.0f), expected_sum);
    }
  }
}

// Rows [3, 4) of an F16 cache = (2 * x or a buffer) cast to F16. The CPY into
// the cache converts a fresh source itself, but a buffer could change before
// the write, so its cast stays.
TEST(GgmlEngineTest, Fp16CacheUpdateSkipsCastOfFreshSource) {
  const auto run = [](const char* rows, std::vector<float>& cache) {
    TestProgram program;
    program.input("x", fbs::ScalarType::FLOAT, {1, 1, 2, 4});
    program.buffer("state", {1, 1, 2, 4});
    program.buffer("cache", {1, 8, 2, 4}, fbs::ScalarType::HALF);
    program.call(
        "doubled",
        "torch.ops.aten.mul.Tensor",
        {{"self", std::string("x")}, {"other", 2.0}},
        fbs::ScalarType::FLOAT,
        {1, 1, 2, 4});
    program.call(
        "half",
        "torch.ops.aten._to_copy.default",
        {{"self", std::string(rows)}},
        fbs::ScalarType::HALF,
        {1, 1, 2, 4});
    program.call(
        "updated",
        "torch.ops.llama.update_cache.default",
        {{"value", std::string("half")},
         {"cache", std::string("cache")},
         {"start_pos", int64_t{3}}},
        fbs::ScalarType::HALF,
        {1, 1, 2, 4});
    program.call(
        "result",
        "torch.ops.aten._to_copy.default",
        {{"self", std::string("cache")}},
        fbs::ScalarType::FLOAT,
        {1, 8, 2, 4});
    auto host =
        GgmlEngineHost::create(GgmlEngineOptions{.print_graph_ops = true});
    auto context = make_context(*host, program.finish("result"));
    auto executable = context->compile("forward");
    const std::vector<float> x(8, 1.0f / 3);
    executable->set_input(0, x.data(), x.size(), ScalarType::Float);
    ::testing::internal::CaptureStderr();
    executable->execute();
    const std::string ops = ::testing::internal::GetCapturedStderr();
    executable->get_output(0, cache.data(), cache.size(), ScalarType::Float);
    const size_t cpy = ops.find(" CPY=");
    return cpy == std::string::npos ? -1 : std::stoi(ops.substr(cpy + 5));
  };
  std::vector<float> cache(64);
  const int fresh_copies = run("doubled", cache);
  for (size_t i = 0; i < cache.size(); ++i) {
    SCOPED_TRACE(i);
    // 2 / 3 rounded to F16.
    EXPECT_EQ(cache[i], i / 8 == 3 ? 1365.0f / 2048 : 0.0f);
  }
  EXPECT_EQ(run("state", cache), fresh_copies + 1);
}

TEST(GgmlEngineTest, RejectsDtypesGgmlCannotConvert) {
  for (const fbs::ScalarType dtype :
       {fbs::ScalarType::BYTE,
        fbs::ScalarType::CHAR,
        fbs::ScalarType::SHORT,
        fbs::ScalarType::DOUBLE}) {
    SCOPED_TRACE(static_cast<int>(dtype));
    TestProgram program;
    program.input("x", dtype, {2, 3});
    program.call(
        "y",
        "torch.ops.aten._to_copy.default",
        {{"self", std::string("x")}},
        fbs::ScalarType::FLOAT,
        {2, 3});
    auto host = GgmlEngineHost::create();
    auto context = make_context(*host, program.finish("y"));
    try {
      context->compile("forward");
      FAIL() << "compile accepted a dtype ggml cannot convert";
    } catch (const std::runtime_error& error) {
      EXPECT_NE(
          std::string(error.what()).find("unsupported tensor dtype"),
          std::string::npos)
          << error.what();
    }
  }
}

// rms_norm of x = 2 with a Half weight of 0.5 gives 0.5, for a Half or a
// Float x.
TEST(GgmlEngineTest, RmsNormWithAHalfWeight) {
  for (const bool half_input : {true, false}) {
    SCOPED_TRACE(half_input);
    const fbs::ScalarType dtype =
        half_input ? fbs::ScalarType::HALF : fbs::ScalarType::FLOAT;
    TestProgram program;
    program.input("x", dtype, {2, 16});
    program.input("w", fbs::ScalarType::HALF, {16});
    program.call(
        "y",
        "torch.ops.aten.rms_norm.default",
        {{"input", std::string("x")},
         {"normalized_shape", std::vector<int64_t>{16}},
         {"weight", std::string("w")},
         {"eps", 1e-6}},
        dtype,
        {2, 16});
    auto host = GgmlEngineHost::create();
    auto context = make_context(*host, program.finish("y"));
    auto executable = context->compile("forward");
    // IEEE half bit patterns of 2.0 and 0.5.
    const std::vector<uint16_t> half_x(32, 0x4000);
    const std::vector<float> float_x(32, 2.0f);
    const std::vector<uint16_t> w(16, 0x3800);
    if (half_input) {
      executable->set_input(0, half_x.data(), 32, ScalarType::Half);
    } else {
      executable->set_input(0, float_x.data(), 32, ScalarType::Float);
    }
    executable->set_input(1, w.data(), w.size(), ScalarType::Half);
    executable->execute();
    std::vector<float> y(32);
    executable->get_output(0, y.data(), y.size(), ScalarType::Float);
    for (const float element : y) {
      EXPECT_NEAR(element, 0.5f, 1e-3f);
    }
  }
}

// x = 1, w = 1 and b = 0.5, all Half, so each output is 4.5.
TEST(GgmlEngineTest, HalfLinearAddsAHalfBias) {
  TestProgram program;
  program.input("x", fbs::ScalarType::HALF, {1, 4});
  program.input("w", fbs::ScalarType::HALF, {2, 4});
  program.input("b", fbs::ScalarType::HALF, {2});
  program.call(
      "y",
      "torch.ops.aten.linear.default",
      {{"input", std::string("x")},
       {"weight", std::string("w")},
       {"bias", std::string("b")}},
      fbs::ScalarType::HALF,
      {1, 2});
  auto host = GgmlEngineHost::create();
  auto context = make_context(*host, program.finish("y"));
  auto executable = context->compile("forward");
  // IEEE half bit patterns of 1.0 and 0.5.
  const std::vector<uint16_t> x(4, 0x3C00);
  const std::vector<uint16_t> w(8, 0x3C00);
  const std::vector<uint16_t> b(2, 0x3800);
  executable->set_input(0, x.data(), x.size(), ScalarType::Half);
  executable->set_input(1, w.data(), w.size(), ScalarType::Half);
  executable->set_input(2, b.data(), b.size(), ScalarType::Half);
  executable->execute();
  std::vector<float> y(2);
  executable->get_output(0, y.data(), y.size(), ScalarType::Float);
  EXPECT_EQ(y, std::vector<float>(2, 4.5f));
}

// make_context's Q4_K weight against a Half input of 1s, as in
// ExecutesPackedQ4KLinear.
TEST(GgmlEngineTest, Q4KLinearOfAHalfInput) {
  TestProgram program;
  program.q4_k_weight("weight");
  program.input("x", fbs::ScalarType::HALF, {1, 256});
  program.call(
      "y",
      "torch.ops.aten.linear.default",
      {{"input", std::string("x")}, {"weight", std::string("weight")}},
      fbs::ScalarType::HALF,
      {1, 1});
  auto host = GgmlEngineHost::create();
  auto context = make_context(*host, program.finish("y"));
  auto executable = context->compile("forward");
  // The IEEE half bit pattern of 1.0.
  const std::vector<uint16_t> x(256, 0x3C00);
  executable->set_input(0, x.data(), x.size(), ScalarType::Half);
  executable->execute();
  float y = 0.0f;
  executable->get_output(0, &y, 1, ScalarType::Float);
  EXPECT_NEAR(y, 128.0f * 1.0f + 128.0f * 2.0f, 0.5f);
}

TEST(GgmlEngineTest, RejectsKvCacheUpdateOutsideCache) {
  auto host = GgmlEngineHost::create();
  EXPECT_NO_THROW(
      make_context(*host, make_cache_update_program(7))->compile("forward"));
  for (const int64_t start :
       {int64_t{8}, int64_t{-1}, std::numeric_limits<int64_t>::max()}) {
    SCOPED_TRACE(start);
    EXPECT_THROW(
        make_context(*host, make_cache_update_program(start))
            ->compile("forward"),
        std::runtime_error);
  }
}

TEST(GgmlEngineTest, RejectsCustomSdpaStartOutsideCache) {
  for (const auto& [start, valid] :
       {std::pair{int64_t{6}, true},
        std::pair{int64_t{7}, false},
        std::pair{int64_t{-1}, false}}) {
    SCOPED_TRACE(start);
    TestProgram program;
    program.input("query", fbs::ScalarType::FLOAT, {1, 2, 2, 8});
    program.buffer("key", {1, 8, 2, 8});
    program.buffer("value", {1, 8, 2, 8});
    program.call(
        "attention",
        "torch.ops.llama.custom_sdpa.default",
        {{"query", std::string("query")},
         {"key", std::string("key")},
         {"value", std::string("value")},
         {"start_pos", start}},
        fbs::ScalarType::FLOAT,
        {1, 2, 2, 8});
    auto host = GgmlEngineHost::create();
    auto context = make_context(*host, program.finish("attention"));
    if (valid) {
      EXPECT_NO_THROW(context->compile("forward"));
    } else {
      EXPECT_THROW(context->compile("forward"), std::runtime_error);
    }
  }
}

// With a mask, custom_sdpa attends the whole cache whatever start_pos is. The
// query is zero, so each output is the mean of the value rows the mask leaves
// open, rows 0 to 3, where row r holds r.
TEST(GgmlEngineTest, MaskedCustomSdpaAttendsTheWholeCache) {
  TestProgram program;
  program.input("query", fbs::ScalarType::FLOAT, {1, 1, 2, 8});
  program.input("key", fbs::ScalarType::FLOAT, {1, 8, 2, 8});
  program.input("value", fbs::ScalarType::FLOAT, {1, 8, 2, 8});
  program.input("mask", fbs::ScalarType::FLOAT, {1, 8});
  program.call(
      "attention",
      "torch.ops.llama.custom_sdpa.default",
      {{"query", std::string("query")},
       {"key", std::string("key")},
       {"value", std::string("value")},
       {"start_pos", int64_t{0}},
       {"attn_mask", std::string("mask")}},
      fbs::ScalarType::FLOAT,
      {1, 1, 2, 8});
  auto host = GgmlEngineHost::create();
  auto context = make_context(*host, program.finish("attention"));
  auto executable = context->compile("forward");
  const std::vector<float> query(16, 0.0f);
  const std::vector<float> key(128, 0.0f);
  std::vector<float> value(128);
  for (size_t i = 0; i < value.size(); ++i) {
    value[i] = static_cast<float>(i / 16);
  }
  const float inf = std::numeric_limits<float>::infinity();
  const std::vector<float> mask = {
      0.0f, 0.0f, 0.0f, 0.0f, -inf, -inf, -inf, -inf};
  executable->set_input(0, query.data(), query.size(), ScalarType::Float);
  executable->set_input(1, key.data(), key.size(), ScalarType::Float);
  executable->set_input(2, value.data(), value.size(), ScalarType::Float);
  executable->set_input(3, mask.data(), mask.size(), ScalarType::Float);
  executable->execute();
  std::vector<float> output(16);
  executable->get_output(0, output.data(), output.size(), ScalarType::Float);
  for (const float element : output) {
    EXPECT_NEAR(element, 1.5f, 1e-5f);
  }
}

// A Half query of 1s against Half caches whose row 1 holds keys of 0.25 and
// values of 1, other rows 0: at start_pos 1 with scale 1, each output is
// sigmoid(8 * 0.25).
TEST(GgmlEngineTest, CustomSdpaReadsAHalfQuery) {
  TestProgram program;
  program.input("query", fbs::ScalarType::HALF, {1, 1, 2, 8});
  program.input("key", fbs::ScalarType::HALF, {1, 8, 2, 8});
  program.input("value", fbs::ScalarType::HALF, {1, 8, 2, 8});
  program.call(
      "attention",
      "torch.ops.llama.custom_sdpa.default",
      {{"query", std::string("query")},
       {"key", std::string("key")},
       {"value", std::string("value")},
       {"start_pos", int64_t{1}},
       {"scale", 1.0}},
      fbs::ScalarType::HALF,
      {1, 1, 2, 8});
  auto host = GgmlEngineHost::create();
  auto context = make_context(*host, program.finish("attention"));
  auto executable = context->compile("forward");
  // IEEE half bit patterns of 1.0 and 0.25.
  constexpr uint16_t kOne = 0x3C00;
  constexpr uint16_t kQuarter = 0x3400;
  const std::vector<uint16_t> query(16, kOne);
  std::vector<uint16_t> key(128, 0);
  std::vector<uint16_t> value(128, 0);
  std::fill_n(key.begin() + 16, 16, kQuarter);
  std::fill_n(value.begin() + 16, 16, kOne);
  executable->set_input(0, query.data(), query.size(), ScalarType::Half);
  executable->set_input(1, key.data(), key.size(), ScalarType::Half);
  executable->set_input(2, value.data(), value.size(), ScalarType::Half);
  executable->execute();
  std::vector<float> output(16);
  executable->get_output(0, output.data(), output.size(), ScalarType::Float);
  for (const float element : output) {
    EXPECT_NEAR(element, 0.8808f, 1e-2f);
  }
}

TEST(GgmlEngineTest, RejectsCustomSdpaIncompatibleShapes) {
  struct Shapes {
    std::vector<int64_t> query;
    std::vector<int64_t> key;
    std::vector<int64_t> value;
    bool valid;
  };
  // After the valid case: head dim, batch, query heads that are not a
  // multiple of the KV heads, and a value cache shaped unlike the key cache.
  const std::vector<Shapes> cases = {
      {{1, 2, 2, 8}, {1, 8, 2, 8}, {1, 8, 2, 8}, true},
      {{1, 2, 2, 8}, {1, 8, 2, 4}, {1, 8, 2, 4}, false},
      {{2, 2, 2, 8}, {1, 8, 2, 8}, {1, 8, 2, 8}, false},
      {{1, 2, 3, 8}, {1, 8, 2, 8}, {1, 8, 2, 8}, false},
      {{1, 2, 2, 8}, {1, 8, 2, 8}, {1, 8, 1, 8}, false},
  };
  for (size_t i = 0; i < cases.size(); ++i) {
    SCOPED_TRACE(i);
    TestProgram program;
    program.input("query", fbs::ScalarType::FLOAT, cases[i].query);
    program.buffer("key", cases[i].key);
    program.buffer("value", cases[i].value);
    program.call(
        "attention",
        "torch.ops.llama.custom_sdpa.default",
        {{"query", std::string("query")},
         {"key", std::string("key")},
         {"value", std::string("value")},
         {"start_pos", int64_t{0}}},
        fbs::ScalarType::FLOAT,
        cases[i].query);
    auto host = GgmlEngineHost::create();
    auto context = make_context(*host, program.finish("attention"));
    if (cases[i].valid) {
      EXPECT_NO_THROW(context->compile("forward"));
    } else {
      EXPECT_THROW(context->compile("forward"), std::runtime_error);
    }
  }
}

// softmax(q k^T / sqrt(D)) v over cache rows [0, keys) for every query, with
// query head h reading KV head h / (H / Hkv). q is [T, H, D]; k and v are
// [S, Hkv, D].
std::vector<float> reference_custom_sdpa(
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    int64_t heads,
    int64_t kv_heads,
    int64_t dim,
    int64_t keys) {
  std::vector<float> out(q.size());
  for (int64_t row = 0; row < static_cast<int64_t>(q.size()) / dim; ++row) {
    const float* query = &q[row * dim];
    const int64_t kv = row % heads / (heads / kv_heads);
    std::vector<double> weights(keys);
    for (int64_t s = 0; s < keys; ++s) {
      weights[s] = std::exp(
          std::inner_product(
              query, query + dim, &k[(s * kv_heads + kv) * dim], 0.0) /
          std::sqrt(static_cast<double>(dim)));
    }
    const double total = std::accumulate(weights.begin(), weights.end(), 0.0);
    for (int64_t d = 0; d < dim; ++d) {
      double sum = 0.0;
      for (int64_t s = 0; s < keys; ++s) {
        sum += weights[s] * v[(s * kv_heads + kv) * dim + d];
      }
      out[row * dim + d] = static_cast<float>(sum / total);
    }
  }
  return out;
}

TEST(GgmlEngineTest, CustomSdpaMatchesReferenceWithoutMask) {
  constexpr int64_t kHeads = 4;
  constexpr int64_t kKvHeads = 2;
  constexpr int64_t kDim = 8;
  constexpr int64_t kCache = 8;
  // A decode step and a two-token prefill, both past the start of the cache.
  for (const auto& [tokens, start] :
       {std::pair{int64_t{1}, int64_t{3}}, std::pair{int64_t{2}, int64_t{2}}}) {
    SCOPED_TRACE(tokens);
    TestProgram program;
    program.input("query", fbs::ScalarType::FLOAT, {1, tokens, kHeads, kDim});
    program.input("key", fbs::ScalarType::FLOAT, {1, kCache, kKvHeads, kDim});
    program.input("value", fbs::ScalarType::FLOAT, {1, kCache, kKvHeads, kDim});
    program.call(
        "attention",
        "torch.ops.llama.custom_sdpa.default",
        {{"query", std::string("query")},
         {"key", std::string("key")},
         {"value", std::string("value")},
         {"start_pos", start}},
        fbs::ScalarType::FLOAT,
        {1, tokens, kHeads, kDim});
    auto host = GgmlEngineHost::create();
    auto context = make_context(*host, program.finish("attention"));
    auto executable = context->compile("forward");
    std::vector<float> q(tokens * kHeads * kDim);
    std::vector<float> k(kCache * kKvHeads * kDim);
    std::vector<float> v(k.size());
    for (size_t i = 0; i < q.size(); ++i) {
      q[i] = static_cast<float>(i * 5 % 7) / 7.0f - 0.5f;
    }
    for (size_t i = 0; i < k.size(); ++i) {
      k[i] = static_cast<float>(i * 3 % 11) / 11.0f - 0.5f;
      v[i] = static_cast<float>(i % 13);
    }
    executable->set_input(0, q.data(), q.size(), ScalarType::Float);
    executable->set_input(1, k.data(), k.size(), ScalarType::Float);
    executable->set_input(2, v.data(), v.size(), ScalarType::Float);
    executable->execute();
    std::vector<float> out(q.size());
    executable->get_output(0, out.data(), out.size(), ScalarType::Float);
    const std::vector<float> expected =
        reference_custom_sdpa(q, k, v, kHeads, kKvHeads, kDim, start + tokens);
    for (size_t i = 0; i < out.size(); ++i) {
      EXPECT_NEAR(out[i], expected[i], 1e-3f) << i;
    }
  }
}

TEST(GgmlEngineTest, RejectsQuantizedGraphInputsAndOutputs) {
  TestProgram as_input;
  as_input.input("x", fbs::ScalarType::FLOAT, {1, 256});
  as_input.input("w", fbs::ScalarType::BYTE, {1, 256}, "gguf:q4_k");
  as_input.call(
      "y",
      "torch.ops.aten.linear.default",
      {{"input", std::string("x")}, {"weight", std::string("w")}},
      fbs::ScalarType::FLOAT,
      {1, 1});
  TestProgram as_output;
  as_output.q4_k_weight("weight");
  auto host = GgmlEngineHost::create();
  for (const std::vector<uint8_t>& program :
       {as_input.finish("y"), as_output.finish("weight")}) {
    auto context = make_context(*host, program);
    EXPECT_THROW(context->compile("forward"), std::runtime_error);
  }
}

TEST(GgmlEngineTest, ChecksEmbeddingIndicesOnTheHost) {
  TestProgram program;
  program.q4_k_weight("weight");
  program.input("ids", fbs::ScalarType::LONG, {1, 2});
  program.call(
      "rows",
      "torch.ops.aten.embedding.default",
      {{"weight", std::string("weight")}, {"indices", std::string("ids")}},
      fbs::ScalarType::FLOAT,
      {1, 2, 256});
  auto host = GgmlEngineHost::create();
  auto context = make_context(*host, program.finish("rows"));
  auto executable = context->compile("forward");
  std::vector<float> rows(512);
  const std::vector<int64_t> in_range = {0, 0};
  executable->set_input(0, in_range.data(), 2, ScalarType::Long);
  executable->execute();
  executable->get_output(0, rows.data(), rows.size(), ScalarType::Float);
  EXPECT_EQ(std::accumulate(rows.begin(), rows.end(), 0.0f), 768.0f);
  for (const int64_t bad : {int64_t{1}, int64_t{-1}}) {
    SCOPED_TRACE(bad);
    const std::vector<int64_t> ids = {0, bad};
    executable->set_input(0, ids.data(), 2, ScalarType::Long);
    EXPECT_THROW(executable->execute(), std::out_of_range);
  }
}

TEST(GgmlEngineTest, RejectsPackedWeightOutsideLinearOrEmbedding) {
  TestProgram program;
  program.q4_k_weight("weight");
  program.input("x", fbs::ScalarType::FLOAT, {1, 1});
  program.call(
      "product",
      "torch.ops.aten.mm.default",
      {{"self", std::string("x")}, {"other", std::string("weight")}},
      fbs::ScalarType::FLOAT,
      {1, 256});
  expect_compile_error(program.finish("product"), "linear or embedding");
}

TEST(GgmlEngineTest, RejectsPackedWeightWithPartialBlocks) {
  TestProgram program;
  program.q4_k_weight("weight", {1, 100});
  program.input("x", fbs::ScalarType::FLOAT, {1, 100});
  program.call(
      "y",
      "torch.ops.aten.linear.default",
      {{"input", std::string("x")}, {"weight", std::string("weight")}},
      fbs::ScalarType::FLOAT,
      {1, 1});
  expect_compile_error(program.finish("y"), "divisible by 256");
}

TEST(GgmlEngineTest, RejectsUnsupportedOpOnUnquantizedInput) {
  TestProgram program;
  program.input("x", fbs::ScalarType::FLOAT, {1, 256});
  program.call(
      "y",
      "torch.ops.aten.unsupported.default",
      {{"self", std::string("x")}},
      fbs::ScalarType::FLOAT,
      {1, 256});
  expect_compile_error(program.finish("y"), "unsupported PTG op");
}

TEST(GgmlEngineTest, ChecksEmbeddingIndicesThroughAliasingViews) {
  TestProgram program;
  program.q4_k_weight("weight");
  program.input("ids", fbs::ScalarType::LONG, {2});
  program.call(
      "batched",
      "torch.ops.aten.unsqueeze.default",
      {{"self", std::string("ids")}, {"dim", int64_t{0}}},
      fbs::ScalarType::LONG,
      {1, 2});
  program.call(
      "rows",
      "torch.ops.aten.embedding.default",
      {{"weight", std::string("weight")}, {"indices", std::string("batched")}},
      fbs::ScalarType::FLOAT,
      {1, 2, 256});
  auto host = GgmlEngineHost::create();
  auto context = make_context(*host, program.finish("rows"));
  auto executable = context->compile("forward");
  std::vector<float> rows(512);
  const std::vector<int64_t> in_range = {0, 0};
  executable->set_input(0, in_range.data(), 2, ScalarType::Long);
  executable->execute();
  executable->get_output(0, rows.data(), rows.size(), ScalarType::Float);
  EXPECT_EQ(std::accumulate(rows.begin(), rows.end(), 0.0f), 768.0f);
  const std::vector<int64_t> out_of_range = {0, 1};
  executable->set_input(0, out_of_range.data(), 2, ScalarType::Long);
  EXPECT_THROW(executable->execute(), std::out_of_range);
}

TEST(GgmlEngineTest, RejectsEmbeddingIndicesNotFromAnInput) {
  TestProgram program;
  program.q4_k_weight("weight");
  program.input("ids", fbs::ScalarType::LONG, {1, 2});
  program.call(
      "shifted",
      "torch.ops.aten.add.Tensor",
      {{"self", std::string("ids")}, {"other", std::string("ids")}},
      fbs::ScalarType::LONG,
      {1, 2});
  program.call(
      "rows",
      "torch.ops.aten.embedding.default",
      {{"weight", std::string("weight")}, {"indices", std::string("shifted")}},
      fbs::ScalarType::FLOAT,
      {1, 2, 256});
  auto host = GgmlEngineHost::create();
  auto context = make_context(*host, program.finish("rows"));
  try {
    context->compile("forward");
    FAIL() << "compile accepted computed embedding indices";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("graph input"), std::string::npos)
        << error.what();
  }
}

TEST(GgmlEngineTest, ViewsQ4_0ScalesOfMappedPackage) {
  for (const bool misaligned : {false, true}) {
    SCOPED_TRACE(misaligned);
    // Every weight is q = 1, stored as the nibble q + 8. The eight groups of 32
    // alternate fp16 scales 0.5 and 1.0.
    constexpr size_t kScaleBytes = 16;
    constexpr size_t kWeightBytes = 128;
    std::string header =
        R"({"weight_scale":{"dtype":"F16","shape":[1,8],"data_offsets":[0,16]},)"
        R"("weight":{"dtype":"U8","shape":[128],"data_offsets":[16,144]}})";
    // The safetensors member comes first, so its payload starts right after
    // the local header, name, size prefix and JSON header. read_constant views
    // the scales in place only at an even offset.
    const size_t payload_offset =
        30 + std::strlen(kSafeTensorsEntry) + sizeof(uint64_t) + header.size();
    header.append((payload_offset + misaligned) % 2, ' ');
    std::vector<uint8_t> safetensors =
        testing::make_safetensors(header, kScaleBytes + kWeightBytes);
    uint8_t* scales =
        safetensors.data() + safetensors.size() - kScaleBytes - kWeightBytes;
    for (size_t group = 0; group < 8; ++group) {
      scales[2 * group + 1] = group % 2 == 0 ? 0x38 : 0x3C;
    }
    std::fill_n(scales + kScaleBytes, kWeightBytes, uint8_t{0x99});
    const std::vector<uint8_t> program_bytes = make_linear_program(
        "torch.ops.aten.linear.default", /*with_cache=*/false, /*affine=*/true);
    const std::vector<uint8_t> zip = testing::make_zip({
        {kSafeTensorsEntry, std::move(safetensors)},
        {kProgramEntry, program_bytes},
    });

    struct TempFile {
      std::filesystem::path path;
      ~TempFile() {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
      }
    } file{
        ::testing::TempDir() + "q4_0_linear_" + std::to_string(::getpid()) +
        (misaligned ? "_misaligned.ptn" : ".ptn")};
    std::ofstream(file.path, std::ios::binary)
        .write(
            reinterpret_cast<const char*>(zip.data()),
            static_cast<std::streamsize>(zip.size()));
    auto package =
        std::make_shared<const Package>(Package::load(file.path.string()));
    const std::optional<OwnedBytes> view =
        package->view_constant("weight_scale");
    ASSERT_TRUE(view.has_value());
    ASSERT_EQ(reinterpret_cast<uintptr_t>(view->span().data()) % 2, misaligned);

    auto host = GgmlEngineHost::create();
    auto context = host->create_ggml_context(
        std::make_shared<const Program>(
            Program::load(program_bytes.data(), program_bytes.size())),
        std::move(package));
    auto executable = context->compile("forward");
    const std::vector<float> input(256, 1.0f);
    executable->set_input(0, input.data(), input.size(), ScalarType::Float);
    executable->execute();
    float output = 0.0f;
    executable->get_output(0, &output, 1, ScalarType::Float);
    // Q4_0 matmuls quantize the activations to Q8_0, whose scale is fp16.
    EXPECT_NEAR(output, 128.0f * 0.5f + 128.0f * 1.0f, 5e-2f);
    // Only misaligned fp16 scales are copied; byte-sized weights are aligned.
    EXPECT_EQ(
        context->heap_copied_constant_bytes(), misaligned ? kScaleBytes : 0);
  }
}

TEST(GgmlEngineTest, GpuUnavailableOnCpuVariant) {
  GgmlEngineOptions options;
  options.device = GgmlDevice::Gpu;
  EXPECT_THROW(GgmlEngineHost::create(options), std::runtime_error);
}

} // namespace
} // namespace ptn
