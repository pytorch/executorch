// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/engine/ggml/GgmlEngine.h>

#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <executorch/backends/native/runtime/Program.h>
#include <executorch/backends/native/runtime/deserialize/OwnedBytes.h>
#include <executorch/backends/native/runtime/deserialize/Package.h>
#include <executorch/backends/native/runtime/deserialize/test/PackageTestData.h>
#include <executorch/backends/native/runtime/engine/Engine.h>
#include <executorch/backends/native/runtime/native_graph_generated.h>
#include <executorch/extension/llm/cache/cache_registry.h>
#include <executorch/runtime/backend/options.h>
#include <flatbuffers/flatbuffers.h>
#include <ggml.h>
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
constexpr size_t kQ4_0BlockBytes = 18;

// The safetensors header of the affine linear program's constants.
constexpr const char* kQ4_0SafeTensorsHeader =
    R"({"weight_scale":{"dtype":"F16","shape":[1,8],"data_offsets":[0,16]},)"
    R"("weight":{"dtype":"U8","shape":[128],"data_offsets":[16,144]}})";

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

// For tests of quantized linears whose activation numerics aren't under test.
constexpr GgmlEngineOptions kAllowActivationQuant{
    .allow_activation_quant_for_weight_only = true};

std::unique_ptr<GgmlEngineContext> make_context(
    GgmlEngineHost& host,
    const std::vector<uint8_t>& program_bytes,
    const EngineContextOptions& options = {}) {
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
  return host.create_ggml_context(
      std::move(program), std::move(package), options);
}

// A bool argument, wrapped so that string literals never convert to it.
struct BoolValue {
  bool value = false;
};

// An optional tensor list: a tensor name per entry, or nullopt for None.
using TestOptionalTensors = std::vector<std::optional<std::string>>;

// A tensor (by name), an int, a float, an int list, None, a bool, or an
// optional tensor list.
using TestArg = std::variant<
    std::string,
    int64_t,
    double,
    std::vector<int64_t>,
    std::monostate,
    BoolValue,
    TestOptionalTensors>;

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

  // make_context's packaged block_q4_K as a parameter, by default [1, 256],
  // labeled with `codec`.
  void q4_k_weight(
      const std::string& name,
      const std::vector<int64_t>& shape = {1, 256},
      const char* codec = "gguf:q4_k") {
    const auto quant = fbs::CreateQuantSpec(
        builder_,
        fbs::QuantScheme::PackedQuant,
        fbs::CreatePackedQuantDirect(builder_, codec).Union());
    const auto meta =
        create_tensor_meta(builder_, fbs::ScalarType::BYTE, shape, quant);
    placeholder(name, meta);
    bindings_.push_back(fbs::CreateNamedTensorRefDirect(
        builder_, name.c_str(), "weight", meta, fbs::InputKind::PARAMETER));
  }

  // Symmetric int4 AffineGroup weights `weight` with float scales
  // `weight_scale` per group of 32, in a persistent buffer if `mutated`.
  void affine_weight(
      const std::string& name,
      const std::vector<int64_t>& shape,
      bool mutated = false) {
    const auto quant = fbs::CreateQuantSpec(
        builder_,
        fbs::QuantScheme::AffineGroup,
        fbs::CreateAffineGroupDirect(
            builder_, "weight_scale", fbs::ScalarType::FLOAT, -8, 7, 32)
            .Union());
    const auto meta =
        create_tensor_meta(builder_, fbs::ScalarType::BYTE, shape, quant);
    placeholder(name, meta);
    bindings_.push_back(fbs::CreateNamedTensorRefDirect(
        builder_,
        name.c_str(),
        "weight",
        meta,
        mutated ? fbs::InputKind::BUFFER : fbs::InputKind::PARAMETER,
        mutated));
    const std::string scale = name + "_scale";
    const auto scale_meta = create_tensor_meta(
        builder_, fbs::ScalarType::FLOAT, {shape.at(0), shape.at(1) / 32});
    placeholder(scale, scale_meta);
    bindings_.push_back(fbs::CreateNamedTensorRefDirect(
        builder_,
        scale.c_str(),
        "weight_scale",
        scale_meta,
        fbs::InputKind::PARAMETER));
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

  // A node producing one tensor list whose elements, named `names`, are F32
  // tensors of `shape`.
  void call_list(
      const std::string& name,
      const char* target,
      const std::vector<std::pair<std::string, TestArg>>& args,
      const std::vector<std::string>& names,
      const std::vector<int64_t>& shape) {
    std::vector<flatbuffers::Offset<fbs::NamedArgument>> named;
    for (const auto& [arg_name, arg] : args) {
      named.push_back(fbs::CreateNamedArgumentDirect(
          builder_, arg_name.c_str(), create_arg(arg)));
    }
    for (const std::string& element : names) {
      values_.push_back(fbs::CreateTensorValueDirect(
          builder_,
          element.c_str(),
          create_tensor_meta(builder_, fbs::ScalarType::FLOAT, shape)));
    }
    const std::vector<flatbuffers::Offset<fbs::Output>> outputs = {
        fbs::CreateOutput(
            builder_,
            builder_.CreateString(""),
            0,
            fbs::OutputValueKind::TENSOR_LIST,
            create_strings(builder_, names))};
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
    if (const auto* flag = std::get_if<BoolValue>(&arg)) {
      return fbs::CreateArgument(
          builder_,
          fbs::ArgumentValue::BoolArg,
          fbs::CreateBoolArgDirect(builder_, flag->value).Union());
    }
    if (const auto* list = std::get_if<std::vector<int64_t>>(&arg)) {
      return fbs::CreateArgument(
          builder_,
          fbs::ArgumentValue::IntListArg,
          fbs::CreateIntListArgDirect(builder_, list).Union());
    }
    if (const auto* tensors = std::get_if<TestOptionalTensors>(&arg)) {
      std::vector<flatbuffers::Offset<flatbuffers::String>> names;
      std::vector<uint8_t> has_value;
      for (const auto& name : *tensors) {
        names.push_back(builder_.CreateString(name.value_or("")));
        has_value.push_back(name.has_value());
      }
      return fbs::CreateArgument(
          builder_,
          fbs::ArgumentValue::OptionalTensorListArg,
          fbs::CreateOptionalTensorListArg(
              builder_,
              builder_.CreateVector(names),
              builder_.CreateVector(has_value))
              .Union());
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
  auto host = GgmlEngineHost::create(kAllowActivationQuant);
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
    auto host = GgmlEngineHost::create(GgmlEngineOptions{
        .max_cached_graphs = max_cached_graphs,
        .allow_activation_quant_for_weight_only = true});
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

TEST(GgmlEngineTest, RejectsOutputReadsAfterInputChangesOrFailedExecute) {
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
  const std::vector<int64_t> good = {0, 0};
  const std::vector<int64_t> bad = {0, 1};
  std::vector<float> rows(512);
  const auto read = [&] {
    executable->get_output(0, rows.data(), rows.size(), ScalarType::Float);
  };
  executable->set_input(0, good.data(), 2, ScalarType::Long);
  executable->execute();
  EXPECT_NO_THROW(read());
  executable->set_input(0, good.data(), 2, ScalarType::Long);
  EXPECT_THROW(read(), std::logic_error);
  executable->execute();
  executable->resize_input(0, {1, 2});
  EXPECT_THROW(read(), std::logic_error);
  executable->set_input(0, bad.data(), 2, ScalarType::Long);
  EXPECT_THROW(executable->execute(), std::out_of_range);
  EXPECT_THROW(read(), std::logic_error);
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
  auto host = GgmlEngineHost::create(kAllowActivationQuant);
  auto context = make_context(
      *host,
      make_linear_program(
          "torch.ops.aten.linear.default", /*with_cache=*/true));
  auto executable = context->compile("forward");
  EXPECT_THROW(context->compile("forward"), std::runtime_error);
}

void expect_compile_error(
    GgmlEngineContext& context,
    const std::string& message) {
  try {
    context.compile("forward");
    FAIL() << "compile accepted the program";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find(message), std::string::npos)
        << error.what();
  }
}

void expect_compile_error(
    const std::vector<uint8_t>& program,
    const std::string& message) {
  auto host = GgmlEngineHost::create();
  expect_compile_error(*make_context(*host, program), message);
}

TEST(GgmlEngineTest, RejectsWeightOnlyQuantizedLinearsByDefault) {
  expect_compile_error(
      make_linear_program(), "allow_activation_quant_for_weight_only");
  const std::vector<uint8_t> program_bytes = make_linear_program(
      "torch.ops.aten.linear.default", /*with_cache=*/false, /*affine=*/true);
  auto host = GgmlEngineHost::create();
  auto context = host->create_ggml_context(
      std::make_shared<const Program>(
          Program::load(program_bytes.data(), program_bytes.size())),
      std::make_shared<const Package>(
          Package::load(OwnedBytes::from_vector(testing::make_zip({
              {kProgramEntry, program_bytes},
              {kSafeTensorsEntry,
               testing::make_safetensors(
                   kQ4_0SafeTensorsHeader, /*scales + weights=*/16 + 128)},
          })))));
  expect_compile_error(*context, "allow_activation_quant_for_weight_only");
}

TEST(GgmlEngineTest, SqrtPreservesNonFiniteValuesAndSignedZeroOnReplay) {
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  for (const size_t cached : {size_t{0}, size_t{1}}) {
    for (const bool transposed : {false, true}) {
      TestProgram program;
      program.input("x", fbs::ScalarType::FLOAT, {2, 4});
      std::string input = "x";
      if (transposed) {
        program.call(
            "xt",
            "torch.ops.aten.permute_copy.default",
            {{"self", input}, {"dims", std::vector<int64_t>{1, 0}}},
            fbs::ScalarType::FLOAT,
            {4, 2});
        input = "xt";
      }
      program.call(
          "y",
          "torch.ops.aten.sqrt.default",
          {{"self", input}},
          fbs::ScalarType::FLOAT,
          transposed ? std::vector<int64_t>{4, 2} : std::vector<int64_t>{2, 4});
      auto host = GgmlEngineHost::create(
          GgmlEngineOptions{.cpu_threads = 4, .max_cached_graphs = cached});
      auto context = make_context(*host, program.finish("y"));
      auto executable = context->compile("forward");
      for (const float scale : {1.0f, 4.0f}) {
        const std::vector<float> x{
            0.0f,
            -0.0f,
            scale * 4.0f,
            scale * 2.0f,
            inf,
            -scale,
            nan,
            scale * 9.0f};
        executable->set_input(0, x.data(), x.size(), ScalarType::Float);
        executable->execute();
        std::vector<float> y(x.size());
        executable->get_output(0, y.data(), y.size(), ScalarType::Float);
        for (size_t i = 0; i < y.size(); ++i) {
          const size_t source = transposed ? (i % 2) * 4 + i / 2 : i;
          const float expected =
              static_cast<float>(std::sqrt(static_cast<double>(x[source])));
          if (std::isnan(expected)) {
            EXPECT_TRUE(std::isnan(y[i]));
          } else {
            EXPECT_EQ(y[i], expected);
            EXPECT_EQ(std::signbit(y[i]), std::signbit(expected));
          }
        }
      }
    }
  }
}

TEST(GgmlEngineTest, SqrtRejectsUnsupportedTypes) {
  TestProgram program;
  program.input("x", fbs::ScalarType::LONG, {2, 4});
  program.call(
      "y",
      "torch.ops.aten.sqrt.default",
      {{"self", std::string("x")}},
      fbs::ScalarType::FLOAT,
      {2, 4});
  expect_compile_error(program.finish("y"), "F32");
}

// tanh of x [4, 3], or of its transpose.
std::vector<uint8_t> make_tanh_program(
    bool transposed,
    fbs::ScalarType dtype = fbs::ScalarType::FLOAT) {
  TestProgram program;
  program.input("x", dtype, {4, 3});
  std::string self = "x";
  if (transposed) {
    program.call(
        "xt",
        "torch.ops.aten.permute_copy.default",
        {{"self", std::string("x")}, {"dims", std::vector<int64_t>{1, 0}}},
        dtype,
        {3, 4});
    self = "xt";
  }
  program.call(
      "y",
      "torch.ops.aten.tanh.default",
      {{"self", self}},
      dtype,
      transposed ? std::vector<int64_t>{3, 4} : std::vector<int64_t>{4, 3});
  return program.finish("y");
}

// Within 1e-6 relative of tanh in double; NaN, +-1 and signed zero exactly.
TEST(GgmlEngineTest, TanhMatchesLibmIncludingNonFiniteValues) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  const std::vector<float> x = {
      -0.0f,
      0.0f,
      1.5f,
      -2.25f,
      30.0f,
      -30.0f,
      kInf,
      -kInf,
      std::nanf(""),
      1e-20f,
      3e38f,
      0.75f};
  for (const bool transposed : {false, true}) {
    for (const size_t cached : {size_t{0}, size_t{1}}) {
      SCOPED_TRACE(std::to_string(transposed) + std::to_string(cached));
      auto host = GgmlEngineHost::create(
          GgmlEngineOptions{.max_cached_graphs = cached});
      auto context = make_context(*host, make_tanh_program(transposed));
      auto executable = context->compile("forward");
      for (const float scale : {1.0f, -0.5f}) {
        std::vector<float> input = x;
        for (float& value : input) {
          value *= scale;
        }
        executable->set_input(0, input.data(), input.size(), ScalarType::Float);
        executable->execute();
        std::vector<float> y(input.size());
        executable->get_output(0, y.data(), y.size(), ScalarType::Float);
        for (size_t i = 0; i < y.size(); ++i) {
          // y[r][c] reads x[c][r] when transposed.
          const size_t source = transposed ? (i % 4) * 3 + i / 4 : i;
          const auto expected =
              static_cast<float>(std::tanh(static_cast<double>(input[source])));
          SCOPED_TRACE(input[source]);
          if (std::isnan(expected)) {
            EXPECT_TRUE(std::isnan(y[i])) << y[i];
          } else if (std::abs(expected) == 1.0f || expected == 0.0f) {
            EXPECT_EQ(y[i], expected);
            EXPECT_EQ(std::signbit(y[i]), std::signbit(expected));
          } else {
            EXPECT_NEAR(y[i], expected, 1e-6 * std::abs(expected));
          }
        }
      }
    }
  }
}

TEST(GgmlEngineTest, RejectsIntegerTanh) {
  expect_compile_error(
      make_tanh_program(false, fbs::ScalarType::INT),
      "supports floating-point tensors only");
}

std::vector<uint8_t> make_elu_program(
    bool transposed,
    double alpha = 1.0,
    double scale = 1.0,
    double input_scale = 1.0) {
  TestProgram program;
  program.input("x", fbs::ScalarType::FLOAT, {4, 3});
  std::string self = "x";
  if (transposed) {
    program.call(
        "xt",
        "torch.ops.aten.permute_copy.default",
        {{"self", std::string("x")}, {"dims", std::vector<int64_t>{1, 0}}},
        fbs::ScalarType::FLOAT,
        {3, 4});
    self = "xt";
  }
  program.call(
      "y",
      "torch.ops.aten.elu.default",
      {{"self", self},
       {"alpha", alpha},
       {"scale", scale},
       {"input_scale", input_scale}},
      fbs::ScalarType::FLOAT,
      transposed ? std::vector<int64_t>{3, 4} : std::vector<int64_t>{4, 3});
  return program.finish("y");
}

// Within 1e-6 relative of expm1 in double; NaN, -1, infinities and zero
// exactly.
TEST(GgmlEngineTest, EluMatchesExpm1IncludingNonFiniteValues) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  const std::vector<float> x = {
      -0.0f,
      0.0f,
      1.5f,
      -2.25f,
      -1e-7f,
      -120.0f,
      kInf,
      -kInf,
      std::nanf(""),
      -1e-30f,
      3e38f,
      -0.75f};
  for (const bool transposed : {false, true}) {
    for (const size_t cached : {size_t{0}, size_t{1}}) {
      SCOPED_TRACE(std::to_string(transposed) + std::to_string(cached));
      auto host = GgmlEngineHost::create(
          GgmlEngineOptions{.max_cached_graphs = cached});
      auto context = make_context(*host, make_elu_program(transposed));
      auto executable = context->compile("forward");
      for (const float scale : {1.0f, -0.5f}) {
        std::vector<float> input = x;
        for (float& value : input) {
          value *= scale;
        }
        executable->set_input(0, input.data(), input.size(), ScalarType::Float);
        executable->execute();
        std::vector<float> y(input.size());
        executable->get_output(0, y.data(), y.size(), ScalarType::Float);
        for (size_t i = 0; i < y.size(); ++i) {
          // y[r][c] reads x[c][r] when transposed.
          const double v = input[transposed ? (i % 4) * 3 + i / 4 : i];
          const auto expected = static_cast<float>(v > 0 ? v : std::expm1(v));
          SCOPED_TRACE(v);
          if (std::isnan(expected)) {
            EXPECT_TRUE(std::isnan(y[i])) << y[i];
          } else if (
              std::isinf(expected) || expected == -1.0f || expected == 0.0f) {
            EXPECT_EQ(y[i], expected);
          } else {
            EXPECT_NEAR(y[i], expected, 1e-6 * std::abs(expected));
          }
        }
      }
    }
  }
}

TEST(GgmlEngineTest, RejectsEluWithOtherParameters) {
  for (const auto& [alpha, scale, input_scale] :
       {std::tuple{0.5, 1.0, 1.0},
        std::tuple{1.0, 2.0, 1.0},
        std::tuple{1.0, 1.0, 0.5}}) {
    expect_compile_error(
        make_elu_program(false, alpha, scale, input_scale),
        "supports only alpha, scale and input_scale 1");
  }
}

TEST(GgmlEngineTest, RejectsIntegerGelu) {
  TestProgram program;
  program.input("x", fbs::ScalarType::LONG, {2, 3});
  program.call(
      "y",
      "torch.ops.aten.gelu.default",
      {{"self", std::string("x")}},
      fbs::ScalarType::LONG,
      {2, 3});
  expect_compile_error(program.finish("y"), "floating-point");
}

// exp of x [4, 3], or of its transpose.
std::vector<uint8_t> make_exp_program(
    bool transposed,
    fbs::ScalarType dtype = fbs::ScalarType::FLOAT) {
  TestProgram program;
  program.input("x", dtype, {4, 3});
  std::string self = "x";
  if (transposed) {
    program.call(
        "xt",
        "torch.ops.aten.permute_copy.default",
        {{"self", std::string("x")}, {"dims", std::vector<int64_t>{1, 0}}},
        dtype,
        {3, 4});
    self = "xt";
  }
  program.call(
      "y",
      "torch.ops.aten.exp.default",
      {{"self", self}},
      dtype,
      transposed ? std::vector<int64_t>{3, 4} : std::vector<int64_t>{4, 3});
  return program.finish("y");
}

// Within 1e-6 relative of exp in double; NaN, infinities and zero exactly.
TEST(GgmlEngineTest, ExpMatchesLibmIncludingNonFiniteValues) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  const std::vector<float> x = {
      -0.0f,
      0.0f,
      1.5f,
      -2.25f,
      30.0f,
      -30.0f,
      100.0f,
      -80.0f,
      kInf,
      -kInf,
      std::nanf(""),
      1e-20f};
  for (const bool transposed : {false, true}) {
    for (const size_t cached : {size_t{0}, size_t{1}}) {
      SCOPED_TRACE(std::to_string(transposed) + std::to_string(cached));
      auto host = GgmlEngineHost::create(
          GgmlEngineOptions{.max_cached_graphs = cached});
      auto context = make_context(*host, make_exp_program(transposed));
      auto executable = context->compile("forward");
      for (const float scale : {1.0f, -0.5f}) {
        std::vector<float> input = x;
        for (float& value : input) {
          value *= scale;
        }
        executable->set_input(0, input.data(), input.size(), ScalarType::Float);
        executable->execute();
        std::vector<float> y(input.size());
        executable->get_output(0, y.data(), y.size(), ScalarType::Float);
        for (size_t i = 0; i < y.size(); ++i) {
          // y[r][c] reads x[c][r] when transposed.
          const size_t source = transposed ? (i % 4) * 3 + i / 4 : i;
          const auto expected =
              static_cast<float>(std::exp(static_cast<double>(input[source])));
          SCOPED_TRACE(input[source]);
          if (std::isnan(expected)) {
            EXPECT_TRUE(std::isnan(y[i])) << y[i];
          } else if (std::isinf(expected) || expected == 0.0f) {
            EXPECT_EQ(y[i], expected);
            EXPECT_EQ(std::signbit(y[i]), std::signbit(expected));
          } else {
            EXPECT_NEAR(y[i], expected, 1e-6 * std::abs(expected));
          }
        }
      }
    }
  }
}

// The Half or BFloat16 bits of x, which must be exact in that dtype.
uint16_t to_16_bit(float x, fbs::ScalarType dtype) {
  uint32_t bits = 0;
  std::memcpy(&bits, &x, sizeof(bits));
  if (dtype == fbs::ScalarType::BFLOAT16) {
    return static_cast<uint16_t>(bits >> 16);
  }
  const auto sign = static_cast<uint16_t>((bits >> 16) & 0x8000);
  if (std::isnan(x)) {
    return sign | 0x7E00;
  }
  if (std::isinf(x)) {
    return sign | 0x7C00;
  }
  if (x == 0.0f) {
    return sign;
  }
  const uint32_t exponent = ((bits >> 23) & 0xFF) - 127 + 15;
  return static_cast<uint16_t>(
      sign | (exponent << 10) | ((bits >> 13) & 0x3FF));
}

// Within one unit in the last place of exp in double, overflowing to infinity
// and underflowing to zero at the range of each dtype.
TEST(GgmlEngineTest, ExpOfHalfAndBFloat16) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  struct Case {
    fbs::ScalarType dtype;
    ScalarType runtime_dtype;
    // Each exact in dtype.
    std::vector<float> x;
    // The smallest exp that rounds to infinity in dtype.
    float overflow;
    float epsilon;
    float min_subnormal;
  };
  const std::vector<Case> cases = {
      {fbs::ScalarType::HALF,
       ScalarType::Half,
       {-0.0f,
        0.0f,
        1.5f,
        -2.25f,
        0.75f,
        -8.0f,
        11.0f,
        12.0f,
        -20.0f,
        kInf,
        -kInf,
        std::nanf("")},
       65520.0f,
       0x1p-10f,
       0x1p-24f},
      {fbs::ScalarType::BFLOAT16,
       ScalarType::BFloat16,
       {-0.0f,
        0.0f,
        1.5f,
        -2.25f,
        -8.0f,
        80.0f,
        100.0f,
        -80.0f,
        -110.0f,
        kInf,
        -kInf,
        std::nanf("")},
       std::numeric_limits<float>::max(),
       0x1p-7f,
       0x1p-133f},
  };
  for (const Case& test_case : cases) {
    for (const bool transposed : {false, true}) {
      SCOPED_TRACE(
          std::to_string(static_cast<int>(test_case.dtype)) +
          std::to_string(transposed));
      auto host = GgmlEngineHost::create();
      auto context =
          make_context(*host, make_exp_program(transposed, test_case.dtype));
      auto executable = context->compile("forward");
      std::vector<uint16_t> input(test_case.x.size());
      std::ranges::transform(
          test_case.x, input.begin(), [&test_case](float value) {
            return to_16_bit(value, test_case.dtype);
          });
      executable->set_input(
          0, input.data(), input.size(), test_case.runtime_dtype);
      executable->execute();
      std::vector<float> y(input.size());
      executable->get_output(0, y.data(), y.size(), ScalarType::Float);
      for (size_t i = 0; i < y.size(); ++i) {
        // y[r][c] reads x[c][r] when transposed.
        const size_t source = transposed ? (i % 4) * 3 + i / 4 : i;
        const double expected =
            std::exp(static_cast<double>(test_case.x[source]));
        SCOPED_TRACE(test_case.x[source]);
        if (std::isnan(expected)) {
          EXPECT_TRUE(std::isnan(y[i])) << y[i];
        } else if (expected >= test_case.overflow) {
          EXPECT_EQ(y[i], kInf);
        } else {
          EXPECT_NEAR(
              y[i],
              expected,
              test_case.epsilon * expected + test_case.min_subnormal);
        }
      }
    }
  }
}

TEST(GgmlEngineTest, RejectsIntegerExp) {
  for (const fbs::ScalarType dtype :
       {fbs::ScalarType::INT, fbs::ScalarType::LONG}) {
    SCOPED_TRACE(static_cast<int>(dtype));
    expect_compile_error(
        make_exp_program(false, dtype), "supports floating-point tensors only");
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
  auto host = GgmlEngineHost::create(kAllowActivationQuant);
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
  auto host = GgmlEngineHost::create(kAllowActivationQuant);
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

namespace kv = ::executorch::extension::llm::cache;

constexpr int64_t kOffGraphHeads = 4;
constexpr int64_t kOffGraphKvHeads = 2;
constexpr int64_t kOffGraphDim = 8;
const double kOffGraphScale =
    1.0 / std::sqrt(static_cast<double>(kOffGraphDim));

// One kvcache::update_and_attend over q [1, 4, T, 8] and k, v [1, 2, T, 8] at
// positions [T, 1], addressing cache layer `layer`. With `gather_row`, the
// program returns the row of the output, viewed as [4 * T, 8], that a fifth
// input [1] names; the engine checks it on the host while building the graph.
std::vector<uint8_t> make_update_and_attend_program(
    int64_t tokens,
    int64_t layer = 0,
    bool gather_row = false) {
  TestProgram program;
  program.input(
      "q", fbs::ScalarType::FLOAT, {1, kOffGraphHeads, tokens, kOffGraphDim});
  program.input(
      "k", fbs::ScalarType::FLOAT, {1, kOffGraphKvHeads, tokens, kOffGraphDim});
  program.input(
      "v", fbs::ScalarType::FLOAT, {1, kOffGraphKvHeads, tokens, kOffGraphDim});
  program.input("position", fbs::ScalarType::LONG, {tokens, 1});
  if (gather_row) {
    program.input("row", fbs::ScalarType::LONG, {1});
  }
  program.call(
      "attention",
      "torch.ops.kvcache.update_and_attend.default",
      {{"q", std::string("q")},
       {"k", std::string("k")},
       {"v", std::string("v")},
       {"position", std::string("position")},
       {"layer_id", layer},
       {"scale", kOffGraphScale}},
      fbs::ScalarType::FLOAT,
      {1, kOffGraphHeads, tokens, kOffGraphDim});
  if (!gather_row) {
    return program.finish("attention");
  }
  program.call(
      "rows",
      "torch.ops.aten.view_copy.default",
      {{"self", std::string("attention")},
       {"size", std::vector<int64_t>{kOffGraphHeads * tokens, kOffGraphDim}}},
      fbs::ScalarType::FLOAT,
      {kOffGraphHeads * tokens, kOffGraphDim});
  program.call(
      "selected",
      "torch.ops.aten.embedding.default",
      {{"weight", std::string("rows")}, {"indices", std::string("row")}},
      fbs::ScalarType::FLOAT,
      {1, kOffGraphDim});
  return program.finish("selected");
}

// make_update_and_attend_program's attention once per cache layer in
// [0, layers), over the same q, k and v, summed.
std::vector<uint8_t> make_layered_attention_program(
    int64_t tokens,
    int64_t layers) {
  TestProgram program;
  program.input(
      "q", fbs::ScalarType::FLOAT, {1, kOffGraphHeads, tokens, kOffGraphDim});
  program.input(
      "k", fbs::ScalarType::FLOAT, {1, kOffGraphKvHeads, tokens, kOffGraphDim});
  program.input(
      "v", fbs::ScalarType::FLOAT, {1, kOffGraphKvHeads, tokens, kOffGraphDim});
  program.input("position", fbs::ScalarType::LONG, {tokens, 1});
  std::string sum;
  for (int64_t layer = 0; layer < layers; ++layer) {
    const std::string attention = "attention" + std::to_string(layer);
    program.call(
        attention,
        "torch.ops.kvcache.update_and_attend.default",
        {{"q", std::string("q")},
         {"k", std::string("k")},
         {"v", std::string("v")},
         {"position", std::string("position")},
         {"layer_id", layer},
         {"scale", kOffGraphScale}},
        fbs::ScalarType::FLOAT,
        {1, kOffGraphHeads, tokens, kOffGraphDim});
    if (sum.empty()) {
      sum = attention;
      continue;
    }
    const std::string next = "sum" + std::to_string(layer);
    program.call(
        next,
        "torch.ops.aten.add.Tensor",
        {{"self", sum}, {"other", attention}},
        fbs::ScalarType::FLOAT,
        {1, kOffGraphHeads, tokens, kOffGraphDim});
    sum = next;
  }
  return program.finish(sum);
}

// A ggml cache built through the factory and published under a registry key,
// with the context option that names it.
struct InstalledCache {
  std::shared_ptr<kv::Cache> cache;
  std::unique_ptr<kv::InstallGuard> guard;
  EngineContextOptions options;

  kv::SequenceControl& control() {
    return *cache->as<kv::SequenceControl>();
  }
  kv::BatchControl& batch() {
    return *cache->as<kv::BatchControl>();
  }
};

// One layer per entry of `windows`: flat for 0, else a ring of that window.
InstalledCache install_cache(
    int capacity,
    int initial_capacity,
    ScalarType kv_dtype = ScalarType::Float,
    int head_dim = static_cast<int>(kOffGraphDim),
    const std::vector<int>& windows = {0},
    std::optional<int> max_write = std::nullopt,
    const char* kind = kv::kind::kSingle) {
  register_ggml_cache_builders();
  kv::CacheGeometry geometry;
  for (const int window : windows) {
    geometry.layers.push_back(kv::LayerGeometry{
        .policy = window > 0
            ? kv::LayerPolicy{kv::LayerPolicy::Kind::Ring, window}
            : kv::LayerPolicy{},
        .n_kv_heads = static_cast<int>(kOffGraphKvHeads),
        .head_dim = head_dim});
  }
  const kv::CacheConfig cfg{
      .capacity = capacity,
      .kv_dtype = static_cast<int>(kv_dtype),
      .initial_capacity = initial_capacity,
      .max_write = max_write};
  auto built = kv::CacheFactory::global().build(
      kGgmlCacheBackendId, kind, geometry, cfg);
  if (!built.ok()) {
    throw std::runtime_error("failed to build the test cache");
  }
  InstalledCache out;
  out.cache = built.get();
  out.guard = std::make_unique<kv::InstallGuard>(out.cache);
  ::executorch::runtime::BackendOptions<1> options;
  const char* key = nullptr;
  if (out.guard->set_option(options) != ::executorch::runtime::Error::Ok ||
      options.get_option(kv::kCacheKeyOption, key) !=
          ::executorch::runtime::Error::Ok) {
    throw std::runtime_error("failed to publish the test cache");
  }
  out.options.emplace(kv::kCacheKeyOption, key);
  return out;
}

// What the cache should hold: K and V [Hkv, D] per position, written by each
// step, attended causally by naive softmax attention once per entry of
// `windows`, over the last `window` positions when it is nonzero, and summed.
class OffGraphReference {
 public:
  explicit OffGraphReference(std::vector<int> windows = {0})
      : windows_(std::move(windows)) {}

  // Runs `tokens` tokens from `start` through `executable`, with inputs that
  // depend on the position and `seed`, and returns the largest difference
  // from the reference.
  float step(
      EngineExecutable& executable,
      int64_t start,
      int64_t tokens,
      int seed = 0) {
    const int64_t kv_row = kOffGraphKvHeads * kOffGraphDim;
    std::vector<float> q(kOffGraphHeads * tokens * kOffGraphDim);
    std::vector<float> k(kOffGraphKvHeads * tokens * kOffGraphDim);
    std::vector<float> v(k.size());
    std::vector<int64_t> positions(tokens);
    for (int64_t t = 0; t < tokens; ++t) {
      const int64_t pos = start + t;
      positions[t] = pos;
      if (static_cast<int64_t>(k_.size()) <= pos) {
        k_.resize(pos + 1, std::vector<float>(kv_row));
        v_.resize(pos + 1, std::vector<float>(kv_row));
      }
      for (int64_t h = 0; h < kOffGraphHeads; ++h) {
        for (int64_t d = 0; d < kOffGraphDim; ++d) {
          q[(h * tokens + t) * kOffGraphDim + d] = static_cast<float>(
              std::sin(0.17 * pos + 0.9 * h - 0.21 * d + seed));
        }
      }
      for (int64_t h = 0; h < kOffGraphKvHeads; ++h) {
        for (int64_t d = 0; d < kOffGraphDim; ++d) {
          const auto key = static_cast<float>(
              std::sin(0.3 * pos + 0.7 * h + 0.13 * d + 2 * seed));
          const auto value = static_cast<float>(
              std::cos(0.2 * pos - 0.5 * h + 0.31 * d + seed) * (1 + pos % 3));
          k[(h * tokens + t) * kOffGraphDim + d] = key;
          v[(h * tokens + t) * kOffGraphDim + d] = value;
          k_[pos][h * kOffGraphDim + d] = key;
          v_[pos][h * kOffGraphDim + d] = value;
        }
      }
    }
    executable.set_input(0, q.data(), q.size(), ScalarType::Float);
    executable.set_input(1, k.data(), k.size(), ScalarType::Float);
    executable.set_input(2, v.data(), v.size(), ScalarType::Float);
    executable.set_input(
        3, positions.data(), positions.size(), ScalarType::Long);
    executable.execute();
    std::vector<float> out(q.size());
    executable.get_output(0, out.data(), out.size(), ScalarType::Float);

    float worst = 0.0f;
    for (int64_t h = 0; h < kOffGraphHeads; ++h) {
      const int64_t kv_head = h / (kOffGraphHeads / kOffGraphKvHeads);
      for (int64_t t = 0; t < tokens; ++t) {
        const float* query = &q[(h * tokens + t) * kOffGraphDim];
        const int64_t keys = start + t + 1;
        std::vector<double> want(kOffGraphDim, 0.0);
        for (const int window : windows_) {
          const int64_t first =
              window > 0 ? std::max<int64_t>(0, keys - window) : 0;
          std::vector<double> weights(keys);
          for (int64_t s = first; s < keys; ++s) {
            weights[s] = std::exp(
                std::inner_product(
                    query,
                    query + kOffGraphDim,
                    &k_[s][kv_head * kOffGraphDim],
                    0.0) *
                kOffGraphScale);
          }
          const double total =
              std::accumulate(weights.begin() + first, weights.end(), 0.0);
          for (int64_t d = 0; d < kOffGraphDim; ++d) {
            double sum = 0.0;
            for (int64_t s = first; s < keys; ++s) {
              sum += weights[s] * v_[s][kv_head * kOffGraphDim + d];
            }
            want[d] += sum / total;
          }
        }
        for (int64_t d = 0; d < kOffGraphDim; ++d) {
          worst = std::max(
              worst,
              std::abs(
                  out[(h * tokens + t) * kOffGraphDim + d] -
                  static_cast<float>(want[d])));
        }
      }
    }
    return worst;
  }

 private:
  std::vector<int> windows_;
  std::vector<std::vector<float>> k_;
  std::vector<std::vector<float>> v_;
};

// A three-token prefill and decode steps in two contexts sharing one cache,
// across graph caching modes and both storage types. Storage starts at one
// row, so it regrows in the prefill and twice while decoding.
TEST(GgmlEngineTest, OffGraphAttentionMatchesReferenceAsTheCacheGrows) {
  for (const size_t graphs : {size_t{0}, size_t{1}, size_t{4}}) {
    for (const auto& [dtype, tol] :
         {std::pair{ScalarType::Float, 1e-4f},
          std::pair{ScalarType::Half, 2e-2f}}) {
      SCOPED_TRACE(
          "graphs " + std::to_string(graphs) + ", " +
          std::string(scalar_type_name(dtype)));
      InstalledCache cache = install_cache(64, 1, dtype);
      auto host = GgmlEngineHost::create({.max_cached_graphs = graphs});
      auto prefill_context =
          make_context(*host, make_update_and_attend_program(3), cache.options);
      auto decode_context =
          make_context(*host, make_update_and_attend_program(1), cache.options);
      auto prefill = prefill_context->compile("forward");
      auto decode = decode_context->compile("forward");
      OffGraphReference reference;
      EXPECT_LT(reference.step(*prefill, 0, 3), tol);
      for (int64_t pos = 3; pos < 12; ++pos) {
        SCOPED_TRACE(pos);
        EXPECT_LT(reference.step(*decode, pos, 1), tol);
      }
    }
  }
}

// After a clear, the old rows are masked out, including in an executable
// whose storage binding went stale when another regrew the cache.
TEST(GgmlEngineTest, OffGraphAttentionReplaysFromEmptyAfterClear) {
  InstalledCache cache = install_cache(64, 2);
  auto host = GgmlEngineHost::create();
  auto prefill_context =
      make_context(*host, make_update_and_attend_program(3), cache.options);
  auto decode_context =
      make_context(*host, make_update_and_attend_program(1), cache.options);
  auto prefill = prefill_context->compile("forward");
  auto decode = decode_context->compile("forward");
  {
    OffGraphReference reference;
    EXPECT_LT(reference.step(*prefill, 0, 3), 1e-4f);
    for (int64_t pos = 3; pos < 10; ++pos) {
      EXPECT_LT(reference.step(*decode, pos, 1), 1e-4f);
    }
  }
  cache.control().clear();
  OffGraphReference reference;
  EXPECT_LT(reference.step(*prefill, 0, 3, /*seed=*/5), 1e-4f);
  EXPECT_LT(reference.step(*decode, 3, 1, /*seed=*/5), 1e-4f);
}

// After a rewind, the next step overwrites the rewound rows and attends
// nothing past them.
TEST(GgmlEngineTest, OffGraphAttentionContinuesFromARewind) {
  InstalledCache cache = install_cache(64, 4);
  auto host = GgmlEngineHost::create();
  auto prefill_context =
      make_context(*host, make_update_and_attend_program(3), cache.options);
  auto decode_context =
      make_context(*host, make_update_and_attend_program(1), cache.options);
  auto prefill = prefill_context->compile("forward");
  auto decode = decode_context->compile("forward");
  OffGraphReference reference;
  EXPECT_LT(reference.step(*prefill, 0, 3), 1e-4f);
  for (int64_t pos = 3; pos < 8; ++pos) {
    EXPECT_LT(reference.step(*decode, pos, 1), 1e-4f);
  }
  ASSERT_TRUE(cache.control().rewind(2));
  EXPECT_LT(reference.step(*decode, 2, 1, /*seed=*/9), 1e-4f);
  EXPECT_LT(reference.step(*decode, 3, 1, /*seed=*/9), 1e-4f);
}

// A step starts at the sequence's length: one past it would attend rows
// nothing wrote, and running earlier positions again takes an explicit rewind.
TEST(GgmlEngineTest, OffGraphStepsStartAtTheLength) {
  InstalledCache cache = install_cache(64, 4);
  auto host = GgmlEngineHost::create();
  auto prefill_context =
      make_context(*host, make_update_and_attend_program(3), cache.options);
  auto decode_context =
      make_context(*host, make_update_and_attend_program(1), cache.options);
  auto prefill = prefill_context->compile("forward");
  auto decode = decode_context->compile("forward");
  OffGraphReference reference;
  EXPECT_LT(reference.step(*prefill, 0, 3), 1e-4f);
  EXPECT_THROW(reference.step(*prefill, 0, 3), std::invalid_argument);
  EXPECT_THROW(reference.step(*decode, 4, 1), std::invalid_argument);
  for (int64_t pos = 3; pos < 8; ++pos) {
    EXPECT_LT(reference.step(*decode, pos, 1), 1e-4f);
  }
  ASSERT_TRUE(cache.control().rewind(0));
  EXPECT_LT(reference.step(*prefill, 0, 3, /*seed=*/3), 1e-4f);
  EXPECT_LT(reference.step(*decode, 3, 1, /*seed=*/3), 1e-4f);
}

// A step that fails, here on a row outside the output, leaves the sequence's
// length where it was.
TEST(GgmlEngineTest, FailedOffGraphStepLeavesTheSequence) {
  InstalledCache cache = install_cache(64, 4);
  auto host = GgmlEngineHost::create();
  auto prefill_context =
      make_context(*host, make_update_and_attend_program(3), cache.options);
  auto decode_context =
      make_context(*host, make_update_and_attend_program(1), cache.options);
  auto gather_context = make_context(
      *host, make_update_and_attend_program(1, 0, true), cache.options);
  auto prefill = prefill_context->compile("forward");
  auto decode = decode_context->compile("forward");
  auto gather = gather_context->compile("forward");
  OffGraphReference reference;
  EXPECT_LT(reference.step(*prefill, 0, 3), 1e-4f);
  for (int64_t pos = 3; pos < 6; ++pos) {
    EXPECT_LT(reference.step(*decode, pos, 1), 1e-4f);
  }
  const std::vector<float> q(kOffGraphHeads * kOffGraphDim);
  const std::vector<float> kv_step(kOffGraphKvHeads * kOffGraphDim);
  const int64_t position = 6;
  const int64_t row = kOffGraphHeads;
  gather->set_input(0, q.data(), q.size(), ScalarType::Float);
  gather->set_input(1, kv_step.data(), kv_step.size(), ScalarType::Float);
  gather->set_input(2, kv_step.data(), kv_step.size(), ScalarType::Float);
  gather->set_input(3, &position, 1, ScalarType::Long);
  gather->set_input(4, &row, 1, ScalarType::Long);
  EXPECT_THROW(gather->execute(), std::out_of_range);
  EXPECT_LT(reference.step(*decode, 6, 1), 1e-4f);
}

TEST(GgmlEngineTest, RejectsOffGraphStepsPastCapacityOrNotConsecutive) {
  InstalledCache cache = install_cache(4, 4);
  auto host = GgmlEngineHost::create();
  auto prefill_context =
      make_context(*host, make_update_and_attend_program(3), cache.options);
  auto decode_context =
      make_context(*host, make_update_and_attend_program(1), cache.options);
  auto prefill = prefill_context->compile("forward");
  auto decode = decode_context->compile("forward");
  OffGraphReference reference;
  EXPECT_THROW(reference.step(*decode, 1, 1), std::invalid_argument);
  EXPECT_LT(reference.step(*prefill, 0, 3), 1e-4f);
  EXPECT_LT(reference.step(*decode, 3, 1), 1e-4f);
  EXPECT_THROW(reference.step(*decode, 4, 1), std::out_of_range);
  EXPECT_THROW(reference.step(*prefill, 2, 3), std::out_of_range);

  const std::vector<float> q(kOffGraphHeads * 3 * kOffGraphDim);
  const std::vector<float> kv_step(kOffGraphKvHeads * 3 * kOffGraphDim);
  const std::vector<int64_t> positions = {0, 2, 3};
  prefill->set_input(0, q.data(), q.size(), ScalarType::Float);
  prefill->set_input(1, kv_step.data(), kv_step.size(), ScalarType::Float);
  prefill->set_input(2, kv_step.data(), kv_step.size(), ScalarType::Float);
  prefill->set_input(3, positions.data(), positions.size(), ScalarType::Long);
  EXPECT_THROW(prefill->execute(), std::invalid_argument);
}

// A ring layer attends each query's window, within a prefill wider than the
// window too, while decoding wraps the ring several times.
TEST(GgmlEngineTest, OffGraphRingAttentionMatchesReference) {
  for (const size_t graphs : {size_t{0}, size_t{4}}) {
    for (const int window : {2, 5}) {
      for (const auto& [dtype, tol] :
           {std::pair{ScalarType::Float, 1e-4f},
            std::pair{ScalarType::Half, 2e-2f}}) {
        SCOPED_TRACE(
            "graphs " + std::to_string(graphs) + ", window " +
            std::to_string(window) + ", " +
            std::string(scalar_type_name(dtype)));
        InstalledCache cache = install_cache(
            64, 1, dtype, static_cast<int>(kOffGraphDim), {window}, 3);
        auto host = GgmlEngineHost::create({.max_cached_graphs = graphs});
        auto prefill_context = make_context(
            *host, make_update_and_attend_program(3), cache.options);
        auto decode_context = make_context(
            *host, make_update_and_attend_program(1), cache.options);
        auto prefill = prefill_context->compile("forward");
        auto decode = decode_context->compile("forward");
        OffGraphReference reference({window});
        EXPECT_LT(reference.step(*prefill, 0, 3), tol);
        for (int64_t pos = 3; pos < 20; ++pos) {
          SCOPED_TRACE(pos);
          EXPECT_LT(reference.step(*decode, pos, 1), tol);
        }
        EXPECT_LT(reference.step(*prefill, 20, 3), tol);
      }
    }
  }
}

// A flat and a ring layer share one sequence in one forward.
TEST(GgmlEngineTest, OffGraphFlatAndRingLayersShareASequence) {
  InstalledCache cache = install_cache(
      64, 1, ScalarType::Float, static_cast<int>(kOffGraphDim), {0, 3}, 3);
  auto host = GgmlEngineHost::create();
  auto prefill_context =
      make_context(*host, make_layered_attention_program(3, 2), cache.options);
  auto decode_context =
      make_context(*host, make_layered_attention_program(1, 2), cache.options);
  auto prefill = prefill_context->compile("forward");
  auto decode = decode_context->compile("forward");
  OffGraphReference reference({0, 3});
  EXPECT_LT(reference.step(*prefill, 0, 3), 1e-4f);
  for (int64_t pos = 3; pos < 12; ++pos) {
    SCOPED_TRACE(pos);
    EXPECT_LT(reference.step(*decode, pos, 1), 1e-4f);
  }
}

// A ring layer can rewind only as far as the rows it still holds.
TEST(GgmlEngineTest, OffGraphRingRewindsWithinTheRowsItHolds) {
  InstalledCache cache = install_cache(
      64, 1, ScalarType::Float, static_cast<int>(kOffGraphDim), {4}, 3);
  auto host = GgmlEngineHost::create();
  auto prefill_context =
      make_context(*host, make_update_and_attend_program(3), cache.options);
  auto decode_context =
      make_context(*host, make_update_and_attend_program(1), cache.options);
  auto prefill = prefill_context->compile("forward");
  auto decode = decode_context->compile("forward");
  OffGraphReference reference({4});
  EXPECT_LT(reference.step(*prefill, 0, 3), 1e-4f);
  for (int64_t pos = 3; pos < 10; ++pos) {
    EXPECT_LT(reference.step(*decode, pos, 1), 1e-4f);
  }
  // The ring holds positions 4..9; the step after a rewind to 7 reads 4..7.
  EXPECT_FALSE(cache.control().rewind(6));
  ASSERT_TRUE(cache.control().rewind(7));
  EXPECT_LT(reference.step(*decode, 7, 1, /*seed=*/4), 1e-4f);
  EXPECT_LT(reference.step(*decode, 8, 1, /*seed=*/4), 1e-4f);
}

// What a cell cache should hold: K and V [Hkv, D] per (sequence, position),
// a history per sequence that a clone copies, attended causally within the
// sequence by naive softmax attention once per entry of `windows`, over the
// last `window` positions when it is nonzero, and summed.
class CellReference {
 public:
  explicit CellReference(std::vector<int> windows = {0})
      : windows_(std::move(windows)) {}

  void clone(int32_t source, int32_t target, int upto) {
    history_[target].assign(
        history_[source].begin(), history_[source].begin() + upto);
  }
  void rewind(int32_t seq, int position) {
    history_[seq].resize(static_cast<size_t>(position));
  }

  // Runs one token per (sequence, position) through `executable`, after
  // `batch` declares their sequences, and returns the largest difference from
  // the reference.
  float step(
      EngineExecutable& executable,
      kv::BatchControl& batch,
      const std::vector<std::pair<int32_t, int64_t>>& tokens,
      int seed = 0) {
    const auto n = static_cast<int64_t>(tokens.size());
    const int64_t kv_row = kOffGraphKvHeads * kOffGraphDim;
    std::vector<float> q(kOffGraphHeads * n * kOffGraphDim);
    std::vector<float> k(kOffGraphKvHeads * n * kOffGraphDim);
    std::vector<float> v(k.size());
    std::vector<int64_t> positions;
    std::vector<int32_t> seq_ids;
    for (int64_t t = 0; t < n; ++t) {
      const auto [seq, pos] = tokens[t];
      positions.push_back(pos);
      seq_ids.push_back(seq);
      const double salt = 1.3 * seq + seed;
      std::vector<std::vector<float>>& history = history_[seq];
      if (static_cast<int64_t>(history.size()) != pos) {
        throw std::logic_error("reference: positions must continue");
      }
      history.emplace_back(2 * kv_row);
      for (int64_t h = 0; h < kOffGraphHeads; ++h) {
        for (int64_t d = 0; d < kOffGraphDim; ++d) {
          q[(h * n + t) * kOffGraphDim + d] = static_cast<float>(
              std::sin(0.17 * pos + 0.9 * h - 0.21 * d + salt));
        }
      }
      for (int64_t h = 0; h < kOffGraphKvHeads; ++h) {
        for (int64_t d = 0; d < kOffGraphDim; ++d) {
          const auto key = static_cast<float>(
              std::sin(0.3 * pos + 0.7 * h + 0.13 * d + 2 * salt));
          const auto value = static_cast<float>(
              std::cos(0.2 * pos - 0.5 * h + 0.31 * d + salt) * (1 + pos % 3));
          k[(h * n + t) * kOffGraphDim + d] = key;
          v[(h * n + t) * kOffGraphDim + d] = value;
          history.back()[h * kOffGraphDim + d] = key;
          history.back()[kv_row + h * kOffGraphDim + d] = value;
        }
      }
    }
    if (!batch.declare_step(seq_ids)) {
      throw std::runtime_error("reference: the cache refused the step");
    }
    executable.set_input(0, q.data(), q.size(), ScalarType::Float);
    executable.set_input(1, k.data(), k.size(), ScalarType::Float);
    executable.set_input(2, v.data(), v.size(), ScalarType::Float);
    executable.set_input(
        3, positions.data(), positions.size(), ScalarType::Long);
    executable.execute();
    std::vector<float> out(q.size());
    executable.get_output(0, out.data(), out.size(), ScalarType::Float);

    float worst = 0.0f;
    for (int64_t t = 0; t < n; ++t) {
      const auto [seq, pos] = tokens[t];
      const std::vector<std::vector<float>>& history = history_[seq];
      for (int64_t h = 0; h < kOffGraphHeads; ++h) {
        const int64_t kv_head = h / (kOffGraphHeads / kOffGraphKvHeads);
        const float* query = &q[(h * n + t) * kOffGraphDim];
        std::vector<double> want(kOffGraphDim, 0.0);
        for (const int window : windows_) {
          const int64_t first =
              window > 0 ? std::max<int64_t>(0, pos + 1 - window) : 0;
          std::vector<double> weights(static_cast<size_t>(pos + 1), 0.0);
          double total = 0.0;
          for (int64_t s = first; s <= pos; ++s) {
            weights[s] = std::exp(
                std::inner_product(
                    query,
                    query + kOffGraphDim,
                    &history[s][kv_head * kOffGraphDim],
                    0.0) *
                kOffGraphScale);
            total += weights[s];
          }
          for (int64_t d = 0; d < kOffGraphDim; ++d) {
            double sum = 0.0;
            for (int64_t s = first; s <= pos; ++s) {
              sum +=
                  weights[s] * history[s][kv_row + kv_head * kOffGraphDim + d];
            }
            want[d] += sum / total;
          }
        }
        for (int64_t d = 0; d < kOffGraphDim; ++d) {
          worst = std::max(
              worst,
              std::abs(
                  out[(h * n + t) * kOffGraphDim + d] -
                  static_cast<float>(want[d])));
        }
      }
    }
    return worst;
  }

 private:
  std::vector<int> windows_;
  std::map<int32_t, std::vector<std::vector<float>>> history_;
};

InstalledCache install_cell_cache(
    int capacity,
    int initial_capacity,
    ScalarType kv_dtype = ScalarType::Float,
    const std::vector<int>& windows = {0}) {
  return install_cache(
      capacity,
      initial_capacity,
      kv_dtype,
      static_cast<int>(kOffGraphDim),
      windows,
      std::nullopt,
      kv::kind::kBatchedCell);
}

// Two sequences prefill separately, then decode together in one forward,
// across graph caching modes and both storage types. Storage starts at one
// row, so it regrows as the cells fill.
TEST(GgmlEngineTest, CellCacheServesInterleavedSequences) {
  for (const size_t graphs : {size_t{0}, size_t{4}}) {
    for (const auto& [dtype, tol] :
         {std::pair{ScalarType::Float, 1e-4f},
          std::pair{ScalarType::Half, 2e-2f}}) {
      SCOPED_TRACE(
          "graphs " + std::to_string(graphs) + ", " +
          std::string(scalar_type_name(dtype)));
      InstalledCache cache = install_cell_cache(64, 1, dtype);
      auto host = GgmlEngineHost::create({.max_cached_graphs = graphs});
      auto prefill_context =
          make_context(*host, make_update_and_attend_program(3), cache.options);
      auto decode_context =
          make_context(*host, make_update_and_attend_program(2), cache.options);
      auto prefill = prefill_context->compile("forward");
      auto decode = decode_context->compile("forward");
      const int32_t a = *cache.batch().seq_new();
      const int32_t b = *cache.batch().seq_new();
      CellReference reference;
      EXPECT_LT(
          reference.step(*prefill, cache.batch(), {{a, 0}, {a, 1}, {a, 2}}),
          tol);
      EXPECT_LT(
          reference.step(*prefill, cache.batch(), {{b, 0}, {b, 1}, {b, 2}}),
          tol);
      for (int64_t pos = 3; pos < 12; ++pos) {
        SCOPED_TRACE(pos);
        EXPECT_LT(
            reference.step(*decode, cache.batch(), {{a, pos}, {b, pos}}), tol);
      }
      EXPECT_EQ(cache.batch().pos(a), 12);
      EXPECT_EQ(cache.batch().pos(b), 12);
    }
  }
}

// A clone shares its source's cells below the fork, and each continues on its
// own; a rewind drops the positions it passes.
TEST(GgmlEngineTest, CellCacheClonesAndRewindsSequences) {
  InstalledCache cache = install_cell_cache(64, 4);
  auto host = GgmlEngineHost::create();
  auto prefill_context =
      make_context(*host, make_update_and_attend_program(3), cache.options);
  auto decode_context =
      make_context(*host, make_update_and_attend_program(2), cache.options);
  auto single_context =
      make_context(*host, make_update_and_attend_program(1), cache.options);
  auto prefill = prefill_context->compile("forward");
  auto decode = decode_context->compile("forward");
  auto single = single_context->compile("forward");
  const int32_t a = *cache.batch().seq_new();
  CellReference reference;
  EXPECT_LT(
      reference.step(*prefill, cache.batch(), {{a, 0}, {a, 1}, {a, 2}}), 1e-4f);
  for (int64_t pos = 3; pos < 6; ++pos) {
    EXPECT_LT(reference.step(*single, cache.batch(), {{a, pos}}), 1e-4f);
  }
  const auto b = cache.batch().seq_clone(a, 4);
  ASSERT_TRUE(b.has_value());
  reference.clone(a, *b, 4);
  EXPECT_EQ(cache.batch().pos(*b), 4);
  for (int64_t i = 0; i < 3; ++i) {
    SCOPED_TRACE(i);
    EXPECT_LT(
        reference.step(*decode, cache.batch(), {{a, 6 + i}, {*b, 4 + i}}),
        1e-4f);
  }
  ASSERT_TRUE(cache.batch().rewind(a, 3));
  reference.rewind(a, 3);
  EXPECT_LT(
      reference.step(*single, cache.batch(), {{a, 3}}, /*seed=*/7), 1e-4f);
  EXPECT_LT(
      reference.step(*decode, cache.batch(), {{a, 4}, {*b, 7}}, /*seed=*/7),
      1e-4f);
  // Closing the clone frees its own cells and leaves the source's.
  ASSERT_TRUE(cache.batch().seq_rm(*b));
  EXPECT_LT(
      reference.step(*single, cache.batch(), {{a, 5}}, /*seed=*/7), 1e-4f);
}

// A windowed layer keeps every cell and its mask narrows to the window, within
// a prefill wider than the window too.
TEST(GgmlEngineTest, CellCacheMasksSlidingWindows) {
  for (const int window : {2, 5}) {
    SCOPED_TRACE(window);
    InstalledCache cache =
        install_cell_cache(64, 2, ScalarType::Float, {window});
    auto host = GgmlEngineHost::create();
    auto prefill_context =
        make_context(*host, make_update_and_attend_program(3), cache.options);
    auto decode_context =
        make_context(*host, make_update_and_attend_program(2), cache.options);
    auto prefill = prefill_context->compile("forward");
    auto decode = decode_context->compile("forward");
    const int32_t a = *cache.batch().seq_new();
    const int32_t b = *cache.batch().seq_new();
    CellReference reference({window});
    EXPECT_LT(
        reference.step(*prefill, cache.batch(), {{a, 0}, {a, 1}, {a, 2}}),
        1e-4f);
    EXPECT_LT(
        reference.step(*prefill, cache.batch(), {{b, 0}, {a, 3}, {b, 1}}),
        1e-4f);
    for (int64_t i = 0; i < 6; ++i) {
      EXPECT_LT(
          reference.step(*decode, cache.batch(), {{a, 4 + i}, {b, 2 + i}}),
          1e-4f);
    }
  }
}

// The cache places only a declared step whose positions continue each
// sequence.
TEST(GgmlEngineTest, CellCacheRefusesUndeclaredSteps) {
  InstalledCache cache = install_cell_cache(64, 4);
  auto host = GgmlEngineHost::create();
  auto context =
      make_context(*host, make_update_and_attend_program(1), cache.options);
  auto executable = context->compile("forward");
  const int32_t a = *cache.batch().seq_new();
  CellReference reference;
  const std::vector<float> q(kOffGraphHeads * kOffGraphDim);
  const std::vector<float> kv_step(kOffGraphKvHeads * kOffGraphDim);
  const auto run_at = [&](int64_t position) {
    executable->set_input(0, q.data(), q.size(), ScalarType::Float);
    executable->set_input(1, kv_step.data(), kv_step.size(), ScalarType::Float);
    executable->set_input(2, kv_step.data(), kv_step.size(), ScalarType::Float);
    executable->set_input(3, &position, 1, ScalarType::Long);
    executable->execute();
  };
  EXPECT_THROW(run_at(0), std::invalid_argument);
  ASSERT_TRUE(cache.batch().declare_step({a}));
  EXPECT_THROW(run_at(1), std::invalid_argument);
  EXPECT_LT(reference.step(*executable, cache.batch(), {{a, 0}}), 1e-4f);
}

// One forward serves a sliding and a full layer from the same cells.
TEST(GgmlEngineTest, CellCacheServesSlidingAndFullLayersInOneForward) {
  InstalledCache cache = install_cell_cache(64, 2, ScalarType::Float, {3, 0});
  auto host = GgmlEngineHost::create();
  auto prefill_context =
      make_context(*host, make_layered_attention_program(3, 2), cache.options);
  auto decode_context =
      make_context(*host, make_layered_attention_program(2, 2), cache.options);
  auto prefill = prefill_context->compile("forward");
  auto decode = decode_context->compile("forward");
  const int32_t a = *cache.batch().seq_new();
  const int32_t b = *cache.batch().seq_new();
  CellReference reference({3, 0});
  EXPECT_LT(
      reference.step(*prefill, cache.batch(), {{a, 0}, {a, 1}, {a, 2}}), 1e-4f);
  EXPECT_LT(
      reference.step(*prefill, cache.batch(), {{b, 0}, {a, 3}, {b, 1}}), 1e-4f);
  for (int64_t i = 0; i < 5; ++i) {
    EXPECT_LT(
        reference.step(*decode, cache.batch(), {{a, 4 + i}, {b, 2 + i}}),
        1e-4f);
  }
}

// A full pool refuses a step until a rewind frees cells.
TEST(GgmlEngineTest, CellCacheRefusesStepsPastAFullPool) {
  InstalledCache cache = install_cell_cache(4, 1);
  auto host = GgmlEngineHost::create();
  auto prefill_context =
      make_context(*host, make_update_and_attend_program(3), cache.options);
  auto decode_context =
      make_context(*host, make_update_and_attend_program(2), cache.options);
  auto single_context =
      make_context(*host, make_update_and_attend_program(1), cache.options);
  auto prefill = prefill_context->compile("forward");
  auto decode = decode_context->compile("forward");
  auto single = single_context->compile("forward");
  const int32_t a = *cache.batch().seq_new();
  const int32_t b = *cache.batch().seq_new();
  CellReference reference;
  EXPECT_LT(
      reference.step(*prefill, cache.batch(), {{a, 0}, {a, 1}, {a, 2}}), 1e-4f);
  EXPECT_THROW(
      reference.step(*decode, cache.batch(), {{a, 3}, {b, 0}}),
      std::runtime_error);
  reference.rewind(a, 3);
  reference.rewind(b, 0);
  EXPECT_LT(reference.step(*single, cache.batch(), {{b, 0}}), 1e-4f);
  EXPECT_EQ(cache.batch().pos(a), 3);
  ASSERT_TRUE(cache.batch().rewind(a, 1));
  reference.rewind(a, 1);
  EXPECT_LT(
      reference.step(*decode, cache.batch(), {{a, 1}, {b, 1}}, /*seed=*/2),
      1e-4f);
}

// A forward that fails after placing its step, here on a row outside the
// output, gives its cells back, so the sequences continue where they were.
TEST(GgmlEngineTest, FailedCellStepGivesBackItsCells) {
  InstalledCache cache = install_cell_cache(64, 4);
  auto host = GgmlEngineHost::create();
  auto prefill_context =
      make_context(*host, make_update_and_attend_program(3), cache.options);
  auto gather_context = make_context(
      *host, make_update_and_attend_program(2, 0, true), cache.options);
  auto decode_context =
      make_context(*host, make_update_and_attend_program(2), cache.options);
  auto prefill = prefill_context->compile("forward");
  auto gather = gather_context->compile("forward");
  auto decode = decode_context->compile("forward");
  const int32_t a = *cache.batch().seq_new();
  const int32_t b = *cache.batch().seq_new();
  CellReference reference;
  EXPECT_LT(
      reference.step(*prefill, cache.batch(), {{a, 0}, {a, 1}, {a, 2}}), 1e-4f);
  const std::vector<float> q(kOffGraphHeads * 2 * kOffGraphDim);
  const std::vector<float> kv_step(kOffGraphKvHeads * 2 * kOffGraphDim);
  const std::vector<int64_t> positions = {3, 0};
  const int64_t row = 2 * kOffGraphHeads;
  ASSERT_TRUE(cache.batch().declare_step({a, b}));
  gather->set_input(0, q.data(), q.size(), ScalarType::Float);
  gather->set_input(1, kv_step.data(), kv_step.size(), ScalarType::Float);
  gather->set_input(2, kv_step.data(), kv_step.size(), ScalarType::Float);
  gather->set_input(3, positions.data(), positions.size(), ScalarType::Long);
  gather->set_input(4, &row, 1, ScalarType::Long);
  EXPECT_THROW(gather->execute(), std::out_of_range);
  EXPECT_EQ(cache.batch().pos(a), 3);
  EXPECT_EQ(cache.batch().pos(b), 0);
  EXPECT_LT(reference.step(*decode, cache.batch(), {{a, 3}, {b, 0}}), 1e-4f);
}

// A forward ends its step, so running it again without a declaration is
// refused rather than re-served, and cannot undo the step that ran.
TEST(GgmlEngineTest, CellStepEndsWithItsForward) {
  InstalledCache cache = install_cell_cache(64, 4);
  auto host = GgmlEngineHost::create();
  auto gather_context = make_context(
      *host, make_update_and_attend_program(1, 0, true), cache.options);
  auto gather = gather_context->compile("forward");
  const int32_t a = *cache.batch().seq_new();
  const std::vector<float> q(kOffGraphHeads * kOffGraphDim);
  const std::vector<float> kv_step(kOffGraphKvHeads * kOffGraphDim);
  const int64_t position = 0;
  int64_t row = 0;
  gather->set_input(0, q.data(), q.size(), ScalarType::Float);
  gather->set_input(1, kv_step.data(), kv_step.size(), ScalarType::Float);
  gather->set_input(2, kv_step.data(), kv_step.size(), ScalarType::Float);
  gather->set_input(3, &position, 1, ScalarType::Long);
  gather->set_input(4, &row, 1, ScalarType::Long);
  ASSERT_TRUE(cache.batch().declare_step({a}));
  gather->execute();
  row = kOffGraphHeads;
  gather->set_input(4, &row, 1, ScalarType::Long);
  EXPECT_THROW(gather->execute(), std::invalid_argument);
  EXPECT_EQ(cache.batch().pos(a), 1);
}

TEST(GgmlEngineTest, OffGraphAttentionNeedsAMatchingCache) {
  auto host = GgmlEngineHost::create();
  // No cache option.
  EXPECT_THROW(
      make_context(*host, make_update_and_attend_program(1))
          ->compile("forward"),
      std::runtime_error);
  // A key nothing is installed under, and an option the engine does not know.
  EXPECT_THROW(
      make_context(
          *host,
          make_update_and_attend_program(1),
          {{std::string(kv::kCacheKeyOption), "no-such-cache"}}),
      std::invalid_argument);
  EXPECT_THROW(
      make_context(
          *host, make_update_and_attend_program(1), {{"unknown", "value"}}),
      std::invalid_argument);
  // A layer the cache does not have, and K/V shaped unlike the cache.
  InstalledCache cache = install_cache(8, 8);
  EXPECT_THROW(
      make_context(*host, make_update_and_attend_program(1, 1), cache.options)
          ->compile("forward"),
      std::runtime_error);
  InstalledCache narrow = install_cache(8, 8, ScalarType::Float, 4);
  EXPECT_THROW(
      make_context(*host, make_update_and_attend_program(1), narrow.options)
          ->compile("forward"),
      std::runtime_error);
}

// x[:, rows] of a [1, T, D] x, as Hugging Face selects logits_to_keep rows.
TEST(GgmlEngineTest, GathersRowsAfterALeadingOne) {
  TestProgram program;
  program.input("x", fbs::ScalarType::FLOAT, {1, 6, 8});
  program.input("rows", fbs::ScalarType::LONG, {2});
  program.call(
      "selected",
      "torch.ops.aten.index.Tensor",
      {{"self", std::string("x")},
       {"indices", TestOptionalTensors{std::nullopt, "rows"}}},
      fbs::ScalarType::FLOAT,
      {1, 2, 8});
  auto host = GgmlEngineHost::create();
  auto context = make_context(*host, program.finish("selected"));
  auto executable = context->compile("forward");
  std::vector<float> x(48);
  std::iota(x.begin(), x.end(), 0.0f);
  const std::vector<int64_t> rows = {4, 1};
  executable->set_input(0, x.data(), x.size(), ScalarType::Float);
  executable->set_input(1, rows.data(), rows.size(), ScalarType::Long);
  executable->execute();
  EXPECT_EQ(executable->output_sizes(0), (std::vector<int64_t>{1, 2, 8}));
  std::vector<float> out(16);
  executable->get_output(0, out.data(), out.size(), ScalarType::Float);
  for (int64_t i = 0; i < 8; ++i) {
    EXPECT_EQ(out[i], x[4 * 8 + i]);
    EXPECT_EQ(out[8 + i], x[1 * 8 + i]);
  }
  const std::vector<int64_t> outside = {6, 0};
  executable->set_input(1, outside.data(), outside.size(), ScalarType::Long);
  EXPECT_ANY_THROW(executable->execute());
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

// A context for `program_bytes` with a `rows`-row int4 weight of 32-weight
// groups and float `scales`. Row r holds q = r + 1, stored as the nibble q + 8.
std::unique_ptr<GgmlEngineContext> make_int4_context(
    GgmlEngineHost& host,
    const std::vector<uint8_t>& program_bytes,
    size_t rows,
    const std::vector<float>& scales) {
  const size_t scale_bytes = scales.size() * sizeof(float);
  const size_t weight_bytes = scales.size() * 16;
  std::vector<uint8_t> safetensors = testing::make_safetensors(
      R"({"weight_scale":{"dtype":"F32","shape":[)" + std::to_string(rows) +
          "," + std::to_string(scales.size() / rows) +
          R"(],"data_offsets":[0,)" + std::to_string(scale_bytes) +
          R"(]},"weight":{"dtype":"U8","shape":[)" +
          std::to_string(weight_bytes) + R"(],"data_offsets":[)" +
          std::to_string(scale_bytes) + "," +
          std::to_string(scale_bytes + weight_bytes) + "]}}",
      scale_bytes + weight_bytes);
  uint8_t* payload =
      safetensors.data() + safetensors.size() - scale_bytes - weight_bytes;
  if (!scales.empty()) {
    std::memcpy(payload, scales.data(), scale_bytes);
  }
  const size_t row_bytes = weight_bytes / rows;
  for (size_t r = 0; r < rows; ++r) {
    std::fill_n(
        payload + scale_bytes + r * row_bytes,
        row_bytes,
        static_cast<uint8_t>(0x99 + 0x11 * r));
  }
  return host.create_ggml_context(
      std::make_shared<const Program>(
          Program::load(program_bytes.data(), program_bytes.size())),
      std::make_shared<const Package>(
          Package::load(OwnedBytes::from_vector(testing::make_zip({
              {kProgramEntry, program_bytes},
              {kSafeTensorsEntry, std::move(safetensors)},
          })))));
}

TEST(GgmlEngineTest, EmbedsInt4RowsWithExactFloatScales) {
  // Three rows of two groups, with scales fp16 can't hold.
  const std::vector<float> scales = {0.1f, 0.3f, 0.7f, 1.1f, 1.3f, 1.7f};
  TestProgram program;
  program.affine_weight("weight", {3, 64});
  program.input("ids", fbs::ScalarType::LONG, {2, 2});
  program.call(
      "rows",
      "torch.ops.aten.embedding.default",
      {{"weight", std::string("weight")}, {"indices", std::string("ids")}},
      fbs::ScalarType::FLOAT,
      {2, 2, 64});
  auto host = GgmlEngineHost::create();
  auto context = make_int4_context(*host, program.finish("rows"), 3, scales);
  auto executable = context->compile("forward");
  EXPECT_EQ(
      context->materialized_constant_bytes(),
      6 * kQ4_0BlockBytes + 6 * sizeof(float));
  const std::vector<int64_t> ids = {2, 0, 1, 2};
  executable->set_input(0, ids.data(), ids.size(), ScalarType::Long);
  executable->execute();
  std::vector<float> rows(4 * 64);
  executable->get_output(0, rows.data(), rows.size(), ScalarType::Float);
  for (size_t i = 0; i < rows.size(); ++i) {
    const int64_t id = ids[i / 64];
    EXPECT_EQ(
        rows[i], static_cast<float>(id + 1) * scales[2 * id + i % 64 / 32])
        << i;
  }
}

TEST(GgmlEngineTest, KeepsFp16ExactScalesOfInt4EmbeddingsInBlocks) {
  const std::vector<float> scales = {0.5f, 0.25f, 0.125f, 2.0f};
  TestProgram program;
  program.affine_weight("weight", {2, 64});
  program.input("ids", fbs::ScalarType::LONG, {3});
  program.call(
      "rows",
      "torch.ops.aten.embedding.default",
      {{"weight", std::string("weight")}, {"indices", std::string("ids")}},
      fbs::ScalarType::FLOAT,
      {3, 64});
  auto host = GgmlEngineHost::create();
  auto context = make_int4_context(*host, program.finish("rows"), 2, scales);
  auto executable = context->compile("forward");
  EXPECT_EQ(context->materialized_constant_bytes(), 4 * kQ4_0BlockBytes);
  const std::vector<int64_t> ids = {1, 0, 1};
  executable->set_input(0, ids.data(), ids.size(), ScalarType::Long);
  executable->execute();
  std::vector<float> rows(3 * 64);
  executable->get_output(0, rows.data(), rows.size(), ScalarType::Float);
  for (size_t i = 0; i < rows.size(); ++i) {
    const int64_t id = ids[i / 64];
    EXPECT_EQ(
        rows[i], static_cast<float>(id + 1) * scales[2 * id + i % 64 / 32])
        << i;
  }
}

// Only an embedding reads the weight, but its write-back updates the Q4_0
// blocks, so they keep the scales: 0.1 and 0.3 narrow to fp16.
TEST(GgmlEngineTest, KeepsScalesOfMutableInt4EmbeddingsInBlocks) {
  const std::vector<float> narrowed = {0.0999755859375f, 0.300048828125f};
  TestProgram program;
  program.affine_weight("weight", {2, 32}, /*mutated=*/true);
  program.input("ids", fbs::ScalarType::LONG, {2});
  program.input("update", fbs::ScalarType::FLOAT, {2, 32});
  program.call(
      "rows",
      "torch.ops.aten.embedding.default",
      {{"weight", std::string("weight")}, {"indices", std::string("ids")}},
      fbs::ScalarType::FLOAT,
      {2, 32});
  program.mutate("weight", "update");
  auto host = GgmlEngineHost::create();
  auto context =
      make_int4_context(*host, program.finish("rows"), 2, {0.1f, 0.3f});
  auto executable = context->compile("forward");
  const std::vector<int64_t> ids = {0, 1};
  const std::vector<float> update(2 * 32, 2.0f);
  std::vector<float> initial;
  for (const int64_t id : ids) {
    initial.insert(
        initial.end(), 32, static_cast<float>(id + 1) * narrowed[id]);
  }
  std::vector<float> rows(2 * 32);
  for (const std::vector<float>& expected : {initial, update}) {
    executable->set_input(0, ids.data(), ids.size(), ScalarType::Long);
    executable->set_input(1, update.data(), update.size(), ScalarType::Float);
    executable->execute();
    executable->get_output(0, rows.data(), rows.size(), ScalarType::Float);
    EXPECT_EQ(rows, expected);
  }
}

// It passes with or without the float row scales;
// EmbedsInt4RowsWithExactFloatScales is the test that needs them.
TEST(GgmlEngineTest, NarrowsScalesOfInt4EmbeddingsSharedWithLinear) {
  // A linear also reads the weight, so its Q4_0 blocks hold the scales in
  // fp16: 0.1 and 0.3 narrow to these.
  const std::vector<float> narrowed = {0.0999755859375f, 0.300048828125f};
  TestProgram program;
  program.affine_weight("weight", {2, 32});
  program.input("ids", fbs::ScalarType::LONG, {3});
  program.input("x", fbs::ScalarType::FLOAT, {1, 32});
  program.call(
      "y",
      "torch.ops.aten.linear.default",
      {{"input", std::string("x")}, {"weight", std::string("weight")}},
      fbs::ScalarType::FLOAT,
      {1, 2});
  program.call(
      "rows",
      "torch.ops.aten.embedding.default",
      {{"weight", std::string("weight")}, {"indices", std::string("ids")}},
      fbs::ScalarType::FLOAT,
      {3, 32});
  auto host = GgmlEngineHost::create(kAllowActivationQuant);
  auto context =
      make_int4_context(*host, program.finish("rows"), 2, {0.1f, 0.3f});
  auto executable = context->compile("forward");
  const std::vector<int64_t> ids = {1, 0, 1};
  const std::vector<float> x(32, 1.0f);
  executable->set_input(0, ids.data(), ids.size(), ScalarType::Long);
  executable->set_input(1, x.data(), x.size(), ScalarType::Float);
  executable->execute();
  std::vector<float> rows(3 * 32);
  executable->get_output(0, rows.data(), rows.size(), ScalarType::Float);
  for (size_t i = 0; i < rows.size(); ++i) {
    const int64_t id = ids[i / 32];
    EXPECT_EQ(rows[i], static_cast<float>(id + 1) * narrowed[id]) << i;
  }
}

TEST(GgmlEngineTest, RejectsZeroWidthInt4Weights) {
  TestProgram program;
  program.affine_weight("weight", {2, 0});
  program.input("ids", fbs::ScalarType::LONG, {1});
  program.call(
      "rows",
      "torch.ops.aten.embedding.default",
      {{"weight", std::string("weight")}, {"indices", std::string("ids")}},
      fbs::ScalarType::FLOAT,
      {1, 0});
  auto host = GgmlEngineHost::create();
  auto context = make_int4_context(*host, program.finish("rows"), 2, {});
  expect_compile_error(*context, "affine weights must be int4 [N, K]");
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
  auto host = GgmlEngineHost::create(kAllowActivationQuant);
  expect_compile_error(
      *make_context(*host, program.finish("y")), "divisible by 256");
}

// A linear of make_context's packaged block, labeled with `codec`.
std::vector<uint8_t> make_packed_linear_program(const char* codec) {
  TestProgram program;
  program.q4_k_weight("weight", {1, 256}, codec);
  program.input("x", fbs::ScalarType::FLOAT, {1, 256});
  program.call(
      "y",
      "torch.ops.aten.linear.default",
      {{"input", std::string("x")}, {"weight", std::string("weight")}},
      fbs::ScalarType::FLOAT,
      {1, 1});
  return program.finish("y");
}

TEST(GgmlEngineTest, RejectsCodecsOfNoGgmlWeightType) {
  // q8_K only quantizes activations, f16 stores no blocks, and the others
  // are not "gguf:" and a lowercase ggml type name.
  auto host = GgmlEngineHost::create(kAllowActivationQuant);
  for (const char* codec :
       {"gguf:q8_k", "gguf:f16", "gguf:q4_2", "gguf:Q4_K", "gguf:", "q4_k"}) {
    SCOPED_TRACE(codec);
    expect_compile_error(
        *make_context(*host, make_packed_linear_program(codec)),
        "unsupported PackedQuant codec");
  }
}

TEST(GgmlEngineTest, RejectsGgufTypesNotYetValidated) {
  // tq1_0 is a ggml weight type that no op sample validates.
  auto host = GgmlEngineHost::create(kAllowActivationQuant);
  expect_compile_error(
      *make_context(*host, make_packed_linear_program("gguf:tq1_0")),
      "codec gguf:tq1_0 not yet validated");
}

TEST(GgmlEngineTest, SizesGgufWeightsWithGgmlBlockGeometry) {
  // make_context packages one 144-byte block_q4_K, while a [1, 256]
  // gguf:q6_k weight is one 210-byte block_q6_K.
  auto host = GgmlEngineHost::create(kAllowActivationQuant);
  expect_compile_error(
      *make_context(*host, make_packed_linear_program("gguf:q6_k")),
      "stored byte count disagrees");
}

// Values and bytes per block of each gguf:<type> codec the engine accepts,
// summed from the fields of its block struct in ggml-common.h (fp16 scales
// take 2 bytes). The exporter's tests check gguf-py against the same table.
struct GgufBlockLayout {
  std::string_view type;
  int64_t values;
  size_t bytes;
};

constexpr GgufBlockLayout kGgufBlockLayouts[] = {
    {"q4_0", 32, 2 + 32 / 2}, // d, qs
    {"q5_0", 32, 2 + 4 + 32 / 2}, // d, qh, qs
    {"q8_0", 32, 2 + 32}, // d, qs
    {"q2_k", 256, 256 / 16 + 256 / 4 + 2 + 2}, // scales, qs, d, dmin
    {"q3_k", 256, 256 / 8 + 256 / 4 + 12 + 2}, // hmask, qs, scales, d
    {"q4_k", 256, 2 + 2 + 12 + 256 / 2}, // d, dmin, scales, qs
    {"q5_k", 256, 2 + 2 + 12 + 256 / 8 + 256 / 2}, // d, dmin, scales, qh, qs
    {"q6_k", 256, 256 / 2 + 256 / 4 + 256 / 16 + 2}, // ql, qh, scales, d
};

// Whether compiling a linear of a `codec` weight gets past the codec check.
bool accepts_codec(GgmlEngineHost& host, const std::string& codec) {
  try {
    make_context(host, make_packed_linear_program(codec.c_str()))
        ->compile("forward");
  } catch (const std::runtime_error& error) {
    const std::string_view message = error.what();
    return message.find("not yet validated") == std::string_view::npos &&
        message.find("unsupported PackedQuant codec") == std::string_view::npos;
  }
  return true;
}

TEST(GgmlEngineTest, AcceptsTheGgufTypesOfKnownBlockLayoutOnly) {
  auto host = GgmlEngineHost::create(kAllowActivationQuant);
  std::set<std::string> accepted;
  for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
    const auto type = static_cast<ggml_type>(i);
    const char* type_name = ggml_type_name(type);
    if (type_name == nullptr) {
      continue;
    }
    std::string name = type_name;
    std::ranges::transform(name, name.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    if (!accepts_codec(*host, "gguf:" + name)) {
      continue;
    }
    SCOPED_TRACE(name);
    accepted.insert(name);
    const auto* layout =
        std::ranges::find(kGgufBlockLayouts, name, &GgufBlockLayout::type);
    ASSERT_NE(layout, std::ranges::end(kGgufBlockLayouts))
        << "no expected block layout";
    EXPECT_EQ(ggml_blck_size(type), layout->values);
    EXPECT_EQ(ggml_type_size(type), layout->bytes);
  }
  std::set<std::string> expected;
  for (const GgufBlockLayout& layout : kGgufBlockLayouts) {
    expected.emplace(layout.type);
  }
  EXPECT_EQ(accepted, expected);
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
    std::string header = kQ4_0SafeTensorsHeader;
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

    auto host = GgmlEngineHost::create(kAllowActivationQuant);
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

TEST(GgmlEngineTest, CastsInt64ToHalfAndBFloat16) {
  for (const fbs::ScalarType dtype :
       {fbs::ScalarType::HALF, fbs::ScalarType::BFLOAT16}) {
    SCOPED_TRACE(static_cast<int>(dtype));
    TestProgram program;
    program.input("x", fbs::ScalarType::LONG, {2, 3});
    program.call(
        "y",
        "torch.ops.aten._to_copy.default",
        {{"self", std::string("x")}},
        dtype,
        {2, 3});
    auto host = GgmlEngineHost::create();
    auto context = make_context(*host, program.finish("y"));
    auto executable = context->compile("forward");
    const std::vector<int64_t> x = {0, 1, 2, 3, 4, 5};
    executable->set_input(0, x.data(), x.size(), ScalarType::Long);
    executable->execute();
    std::vector<float> y(6);
    executable->get_output(0, y.data(), y.size(), ScalarType::Float);
    EXPECT_EQ(y, (std::vector<float>{0, 1, 2, 3, 4, 5}));
  }
}

TEST(GgmlEngineTest, SumOverTrailingDims) {
  const std::vector<int64_t> shape = {2, 3, 4, 5};
  struct Case {
    std::vector<int64_t> dims;
    bool keepdim;
    size_t width;
    std::vector<int64_t> out_shape;
  };
  for (const Case& c :
       {Case{{3}, true, 5, {2, 3, 4, 1}},
        Case{{-1}, false, 5, {2, 3, 4}},
        Case{{2, 3}, true, 20, {2, 3, 1, 1}},
        Case{{3, -2}, false, 20, {2, 3}}}) {
    for (const size_t cached : {size_t{0}, size_t{1}}) {
      SCOPED_TRACE(std::to_string(c.width) + std::to_string(cached));
      TestProgram program;
      program.input("x", fbs::ScalarType::FLOAT, shape);
      program.call(
          "y",
          "torch.ops.aten.sum.dim_IntList",
          {{"self", std::string("x")},
           {"dim", c.dims},
           {"keepdim", BoolValue{c.keepdim}}},
          fbs::ScalarType::FLOAT,
          c.out_shape);
      auto host = GgmlEngineHost::create(
          GgmlEngineOptions{.max_cached_graphs = cached});
      auto context = make_context(*host, program.finish("y"));
      auto executable = context->compile("forward");
      for (const float scale : {1.0f, -3.0f}) {
        std::vector<float> x(120);
        for (size_t i = 0; i < x.size(); ++i) {
          x[i] = scale * std::cos(static_cast<float>(i) * 1.3f);
        }
        executable->set_input(0, x.data(), x.size(), ScalarType::Float);
        executable->execute();
        std::vector<float> y(120 / c.width);
        executable->get_output(0, y.data(), y.size(), ScalarType::Float);
        for (size_t row = 0; row < y.size(); ++row) {
          double expected = 0.0;
          for (size_t i = 0; i < c.width; ++i) {
            expected += x[row * c.width + i];
          }
          EXPECT_NEAR(y[row], expected, 1e-5) << row;
        }
      }
    }
  }
}

TEST(GgmlEngineTest, RejectsSumOverARepeatedDim) {
  TestProgram program;
  program.input("x", fbs::ScalarType::FLOAT, {2, 3, 4});
  program.call(
      "y",
      "torch.ops.aten.sum.dim_IntList",
      {{"self", std::string("x")},
       {"dim", std::vector<int64_t>{1, -2}},
       {"keepdim", BoolValue{false}}},
      fbs::ScalarType::FLOAT,
      {2, 4});
  expect_compile_error(program.finish("y"), "sum repeats a dimension");
}

TEST(GgmlEngineTest, AmaxOverTrailingDims) {
  const std::vector<int64_t> shape = {2, 3, 4, 5};
  struct Case {
    std::vector<int64_t> dims;
    bool keepdim;
    size_t width;
    std::vector<int64_t> out_shape;
  };
  constexpr float kInf = std::numeric_limits<float>::infinity();
  for (const Case& c :
       {Case{{-1}, true, 5, {2, 3, 4, 1}},
        Case{{2, 3}, false, 20, {2, 3}},
        Case{{0, 1, 2, 3}, false, 120, {}}}) {
    for (const size_t cached : {size_t{0}, size_t{1}}) {
      SCOPED_TRACE(std::to_string(c.width) + std::to_string(cached));
      TestProgram program;
      program.input("x", fbs::ScalarType::FLOAT, shape);
      program.call(
          "y",
          "torch.ops.aten.amax.default",
          {{"self", std::string("x")},
           {"dim", c.dims},
           {"keepdim", BoolValue{c.keepdim}}},
          fbs::ScalarType::FLOAT,
          c.out_shape);
      auto host = GgmlEngineHost::create(
          GgmlEngineOptions{.max_cached_graphs = cached});
      auto context = make_context(*host, program.finish("y"));
      auto executable = context->compile("forward");
      for (const float scale : {1.0f, -3.0f}) {
        std::vector<float> x(120);
        for (size_t i = 0; i < x.size(); ++i) {
          x[i] = scale * std::cos(static_cast<float>(i) * 1.3f);
        }
        // The first row is all -inf, and the last holds +inf.
        std::fill_n(x.begin(), c.width, -kInf);
        x.back() = kInf;
        executable->set_input(0, x.data(), x.size(), ScalarType::Float);
        executable->execute();
        std::vector<float> y(120 / c.width);
        executable->get_output(0, y.data(), y.size(), ScalarType::Float);
        std::vector<float> expected(y.size(), -kInf);
        for (size_t i = 0; i < x.size(); ++i) {
          expected[i / c.width] = std::max(expected[i / c.width], x[i]);
        }
        EXPECT_EQ(y, expected);
      }
    }
  }
}

// A split of an [extent] F32 input into `pieces` declared [1] outputs, with
// `args` after self and dim 0; returns the first piece.
std::vector<uint8_t> make_split_program(
    int64_t extent,
    const char* target,
    const std::pair<std::string, TestArg>& sizes,
    size_t pieces) {
  TestProgram program;
  program.input("x", fbs::ScalarType::FLOAT, {extent});
  std::vector<std::string> names;
  for (size_t i = 0; i < pieces; ++i) {
    names.push_back("piece" + std::to_string(i));
  }
  program.call_list(
      "split",
      target,
      {{"self", std::string("x")}, sizes, {"dim", int64_t{0}}},
      names,
      {1});
  return program.finish(names.front());
}

TEST(GgmlEngineTest, SplitRefusesSizesThatOverflowOrOutnumberItsOutputs) {
  // The sizes wrap around to 4 in int64.
  constexpr int64_t kQuarter = int64_t{1} << 62;
  expect_compile_error(
      make_split_program(
          4,
          "torch.ops.aten.split_with_sizes_copy.default",
          {"split_sizes",
           std::vector<int64_t>{kQuarter, kQuarter, kQuarter, kQuarter + 4}},
          4),
      "add up");
  // 2^31 - 1 pieces of one element for two declared outputs, refused before
  // the pieces are listed.
  expect_compile_error(
      make_split_program(
          (int64_t{1} << 31) - 1,
          "torch.ops.aten.split_copy.Tensor",
          {"split_size", int64_t{1}},
          2),
      "one tensor per piece");
  auto host = GgmlEngineHost::create();
  auto context = make_context(
      *host,
      make_split_program(
          4,
          "torch.ops.aten.split_with_sizes_copy.default",
          {"split_sizes", std::vector<int64_t>{1, 1, 1, 1}},
          4));
  auto executable = context->compile("forward");
  const std::vector<float> x{1.0f, 2.0f, 3.0f, 4.0f};
  executable->set_input(0, x.data(), x.size(), ScalarType::Float);
  executable->execute();
  float first = 0.0f;
  executable->get_output(0, &first, 1, ScalarType::Float);
  EXPECT_EQ(first, 1.0f);
}

// y = x + expand(z) + expand(z) + ..., `count` adds: each a REPEAT node, its
// shape leaf and an ADD node.
std::vector<uint8_t> make_expand_add_chain(int count) {
  TestProgram program;
  program.input("x", fbs::ScalarType::FLOAT, {1, 64});
  program.input("z", fbs::ScalarType::FLOAT, {1, 1});
  std::string last = "x";
  for (int i = 0; i < count; ++i) {
    const std::string expanded = "e" + std::to_string(i);
    const std::string sum = "y" + std::to_string(i);
    program.call(
        expanded,
        "torch.ops.aten.expand_copy.default",
        {{"self", std::string("z")}, {"size", std::vector<int64_t>{1, 64}}},
        fbs::ScalarType::FLOAT,
        {1, 64});
    program.call(
        sum,
        "torch.ops.aten.add.Tensor",
        {{"self", last}, {"other", expanded}},
        fbs::ScalarType::FLOAT,
        {1, 64});
    last = sum;
  }
  return program.finish(last);
}

// GCC defines __SANITIZE_*__ but has __has_feature only from version 14.
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define PTN_TEST_SANITIZED
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer) || \
    __has_feature(memory_sanitizer)
#define PTN_TEST_SANITIZED
#endif
#endif

bool limit_address_space_to_4gib() {
  const rlimit limit{.rlim_cur = 4ULL << 30, .rlim_max = 4ULL << 30};
  rlimit applied{};
  return setrlimit(RLIMIT_AS, &limit) == 0 &&
      getrlimit(RLIMIT_AS, &applied) == 0 && applied.rlim_cur == limit.rlim_cur;
}

// macOS refuses any finite RLIMIT_AS. The probe child only makes system calls,
// so forking this threaded process is safe for it.
bool can_limit_address_space() {
  const pid_t pid = fork();
  if (pid == 0) {
    _exit(limit_address_space_to_4gib() ? 0 : 2);
  }
  int status = 0;
  return pid > 0 && waitpid(pid, &status, 0) == pid && WIFEXITED(status) &&
      WEXITSTATUS(status) == 0;
}

// Runs make_expand_add_chain(count) once under a 4 GiB address-space limit and
// exits 0 if its result is right, 2 if the limit could not be set.
[[noreturn]] void run_chain_under_limit(
    const std::vector<uint8_t>& program,
    int count,
    size_t cached) {
  if (!limit_address_space_to_4gib()) {
    std::exit(2);
  }
  auto host = GgmlEngineHost::create(
      GgmlEngineOptions{.cpu_threads = 2, .max_cached_graphs = cached});
  auto context = make_context(*host, program);
  auto executable = context->compile("forward");
  const std::vector<float> x(64, 1.0f);
  const float z = 0.5f;
  executable->set_input(0, x.data(), x.size(), ScalarType::Float);
  executable->set_input(1, &z, 1, ScalarType::Float);
  executable->execute();
  std::vector<float> y(64);
  executable->get_output(0, y.data(), y.size(), ScalarType::Float);
  std::exit(y[63] == 1.0f + 0.5f * static_cast<float>(count) ? 0 : 1);
}

// ggml mallocs a scheduler's split context in proportion to its size, about
// 20 KB per unit, so a scheduler sized from a method's metadata budget, 80
// tensors per node, takes gigabytes. An address-space limit refuses that even
// though the pages are never touched; each chain runs in a child under 4 GiB.
// Earlier tests leave the CPU backend's OpenMP threads running, and a forked
// child of a threaded process can deadlock in them, so each child re-executes
// the test binary and runs only this test ("threadsafe" death tests).
TEST(GgmlEngineTest, SizesTheSchedulerFromTheBuiltGraph) {
#ifdef PTN_TEST_SANITIZED
  GTEST_SKIP() << "sanitizers reserve more address space than the limit";
#endif
  if (!can_limit_address_space()) {
    GTEST_SKIP() << "this host refuses a 4 GiB address-space limit";
  }
  ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
  // 2,000 and 10,000 method nodes; the second builds about 10,000 graph nodes
  // and 5,000 leafs.
  for (const int count : {1000, 5000}) {
    const std::vector<uint8_t> program = make_expand_add_chain(count);
    for (const size_t cached : {size_t{0}, size_t{1}}) {
      EXPECT_EXIT(
          run_chain_under_limit(program, count, cached),
          ::testing::ExitedWithCode(0),
          "")
          << count << " adds, " << cached << " cached graphs";
    }
  }
}

TEST(GgmlEngineTest, GpuUnavailableOnCpuVariant) {
  GgmlEngineOptions options;
  options.device = GgmlDevice::Gpu;
  EXPECT_THROW(GgmlEngineHost::create(options), std::runtime_error);
}

} // namespace
} // namespace ptn
