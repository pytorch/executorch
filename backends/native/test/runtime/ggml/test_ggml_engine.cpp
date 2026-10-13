// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/engine/ggml/GgmlEngine.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
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

// `cache` is an unused mutable buffer.
std::vector<uint8_t> make_q4_k_linear_program(
    const char* target = "torch.ops.aten.linear.default",
    bool with_cache = false) {
  flatbuffers::FlatBufferBuilder builder;
  const auto packed = fbs::CreatePackedQuantDirect(builder, "gguf:q4_k");
  const auto quant = fbs::CreateQuantSpec(
      builder, fbs::QuantScheme::PackedQuant, packed.Union());
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
  const auto graph = fbs::CreateGraph(
      builder,
      builder.CreateVector(nodes),
      create_strings(builder, {"input"}),
      create_strings(builder, {"output"}),
      builder.CreateVector(values));
  const auto binding = fbs::CreateNamedTensorRefDirect(
      builder, "weight", "weight", weight_meta, fbs::InputKind::PARAMETER);
  const auto output_spec = fbs::CreateOutputSpecDirect(builder, "output");
  const auto method = fbs::CreateMethod(
      builder,
      builder.CreateString("forward"),
      graph,
      builder.CreateVector(
          std::vector<flatbuffers::Offset<fbs::NamedTensorRef>>{binding}),
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

  void buffer(const std::string& name, const std::vector<int64_t>& shape) {
    placeholder(
        name, create_tensor_meta(builder_, fbs::ScalarType::FLOAT, shape));
    buffers_.push_back(fbs::CreateMutableBufferSpecDirect(
        builder_, name.c_str(), name.c_str()));
  }

  // Each execution ends by writing value `value` to buffer `name`.
  void mutate(const std::string& name, const std::string& value) {
    mutations_.emplace_back(name, value);
  }

  // make_context's packaged block_q4_K as a [1, 256] parameter.
  void q4_k_weight(const std::string& name) {
    const auto quant = fbs::CreateQuantSpec(
        builder_,
        fbs::QuantScheme::PackedQuant,
        fbs::CreatePackedQuantDirect(builder_, "gguf:q4_k").Union());
    const auto meta =
        create_tensor_meta(builder_, fbs::ScalarType::BYTE, {1, 256}, quant);
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

  std::vector<uint8_t> finish(const std::string& output) {
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
        fbs::CreateOutputSpecDirect(builder_, output.c_str());
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
  auto context = make_context(*host, make_q4_k_linear_program());
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

TEST(GgmlEngineTest, RejectsUnsupportedOpAtCompile) {
  auto host = GgmlEngineHost::create();
  auto context = make_context(
      *host, make_q4_k_linear_program("torch.ops.aten.unsupported.default"));
  EXPECT_THROW(context->compile("forward"), std::runtime_error);
}

TEST(GgmlEngineTest, RejectsSecondExecutableBindingMutableBuffer) {
  auto host = GgmlEngineHost::create();
  auto context = make_context(
      *host,
      make_q4_k_linear_program(
          "torch.ops.aten.linear.default", /*with_cache=*/true));
  auto executable = context->compile("forward");
  EXPECT_THROW(context->compile("forward"), std::runtime_error);
}

TEST(GgmlEngineTest, ReleasesMutableBufferWithItsExecutable) {
  auto host = GgmlEngineHost::create();
  auto context = make_context(
      *host,
      make_q4_k_linear_program(
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

} // namespace
} // namespace ptn
