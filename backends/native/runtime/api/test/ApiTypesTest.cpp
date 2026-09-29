// cppcheck-suppress-file syntaxError

// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/api/MethodInfo.h>
#include <executorch/backends/native/runtime/api/Model.h>
#include <executorch/backends/native/runtime/api/Session.h>
#include <executorch/backends/native/runtime/api/Tensor.h>

#include <array>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>

#include <executorch/backends/native/runtime/deserialize/Package.h>
#include <executorch/backends/native/runtime/deserialize/test/PackageTestData.h>
#include <executorch/backends/native/runtime/engine/Engine.h>
#include <executorch/backends/native/runtime/native_graph_generated.h>

namespace ptn {
namespace {

// The API budget: a Model is a shareable handle, a Session is an exclusive
// state domain. Copying a Session would duplicate a KV-cache owner.
static_assert(std::is_copy_constructible_v<Model>);
static_assert(std::is_move_constructible_v<Model>);
static_assert(!std::is_copy_constructible_v<Session>);
static_assert(std::is_move_constructible_v<Session>);
static_assert(std::is_nothrow_move_constructible_v<Session>);

// Views borrow; they must stay cheap enough to pass by value.
static_assert(std::is_trivially_copyable_v<ConstTensorView>);
static_assert(std::is_trivially_copyable_v<MutableTensorView>);

std::vector<uint8_t> make_empty_package() {
  flatbuffers::FlatBufferBuilder builder;
  const auto version = builder.CreateString("1.0");
  const auto methods = builder.CreateVector(
      std::vector<flatbuffers::Offset<native_backend::Method>>{});
  const auto program = native_backend::CreateProgram(builder, version, methods);
  native_backend::FinishProgramBuffer(builder, program);
  return testing::make_zip({
      {kProgramEntry,
       {builder.GetBufferPointer(),
        builder.GetBufferPointer() + builder.GetSize()}},
  });
}

std::vector<uint8_t> make_tensor_package() {
  flatbuffers::FlatBufferBuilder builder;
  const auto dimensions = builder.CreateVector(
      std::vector<flatbuffers::Offset<native_backend::Dim>>{
          native_backend::CreateDim(builder, /*min=*/2, /*max=*/2)});
  const auto metadata = native_backend::CreateTensorMeta(
      builder, native_backend::ScalarType::FLOAT, dimensions);
  const auto input_name = builder.CreateString("input");
  const auto output_name = builder.CreateString("output");
  const auto graph = native_backend::CreateGraph(
      builder,
      builder.CreateVector(
          std::vector<flatbuffers::Offset<native_backend::Node>>{}),
      builder.CreateVector(
          std::vector<flatbuffers::Offset<flatbuffers::String>>{input_name}),
      builder.CreateVector(
          std::vector<flatbuffers::Offset<flatbuffers::String>>{output_name}),
      builder.CreateVector(
          std::vector<flatbuffers::Offset<native_backend::TensorValue>>{
              native_backend::CreateTensorValue(builder, input_name, metadata),
              native_backend::CreateTensorValue(
                  builder, output_name, metadata)}));
  const auto output_spec = native_backend::CreateOutputSpec(
      builder,
      output_name,
      native_backend::OutputKind::USER_OUTPUT,
      builder.CreateString(""));
  const auto method = native_backend::CreateMethod(
      builder,
      builder.CreateString("forward"),
      graph,
      /*constants=*/0,
      builder.CreateVector(
          std::vector<flatbuffers::Offset<native_backend::OutputSpec>>{
              output_spec}));
  const auto program = native_backend::CreateProgram(
      builder,
      builder.CreateString("1.0"),
      builder.CreateVector(
          std::vector<flatbuffers::Offset<native_backend::Method>>{method}));
  native_backend::FinishProgramBuffer(builder, program);
  return testing::make_zip({
      {kProgramEntry,
       {builder.GetBufferPointer(),
        builder.GetBufferPointer() + builder.GetSize()}},
  });
}

class EmptyContext final : public EngineContext {
 public:
  EmptyContext(
      std::shared_ptr<const Program> program,
      std::shared_ptr<const Package> package)
      : EngineContext(std::move(program), std::move(package)) {}

 private:
  [[noreturn]] std::unique_ptr<EngineExecutable> compile_method(
      const Method&) override {
    throw std::runtime_error("empty program has no methods");
  }
};

class EmptyHost final : public EngineHost {
 public:
  const std::string& name() const override {
    return name_;
  }

  const std::string& device_name() const override {
    return device_name_;
  }

  std::unique_ptr<EngineContext> create_context(
      std::shared_ptr<const Program> program,
      std::shared_ptr<const Package> package) override {
    return std::make_unique<EmptyContext>(
        std::move(program), std::move(package));
  }

 private:
  std::string name_ = "empty";
  std::string device_name_ = "none";
};

class ThrowingHost final : public EngineHost {
 public:
  const std::string& name() const override {
    return name_;
  }

  const std::string& device_name() const override {
    return device_name_;
  }

  [[noreturn]] std::unique_ptr<EngineContext> create_context(
      std::shared_ptr<const Program>,
      std::shared_ptr<const Package>) override {
    throw std::runtime_error("context creation failed");
  }

 private:
  std::string name_ = "throwing";
  std::string device_name_ = "none";
};

class RunningExecutable final : public EngineExecutable {
 public:
  size_t num_inputs() const override {
    return 1;
  }

  size_t num_outputs() const override {
    return 1;
  }

  std::vector<int64_t> input_sizes(size_t) const override {
    return {2};
  }

  std::vector<int64_t> output_sizes(size_t) const override {
    return {2};
  }

  ScalarType input_dtype(size_t) const override {
    return kFloat;
  }

  ScalarType output_dtype(size_t) const override {
    return kFloat;
  }

  void set_input(size_t, const void* data, size_t numel, ScalarType) override {
    if (data == nullptr || numel != input_.size()) {
      throw std::runtime_error("invalid test input");
    }
    std::memcpy(input_.data(), data, sizeof(input_));
  }

  void execute() override {
    output_[0] = input_[0] + 1.0f;
    output_[1] = input_[1] + 1.0f;
  }

  void get_output(size_t, void* data, size_t numel, ScalarType) override {
    if (data == nullptr || numel != output_.size()) {
      throw std::runtime_error("invalid test output");
    }
    std::memcpy(data, output_.data(), sizeof(output_));
  }

 private:
  std::array<float, 2> input_{};
  std::array<float, 2> output_{};
};

class RunningContext final : public EngineContext {
 public:
  RunningContext(
      std::shared_ptr<const Program> program,
      std::shared_ptr<const Package> package)
      : EngineContext(std::move(program), std::move(package)) {}

 private:
  std::unique_ptr<EngineExecutable> compile_method(const Method&) override {
    return std::make_unique<RunningExecutable>();
  }
};

class RunningHost final : public EngineHost {
 public:
  const std::string& name() const override {
    return name_;
  }

  const std::string& device_name() const override {
    return device_name_;
  }

  std::unique_ptr<EngineContext> create_context(
      std::shared_ptr<const Program> program,
      std::shared_ptr<const Package> package) override {
    return std::make_unique<RunningContext>(
        std::move(program), std::move(package));
  }

 private:
  std::string name_ = "running";
  std::string device_name_ = "test-device";
};

TEST(ModelTest, LoadsAndCreatesAnIsolatedSession) {
  Model model = Model::load_bytes(make_empty_package());
  EXPECT_TRUE(model.method_names().empty());
  EXPECT_THROW(model.method_info("missing"), std::invalid_argument);

  EmptyHost host;
  Session session = model.create_session(host);
  EXPECT_FALSE(session.is_prepared("missing"));
  EXPECT_THROW(session.prepare("missing"), std::invalid_argument);
  EXPECT_THROW(session.release("missing"), std::invalid_argument);
}

TEST(ModelTest, RejectsMalformedBytes) {
  EXPECT_THROW(Model::load_bytes({0, 1, 2, 3}), std::runtime_error);
}

TEST(ModelTest, PropagatesEngineExceptions) {
  Model model = Model::load_bytes(make_empty_package());
  ThrowingHost host;

  EXPECT_THROW(model.create_session(host), std::runtime_error);
}

TEST(SessionTest, RunExecutesPreparedMethod) {
  Model model = Model::load_bytes(make_tensor_package());
  RunningHost host;
  Session session = model.create_session(host);
  const std::array<int64_t, 1> sizes{2};
  const std::array<int64_t, 1> strides{1};
  const std::array<float, 2> input{1.0f, 2.0f};
  std::array<float, 2> output{};
  const std::array<ConstTensorView, 1> inputs{
      ConstTensorView(kFloat, sizes, strides, input.data(), sizeof(input))};
  std::array<MutableTensorView, 1> outputs{
      MutableTensorView(kFloat, sizes, strides, output.data(), sizeof(output))};

  session.run("forward", inputs, outputs);

  EXPECT_TRUE(session.is_prepared("forward"));
  EXPECT_EQ(output, (std::array<float, 2>{2.0f, 3.0f}));
}

TEST(SessionTest, RunRejectsMismatchedTensorDtypes) {
  Model model = Model::load_bytes(make_tensor_package());
  RunningHost host;
  Session session = model.create_session(host);
  const std::array<int64_t, 1> sizes{2};
  const std::array<int64_t, 1> strides{1};
  const std::array<int64_t, 2> wrong_input{1, 2};
  std::array<float, 2> output{};
  const std::array<ConstTensorView, 1> wrong_inputs{ConstTensorView(
      kLong, sizes, strides, wrong_input.data(), sizeof(wrong_input))};
  std::array<MutableTensorView, 1> outputs{
      MutableTensorView(kFloat, sizes, strides, output.data(), sizeof(output))};

  EXPECT_THROW(
      session.run("forward", wrong_inputs, outputs), std::invalid_argument);

  const std::array<float, 2> input{1.0f, 2.0f};
  std::array<int64_t, 2> wrong_output{};
  const std::array<ConstTensorView, 1> inputs{
      ConstTensorView(kFloat, sizes, strides, input.data(), sizeof(input))};
  std::array<MutableTensorView, 1> wrong_outputs{MutableTensorView(
      kLong, sizes, strides, wrong_output.data(), sizeof(wrong_output))};

  EXPECT_THROW(
      session.run("forward", inputs, wrong_outputs), std::invalid_argument);
}

TEST(MethodInfoTest, ReportsSignature) {
  const MethodInfo info(
      "forward", {TensorInfo(kFloat, {2, 3})}, {TensorInfo(kLong, {2})});

  EXPECT_EQ(info.name(), "forward");
  ASSERT_EQ(info.inputs().size(), 1u);
  ASSERT_EQ(info.outputs().size(), 1u);

  const std::vector<int64_t> expected_input_sizes = {2, 3};
  const std::vector<int64_t> expected_input_strides = {3, 1};
  const TensorInfo& input = info.inputs()[0];
  EXPECT_EQ(input.dtype(), kFloat);
  EXPECT_EQ(
      std::vector<int64_t>(input.sizes().begin(), input.sizes().end()),
      expected_input_sizes);
  EXPECT_EQ(
      std::vector<int64_t>(input.strides().begin(), input.strides().end()),
      expected_input_strides);
  EXPECT_EQ(input.nbytes(), 24u);
  const TensorInfo& output = info.outputs()[0];
  EXPECT_EQ(output.dtype(), kLong);
  EXPECT_EQ(
      std::vector<int64_t>(output.sizes().begin(), output.sizes().end()),
      (std::vector<int64_t>{2}));
  EXPECT_EQ(
      std::vector<int64_t>(output.strides().begin(), output.strides().end()),
      (std::vector<int64_t>{1}));
  EXPECT_EQ(output.nbytes(), 16u);
}

TEST(TensorViewTest, BorrowsCallerStorage) {
  const std::vector<int64_t> sizes = {2, 2};
  const std::vector<int64_t> strides = {2, 1};
  constexpr size_t kElementCount = 4;
  std::vector<float> storage(kElementCount, 1.0f);

  const ConstTensorView input(
      kFloat, sizes, strides, storage.data(), storage.size() * sizeof(float));
  EXPECT_EQ(input.dtype(), kFloat);
  EXPECT_EQ(input.data(), storage.data());
  EXPECT_EQ(input.nbytes(), 16u);
  EXPECT_EQ(
      std::vector<int64_t>(input.strides().begin(), input.strides().end()),
      strides);

  const MutableTensorView output(
      kFloat, sizes, strides, storage.data(), storage.size() * sizeof(float));
  EXPECT_EQ(output.data(), storage.data());
}

} // namespace
} // namespace ptn
