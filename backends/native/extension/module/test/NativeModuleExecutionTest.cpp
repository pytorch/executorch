// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/extension/module/NativeModule.h>

#include <array>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <executorch/backends/native/extension/module/test/TestData.h>
#include <executorch/backends/native/runtime/MethodMeta.h>
#include <executorch/backends/native/runtime/engine/Engine.h>
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <gtest/gtest.h>

namespace executorch::extension::native_module {
namespace {

using executorch::aten::ScalarType;
using torch::executor::testing::TensorFactory;

struct EngineBehavior {
  bool fail_compile = false;
  bool extra_output = false;
  bool bad_input_dtype = false;
  bool bad_output_shape = false;
  bool fail_get_output = false;
};

struct TestEngineState {
  void reset() {
    behavior = {};
    hosts_created = 0;
    contexts_created = 0;
  }

  EngineBehavior behavior;
  size_t hosts_created = 0;
  size_t contexts_created = 0;
};

TestEngineState& engine_state() {
  static TestEngineState state;
  return state;
}

class FakeExecutable final : public ptn::EngineExecutable {
 public:
  FakeExecutable(
      const ptn::Method& method,
      int32_t& shared_state,
      EngineBehavior& behavior)
      : metadata_(ptn::MethodMeta::from_method(method)),
        shared_state_(shared_state),
        behavior_(behavior),
        inputs_(metadata_.inputs().size()) {}

  size_t num_inputs() const override {
    return metadata_.inputs().size();
  }

  size_t num_outputs() const override {
    return metadata_.outputs().size() + (behavior_.extra_output ? 1 : 0);
  }

  std::vector<int64_t> input_sizes(size_t index) const override {
    const auto sizes = metadata_.inputs().at(index).sizes();
    return {sizes.begin(), sizes.end()};
  }

  std::vector<int64_t> output_sizes(size_t index) const override {
    const auto sizes = metadata_.outputs().at(index).sizes();
    std::vector<int64_t> result{sizes.begin(), sizes.end()};
    if (behavior_.bad_output_shape) {
      result.at(0) += 1;
    }
    return result;
  }

  ptn::ScalarType input_dtype(size_t index) const override {
    return behavior_.bad_input_dtype ? ptn::kInt
                                     : metadata_.inputs().at(index).dtype();
  }

  ptn::ScalarType output_dtype(size_t index) const override {
    return metadata_.outputs().at(index).dtype();
  }

  void set_input(
      size_t index,
      const void* data,
      size_t numel,
      ptn::ScalarType dtype) override {
    if (dtype != ptn::kFloat || numel != metadata_.inputs().at(index).numel() ||
        data == nullptr) {
      throw std::runtime_error("bad input transfer");
    }
    const auto* values = static_cast<const float*>(data);
    inputs_.at(index).assign(values, values + numel);
  }

  void execute() override {
    ++shared_state_;
    output_.assign(
        metadata_.outputs().at(0).numel(), static_cast<float>(shared_state_));
    for (const auto& input : inputs_) {
      if (input.size() != output_.size()) {
        throw std::runtime_error("missing input");
      }
      for (size_t i = 0; i < output_.size(); ++i) {
        output_[i] += input[i];
      }
    }
  }

  void get_output(size_t index, void* data, size_t numel, ptn::ScalarType dtype)
      override {
    if (behavior_.fail_get_output) {
      throw std::runtime_error("output transfer failed");
    }
    if (index != 0 || dtype != ptn::kFloat || numel != output_.size() ||
        data == nullptr) {
      throw std::runtime_error("bad output transfer");
    }
    std::memcpy(data, output_.data(), output_.size() * sizeof(float));
  }

 private:
  ptn::MethodMeta metadata_;
  int32_t& shared_state_;
  EngineBehavior& behavior_;
  std::vector<std::vector<float>> inputs_;
  std::vector<float> output_;
};

class FakeContext final : public ptn::EngineContext {
 public:
  FakeContext(
      std::shared_ptr<const ptn::Program> program,
      std::shared_ptr<const ptn::Package> package,
      EngineBehavior& behavior)
      : EngineContext(std::move(program), std::move(package)),
        behavior_(behavior) {}

 private:
  std::unique_ptr<ptn::EngineExecutable> compile_method(
      const ptn::Method& method) override {
    if (behavior_.fail_compile) {
      throw std::runtime_error("unsupported graph");
    }
    return std::make_unique<FakeExecutable>(method, shared_state_, behavior_);
  }

  EngineBehavior& behavior_;
  int32_t shared_state_ = 0;
};

class FakeHost final : public ptn::EngineHost {
 public:
  explicit FakeHost(TestEngineState& state) : state_(state) {}

  const std::string& name() const override {
    return name_;
  }

  const std::string& device_name() const override {
    return device_name_;
  }

  std::unique_ptr<ptn::EngineContext> create_context(
      std::shared_ptr<const ptn::Program> program,
      std::shared_ptr<const ptn::Package> package) override {
    ++state_.contexts_created;
    return std::make_unique<FakeContext>(
        std::move(program), std::move(package), state_.behavior);
  }

 private:
  TestEngineState& state_;
  const std::string name_ = "fake";
  const std::string device_name_ = "fake-device";
};

std::shared_ptr<ptn::EngineHost> create_fake_host() {
  TestEngineState& state = engine_state();
  ++state.hosts_created;
  return std::make_shared<FakeHost>(state);
}

std::shared_ptr<ptn::EngineHost> create_another_fake_host() {
  return std::make_shared<FakeHost>(engine_state());
}

class NativeModuleExecutionTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    executorch_native_module_ptn_link_anchor();
    ASSERT_EQ(
        internal::register_engine_host_factory(nullptr),
        runtime::Error::InvalidArgument);
    ASSERT_EQ(
        internal::register_engine_host_factory(create_fake_host),
        runtime::Error::Ok);
  }

  void SetUp() override {
    engine_state().reset();
  }
};

std::unique_ptr<Module> make_module(const std::vector<uint8_t>& bytes) {
  return std::make_unique<Module>(
      std::make_unique<BufferDataLoader>(bytes.data(), bytes.size()));
}

void expect_values(
    const runtime::Result<std::vector<runtime::EValue>>& result,
    float first,
    float second) {
  ASSERT_TRUE(result.ok());
  ASSERT_EQ(result->size(), 1);
  const auto& tensor = result->at(0).toTensor();
  ASSERT_EQ(tensor.numel(), 2);
  EXPECT_FLOAT_EQ(tensor.const_data_ptr<float>()[0], first);
  EXPECT_FLOAT_EQ(tensor.const_data_ptr<float>()[1], second);
}

runtime::Error bind_temporary_output(
    Module& module,
    std::array<float, 2>& data,
    bool use_set_outputs) {
  std::array<executorch::aten::SizesType, 1> sizes{2};
  std::array<executorch::aten::DimOrderType, 1> dim_order{0};
  std::array<executorch::aten::StridesType, 1> strides{1};
  executorch::aten::TensorImpl impl(
      ScalarType::Float,
      1,
      sizes.data(),
      data.data(),
      dim_order.data(),
      strides.data());
  const runtime::EValue output{executorch::aten::Tensor(&impl)};
  return use_set_outputs ? module.set_outputs({output})
                         : module.set_output(output);
}

// cppcheck-suppress-begin syntaxError
TEST_F(NativeModuleExecutionTest, SessionsShareWithinModuleAndIsolateModules) {
  const auto bytes =
      testing::make_tensor_package(std::vector<std::string>{"first", "second"});
  TensorFactory<ScalarType::Float> tensors;
  const auto input = tensors.make({2}, {1.0f, 2.0f});
  auto first_module = make_module(bytes);
  auto second_module = make_module(bytes);

  expect_values(
      first_module->execute("first", input),
      /*first=*/2.0f,
      /*second=*/3.0f);
  expect_values(
      first_module->execute("second", input),
      /*first=*/3.0f,
      /*second=*/4.0f);
  expect_values(
      second_module->execute("first", input),
      /*first=*/2.0f,
      /*second=*/3.0f);
  EXPECT_EQ(engine_state().hosts_created, 1);
  EXPECT_EQ(engine_state().contexts_created, 2);
}

TEST_F(NativeModuleExecutionTest, SetInputsIsTransactional) {
  const auto bytes =
      testing::make_tensor_package({"forward"}, /*num_inputs=*/2);
  TensorFactory<ScalarType::Float> floats;
  TensorFactory<ScalarType::Int> ints;
  const auto first = floats.make({2}, {1.0f, 2.0f});
  const auto second = floats.make({2}, {10.0f, 20.0f});
  const auto replacement = floats.make({2}, {100.0f, 200.0f});
  const auto wrong_dtype = ints.make({2}, {3, 4});
  auto module = make_module(bytes);

  ASSERT_EQ(module->set_inputs({first, second}), runtime::Error::Ok);
  expect_values(module->execute("forward"), /*first=*/12.0f, /*second=*/23.0f);
  EXPECT_EQ(
      module->set_inputs({replacement, wrong_dtype}),
      runtime::Error::InvalidArgument);
  expect_values(module->execute("forward"), /*first=*/13.0f, /*second=*/24.0f);
}

TEST_F(
    NativeModuleExecutionTest,
    FailedOutputTransferInvalidatesCurrentOutput) {
  const auto bytes = testing::make_tensor_package();
  TensorFactory<ScalarType::Float> tensors;
  const auto input = tensors.make({2}, {1.0f, 2.0f});
  auto destination = tensors.zeros({2});
  auto module = make_module(bytes);

  ASSERT_EQ(module->set_output(destination), runtime::Error::Ok);
  const auto published = module->forward(input);
  expect_values(published, /*first=*/2.0f, /*second=*/3.0f);
  const runtime::EValue published_view = published->at(0);
  EXPECT_FLOAT_EQ(destination.const_data_ptr<float>()[0], 2.0f);
  engine_state().behavior.fail_get_output = true;
  EXPECT_EQ(module->forward(input).error(), runtime::Error::Internal);
  EXPECT_EQ(module->get_outputs().error(), runtime::Error::InvalidState);
  EXPECT_FLOAT_EQ(published_view.toTensor().const_data_ptr<float>()[0], 2.0f);
  EXPECT_FLOAT_EQ(destination.const_data_ptr<float>()[0], 2.0f);
}

TEST_F(NativeModuleExecutionTest, CompileFailureIsRetryable) {
  const auto bytes = testing::make_tensor_package();
  auto module = make_module(bytes);

  engine_state().behavior.fail_compile = true;
  EXPECT_EQ(module->load_method("forward"), runtime::Error::NotSupported);
  EXPECT_FALSE(module->is_method_loaded("forward"));
  engine_state().behavior.fail_compile = false;
  EXPECT_EQ(module->load_method("forward"), runtime::Error::Ok);
  EXPECT_TRUE(module->is_method_loaded("forward"));
  EXPECT_TRUE(module->unload_method("forward"));
  EXPECT_FALSE(module->is_method_loaded("forward"));
}

TEST_F(NativeModuleExecutionTest, SignatureMismatchDoesNotPublishMethod) {
  const auto bytes = testing::make_tensor_package();
  auto module = make_module(bytes);

  engine_state().behavior.extra_output = true;
  EXPECT_EQ(module->load_method("forward"), runtime::Error::Internal);
  EXPECT_FALSE(module->is_method_loaded("forward"));
}

TEST_F(NativeModuleExecutionTest, InputDtypeMismatchDoesNotPublishMethod) {
  const auto bytes = testing::make_tensor_package();
  auto module = make_module(bytes);

  engine_state().behavior.bad_input_dtype = true;
  EXPECT_EQ(module->load_method("forward"), runtime::Error::Internal);
  EXPECT_FALSE(module->is_method_loaded("forward"));
}

TEST_F(NativeModuleExecutionTest, OutputShapeMismatchDoesNotPublishMethod) {
  const auto bytes = testing::make_tensor_package();
  auto module = make_module(bytes);

  engine_state().behavior.bad_output_shape = true;
  EXPECT_EQ(module->load_method("forward"), runtime::Error::Internal);
  EXPECT_FALSE(module->is_method_loaded("forward"));
}

TEST_F(NativeModuleExecutionTest, MissingConstantFailsBeforeEngineContext) {
  const auto bytes = testing::make_missing_constant_package();
  auto module = make_module(bytes);

  EXPECT_EQ(module->load(), runtime::Error::Ok);
  EXPECT_EQ(
      module->load_method("forward"), runtime::Error::InvalidExternalData);
  EXPECT_FALSE(module->is_method_loaded("forward"));
  EXPECT_EQ(engine_state().hosts_created, 0);
  EXPECT_EQ(engine_state().contexts_created, 0);
}

TEST_F(NativeModuleExecutionTest, RejectsInvalidInputAndOutputTensors) {
  const auto bytes = testing::make_tensor_package();
  TensorFactory<ScalarType::Float> floats;
  TensorFactory<ScalarType::Int> ints;
  const auto wrong_shape = floats.zeros({3});
  const auto wrong_dtype = ints.zeros({2});
  std::array<int32_t, 1> sizes{2};
  std::array<uint8_t, 1> dim_order{0};
  std::array<int32_t, 1> non_contiguous_stride{2};
  std::array<float, 2> data{1.0f, 2.0f};
  executorch::aten::TensorImpl non_contiguous_impl(
      ScalarType::Float,
      /*dim=*/1,
      sizes.data(),
      data.data(),
      dim_order.data(),
      non_contiguous_stride.data());
  const executorch::aten::Tensor non_contiguous(&non_contiguous_impl);
  const runtime::EValue non_tensor(int64_t{1});
  auto module = make_module(bytes);

  EXPECT_EQ(
      module->set_input(wrong_shape, /*input_index=*/0),
      runtime::Error::InvalidArgument);
  EXPECT_EQ(
      module->set_input(wrong_dtype, /*input_index=*/0),
      runtime::Error::InvalidArgument);
  EXPECT_EQ(
      module->set_input(non_contiguous, /*input_index=*/0),
      runtime::Error::InvalidArgument);
  EXPECT_EQ(module->set_output(wrong_shape), runtime::Error::InvalidArgument);
  EXPECT_EQ(module->set_output(wrong_dtype), runtime::Error::InvalidArgument);
  EXPECT_EQ(
      module->set_output(non_contiguous), runtime::Error::InvalidArgument);
  EXPECT_EQ(
      module->set_input(non_tensor, /*input_index=*/0),
      runtime::Error::InvalidType);
  EXPECT_EQ(module->get_output().error(), runtime::Error::InvalidState);
}

TEST_F(NativeModuleExecutionTest, RejectsMissingTensorLayoutMetadata) {
  const auto bytes = testing::make_tensor_package();
  std::array<executorch::aten::SizesType, 1> sizes{2};
  std::array<executorch::aten::StridesType, 1> strides{1};
  std::array<float, 2> data{};
  executorch::aten::TensorImpl impl(
      ScalarType::Float,
      1,
      sizes.data(),
      data.data(),
      /*dim_order=*/nullptr,
      strides.data());
  const executorch::aten::Tensor tensor(&impl);
  auto module = make_module(bytes);

  EXPECT_EQ(
      module->set_input(tensor, /*input_index=*/0),
      runtime::Error::InvalidArgument);
  EXPECT_EQ(module->set_output(tensor), runtime::Error::InvalidArgument);
}

TEST_F(NativeModuleExecutionTest, RejectsNonCpuInputAndOutputTensors) {
  const auto bytes = testing::make_tensor_package();
  std::array<executorch::aten::SizesType, 1> sizes{2};
  std::array<executorch::aten::DimOrderType, 1> dim_order{0};
  std::array<executorch::aten::StridesType, 1> strides{1};
  std::array<float, 2> data{};
  executorch::aten::TensorImpl impl(
      ScalarType::Float,
      1,
      sizes.data(),
      data.data(),
      dim_order.data(),
      strides.data(),
      executorch::aten::TensorShapeDynamism::STATIC,
      executorch::aten::DeviceType::CUDA);
  const executorch::aten::Tensor tensor(&impl);
  auto module = make_module(bytes);

  EXPECT_EQ(
      module->set_input(tensor, /*input_index=*/0),
      runtime::Error::NotSupported);
  EXPECT_EQ(module->set_output(tensor), runtime::Error::NotSupported);
}

TEST_F(NativeModuleExecutionTest, SetOutputDoesNotRetainTensorDescriptor) {
  const auto bytes = testing::make_tensor_package();
  TensorFactory<ScalarType::Float> tensors;
  const auto input = tensors.make({2}, {1.0f, 2.0f});
  std::array<float, 2> destination{};
  auto module = make_module(bytes);

  ASSERT_EQ(
      bind_temporary_output(*module, destination, /*use_set_outputs=*/false),
      runtime::Error::Ok);
  expect_values(module->forward(input), /*first=*/2.0f, /*second=*/3.0f);
  EXPECT_EQ(destination, (std::array<float, 2>{2.0f, 3.0f}));
}

TEST_F(NativeModuleExecutionTest, SetOutputsDoesNotRetainTensorDescriptor) {
  const auto bytes = testing::make_tensor_package();
  TensorFactory<ScalarType::Float> tensors;
  const auto input = tensors.make({2}, {1.0f, 2.0f});
  std::array<float, 2> destination{};
  auto module = make_module(bytes);

  ASSERT_EQ(
      bind_temporary_output(*module, destination, /*use_set_outputs=*/true),
      runtime::Error::Ok);
  expect_values(module->forward(input), /*first=*/2.0f, /*second=*/3.0f);
  EXPECT_EQ(destination, (std::array<float, 2>{2.0f, 3.0f}));
}

TEST_F(NativeModuleExecutionTest, PreservesDeclaredOutputLayout) {
  const auto bytes = testing::make_tensor_package(
      {"forward"},
      /*num_inputs=*/1,
      /*sizes=*/{2, 3},
      /*bind_missing_constant=*/false,
      /*version=*/"1.0",
      /*dim_order=*/{1, 0});
  std::array<executorch::aten::SizesType, 2> sizes{2, 3};
  std::array<executorch::aten::DimOrderType, 2> dim_order{1, 0};
  std::array<executorch::aten::StridesType, 2> strides{1, 2};
  std::array<float, 6> input_data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::array<float, 6> destination{};
  executorch::aten::TensorImpl input_impl(
      ScalarType::Float,
      2,
      sizes.data(),
      input_data.data(),
      dim_order.data(),
      strides.data());
  executorch::aten::TensorImpl output_impl(
      ScalarType::Float,
      2,
      sizes.data(),
      destination.data(),
      dim_order.data(),
      strides.data());
  auto module = make_module(bytes);

  ASSERT_EQ(
      module->set_output(executorch::aten::Tensor(&output_impl)),
      runtime::Error::Ok);
  const auto result = module->forward(executorch::aten::Tensor(&input_impl));

  ASSERT_TRUE(result.ok());
  ASSERT_EQ(result->size(), 1);
  const executorch::aten::Tensor& output = result->at(0).toTensor();
  EXPECT_EQ(
      std::vector<executorch::aten::DimOrderType>(
          output.dim_order().begin(), output.dim_order().end()),
      (std::vector<executorch::aten::DimOrderType>{1, 0}));
  EXPECT_EQ(
      std::vector<executorch::aten::StridesType>(
          output.strides().begin(), output.strides().end()),
      (std::vector<executorch::aten::StridesType>{1, 2}));
  const std::array<float, 6> expected{2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  EXPECT_EQ(
      std::vector<float>(
          output.const_data_ptr<float>(),
          output.const_data_ptr<float>() + output.numel()),
      std::vector<float>(expected.begin(), expected.end()));
  EXPECT_EQ(destination, expected);
}

TEST_F(NativeModuleExecutionTest, AcceptsZeroSizedOutputTensor) {
  const auto bytes =
      testing::make_tensor_package("forward", std::vector<int64_t>{0});
  std::array<int32_t, 1> sizes{0};
  std::array<uint8_t, 1> dim_order{0};
  std::array<int32_t, 1> arbitrary_stride{7};
  executorch::aten::TensorImpl output_impl(
      ScalarType::Float,
      /*dim=*/1,
      sizes.data(),
      /*data=*/nullptr,
      dim_order.data(),
      arbitrary_stride.data());
  const executorch::aten::Tensor output(&output_impl);
  auto module = make_module(bytes);

  EXPECT_EQ(module->set_output(output), runtime::Error::Ok);
}

TEST_F(NativeModuleExecutionTest, SecondHostFactoryIsRejected) {
  EXPECT_EQ(
      internal::register_engine_host_factory(create_another_fake_host),
      runtime::Error::AlreadyLoaded);
}

TEST_F(NativeModuleExecutionTest, SameHostFactoryCanBeRegisteredAgain) {
  EXPECT_EQ(
      internal::register_engine_host_factory(create_fake_host),
      runtime::Error::Ok);
}
// cppcheck-suppress-end syntaxError

TEST_F(NativeModuleExecutionTest, MetadataOutlivesMethodUnload) {
  const auto bytes = testing::make_tensor_package();
  auto module = make_module(bytes);
  ASSERT_EQ(module->load_method("forward"), runtime::Error::Ok);
  const auto metadata = module->method_meta("forward");
  ASSERT_TRUE(metadata.ok());

  EXPECT_TRUE(module->is_method_loaded("forward"));
  EXPECT_TRUE(module->unload_method("forward"));
  EXPECT_STREQ(metadata->name(), "forward");
  EXPECT_EQ(metadata->num_inputs(), 1);
}

} // namespace
} // namespace executorch::extension::native_module
