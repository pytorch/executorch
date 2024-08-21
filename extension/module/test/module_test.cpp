/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/module/module.h>

#include <array>
#include <thread>

#include <gtest/gtest.h>

#include <executorch/extension/data_loader/file_data_loader.h>

using namespace ::testing;

namespace torch::executor {

class ModuleTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    model_path_ = std::getenv("RESOURCES_PATH") + std::string("/model.pte");
  }

  static std::string model_path_;
};

std::string ModuleTest::model_path_;

TEST_F(ModuleTest, TestLoad) {
  Module module(model_path_);

  EXPECT_FALSE(module.is_loaded());
  const auto error = module.load();
  EXPECT_EQ(error, Error::Ok);
  EXPECT_TRUE(module.is_loaded());
}

TEST_F(ModuleTest, TestLoadNonExistent) {
  Module module("/path/to/nonexistent/file.pte");
  const auto error = module.load();

  EXPECT_NE(error, Error::Ok);
  EXPECT_FALSE(module.is_loaded());
}

TEST_F(ModuleTest, TestLoadCorruptedFile) {
  Module module("/dev/null");
  const auto error = module.load();

  EXPECT_NE(error, Error::Ok);
  EXPECT_FALSE(module.is_loaded());
}

TEST_F(ModuleTest, TestMethodNames) {
  Module module(model_path_);

  const auto method_names = module.method_names();
  EXPECT_TRUE(method_names.ok());
  EXPECT_EQ(method_names.get(), std::unordered_set<std::string>{"forward"});
}

TEST_F(ModuleTest, TestNonExistentMethodNames) {
  Module module("/path/to/nonexistent/file.pte");

  const auto method_names = module.method_names();
  EXPECT_FALSE(method_names.ok());
}

TEST_F(ModuleTest, TestLoadMethod) {
  Module module(model_path_);

  EXPECT_FALSE(module.is_method_loaded("forward"));
  const auto error = module.load_method("forward");
  EXPECT_EQ(error, Error::Ok);
  EXPECT_TRUE(module.is_method_loaded("forward"));
  EXPECT_TRUE(module.is_loaded());
}

TEST_F(ModuleTest, TestLoadNonExistentMethod) {
  Module module(model_path_);

  const auto error = module.load_method("backward");
  EXPECT_NE(error, Error::Ok);
  EXPECT_FALSE(module.is_method_loaded("backward"));
  EXPECT_TRUE(module.is_loaded());
}

TEST_F(ModuleTest, TestMethodMeta) {
  Module module(model_path_);

  const auto meta = module.method_meta("forward");
  EXPECT_TRUE(meta.ok());
  EXPECT_STREQ(meta->name(), "forward");
  EXPECT_EQ(meta->num_inputs(), 1);
  EXPECT_EQ(*(meta->input_tag(0)), Tag::Tensor);
  EXPECT_EQ(meta->num_outputs(), 1);
  EXPECT_EQ(*(meta->output_tag(0)), Tag::Tensor);

  const auto input_meta = meta->input_tensor_meta(0);
  EXPECT_TRUE(input_meta.ok());
  EXPECT_EQ(input_meta->scalar_type(), ScalarType::Float);
  EXPECT_EQ(input_meta->sizes().size(), 2);
  EXPECT_EQ(input_meta->sizes()[0], 1);
  EXPECT_EQ(input_meta->sizes()[1], 2);

  const auto output_meta = meta->output_tensor_meta(0);
  EXPECT_TRUE(output_meta.ok());
  EXPECT_EQ(output_meta->scalar_type(), ScalarType::Float);
  EXPECT_EQ(output_meta->sizes().size(), 1);
  EXPECT_EQ(output_meta->sizes()[0], 1);
}

TEST_F(ModuleTest, TestNonExistentMethodMeta) {
  Module module("/path/to/nonexistent/file.pte");

  const auto meta = module.method_meta("forward");
  EXPECT_FALSE(meta.ok());
}

TEST_F(ModuleTest, TestExecute) {
  Module module(model_path_);

  std::array<float, 2> input{1, 2};
  std::array<int32_t, 2> sizes{1, 2};
  TensorImpl tensor(
      ScalarType::Float, sizes.size(), sizes.data(), input.data());

  const auto result = module.execute("forward", {EValue(Tensor(&tensor))});
  EXPECT_TRUE(result.ok());
  EXPECT_TRUE(module.is_loaded());
  EXPECT_TRUE(module.is_method_loaded("forward"));

  const auto data = result->at(0).toTensor().const_data_ptr<float>();

  EXPECT_NEAR(data[0], 1.5, 1e-5);
}

TEST_F(ModuleTest, TestExecutePreload) {
  Module module(model_path_);

  const auto error = module.load();
  EXPECT_EQ(error, Error::Ok);

  std::array<float, 2> input{1, 2};
  std::array<int32_t, 2> sizes{1, 2};
  TensorImpl tensor(
      ScalarType::Float, sizes.size(), sizes.data(), input.data());

  const auto result = module.execute("forward", {EValue(Tensor(&tensor))});
  EXPECT_TRUE(result.ok());

  const auto data = result->at(0).toTensor().const_data_ptr<float>();

  EXPECT_NEAR(data[0], 1.5, 1e-5);
}

TEST_F(ModuleTest, TestExecutePreload_method) {
  Module module(model_path_);

  const auto error = module.load_method("forward");
  EXPECT_EQ(error, Error::Ok);

  std::array<float, 2> input{1, 2};
  std::array<int32_t, 2> sizes{1, 2};
  TensorImpl tensor(
      ScalarType::Float, sizes.size(), sizes.data(), input.data());

  const auto result = module.execute("forward", {EValue(Tensor(&tensor))});
  EXPECT_TRUE(result.ok());

  const auto data = result->at(0).toTensor().const_data_ptr<float>();

  EXPECT_NEAR(data[0], 1.5, 1e-5);
}

TEST_F(ModuleTest, TestExecutePreloadProgramAndMethod) {
  Module module(model_path_);

  const auto load_error = module.load();
  EXPECT_EQ(load_error, Error::Ok);

  const auto load_method_error = module.load_method("forward");
  EXPECT_EQ(load_method_error, Error::Ok);

  std::array<float, 2> input{1, 2};
  std::array<int32_t, 2> sizes{1, 2};
  TensorImpl tensor(
      ScalarType::Float, sizes.size(), sizes.data(), input.data());

  const auto result = module.execute("forward", {EValue(Tensor(&tensor))});
  EXPECT_TRUE(result.ok());

  const auto data = result->at(0).toTensor().const_data_ptr<float>();

  EXPECT_NEAR(data[0], 1.5, 1e-5);
}

TEST_F(ModuleTest, TestExecuteOnNonExistent) {
  Module module("/path/to/nonexistent/file.pte");

  const auto result = module.execute("forward");

  EXPECT_FALSE(result.ok());
}

TEST_F(ModuleTest, TestExecuteOnCurrupted) {
  Module module("/dev/null");

  const auto result = module.execute("forward");

  EXPECT_FALSE(result.ok());
}

TEST_F(ModuleTest, TestGet) {
  Module module(model_path_);

  std::array<float, 2> input{1, 2};
  std::array<int32_t, 2> sizes{1, 2};
  TensorImpl tensor(
      ScalarType::Float, sizes.size(), sizes.data(), input.data());

  const auto result = module.get("forward", {EValue(Tensor(&tensor))});

  EXPECT_TRUE(result.ok());
  const auto data = result->toTensor().const_data_ptr<float>();
  EXPECT_NEAR(data[0], 1.5, 1e-5);
}

TEST_F(ModuleTest, TestForward) {
  auto module = std::make_unique<Module>(model_path_);

  std::array<float, 2> input{1, 2};
  std::array<int32_t, 2> sizes{1, 2};
  TensorImpl tensor(
      ScalarType::Float, sizes.size(), sizes.data(), input.data());
  const auto result = module->forward({EValue(Tensor(&tensor))});
  EXPECT_TRUE(result.ok());

  const auto data = result->at(0).toTensor().const_data_ptr<float>();

  EXPECT_NEAR(data[0], 1.5, 1e-5);

  std::array<float, 2> input2{2, 3};
  TensorImpl tensor2(
      ScalarType::Float, sizes.size(), sizes.data(), input2.data());
  const auto result2 = module->forward({EValue(Tensor(&tensor2))});
  EXPECT_TRUE(result2.ok());

  const auto data2 = result->at(0).toTensor().const_data_ptr<float>();

  EXPECT_NEAR(data2[0], 2.5, 1e-5);
}

TEST_F(ModuleTest, TestForwardWithInvalidInputs) {
  Module module(model_path_);

  const auto result = module.forward({EValue()});

  EXPECT_FALSE(result.ok());
}

TEST_F(ModuleTest, TestProgramSharingBetweenModules) {
  Module module1(model_path_);
  EXPECT_FALSE(module1.is_loaded());

  auto load_error = module1.load();
  EXPECT_EQ(load_error, Error::Ok);
  EXPECT_TRUE(module1.is_loaded());

  Module module2(module1.program());
  EXPECT_TRUE(module2.is_loaded());

  auto method_names1 = module1.method_names();
  EXPECT_TRUE(method_names1.ok());

  auto method_names2 = module2.method_names();
  EXPECT_TRUE(method_names2.ok());
  EXPECT_EQ(method_names1.get(), method_names2.get());

  auto load_method_error = module1.load_method("forward");
  EXPECT_EQ(load_method_error, Error::Ok);
  EXPECT_TRUE(module1.is_method_loaded("forward"));
  EXPECT_FALSE(module2.is_method_loaded("forward"));

  auto load_method_error2 = module2.load_method("forward");
  EXPECT_EQ(load_method_error2, Error::Ok);
  EXPECT_TRUE(module2.is_method_loaded("forward"));
}

TEST_F(ModuleTest, TestProgramSharingAndDataLoaderManagement) {
  auto loader = util::FileDataLoader::from(model_path_.c_str());
  EXPECT_TRUE(loader.ok());
  auto data_loader =
      std::make_unique<util::FileDataLoader>(std::move(loader.get()));

  auto module1 = std::make_unique<Module>(std::move(data_loader));

  auto load_error = module1->load();
  EXPECT_EQ(load_error, Error::Ok);
  EXPECT_TRUE(module1->is_loaded());

  std::array<float, 2> input{1, 2};
  std::array<int32_t, 2> sizes{1, 2};
  TensorImpl tensor(
      ScalarType::Float, sizes.size(), sizes.data(), input.data());

  auto result1 = module1->execute("forward", {EValue(Tensor(&tensor))});
  EXPECT_TRUE(result1.ok());

  auto module2 = std::make_unique<Module>(module1->program());

  auto result2 = module2->execute("forward", {EValue(Tensor(&tensor))});
  EXPECT_TRUE(result2.ok());

  module1 = std::make_unique<Module>("/path/to/nonexistent/file.pte");
  EXPECT_FALSE(module1->is_loaded());

  auto result3 = module2->execute("forward", {EValue(Tensor(&tensor))});
  EXPECT_TRUE(result3.ok());
}

TEST_F(ModuleTest, TestProgramPersistenceAndReuseAfterModuleDestruction) {
  std::shared_ptr<Program> shared_program;

  {
    auto loader = util::FileDataLoader::from(model_path_.c_str());
    EXPECT_TRUE(loader.ok());
    auto data_loader =
        std::make_unique<util::FileDataLoader>(std::move(loader.get()));
    auto* data_loader_ptr = data_loader.get();

    Module module(std::move(data_loader));

    auto load_error = module.load();
    EXPECT_EQ(load_error, Error::Ok);
    EXPECT_TRUE(module.is_loaded());

    shared_program = module.program();
    EXPECT_NE(shared_program, nullptr);

    EXPECT_NE(data_loader_ptr, nullptr);
  }

  EXPECT_NE(shared_program, nullptr);

  Module module(shared_program);

  EXPECT_EQ(module.program(), shared_program);

  std::array<float, 2> input{1, 2};
  std::array<int32_t, 2> sizes{1, 2};
  TensorImpl tensor(
      ScalarType::Float, sizes.size(), sizes.data(), input.data());

  auto result = module.execute("forward", {EValue(Tensor(&tensor))});
  EXPECT_TRUE(result.ok());

  auto data = result->at(0).toTensor().const_data_ptr<float>();

  EXPECT_NEAR(data[0], 1.5, 1e-5);
}

TEST_F(ModuleTest, TestConcurrentExecutionWithSharedProgram) {
  std::shared_ptr<Program> program;
  {
    Module module(model_path_);
    EXPECT_FALSE(module.is_loaded());

    auto load_error = module.load();
    EXPECT_EQ(load_error, Error::Ok);
    EXPECT_TRUE(module.is_loaded());

    program = module.program();
  }
  EXPECT_TRUE(program != nullptr);

  auto thread = [](std::shared_ptr<Program> program,
                   const std::array<float, 2>& input) {
    Module module(program);
    std::array<int32_t, 2> sizes{1, 2};
    TensorImpl tensor(
        ScalarType::Float, sizes.size(), sizes.data(), (void*)input.data());

    const auto result = module.forward({EValue(Tensor(&tensor))});
    EXPECT_TRUE(result.ok());

    const auto data = result->at(0).toTensor().const_data_ptr<float>();
    EXPECT_NEAR(data[0], (input[0] + input[1]) / 2.0, 1e-5);
  };

  std::thread t1(thread, program, std::array<float, 2>{1, 2});
  std::thread t2(thread, program, std::array<float, 2>{2, 3});
  std::thread t3(thread, program, std::array<float, 2>{3, 4});
  std::thread t4(thread, program, std::array<float, 2>{4, 5});
  std::thread t5(thread, program, std::array<float, 2>{5, 6});

  t1.join();
  t2.join();
  t3.join();
  t4.join();
  t5.join();
}

} // namespace torch::executor
