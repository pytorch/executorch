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
#include <executorch/extension/tensor/tensor.h>

using namespace ::executorch::extension;
using namespace ::executorch::runtime;

class ModuleTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    model_path_ = std::getenv("RESOURCES_PATH") + std::string("/add.pte");
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
  EXPECT_EQ(method_names.error(), Error::Ok);
  EXPECT_EQ(method_names.get(), std::unordered_set<std::string>{"forward"});
}

TEST_F(ModuleTest, TestNonExistentMethodNames) {
  Module module("/path/to/nonexistent/file.pte");

  const auto method_names = module.method_names();
  EXPECT_NE(method_names.error(), Error::Ok);
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
  EXPECT_EQ(meta.error(), Error::Ok);
  EXPECT_STREQ(meta->name(), "forward");
  EXPECT_EQ(meta->num_inputs(), 2);
  EXPECT_EQ(*(meta->input_tag(0)), Tag::Tensor);
  EXPECT_EQ(meta->num_outputs(), 1);
  EXPECT_EQ(*(meta->output_tag(0)), Tag::Tensor);

  const auto input_meta = meta->input_tensor_meta(0);
  EXPECT_EQ(input_meta.error(), Error::Ok);
  EXPECT_EQ(input_meta->scalar_type(), exec_aten::ScalarType::Float);
  EXPECT_EQ(input_meta->sizes().size(), 1);
  EXPECT_EQ(input_meta->sizes()[0], 1);

  const auto output_meta = meta->output_tensor_meta(0);
  EXPECT_EQ(output_meta.error(), Error::Ok);
  EXPECT_EQ(output_meta->scalar_type(), exec_aten::ScalarType::Float);
  EXPECT_EQ(output_meta->sizes().size(), 1);
  EXPECT_EQ(output_meta->sizes()[0], 1);
}

TEST_F(ModuleTest, TestNonExistentMethodMeta) {
  Module module("/path/to/nonexistent/file.pte");

  const auto meta = module.method_meta("forward");
  EXPECT_NE(meta.error(), Error::Ok);
}

TEST_F(ModuleTest, TestExecute) {
  Module module(model_path_);
  auto tensor = make_tensor_ptr({1.f});

  const auto result = module.execute("forward", {tensor, tensor});
  EXPECT_EQ(result.error(), Error::Ok);

  EXPECT_TRUE(module.is_loaded());
  EXPECT_TRUE(module.is_method_loaded("forward"));

  const auto data = result->at(0).toTensor().const_data_ptr<float>();

  EXPECT_NEAR(data[0], 2, 1e-5);
}

TEST_F(ModuleTest, TestExecutePreload) {
  Module module(model_path_);

  const auto error = module.load();
  EXPECT_EQ(error, Error::Ok);

  auto tensor = make_tensor_ptr({1.f});

  const auto result = module.execute("forward", {tensor, tensor});
  EXPECT_EQ(result.error(), Error::Ok);

  const auto data = result->at(0).toTensor().const_data_ptr<float>();

  EXPECT_NEAR(data[0], 2, 1e-5);
}

TEST_F(ModuleTest, TestExecutePreload_method) {
  Module module(model_path_);

  const auto error = module.load_method("forward");
  EXPECT_EQ(error, Error::Ok);

  auto tensor = make_tensor_ptr({1.f});

  const auto result = module.execute("forward", {tensor, tensor});
  EXPECT_EQ(result.error(), Error::Ok);

  const auto data = result->at(0).toTensor().const_data_ptr<float>();

  EXPECT_NEAR(data[0], 2, 1e-5);
}

TEST_F(ModuleTest, TestExecutePreloadProgramAndMethod) {
  Module module(model_path_);

  const auto load_error = module.load();
  EXPECT_EQ(load_error, Error::Ok);

  const auto load_method_error = module.load_method("forward");
  EXPECT_EQ(load_method_error, Error::Ok);

  auto tensor = make_tensor_ptr({1.f});

  const auto result = module.execute("forward", {tensor, tensor});
  EXPECT_EQ(result.error(), Error::Ok);

  const auto data = result->at(0).toTensor().const_data_ptr<float>();

  EXPECT_NEAR(data[0], 2, 1e-5);
}

TEST_F(ModuleTest, TestExecuteOnNonExistent) {
  Module module("/path/to/nonexistent/file.pte");

  const auto result = module.execute("forward");

  EXPECT_NE(result.error(), Error::Ok);
}

TEST_F(ModuleTest, TestExecuteOnCurrupted) {
  Module module("/dev/null");

  const auto result = module.execute("forward");

  EXPECT_NE(result.error(), Error::Ok);
}

TEST_F(ModuleTest, TestGet) {
  Module module(model_path_);

  auto tensor = make_tensor_ptr({1.f});

  const auto result = module.get("forward", {tensor, tensor});
  EXPECT_EQ(result.error(), Error::Ok);
  const auto data = result->toTensor().const_data_ptr<float>();
  EXPECT_NEAR(data[0], 2, 1e-5);
}

TEST_F(ModuleTest, TestForward) {
  auto module = std::make_unique<Module>(model_path_);
  auto tensor = make_tensor_ptr({21.f});

  const auto result = module->forward({tensor, tensor});
  EXPECT_EQ(result.error(), Error::Ok);

  const auto data = result->at(0).toTensor().const_data_ptr<float>();

  EXPECT_NEAR(data[0], 42, 1e-5);

  auto tensor2 = make_tensor_ptr({2.f});
  const auto result2 = module->forward({tensor2, tensor2});
  EXPECT_EQ(result2.error(), Error::Ok);

  const auto data2 = result->at(0).toTensor().const_data_ptr<float>();

  EXPECT_NEAR(data2[0], 4, 1e-5);
}

TEST_F(ModuleTest, TestForwardWithInvalidInputs) {
  Module module(model_path_);

  const auto result = module.forward(EValue());

  EXPECT_NE(result.error(), Error::Ok);
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
  EXPECT_EQ(method_names1.error(), Error::Ok);

  auto method_names2 = module2.method_names();
  EXPECT_EQ(method_names2.error(), Error::Ok);
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
  auto loader = FileDataLoader::from(model_path_.c_str());
  EXPECT_EQ(loader.error(), Error::Ok);
  auto data_loader = std::make_unique<FileDataLoader>(std::move(loader.get()));

  auto module1 = std::make_unique<Module>(std::move(data_loader));

  auto load_error = module1->load();
  EXPECT_EQ(load_error, Error::Ok);
  EXPECT_TRUE(module1->is_loaded());

  auto tensor = make_tensor_ptr({1.f});

  const auto result1 = module1->execute("forward", {tensor, tensor});
  EXPECT_EQ(result1.error(), Error::Ok);

  auto module2 = std::make_unique<Module>(module1->program());

  const auto result2 = module2->execute("forward", {tensor, tensor});
  EXPECT_EQ(result2.error(), Error::Ok);

  module1 = std::make_unique<Module>("/path/to/nonexistent/file.pte");
  EXPECT_FALSE(module1->is_loaded());

  const auto result3 = module2->execute("forward", {tensor, tensor});
  EXPECT_EQ(result3.error(), Error::Ok);
}

TEST_F(ModuleTest, TestProgramPersistenceAndReuseAfterModuleDestruction) {
  std::shared_ptr<Program> shared_program;

  {
    auto loader = FileDataLoader::from(model_path_.c_str());
    EXPECT_EQ(loader.error(), Error::Ok);
    auto data_loader =
        std::make_unique<FileDataLoader>(std::move(loader.get()));
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

  auto tensor = make_tensor_ptr({1.f});

  const auto result = module.execute("forward", {tensor, tensor});
  EXPECT_EQ(result.error(), Error::Ok);

  auto data = result->at(0).toTensor().const_data_ptr<float>();

  EXPECT_NEAR(data[0], 2, 1e-5);
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
                   const std::array<float, 1>& input) {
    Module module(program);
    auto tensor = from_blob((void*)input.data(), {1});

    const auto result = module.forward({tensor, tensor});
    EXPECT_EQ(result.error(), Error::Ok);

    const auto data = result->at(0).toTensor().const_data_ptr<float>();
    EXPECT_NEAR(data[0], (input[0] * 2), 1e-5);
  };

  std::thread t1(thread, program, std::array<float, 1>{1});
  std::thread t2(thread, program, std::array<float, 1>{2});
  std::thread t3(thread, program, std::array<float, 1>{3});
  std::thread t4(thread, program, std::array<float, 1>{4});
  std::thread t5(thread, program, std::array<float, 1>{5});

  t1.join();
  t2.join();
  t3.join();
  t4.join();
  t5.join();
}

TEST_F(ModuleTest, TestSetInputsBeforeExecute) {
  Module module(model_path_);

  auto tensor1 = make_tensor_ptr({4.f});
  auto tensor2 = make_tensor_ptr({5.f});

  EXPECT_EQ(module.set_inputs({tensor1, tensor2}), Error::Ok);

  const auto result = module.forward();
  EXPECT_EQ(result.error(), Error::Ok);

  const auto data = result->at(0).toTensor().const_data_ptr<float>();
  EXPECT_NEAR(data[0], 9, 1e-5);
}

TEST_F(ModuleTest, TestSetInputCombinedWithExecute) {
  Module module(model_path_);

  auto tensor1 = make_tensor_ptr({2.f});
  auto tensor2 = make_tensor_ptr({3.f});

  EXPECT_EQ(module.set_input(tensor2, 1), Error::Ok);

  const auto result = module.forward(tensor1);
  EXPECT_EQ(result.error(), Error::Ok);

  const auto data = result->at(0).toTensor().const_data_ptr<float>();
  EXPECT_NEAR(data[0], 5, 1e-5);
}

TEST_F(ModuleTest, TestPartiallySetInputs) {
  Module module(model_path_);

  auto tensor = make_tensor_ptr({1.f});

  EXPECT_EQ(module.set_input(tensor, 0), Error::Ok);

  const auto result = module.forward();
  EXPECT_NE(result.error(), Error::Ok);
}

TEST_F(ModuleTest, TestUnsetInputs) {
  Module module(model_path_);

  const auto result = module.forward();
  EXPECT_NE(result.error(), Error::Ok);
}

TEST_F(ModuleTest, TestSetOutputInvalidIndex) {
  Module module(model_path_);

  auto output_tensor = empty({1});

  EXPECT_NE(module.set_output(output_tensor, 1), Error::Ok);
}

TEST_F(ModuleTest, TestSetOutputInvalidType) {
  Module module(model_path_);

  EXPECT_NE(module.set_output(EValue()), Error::Ok);
}
