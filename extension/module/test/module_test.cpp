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
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

using namespace ::executorch::extension;
using namespace ::executorch::runtime;

class ModuleTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    model_path_ = std::getenv("ET_MODULE_ADD_PATH");
    add_mul_path_ = std::getenv("ET_MODULE_ADD_MUL_PROGRAM_PATH");
    add_mul_data_path_ = std::getenv("ET_MODULE_ADD_MUL_DATA_PATH");
    linear_path_ = std::getenv("ET_MODULE_LINEAR_PROGRAM_PATH");
    linear_data_path_ = std::getenv("ET_MODULE_LINEAR_DATA_PATH");
  }

  static inline std::string model_path_;
  static inline std::string add_mul_path_;
  static inline std::string add_mul_data_path_;
  static inline std::string linear_path_;
  static inline std::string linear_data_path_;
};

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

TEST_F(ModuleTest, TestNumMethods) {
  Module module(model_path_);

  const auto num_methods = module.num_methods();
  EXPECT_EQ(num_methods.error(), Error::Ok);
  EXPECT_EQ(num_methods.get(), 1);
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

TEST_F(ModuleTest, TestUnloadMethod) {
  Module module(model_path_);

  EXPECT_FALSE(module.is_method_loaded("forward"));
  const auto errorLoad = module.load_method("forward");
  EXPECT_EQ(errorLoad, Error::Ok);
  EXPECT_TRUE(module.is_method_loaded("forward"));
  // Unload method
  EXPECT_TRUE(module.unload_method("forward"));
  EXPECT_FALSE(module.is_method_loaded("forward"));
  // Try unload method again
  EXPECT_FALSE(module.unload_method("forward"));
  // Load method again
  const auto errorReload = module.load_method("forward");
  EXPECT_EQ(errorReload, Error::Ok);
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
  // tensor, tensor, alpha
  EXPECT_EQ(meta->num_inputs(), 3);
  EXPECT_EQ(*(meta->input_tag(0)), Tag::Tensor);
  EXPECT_EQ(meta->num_outputs(), 1);
  EXPECT_EQ(*(meta->output_tag(0)), Tag::Tensor);

  const auto input_meta = meta->input_tensor_meta(0);
  EXPECT_EQ(input_meta.error(), Error::Ok);
  EXPECT_EQ(input_meta->scalar_type(), executorch::aten::ScalarType::Float);
  EXPECT_EQ(input_meta->sizes().size(), 2);
  EXPECT_EQ(input_meta->sizes()[0], 2);

  const auto input_meta1 = meta->input_tensor_meta(1);
  EXPECT_EQ(input_meta1.error(), Error::Ok);
  EXPECT_EQ(input_meta1->scalar_type(), executorch::aten::ScalarType::Float);
  EXPECT_EQ(input_meta1->sizes().size(), 2);
  EXPECT_EQ(input_meta1->sizes()[0], 2);

  const auto output_meta = meta->output_tensor_meta(0);
  EXPECT_EQ(output_meta.error(), Error::Ok);
  EXPECT_EQ(output_meta->scalar_type(), executorch::aten::ScalarType::Float);
  EXPECT_EQ(output_meta->sizes().size(), 2);
  EXPECT_EQ(output_meta->sizes()[0], 2);
}

TEST_F(ModuleTest, TestNonExistentMethodMeta) {
  Module module("/path/to/nonexistent/file.pte");

  const auto meta = module.method_meta("forward");
  EXPECT_NE(meta.error(), Error::Ok);
}

TEST_F(ModuleTest, TestExecute) {
  Module module(model_path_);
  auto tensor = make_tensor_ptr({2, 2}, {1.f, 2.f, 3.f, 4.f});

  const auto result = module.execute("forward", {tensor, tensor, 1.0});
  EXPECT_EQ(result.error(), Error::Ok);

  EXPECT_TRUE(module.is_loaded());
  EXPECT_TRUE(module.is_method_loaded("forward"));

  const auto expected = make_tensor_ptr({2, 2}, {2.f, 4.f, 6.f, 8.f});
  EXPECT_TENSOR_CLOSE(result->at(0).toTensor(), *expected.get());
}

TEST_F(ModuleTest, TestExecutePreload) {
  Module module(model_path_);

  const auto error = module.load();
  EXPECT_EQ(error, Error::Ok);

  auto tensor = make_tensor_ptr({2, 2}, {1.f, 2.f, 3.f, 4.f});

  const auto result = module.execute("forward", {tensor, tensor, 1.0});
  EXPECT_EQ(result.error(), Error::Ok);

  const auto expected = make_tensor_ptr({2, 2}, {2.f, 4.f, 6.f, 8.f});
  EXPECT_TENSOR_CLOSE(result->at(0).toTensor(), *expected.get());
}

TEST_F(ModuleTest, TestExecutePreload_method) {
  Module module(model_path_);

  const auto error = module.load_method("forward");
  EXPECT_EQ(error, Error::Ok);

  auto tensor = make_tensor_ptr({2, 2}, {1.f, 2.f, 3.f, 4.f});

  const auto result = module.execute("forward", {tensor, tensor, 1.0});
  EXPECT_EQ(result.error(), Error::Ok);

  const auto expected = make_tensor_ptr({2, 2}, {2.f, 4.f, 6.f, 8.f});
  EXPECT_TENSOR_CLOSE(result->at(0).toTensor(), *expected.get());
}

TEST_F(ModuleTest, TestExecutePreloadProgramAndMethod) {
  Module module(model_path_);

  const auto load_error = module.load();
  EXPECT_EQ(load_error, Error::Ok);

  const auto load_method_error = module.load_method("forward");
  EXPECT_EQ(load_method_error, Error::Ok);

  auto tensor = make_tensor_ptr({2, 2}, {1.f, 2.f, 3.f, 4.f});

  const auto result = module.execute("forward", {tensor, tensor, 1.0});
  EXPECT_EQ(result.error(), Error::Ok);

  const auto expected = make_tensor_ptr({2, 2}, {2.f, 4.f, 6.f, 8.f});
  EXPECT_TENSOR_CLOSE(result->at(0).toTensor(), *expected.get());
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

TEST_F(ModuleTest, TestExecuteWithTooManyInputs) {
  Module module(model_path_);

  auto tensor = make_tensor_ptr({2, 2}, {1.f, 2.f, 3.f, 4.f});

  const auto result = module.execute("forward", {tensor, tensor, 1.0, 1.0});

  EXPECT_NE(result.error(), Error::Ok);
}

TEST_F(ModuleTest, TestGet) {
  Module module(model_path_);

  auto tensor = make_tensor_ptr({2, 2}, {1.f, 2.f, 3.f, 4.f});

  const auto result = module.get("forward", {tensor, tensor, 1.0});
  EXPECT_EQ(result.error(), Error::Ok);
  const auto expected = make_tensor_ptr({2, 2}, {2.f, 4.f, 6.f, 8.f});
  EXPECT_TENSOR_CLOSE(result->toTensor(), *expected.get());
}

TEST_F(ModuleTest, TestForward) {
  auto module = std::make_unique<Module>(model_path_);
  auto tensor = make_tensor_ptr({2, 2}, {21.f, 22.f, 23.f, 24.f});

  const auto result = module->forward({tensor, tensor, 1.0});
  EXPECT_EQ(result.error(), Error::Ok);

  const auto expected = make_tensor_ptr({2, 2}, {42.f, 44.f, 46.f, 48.f});
  EXPECT_TENSOR_CLOSE(result->at(0).toTensor(), *expected.get());

  auto tensor2 = make_tensor_ptr({2, 2}, {2.f, 3.f, 4.f, 5.f});
  const auto result2 = module->forward({tensor2, tensor2, 1.0});
  EXPECT_EQ(result2.error(), Error::Ok);

  const auto expected2 = make_tensor_ptr({2, 2}, {4.f, 6.f, 8.f, 10.f});
  EXPECT_TENSOR_CLOSE(result2->at(0).toTensor(), *expected2.get());
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

  auto tensor = make_tensor_ptr({2, 2}, {1.f, 2.f, 3.f, 4.f});

  const auto result1 = module1->execute("forward", {tensor, tensor, 1.0});
  EXPECT_EQ(result1.error(), Error::Ok);

  auto module2 = std::make_unique<Module>(module1->program());

  const auto result2 = module2->execute("forward", {tensor, tensor, 1.0});
  EXPECT_EQ(result2.error(), Error::Ok);

  module1 = std::make_unique<Module>("/path/to/nonexistent/file.pte");
  EXPECT_FALSE(module1->is_loaded());

  const auto result3 = module2->execute("forward", {tensor, tensor, 1.0});
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

  auto tensor = make_tensor_ptr({2, 2}, {1.f, 2.f, 3.f, 4.f});

  const auto result = module.execute("forward", {tensor, tensor, 1.0});
  EXPECT_EQ(result.error(), Error::Ok);

  const auto expected = make_tensor_ptr({2, 2}, {2.f, 4.f, 6.f, 8.f});
  EXPECT_TENSOR_CLOSE(result->at(0).toTensor(), *expected.get());
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
                   const std::array<float, 4>& input) {
    Module module(program);
    auto tensor = from_blob((void*)input.data(), {2, 2});

    const auto result = module.forward({tensor, tensor, 1.0});
    EXPECT_EQ(result.error(), Error::Ok);

    const auto data = result->at(0).toTensor().const_data_ptr<float>();
    EXPECT_NEAR(data[0], (input[0] * 2), 1e-5);
  };

  std::thread t1(thread, program, std::array<float, 4>{1, 2, 3, 4});
  std::thread t2(thread, program, std::array<float, 4>{2, 3, 4, 5});
  std::thread t3(thread, program, std::array<float, 4>{3, 4, 5, 6});
  std::thread t4(thread, program, std::array<float, 4>{4, 5, 6, 7});
  std::thread t5(thread, program, std::array<float, 4>{5, 6, 7, 8});

  t1.join();
  t2.join();
  t3.join();
  t4.join();
  t5.join();
}

TEST_F(ModuleTest, TestSetInputsBeforeExecute) {
  Module module(model_path_);

  auto tensor1 = make_tensor_ptr({2, 2}, {1.f, 2.f, 3.f, 4.f});
  auto tensor2 = make_tensor_ptr({2, 2}, {2.f, 3.f, 4.f, 5.f});

  EXPECT_EQ(module.set_inputs({tensor1, tensor2, 1.0}), Error::Ok);

  const auto result = module.forward();
  EXPECT_EQ(result.error(), Error::Ok);

  const auto expected = make_tensor_ptr({2, 2}, {3.f, 5.f, 7.f, 9.f});
  EXPECT_TENSOR_CLOSE(result->at(0).toTensor(), *expected.get());
}

TEST_F(ModuleTest, TestSetInputCombinedWithExecute) {
  Module module(model_path_);

  auto tensor1 = make_tensor_ptr({2, 2}, {1.f, 2.f, 3.f, 4.f});
  auto tensor2 = make_tensor_ptr({2, 2}, {2.f, 3.f, 4.f, 5.f});

  EXPECT_EQ(module.set_input(tensor2, 1), Error::Ok);
  EXPECT_EQ(module.set_input(1.0, 2), Error::Ok); // alpha

  const auto result = module.forward(tensor1);
  EXPECT_EQ(result.error(), Error::Ok);

  const auto expected = make_tensor_ptr({2, 2}, {3.f, 5.f, 7.f, 9.f});
  EXPECT_TENSOR_CLOSE(result->at(0).toTensor(), *expected.get());
}

TEST_F(ModuleTest, TestPartiallySetInputs) {
  Module module(model_path_);

  auto tensor = make_tensor_ptr({2, 2}, {1.f, 2.f, 3.f, 4.f});

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

TEST_F(ModuleTest, TestSetOutputsCountMismatch) {
  Module module(model_path_);

  EXPECT_NE(module.set_outputs(std::vector<EValue>{}), Error::Ok);
}

TEST_F(ModuleTest, TestSetOutputsInvalidType) {
  Module module(model_path_);

  EXPECT_NE(module.set_outputs({EValue()}), Error::Ok);
}

TEST_F(ModuleTest, TestSetOutputsMemoryPlanned) {
  Module module(model_path_);

  EXPECT_NE(module.set_outputs({empty({1})}), Error::Ok);
}

TEST_F(ModuleTest, TestGetOutputAndGetOutputs) {
  Module module(model_path_);

  auto tensor = make_tensor_ptr({2, 2}, {1.f, 2.f, 3.f, 4.f});

  ASSERT_EQ(module.forward({tensor, tensor, 1.0}).error(), Error::Ok);

  const auto single = module.get_output();
  EXPECT_EQ(single.error(), Error::Ok);
  const auto expected = make_tensor_ptr({2, 2}, {2.f, 4.f, 6.f, 8.f});
  EXPECT_TENSOR_CLOSE(single->toTensor(), *expected.get());

  const auto all = module.get_outputs();
  EXPECT_EQ(all.error(), Error::Ok);
  ASSERT_EQ(all->size(), 1);
  EXPECT_TENSOR_CLOSE(all->at(0).toTensor(), *expected.get());
}

TEST_F(ModuleTest, TestGetOutputInvalidIndex) {
  Module module(model_path_);

  ASSERT_EQ(module.load_method("forward"), Error::Ok);

  const auto bad = module.get_output("forward", 99);
  EXPECT_NE(bad.error(), Error::Ok);
}

TEST_F(ModuleTest, TestPTD) {
  Module module(add_mul_path_, add_mul_data_path_);

  ASSERT_EQ(module.load_method("forward"), Error::Ok);

  auto tensor = make_tensor_ptr({2, 2}, {2.f, 3.f, 4.f, 2.f});
  ASSERT_EQ(module.forward(tensor).error(), Error::Ok);
}

TEST_F(ModuleTest, TestPTD_Multiple) {
  std::vector<std::string> data_files = {add_mul_data_path_, linear_data_path_};
  
  // Create module with add mul.
  Module module_add_mul(add_mul_path_, data_files);
  ASSERT_EQ(module_add_mul.load_method("forward"), Error::Ok);
  auto tensor = make_tensor_ptr({2, 2}, {2.f, 3.f, 4.f, 2.f});
  ASSERT_EQ(module_add_mul.forward(tensor).error(), Error::Ok);

  // Confirm that the data_file is not std::move'd away.
  ASSERT_EQ(std::strcmp(data_files[0].c_str(), add_mul_data_path_.c_str()), 0);
  ASSERT_EQ(std::strcmp(data_files[1].c_str(), linear_data_path_.c_str()), 0);

  // Create module with linear.
  Module module_linear(linear_path_, data_files);
  ASSERT_EQ(module_linear.load_method("forward"), Error::Ok);
  auto tensor2 = make_tensor_ptr({3}, {2.f, 3.f, 4.f});
  ASSERT_EQ(module_linear.forward(tensor2).error(), Error::Ok);
}
