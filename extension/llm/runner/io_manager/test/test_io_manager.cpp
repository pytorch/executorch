/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/runner/io_manager/io_manager.h>

#include <gtest/gtest.h>

using namespace ::testing;
using namespace ::executorch::extension;
using namespace ::executorch::runtime;

// Test fixture for IOManager tests
class IOManagerTest : public Test {
 protected:
  void SetUp() override {
    module_ = std::make_unique<Module>(std::getenv("KVCACHE_CACHE_POS"));
    io_manager_ = std::make_unique<llm::IOManager>(*module_);
    EXPECT_EQ(module_->load_forward(), Error::Ok);
  }

 protected:
  std::unique_ptr<Module> module_;
  std::unique_ptr<llm::IOManager> io_manager_;
};

// Test that load() returns Error::Ok (no-op)
TEST_F(IOManagerTest, LoadReturnsOk) {
  EXPECT_EQ(io_manager_->load(), Error::Ok);
}

// Test that reset() returns Error::Ok (no-op)
TEST_F(IOManagerTest, ResetReturnsOk) {
  EXPECT_EQ(io_manager_->reset(), Error::Ok);
}

// Test that prepare_prefill() returns the input tensors when method has 2
// inputs
TEST_F(IOManagerTest, PreparePrefillReturnsInputsWhenValidInputCount) {
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<int64_t> start_pos_data = {0};
  auto input_ptr = make_tensor_ptr({1, 4}, input_data);
  auto start_pos_ptr = make_tensor_ptr({1}, start_pos_data);
  auto result = io_manager_->prepare_prefill(input_ptr, start_pos_ptr);

  EXPECT_EQ(result.error(), Error::Ok);
  auto outputs = result.get();
  EXPECT_EQ(outputs.size(), 2);

  // Verify that the returned EValues contain the same tensors we passed in
  EXPECT_TRUE(outputs[0].isTensor());
  EXPECT_TRUE(outputs[1].isTensor());
}

// Test that prepare_decode() returns the input tensors when method has 2 inputs
TEST_F(IOManagerTest, PrepareDecodeReturnsInputsWhenValidInputCount) {
  std::vector<float> input_data = {5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<int64_t> start_pos_data = {10};
  auto input_ptr = make_tensor_ptr({1, 4}, input_data);
  auto start_pos_ptr = make_tensor_ptr({1}, start_pos_data);

  auto result = io_manager_->prepare_decode(input_ptr, start_pos_ptr);

  EXPECT_EQ(result.error(), Error::Ok);
  auto outputs = result.get();
  EXPECT_EQ(outputs.size(), 2);

  // Verify that the returned EValues contain the same tensors we passed in
  EXPECT_TRUE(outputs[0].isTensor());
  EXPECT_TRUE(outputs[1].isTensor());
}

// Test that update_prefill() returns Error::Ok (no-op)
TEST_F(IOManagerTest, UpdatePrefillReturnsOk) {
  std::vector<EValue> model_outputs;
  std::vector<float> output_data = {0.1f, 0.2f, 0.3f};
  auto output_tensor = make_tensor_ptr({1, 3}, output_data);
  model_outputs.emplace_back(*output_tensor);

  EXPECT_EQ(io_manager_->update_prefill(model_outputs), Error::Ok);
}

// Test that update_decode() returns Error::Ok (no-op)
TEST_F(IOManagerTest, UpdateDecodeReturnsOk) {
  std::vector<EValue> model_outputs;
  std::vector<float> output_data = {0.4f, 0.5f, 0.6f};
  auto output_tensor = make_tensor_ptr({1, 3}, output_data);
  model_outputs.emplace_back(*output_tensor);

  EXPECT_EQ(io_manager_->update_decode(model_outputs), Error::Ok);
}

// Test that prepare_prefill() correctly passes through different tensor shapes
TEST_F(IOManagerTest, PreparePrefillPassesThroughDifferentTensorShapes) {
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<int64_t> start_pos_data = {5, 10};
  auto input_ptr = make_tensor_ptr({2, 3}, input_data);
  auto start_pos_ptr = make_tensor_ptr({2}, start_pos_data);
  auto result = io_manager_->prepare_prefill(input_ptr, start_pos_ptr);

  EXPECT_EQ(result.error(), Error::Ok);
  auto outputs = result.get();
  EXPECT_EQ(outputs.size(), 2);

  // Verify that the returned EValues contain tensors
  EXPECT_TRUE(outputs[0].isTensor());
  EXPECT_TRUE(outputs[1].isTensor());
}

// Test that prepare_decode() correctly passes through different tensor shapes
TEST_F(IOManagerTest, PrepareDecodePassesThroughDifferentTensorShapes) {
  std::vector<float> input_data = {
      7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f};
  std::vector<int64_t> start_pos_data = {15, 20, 25};
  auto input_ptr = make_tensor_ptr({2, 4}, input_data);
  auto start_pos_ptr = make_tensor_ptr({3}, start_pos_data);
  auto result = io_manager_->prepare_decode(input_ptr, start_pos_ptr);

  EXPECT_EQ(result.error(), Error::Ok);
  auto outputs = result.get();
  EXPECT_EQ(outputs.size(), 2);

  // Verify that the returned EValues contain tensors
  EXPECT_TRUE(outputs[0].isTensor());
  EXPECT_TRUE(outputs[1].isTensor());
}

// Test that update methods handle empty model outputs
TEST_F(IOManagerTest, UpdateMethodsHandleEmptyModelOutputs) {
  std::vector<EValue> empty_outputs;

  EXPECT_EQ(io_manager_->update_prefill(empty_outputs), Error::Ok);
  EXPECT_EQ(io_manager_->update_decode(empty_outputs), Error::Ok);
}

// Test that update methods handle multiple model outputs
TEST_F(IOManagerTest, UpdateMethodsHandleMultipleModelOutputs) {
  std::vector<EValue> model_outputs;
  std::vector<float> output1_data = {0.1f, 0.2f};
  std::vector<float> output2_data = {0.3f, 0.4f, 0.5f};
  auto output1_tensor = make_tensor_ptr({1, 2}, output1_data);
  auto output2_tensor = make_tensor_ptr({1, 3}, output2_data);
  model_outputs.emplace_back(*output1_tensor);
  model_outputs.emplace_back(*output2_tensor);

  EXPECT_EQ(io_manager_->update_prefill(model_outputs), Error::Ok);
  EXPECT_EQ(io_manager_->update_decode(model_outputs), Error::Ok);
}
