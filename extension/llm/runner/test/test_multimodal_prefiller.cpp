/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 * @lint-ignore-every CLANGTIDY facebook-hte-Deprecated
 */

#include <executorch/extension/llm/runner/audio.h>
#include <executorch/extension/llm/runner/constants.h>
#include <executorch/extension/llm/runner/image.h>
#include <executorch/extension/llm/runner/io_manager/io_manager.h>
#include <executorch/extension/llm/runner/multimodal_decoder_runner.h>
#include <executorch/extension/llm/runner/multimodal_input.h>
#include <executorch/extension/llm/runner/multimodal_prefiller.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/platform/runtime.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <pytorch/tokenizers/tokenizer.h>

using namespace ::testing;
using executorch::extension::Module;
using executorch::extension::llm::Audio;
using executorch::extension::llm::Image;
using executorch::extension::llm::IOManager;
using executorch::extension::llm::MultimodalDecoderRunner;
using executorch::extension::llm::MultimodalInput;
using executorch::extension::llm::MultimodalPrefiller;
using executorch::extension::llm::kAudioEncoderMethod;
using executorch::extension::llm::kImageEncoderMethod;
using executorch::extension::llm::kTextModelMethod;
using executorch::extension::llm::kTokenEmbeddingMethod;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::Result;
using executorch::runtime::testing::TensorFactory;

namespace {

// Mock classes for dependencies
class MockModule : public Module {
 public:
  MockModule() : Module("") {}
  
  // Only mock the methods actually used by MultimodalPrefiller
  MOCK_METHOD(
      Result<std::vector<EValue>>,
      execute,
      (const std::string&, const std::vector<EValue>&),
      (override));
  MOCK_METHOD(
      Error,
      load_method,
      (const std::string&),
      (override));
  MOCK_METHOD(bool, is_method_loaded, (const std::string&), (const, override));
  MOCK_METHOD(
      Result<std::unordered_set<std::string>>,
      method_names,
      (),
      (override));
};

class MockTokenizer : public ::tokenizers::Tokenizer {
 public:
  // Only mock the encode method which is used by MultimodalPrefiller for text input
  MOCK_METHOD(
      ::tokenizers::Result<std::vector<uint64_t>>,
      encode,
      (const std::string&, int8_t, int8_t),
      (const));
};

class MockMultimodalDecoderRunner : public MultimodalDecoderRunner {
 public:
  MockMultimodalDecoderRunner() : MultimodalDecoderRunner(nullptr, nullptr) {}
  
  // Only mock the logits_to_token method which is used by MultimodalPrefiller
  MOCK_METHOD(int32_t, logits_to_token, (executorch::aten::Tensor&, float), ());
};

// Test fixture
class MultimodalPrefillerTest : public Test {
 protected:
  void SetUp() override {
    executorch::runtime::runtime_init();
    
    mock_module_ = std::make_unique<MockModule>();
    mock_tokenizer_ = std::make_unique<MockTokenizer>();
    mock_decoder_runner_ = std::make_unique<MockMultimodalDecoderRunner>();
    io_manager_ = std::make_unique<IOManager>();
    
    prefiller_ = std::make_unique<MultimodalPrefiller>(
        mock_module_.get(),
        mock_decoder_runner_.get(),
        mock_tokenizer_.get(),
        io_manager_.get());
        
    // Set up tensor factory for creating test tensors
    tf_float_ = std::make_unique<TensorFactory<executorch::aten::ScalarType::Float>>();
  }

  std::unique_ptr<MockModule> mock_module_;
  std::unique_ptr<MockTokenizer> mock_tokenizer_;
  std::unique_ptr<MockMultimodalDecoderRunner> mock_decoder_runner_;
  std::unique_ptr<IOManager> io_manager_;
  std::unique_ptr<MultimodalPrefiller> prefiller_;
  
  std::unique_ptr<TensorFactory<executorch::aten::ScalarType::Float>> tf_float_;
};

// Load Tests
TEST_F(MultimodalPrefillerTest, LoadAllRequiredMethodsExist) {
  // Set up method names to include all required methods
  std::unordered_set<std::string> method_names = {
      kTokenEmbeddingMethod, kTextModelMethod, kImageEncoderMethod, kAudioEncoderMethod
  };
  
  EXPECT_CALL(*mock_module_, method_names())
      .WillOnce(Return(Result<std::unordered_set<std::string>>(method_names)));
  
  EXPECT_CALL(*mock_module_, load_method(kTokenEmbeddingMethod))
      .WillOnce(Return(Error::Ok));
  EXPECT_CALL(*mock_module_, load_method(kTextModelMethod))
      .WillOnce(Return(Error::Ok));
  EXPECT_CALL(*mock_module_, load_method(kImageEncoderMethod))
      .WillOnce(Return(Error::Ok));
  EXPECT_CALL(*mock_module_, load_method(kAudioEncoderMethod))
      .WillOnce(Return(Error::Ok));
  
  EXPECT_CALL(*mock_module_, is_method_loaded(kTokenEmbeddingMethod))
      .WillOnce(Return(false));
  
  Error result = prefiller_->load();
  EXPECT_EQ(result, Error::Ok);
}

TEST_F(MultimodalPrefillerTest, LoadTokenEmbeddingMethodDoesntExist) {
  EXPECT_CALL(*mock_module_, load_method(kTokenEmbeddingMethod))
      .WillOnce(Return(Error::InvalidProgram));
  
  EXPECT_CALL(*mock_module_, is_method_loaded(kTokenEmbeddingMethod))
      .WillOnce(Return(false));
  
  Error result = prefiller_->load();
  EXPECT_EQ(result, Error::InvalidProgram);
}

TEST_F(MultimodalPrefillerTest, LoadTextModelMethodDoesntExist) {
  EXPECT_CALL(*mock_module_, load_method(kTokenEmbeddingMethod))
      .WillOnce(Return(Error::Ok));
  EXPECT_CALL(*mock_module_, load_method(kTextModelMethod))
      .WillOnce(Return(Error::InvalidProgram));
  
  EXPECT_CALL(*mock_module_, is_method_loaded(kTokenEmbeddingMethod))
      .WillOnce(Return(false));
  
  Error result = prefiller_->load();
  EXPECT_EQ(result, Error::InvalidProgram);
}

// Prefill Tests
TEST_F(MultimodalPrefillerTest, PrefillImageInput) {
  // Create test image data
  std::vector<uint8_t> image_data(3 * 224 * 224, 128); // 224x224 RGB image
  Image test_image{std::move(image_data), 224, 224};
  MultimodalInput input(std::move(test_image));
  
  // Create mock encoder output tensor
  auto encoder_output_tensor = tf_float_->make({1, 5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  std::vector<EValue> encoder_outputs = {EValue(encoder_output_tensor)};
  
  // Create mock prefill output tensor
  auto prefill_output_tensor = tf_float_->make({1, 1, 4096}, std::vector<float>(4096, 0.5f));
  std::vector<EValue> prefill_outputs = {EValue(prefill_output_tensor)};
  
  EXPECT_CALL(*mock_module_, execute(kImageEncoderMethod, _))
      .WillOnce(Return(Result<std::vector<EValue>>(encoder_outputs)));
  
  EXPECT_CALL(*mock_module_, execute(kTextModelMethod, _))
      .WillOnce(Return(Result<std::vector<EValue>>(prefill_outputs)));
  
  EXPECT_CALL(*mock_decoder_runner_, logits_to_token(_, _))
      .WillOnce(Return(123));
  
  int64_t start_pos = 0;
  Result<uint64_t> result = prefiller_->prefill(input, start_pos);
  
  EXPECT_TRUE(result.ok());
  EXPECT_EQ(result.get(), 123);
  EXPECT_EQ(start_pos, 5); // Should be incremented by seq_len
}

TEST_F(MultimodalPrefillerTest, PrefillAudioInput) {
  // Create test audio data
  std::vector<float> audio_data(2 * 80 * 100, 0.1f); // batch=2, n_bins=80, n_frames=100
  Audio test_audio{std::move(audio_data), 2, 80, 100};
  MultimodalInput input(std::move(test_audio));
  
  // Create mock encoder output tensor
  auto encoder_output_tensor = tf_float_->make({1, 3}, {1.0f, 2.0f, 3.0f});
  std::vector<EValue> encoder_outputs = {EValue(encoder_output_tensor)};
  
  // Create mock prefill output tensor
  auto prefill_output_tensor = tf_float_->make({1, 1, 4096}, std::vector<float>(4096, 0.7f));
  std::vector<EValue> prefill_outputs = {EValue(prefill_output_tensor)};
  
  EXPECT_CALL(*mock_module_, execute(kAudioEncoderMethod, _))
      .WillOnce(Return(Result<std::vector<EValue>>(encoder_outputs)));
  
  EXPECT_CALL(*mock_module_, execute(kTextModelMethod, _))
      .WillOnce(Return(Result<std::vector<EValue>>(prefill_outputs)));
  
  EXPECT_CALL(*mock_decoder_runner_, logits_to_token(_, _))
      .WillOnce(Return(456));
  
  int64_t start_pos = 0;
  Result<uint64_t> result = prefiller_->prefill(input, start_pos);
  
  EXPECT_TRUE(result.ok());
  EXPECT_EQ(result.get(), 456);
  EXPECT_EQ(start_pos, 3); // Should be incremented by seq_len
}

TEST_F(MultimodalPrefillerTest, PrefillTextInput) {
  // Create test text input
  std::string test_text = "Hello world";
  MultimodalInput input(test_text);
  
  // Mock tokenizer encoding
  std::vector<uint64_t> tokens = {1, 2, 3, 4};
  ::tokenizers::Result<std::vector<uint64_t>> tokenize_result(tokens);
  EXPECT_CALL(*mock_tokenizer_, encode(test_text, _, _))
      .WillOnce(Return(tokenize_result));
  
  // Create mock encoder output tensor
  auto encoder_output_tensor = tf_float_->make({1, 4}, {1.0f, 2.0f, 3.0f, 4.0f});
  std::vector<EValue> encoder_outputs = {EValue(encoder_output_tensor)};
  
  // Create mock prefill output tensor
  auto prefill_output_tensor = tf_float_->make({1, 1, 4096}, std::vector<float>(4096, 0.3f));
  std::vector<EValue> prefill_outputs = {EValue(prefill_output_tensor)};
  
  EXPECT_CALL(*mock_module_, execute(kTokenEmbeddingMethod, _))
      .WillOnce(Return(Result<std::vector<EValue>>(encoder_outputs)));
  
  EXPECT_CALL(*mock_module_, execute(kTextModelMethod, _))
      .WillOnce(Return(Result<std::vector<EValue>>(prefill_outputs)));
  
  EXPECT_CALL(*mock_decoder_runner_, logits_to_token(_, _))
      .WillOnce(Return(789));
  
  int64_t start_pos = 0;
  Result<uint64_t> result = prefiller_->prefill(input, start_pos);
  
  EXPECT_TRUE(result.ok());
  EXPECT_EQ(result.get(), 789);
  EXPECT_EQ(start_pos, 4); // Should be incremented by seq_len
}

TEST_F(MultimodalPrefillerTest, PrefillUnsupportedInputType) {
  // Create an unsupported input type by using RawAudio (which isn't handled in prefill)
  executorch::extension::llm::RawAudio raw_audio{"test_path", 44100, 2, 1000};
  MultimodalInput input(std::move(raw_audio));
  
  int64_t start_pos = 0;
  Result<uint64_t> result = prefiller_->prefill(input, start_pos);
  
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.error(), Error::NotSupported);
  EXPECT_EQ(start_pos, 0); // start_pos should not be modified
}

TEST_F(MultimodalPrefillerTest, PrefillCorrectCachePositionTensor) {
  // Create test image data
  std::vector<uint8_t> image_data(3 * 224 * 224, 128);
  Image test_image{std::move(image_data), 224, 224};
  MultimodalInput input(std::move(test_image));
  
  // Create mock encoder output tensor with specific seq_len
  auto encoder_output_tensor = tf_float_->make({1, 3}, {1.0f, 2.0f, 3.0f});
  std::vector<EValue> encoder_outputs = {EValue(encoder_output_tensor)};
  
  // Create mock prefill output tensor
  auto prefill_output_tensor = tf_float_->make({1, 1, 4096}, std::vector<float>(4096, 0.5f));
  std::vector<EValue> prefill_outputs = {EValue(prefill_output_tensor)};
  
  EXPECT_CALL(*mock_module_, execute(kImageEncoderMethod, _))
      .WillOnce(Return(Result<std::vector<EValue>>(encoder_outputs)));
  
  // Verify that the cache_position_tensor is correctly constructed
  EXPECT_CALL(*mock_module_, execute(kTextModelMethod, _))
      .WillOnce([&](const std::string& method, const std::vector<EValue>& args) {
        EXPECT_EQ(method, kTextModelMethod);
        EXPECT_EQ(args.size(), 2);
        
        // Check cache position tensor - should be [5, 6, 7] since start_pos=5, seq_len=3
        auto cache_pos_tensor = args[0].toTensor();
        EXPECT_EQ(cache_pos_tensor.size(0), 3);
        
        int64_t* cache_pos_data = cache_pos_tensor.data_ptr<int64_t>();
        EXPECT_EQ(cache_pos_data[0], 5);
        EXPECT_EQ(cache_pos_data[1], 6);
        EXPECT_EQ(cache_pos_data[2], 7);
        
        return Result<std::vector<EValue>>(prefill_outputs);
      });
  
  EXPECT_CALL(*mock_decoder_runner_, logits_to_token(_, _))
      .WillOnce(Return(123));
  
  int64_t start_pos = 5; // Start from position 5
  Result<uint64_t> result = prefiller_->prefill(input, start_pos);
  
  EXPECT_TRUE(result.ok());
  EXPECT_EQ(start_pos, 8); // Should be 5 + 3 = 8
}

TEST_F(MultimodalPrefillerTest, PrefillEmptyPrefillResults) {
  // Create test image data
  std::vector<uint8_t> image_data(3 * 224 * 224, 128);
  Image test_image{std::move(image_data), 224, 224};
  MultimodalInput input(std::move(test_image));
  
  // Create mock encoder output tensor
  auto encoder_output_tensor = tf_float_->make({1, 3}, {1.0f, 2.0f, 3.0f});
  std::vector<EValue> encoder_outputs = {EValue(encoder_output_tensor)};
  
  // Return empty prefill outputs
  std::vector<EValue> empty_prefill_outputs = {};
  
  EXPECT_CALL(*mock_module_, execute(kImageEncoderMethod, _))
      .WillOnce(Return(Result<std::vector<EValue>>(encoder_outputs)));
  
  EXPECT_CALL(*mock_module_, execute(kTextModelMethod, _))
      .WillOnce(Return(Result<std::vector<EValue>>(empty_prefill_outputs)));
  
  int64_t start_pos = 0;
  Result<uint64_t> result = prefiller_->prefill(input, start_pos);
  
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.error(), Error::InvalidState);
  EXPECT_EQ(start_pos, 0); // start_pos should not be modified when error occurs
}

TEST_F(MultimodalPrefillerTest, PrefillStartPosIncrementAcrossMultipleCalls) {
  // First call with image input
  std::vector<uint8_t> image_data1(3 * 224 * 224, 128);
  Image test_image1{std::move(image_data1), 224, 224};
  MultimodalInput input1(std::move(test_image1));
  
  auto encoder_output_tensor1 = tf_float_->make({1, 2}, {1.0f, 2.0f});
  std::vector<EValue> encoder_outputs1 = {EValue(encoder_output_tensor1)};
  
  auto prefill_output_tensor1 = tf_float_->make({1, 1, 4096}, std::vector<float>(4096, 0.5f));
  std::vector<EValue> prefill_outputs1 = {EValue(prefill_output_tensor1)};
  
  EXPECT_CALL(*mock_module_, execute(kImageEncoderMethod, _))
      .WillOnce(Return(Result<std::vector<EValue>>(encoder_outputs1)));
  
  EXPECT_CALL(*mock_module_, execute(kTextModelMethod, _))
      .WillOnce(Return(Result<std::vector<EValue>>(prefill_outputs1)));
  
  EXPECT_CALL(*mock_decoder_runner_, logits_to_token(_, _))
      .WillOnce(Return(100));
  
  int64_t start_pos = 0;
  Result<uint64_t> result1 = prefiller_->prefill(input1, start_pos);
  
  EXPECT_TRUE(result1.ok());
  EXPECT_EQ(start_pos, 2); // Should be incremented by 2
  
  // Second call with another image input
  std::vector<uint8_t> image_data2(3 * 224 * 224, 128);
  Image test_image2{std::move(image_data2), 224, 224};
  MultimodalInput input2(std::move(test_image2));
  
  auto encoder_output_tensor2 = tf_float_->make({1, 3}, {1.0f, 2.0f, 3.0f});
  std::vector<EValue> encoder_outputs2 = {EValue(encoder_output_tensor2)};
  
  auto prefill_output_tensor2 = tf_float_->make({1, 1, 4096}, std::vector<float>(4096, 0.7f));
  std::vector<EValue> prefill_outputs2 = {EValue(prefill_output_tensor2)};
  
  EXPECT_CALL(*mock_module_, execute(kImageEncoderMethod, _))
      .WillOnce(Return(Result<std::vector<EValue>>(encoder_outputs2)));
  
  EXPECT_CALL(*mock_module_, execute(kTextModelMethod, _))
      .WillOnce(Return(Result<std::vector<EValue>>(prefill_outputs2)));
  
  EXPECT_CALL(*mock_decoder_runner_, logits_to_token(_, _))
      .WillOnce(Return(200));
  
  Result<uint64_t> result2 = prefiller_->prefill(input2, start_pos);
  
  EXPECT_TRUE(result2.ok());
  EXPECT_EQ(start_pos, 5); // Should be 2 + 3 = 5
}

} // namespace
