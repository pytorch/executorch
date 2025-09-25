/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every CLANGTIDY facebook-hte-Deprecated

#include <executorch/extension/llm/runner/multimodal_input.h>
#include <gtest/gtest.h>

using namespace ::testing;
using executorch::extension::llm::Image;
using executorch::extension::llm::make_image_input;
using executorch::extension::llm::make_text_input;
using executorch::extension::llm::make_token_input;
using executorch::extension::llm::MultimodalInput;

class MultimodalInputTest : public Test {
 protected:
  std::string createTestText() {
    return "Hello, world!";
  }

  std::string createTestTextLong() {
    return "This is a longer test string with multiple words and punctuation.";
  }

  Image createTestImage() {
    std::vector<uint8_t> data(224 * 224 * 3, 128); // Fill with gray
    return Image(std::move(data), 224, 224, 3);
  }

  Image createTestImageSmall() {
    std::vector<uint8_t> data(32 * 32, 255); // Fill with white
    return Image(std::move(data), 32, 32, 1);
  }
};

// Test text constructors
TEST_F(MultimodalInputTest, TextConstructorFromString) {
  std::string text = createTestText();
  MultimodalInput input(text);

  EXPECT_TRUE(input.is_text());
  EXPECT_FALSE(input.is_image());
  EXPECT_EQ(input.get_type(), MultimodalInput::Type::TEXT);
  EXPECT_EQ(input.get_text(), text);
}

TEST_F(MultimodalInputTest, TextConstructorFromRvalueString) {
  std::string text = createTestText();
  std::string original_text = text;
  MultimodalInput input(std::move(text));

  EXPECT_TRUE(input.is_text());
  EXPECT_FALSE(input.is_image());
  EXPECT_EQ(input.get_type(), MultimodalInput::Type::TEXT);
  EXPECT_EQ(input.get_text(), original_text);
}

// Test image constructors
TEST_F(MultimodalInputTest, ImageConstructorFromImage) {
  Image img = createTestImage();
  MultimodalInput input(img);

  EXPECT_FALSE(input.is_text());
  EXPECT_TRUE(input.is_image());
  EXPECT_EQ(input.get_type(), MultimodalInput::Type::IMAGE);
  EXPECT_EQ(input.get_image().width(), 224);
  EXPECT_EQ(input.get_image().height(), 224);
  EXPECT_EQ(input.get_image().channels(), 3);
  EXPECT_EQ(input.get_image().get_uint8_data().size(), 224 * 224 * 3);
}

TEST_F(MultimodalInputTest, ImageConstructorFromRvalueImage) {
  Image img = createTestImage();
  int width = img.width();
  int height = img.height();
  int channels = img.channels();
  size_t data_size = img.get_uint8_data().size();

  MultimodalInput input(std::move(img));

  EXPECT_FALSE(input.is_text());
  EXPECT_TRUE(input.is_image());
  EXPECT_EQ(input.get_type(), MultimodalInput::Type::IMAGE);
  EXPECT_EQ(input.get_image().width(), width);
  EXPECT_EQ(input.get_image().height(), height);
  EXPECT_EQ(input.get_image().channels(), channels);
  EXPECT_EQ(input.get_image().get_uint8_data().size(), data_size);
}

// Test copy constructor and assignment
TEST_F(MultimodalInputTest, CopyConstructorText) {
  std::string text = createTestText();
  MultimodalInput original(text);
  MultimodalInput copy(original);

  EXPECT_TRUE(copy.is_text());
  EXPECT_EQ(copy.get_text(), text);
  EXPECT_EQ(original.get_text(), text); // Original should be unchanged
}

TEST_F(MultimodalInputTest, CopyAssignmentText) {
  std::string text = createTestText();
  MultimodalInput original(text);
  MultimodalInput copy(createTestImage()); // Start with different type

  copy = original;

  EXPECT_TRUE(copy.is_text());
  EXPECT_EQ(copy.get_text(), text);
  EXPECT_EQ(original.get_text(), text); // Original should be unchanged
}

TEST_F(MultimodalInputTest, CopyConstructorImage) {
  Image img = createTestImage();
  MultimodalInput original(img);
  MultimodalInput copy(original);

  EXPECT_TRUE(copy.is_image());
  EXPECT_EQ(copy.get_image().width(), 224);
  EXPECT_EQ(copy.get_image().height(), 224);
  EXPECT_EQ(copy.get_image().channels(), 3);
  EXPECT_EQ(original.get_image().width(), 224); // Original should be unchanged
}

TEST_F(MultimodalInputTest, CopyAssignmentImage) {
  Image img = createTestImage();
  MultimodalInput original(img);
  MultimodalInput copy(createTestText()); // Start with different type

  copy = original;

  EXPECT_TRUE(copy.is_image());
  EXPECT_EQ(copy.get_image().width(), 224);
  EXPECT_EQ(copy.get_image().height(), 224);
  EXPECT_EQ(copy.get_image().channels(), 3);
  EXPECT_EQ(original.get_image().width(), 224); // Original should be unchanged
}

// Test move constructor and assignment
TEST_F(MultimodalInputTest, MoveConstructorText) {
  std::string text = createTestText();
  std::string original_text = text;
  MultimodalInput original(std::move(text));
  MultimodalInput moved(std::move(original));

  EXPECT_TRUE(moved.is_text());
  EXPECT_EQ(moved.get_text(), original_text);
}

TEST_F(MultimodalInputTest, MoveAssignmentText) {
  std::string text = createTestText();
  std::string original_text = text;
  MultimodalInput original(std::move(text));
  MultimodalInput moved(createTestImage()); // Start with different type

  moved = std::move(original);

  EXPECT_TRUE(moved.is_text());
  EXPECT_EQ(moved.get_text(), original_text);
}

TEST_F(MultimodalInputTest, MoveConstructorImage) {
  Image img = createTestImage();
  int width = img.width();
  int height = img.height();
  int channels = img.channels();
  MultimodalInput original(std::move(img));
  MultimodalInput moved(std::move(original));

  EXPECT_TRUE(moved.is_image());
  EXPECT_EQ(moved.get_image().width(), width);
  EXPECT_EQ(moved.get_image().height(), height);
  EXPECT_EQ(moved.get_image().channels(), channels);
}

TEST_F(MultimodalInputTest, MoveAssignmentImage) {
  Image img = createTestImage();
  int width = img.width();
  int height = img.height();
  int channels = img.channels();
  MultimodalInput original(std::move(img));
  MultimodalInput moved(createTestText()); // Start with different type

  moved = std::move(original);

  EXPECT_TRUE(moved.is_image());
  EXPECT_EQ(moved.get_image().width(), width);
  EXPECT_EQ(moved.get_image().height(), height);
  EXPECT_EQ(moved.get_image().channels(), channels);
}

// Test getter methods with correct types
TEST_F(MultimodalInputTest, GetTextWithTextInput) {
  std::string text = createTestText();
  MultimodalInput input(text);

  // Test const lvalue reference version
  const MultimodalInput& const_input = input;
  EXPECT_EQ(const_input.get_text(), text);

  // Test mutable lvalue reference version
  std::string& mutable_text = input.get_text();
  mutable_text += " Modified";
  EXPECT_EQ(input.get_text(), text + " Modified");

  // Test rvalue reference version
  std::string moved_text = std::move(input).get_text();
  EXPECT_EQ(moved_text, text + " Modified");
}

TEST_F(MultimodalInputTest, GetImageWithImageInput) {
  Image img = createTestImage();
  MultimodalInput input(img);

  // Test const lvalue reference version
  const MultimodalInput& const_input = input;
  EXPECT_EQ(const_input.get_image().width(), 224);
  EXPECT_EQ(const_input.get_image().height(), 224);
  EXPECT_EQ(const_input.get_image().channels(), 3);

  // Test rvalue reference version
  Image moved_image = std::move(input).get_image();
  EXPECT_EQ(moved_image.width(), 224);
}

// Test getter methods with wrong types (should throw)
TEST_F(MultimodalInputTest, GetTextWithImageInputThrows) {
  Image img = createTestImage();
  MultimodalInput input(img);

  EXPECT_THROW(input.get_text(), std::bad_variant_access);
  EXPECT_THROW(std::move(input).get_text(), std::bad_variant_access);
}

TEST_F(MultimodalInputTest, GetImageWithTextInputThrows) {
  std::string text = createTestText();
  MultimodalInput input(text);

  EXPECT_THROW(input.get_image(), std::bad_variant_access);
  EXPECT_THROW(std::move(input).get_image(), std::bad_variant_access);
}

// Test safe getter methods (try_get_*)
TEST_F(MultimodalInputTest, TryGetTextWithTextInput) {
  std::string text = createTestText();
  MultimodalInput input(text);

  // Test const version
  const MultimodalInput& const_input = input;
  const std::string* text_ptr = const_input.try_get_text();
  ASSERT_NE(text_ptr, nullptr);
  EXPECT_EQ(*text_ptr, text);

  // Test mutable version
  std::string* mutable_text_ptr = input.try_get_text();
  ASSERT_NE(mutable_text_ptr, nullptr);
  EXPECT_EQ(*mutable_text_ptr, text);

  // Modify through pointer
  *mutable_text_ptr += " Modified";
  EXPECT_EQ(input.get_text(), text + " Modified");
}

TEST_F(MultimodalInputTest, TryGetTextWithImageInput) {
  Image img = createTestImage();
  MultimodalInput input(img);

  // Should return nullptr for wrong type
  EXPECT_EQ(input.try_get_text(), nullptr);

  const MultimodalInput& const_input = input;
  EXPECT_EQ(const_input.try_get_text(), nullptr);
}

TEST_F(MultimodalInputTest, TryGetImageWithImageInput) {
  Image img = createTestImage();
  MultimodalInput input(img);

  // Test const version
  const MultimodalInput& const_input = input;
  const Image* image_ptr = const_input.try_get_image();
  ASSERT_NE(image_ptr, nullptr);
  EXPECT_EQ(image_ptr->width(), 224);
  EXPECT_EQ(image_ptr->height(), 224);
  EXPECT_EQ(image_ptr->channels(), 3);

  // Test mutable version
  Image* mutable_image_ptr = input.try_get_image();
  ASSERT_NE(mutable_image_ptr, nullptr);
  EXPECT_EQ(mutable_image_ptr->width(), 224);
}

TEST_F(MultimodalInputTest, TryGetImageWithTextInput) {
  std::string text = createTestText();
  MultimodalInput input(text);

  // Should return nullptr for wrong type
  EXPECT_EQ(input.try_get_image(), nullptr);

  const MultimodalInput& const_input = input;
  EXPECT_EQ(const_input.try_get_image(), nullptr);
}

// Test convenience factory functions
TEST_F(MultimodalInputTest, MakeTextInputFromString) {
  std::string text = createTestText();
  MultimodalInput input = make_text_input(text);

  EXPECT_TRUE(input.is_text());
  EXPECT_EQ(input.get_text(), text);
}

TEST_F(MultimodalInputTest, MakeTextInputFromRvalueString) {
  std::string text = createTestText();
  std::string original_text = text;
  MultimodalInput input = make_text_input(std::move(text));

  EXPECT_TRUE(input.is_text());
  EXPECT_EQ(input.get_text(), original_text);
}

TEST_F(MultimodalInputTest, MakeImageInputFromImage) {
  Image img = createTestImage();
  MultimodalInput input = make_image_input(img);

  EXPECT_TRUE(input.is_image());
  EXPECT_EQ(input.get_image().width(), 224);
  EXPECT_EQ(input.get_image().height(), 224);
  EXPECT_EQ(input.get_image().channels(), 3);
}

TEST_F(MultimodalInputTest, MakeImageInputFromRvalueImage) {
  Image img = createTestImage();
  int width = img.width();
  int height = img.height();
  int channels = img.channels();
  MultimodalInput input = make_image_input(std::move(img));

  EXPECT_TRUE(input.is_image());
  EXPECT_EQ(input.get_image().width(), width);
  EXPECT_EQ(input.get_image().height(), height);
  EXPECT_EQ(input.get_image().channels(), channels);
}

// Test with different image sizes
TEST_F(MultimodalInputTest, DifferentImageSizes) {
  Image small_img = createTestImageSmall();
  MultimodalInput input(small_img);

  EXPECT_TRUE(input.is_image());
  EXPECT_EQ(input.get_image().width(), 32);
  EXPECT_EQ(input.get_image().height(), 32);
  EXPECT_EQ(input.get_image().channels(), 1);
  EXPECT_EQ(input.get_image().get_uint8_data().size(), 32 * 32);
}

// Test with empty text
TEST_F(MultimodalInputTest, EmptyText) {
  std::string empty_text = "";
  MultimodalInput input(empty_text);

  EXPECT_TRUE(input.is_text());
  EXPECT_EQ(input.get_text(), "");
  EXPECT_EQ(input.get_text().size(), 0);
}

// Test with long text
TEST_F(MultimodalInputTest, LongText) {
  std::string long_text = createTestTextLong();
  MultimodalInput input(long_text);

  EXPECT_TRUE(input.is_text());
  EXPECT_EQ(input.get_text(), long_text);
  EXPECT_GT(input.get_text().size(), 50);
}

// Test type consistency
TEST_F(MultimodalInputTest, TypeConsistency) {
  std::string text = createTestText();
  Image img = createTestImage();

  MultimodalInput text_input(text);
  MultimodalInput image_input(img);

  // Text input should consistently report as text
  EXPECT_TRUE(text_input.is_text());
  EXPECT_FALSE(text_input.is_image());
  EXPECT_EQ(text_input.get_type(), MultimodalInput::Type::TEXT);

  // Image input should consistently report as image
  EXPECT_FALSE(image_input.is_text());
  EXPECT_TRUE(image_input.is_image());
  EXPECT_EQ(image_input.get_type(), MultimodalInput::Type::IMAGE);
}

// Test assignment between different types
TEST_F(MultimodalInputTest, AssignmentBetweenTypes) {
  std::string text = createTestText();
  Image img = createTestImage();

  MultimodalInput input(text);
  EXPECT_TRUE(input.is_text());

  // Assign image to text input
  input = MultimodalInput(img);
  EXPECT_TRUE(input.is_image());
  EXPECT_EQ(input.get_image().width(), 224);

  // Assign text back to image input
  input = MultimodalInput(text);
  EXPECT_TRUE(input.is_text());
  EXPECT_EQ(input.get_text(), text);
}

// Token-related tests
class MultimodalInputTokenTest : public Test {
 protected:
  std::vector<uint64_t> createTestTokens() {
    return {1, 2, 3, 4, 5};
  }
};

// Test token constructors
TEST_F(MultimodalInputTokenTest, TokenConstructorFromVector) {
  std::vector<uint64_t> tokens = createTestTokens();
  MultimodalInput input(tokens);

  EXPECT_TRUE(input.is_tokens());
  EXPECT_FALSE(input.is_text());
  EXPECT_FALSE(input.is_image());
  EXPECT_EQ(input.get_type(), MultimodalInput::Type::TOKENS);
  EXPECT_EQ(input.get_tokens(), tokens);
  EXPECT_EQ(input.get_tokens().size(), 5);
}

TEST_F(MultimodalInputTokenTest, TokenConstructorFromRvalueVector) {
  std::vector<uint64_t> tokens = createTestTokens();
  std::vector<uint64_t> original_tokens = tokens;
  MultimodalInput input(std::move(tokens));

  EXPECT_TRUE(input.is_tokens());
  EXPECT_FALSE(input.is_text());
  EXPECT_FALSE(input.is_image());
  EXPECT_EQ(input.get_type(), MultimodalInput::Type::TOKENS);
  EXPECT_EQ(input.get_tokens(), original_tokens);
  EXPECT_EQ(input.get_tokens().size(), 5);
}

// Test token type checking
TEST_F(MultimodalInputTokenTest, TokenTypeChecking) {
  std::vector<uint64_t> tokens = createTestTokens();
  MultimodalInput input(tokens);

  EXPECT_TRUE(input.is_tokens());
  EXPECT_FALSE(input.is_text());
  EXPECT_FALSE(input.is_image());
  EXPECT_FALSE(input.is_audio());
  EXPECT_FALSE(input.is_raw_audio());
  EXPECT_EQ(input.get_type(), MultimodalInput::Type::TOKENS);
  EXPECT_STREQ(input.type_name(), "tokens");
}

// Test token getters
TEST_F(MultimodalInputTokenTest, GetTokensWithTokenInput) {
  std::vector<uint64_t> tokens = createTestTokens();
  MultimodalInput input(tokens);

  // Test const lvalue reference version
  const MultimodalInput& const_input = input;
  EXPECT_EQ(const_input.get_tokens(), tokens);
  EXPECT_EQ(const_input.get_tokens().size(), 5);

  // Test mutable lvalue reference version
  std::vector<uint64_t>& mutable_tokens = input.get_tokens();
  mutable_tokens.push_back(6);
  EXPECT_EQ(input.get_tokens().size(), 6);
  EXPECT_EQ(input.get_tokens().back(), 6);

  // Test rvalue reference version
  std::vector<uint64_t> moved_tokens = std::move(input).get_tokens();
  EXPECT_EQ(moved_tokens.size(), 6);
  EXPECT_EQ(moved_tokens.back(), 6);
}

// Test token getters with wrong types (should throw)
TEST_F(MultimodalInputTokenTest, GetTokensWithTextInputThrows) {
  std::string text = "Hello";
  MultimodalInput input(text);

  EXPECT_THROW(input.get_tokens(), std::bad_variant_access);
  EXPECT_THROW(std::move(input).get_tokens(), std::bad_variant_access);
}

TEST_F(MultimodalInputTokenTest, GetTextWithTokenInputThrows) {
  std::vector<uint64_t> tokens = createTestTokens();
  MultimodalInput input(tokens);

  EXPECT_THROW(input.get_text(), std::bad_variant_access);
  EXPECT_THROW(std::move(input).get_text(), std::bad_variant_access);
}

// Test safe token getters (try_get_*)
TEST_F(MultimodalInputTokenTest, TryGetTokensWithTokenInput) {
  std::vector<uint64_t> tokens = createTestTokens();
  MultimodalInput input(tokens);

  // Test const version
  const MultimodalInput& const_input = input;
  const std::vector<uint64_t>* tokens_ptr = const_input.try_get_tokens();
  ASSERT_NE(tokens_ptr, nullptr);
  EXPECT_EQ(*tokens_ptr, tokens);

  // Test mutable version
  std::vector<uint64_t>* mutable_tokens_ptr = input.try_get_tokens();
  ASSERT_NE(mutable_tokens_ptr, nullptr);
  EXPECT_EQ(*mutable_tokens_ptr, tokens);

  // Modify through pointer
  mutable_tokens_ptr->push_back(100);
  EXPECT_EQ(input.get_tokens().size(), 6);
  EXPECT_EQ(input.get_tokens().back(), 100);
}

TEST_F(MultimodalInputTokenTest, TryGetTokensWithTextInput) {
  std::string text = "Hello";
  MultimodalInput input(text);

  // Should return nullptr for wrong type
  EXPECT_EQ(input.try_get_tokens(), nullptr);

  const MultimodalInput& const_input = input;
  EXPECT_EQ(const_input.try_get_tokens(), nullptr);
}

// Test token convenience factory functions
TEST_F(MultimodalInputTokenTest, MakeTokenInputFromVector) {
  std::vector<uint64_t> tokens = createTestTokens();
  MultimodalInput input = make_token_input(tokens);

  EXPECT_TRUE(input.is_tokens());
  EXPECT_EQ(input.get_tokens(), tokens);
  EXPECT_EQ(input.get_tokens().size(), 5);
}

TEST_F(MultimodalInputTokenTest, MakeTokenInputFromRvalueVector) {
  std::vector<uint64_t> tokens = createTestTokens();
  std::vector<uint64_t> original_tokens = tokens;
  MultimodalInput input = make_token_input(std::move(tokens));

  EXPECT_TRUE(input.is_tokens());
  EXPECT_EQ(input.get_tokens(), original_tokens);
  EXPECT_EQ(input.get_tokens().size(), 5);
}

// Test token copy semantics
TEST_F(MultimodalInputTokenTest, TokenCopyConstructor) {
  std::vector<uint64_t> tokens = createTestTokens();
  MultimodalInput original(tokens);
  MultimodalInput copy(original);

  EXPECT_TRUE(copy.is_tokens());
  EXPECT_EQ(copy.get_tokens(), tokens);
  EXPECT_EQ(original.get_tokens(), tokens); // Original should be unchanged

  // Modify copy, original should be unaffected
  copy.get_tokens().push_back(999);
  EXPECT_EQ(copy.get_tokens().size(), 6);
  EXPECT_EQ(original.get_tokens().size(), 5);
}

TEST_F(MultimodalInputTokenTest, TokenCopyAssignment) {
  std::vector<uint64_t> tokens = createTestTokens();
  MultimodalInput original(tokens);
  MultimodalInput copy("initial text"); // Start with different type

  copy = original;

  EXPECT_TRUE(copy.is_tokens());
  EXPECT_EQ(copy.get_tokens(), tokens);
  EXPECT_EQ(original.get_tokens(), tokens); // Original should be unchanged
}

// Test token move semantics
TEST_F(MultimodalInputTokenTest, TokenMoveConstructor) {
  std::vector<uint64_t> tokens = createTestTokens();
  std::vector<uint64_t> original_tokens = tokens;
  MultimodalInput original(std::move(tokens));
  MultimodalInput moved(std::move(original));

  EXPECT_TRUE(moved.is_tokens());
  EXPECT_EQ(moved.get_tokens(), original_tokens);
}

TEST_F(MultimodalInputTokenTest, TokenMoveAssignment) {
  std::vector<uint64_t> tokens = createTestTokens();
  std::vector<uint64_t> original_tokens = tokens;
  MultimodalInput original(std::move(tokens));
  MultimodalInput moved("initial text"); // Start with different type

  moved = std::move(original);

  EXPECT_TRUE(moved.is_tokens());
  EXPECT_EQ(moved.get_tokens(), original_tokens);
}

// Test TypeName and TypeToString static methods for TOKENS
TEST_F(MultimodalInputTokenTest, TypeNameAndToString) {
  EXPECT_STREQ(
      MultimodalInput::TypeName(MultimodalInput::Type::TOKENS), "tokens");
  EXPECT_EQ(
      MultimodalInput::TypeToString(MultimodalInput::Type::TOKENS), "tokens");

  std::vector<uint64_t> tokens = createTestTokens();
  MultimodalInput input(tokens);
  EXPECT_STREQ(input.type_name(), "tokens");
}

// Test assignment between token and other types
TEST_F(MultimodalInputTokenTest, AssignmentBetweenTokensAndOtherTypes) {
  std::vector<uint64_t> tokens = createTestTokens();
  std::string text = "Hello";

  MultimodalInput input(tokens);
  EXPECT_TRUE(input.is_tokens());

  // Assign text to token input
  input = MultimodalInput(text);
  EXPECT_TRUE(input.is_text());
  EXPECT_EQ(input.get_text(), text);

  // Assign tokens back to text input
  input = MultimodalInput(tokens);
  EXPECT_TRUE(input.is_tokens());
  EXPECT_EQ(input.get_tokens(), tokens);
}

// Test token values with specific patterns
TEST_F(MultimodalInputTokenTest, SpecificTokenValues) {
  std::vector<uint64_t> tokens = {
      0, 1, 2, 65535, 4294967295ULL, 18446744073709551615ULL};
  MultimodalInput input(tokens);

  EXPECT_TRUE(input.is_tokens());
  EXPECT_EQ(input.get_tokens().size(), 6);
  EXPECT_EQ(input.get_tokens()[0], 0);
  EXPECT_EQ(input.get_tokens()[1], 1);
  EXPECT_EQ(input.get_tokens()[2], 2);
  EXPECT_EQ(input.get_tokens()[3], 65535);
  EXPECT_EQ(input.get_tokens()[4], 4294967295ULL);
  EXPECT_EQ(input.get_tokens()[5], 18446744073709551615ULL); // Max uint64_t
}

// Test token modification through reference
TEST_F(MultimodalInputTokenTest, TokenModificationThroughReference) {
  std::vector<uint64_t> tokens = createTestTokens();
  MultimodalInput input(tokens);

  // Get mutable reference and modify
  std::vector<uint64_t>& token_ref = input.get_tokens();
  token_ref[0] = 999;
  token_ref.push_back(1000);

  // Verify changes
  EXPECT_EQ(input.get_tokens()[0], 999);
  EXPECT_EQ(input.get_tokens().size(), 6);
  EXPECT_EQ(input.get_tokens().back(), 1000);
}
