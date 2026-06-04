/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/runner/util.h>
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>

#include <vector>

namespace {

using ::executorch::aten::ScalarType;
using ::executorch::extension::make_tensor_ptr;
using ::executorch::extension::llm::convert_to_bfloat16;
using ::executorch::extension::llm::utf8_complete_prefix_len;

class ConvertToBFloat16Test : public ::testing::Test {
 protected:
  void SetUp() override {
    executorch::runtime::runtime_init();
  }
};

TEST_F(ConvertToBFloat16Test, ConvertsFloatTensorData) {
  auto source_tensor = make_tensor_ptr<float>(
      {2, 2}, std::vector<float>{0.0f, 1.5f, -2.0f, 3.25f});

  auto result = convert_to_bfloat16(source_tensor);
  ASSERT_TRUE(result.ok());
  auto bf16_tensor = *result;

  EXPECT_EQ(bf16_tensor->scalar_type(), ScalarType::BFloat16);
  EXPECT_EQ(bf16_tensor->numel(), source_tensor->numel());

  auto src_sizes = source_tensor->sizes();
  auto dst_sizes = bf16_tensor->sizes();
  ASSERT_EQ(dst_sizes.size(), src_sizes.size());
  for (size_t dim = 0; dim < dst_sizes.size(); ++dim) {
    EXPECT_EQ(dst_sizes[dim], src_sizes[dim]);
  }

  const auto* converted_data = bf16_tensor->const_data_ptr<::c10::BFloat16>();
  const auto* original_data = source_tensor->const_data_ptr<float>();
  ASSERT_NE(converted_data, nullptr);
  ASSERT_NE(original_data, nullptr);

  for (size_t i = 0; i < static_cast<size_t>(source_tensor->numel()); ++i) {
    EXPECT_NEAR(static_cast<float>(converted_data[i]), original_data[i], 1e-2f);
  }
}

TEST_F(ConvertToBFloat16Test, RejectsNonFloatTensor) {
  auto non_float_tensor =
      make_tensor_ptr<int64_t>({3}, std::vector<int64_t>{1, 2, 3});

  auto result = convert_to_bfloat16(non_float_tensor);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.error(), ::executorch::runtime::Error::InvalidArgument);
}

TEST(Utf8CompletePrefixLenTest, HandlesAsciiAndMultiByteBoundaries) {
  EXPECT_EQ(utf8_complete_prefix_len(""), 0u);
  EXPECT_EQ(utf8_complete_prefix_len("ascii"), 5u);

  // Complete multi-byte characters are fully consumed.
  EXPECT_EQ(utf8_complete_prefix_len("\xc3\xa9"), 2u); // é   (2-byte)
  EXPECT_EQ(utf8_complete_prefix_len("\xe2\x82\xac"), 3u); // €   (3-byte)
  EXPECT_EQ(utf8_complete_prefix_len("\xf0\x9f\x98\x80"), 4u); // 😀 (4-byte)

  // A character split across the end is held back (not counted).
  EXPECT_EQ(utf8_complete_prefix_len("\xc3"), 0u); // 1/2 of é
  EXPECT_EQ(utf8_complete_prefix_len("\xe2\x82"), 0u); // 2/3 of €
  EXPECT_EQ(utf8_complete_prefix_len("\xf0\x9f\x98"), 0u); // 3/4 of 😀

  // A complete prefix followed by a split character keeps the complete part.
  EXPECT_EQ(utf8_complete_prefix_len("hi\xe2\x82"), 2u);
  EXPECT_EQ(utf8_complete_prefix_len("\xe2\x82\xac\xf0\x9f"), 3u);

  // An invalid lead byte counts as length 1 (emitted, not stalled).
  EXPECT_EQ(utf8_complete_prefix_len("\x80"), 1u);
}

} // namespace
