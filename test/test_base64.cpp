/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pytorch/tokenizers/base64.h>
#include "gtest/gtest.h"

namespace tokenizers {

TEST(Base64Test, TestDecodeSmoke) {
  std::string text = "bGxhbWE=";
  auto result = base64::decode(text);
  EXPECT_TRUE(result.ok());
  EXPECT_EQ(result.get(), "llama");
}

TEST(Base64Test, TestDecodeEmptyStringReturnsError) {
  std::string text;
  auto result = base64::decode(text);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.error(), Error::Base64DecodeFailure);
}

TEST(Base64Test, TestInvalidStringDecodeReturnsError) {
  std::string text = "llama";
  auto result = base64::decode(text);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.error(), Error::Base64DecodeFailure);
}

} // namespace tokenizers
