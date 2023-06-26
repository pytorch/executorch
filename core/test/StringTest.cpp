/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Changes:
- Naming
- Use gtest instead of TFLite Micro test framework
==============================================================================*/

#include <gtest/gtest.h>

#include <executorch/core/String.h>

using namespace ::testing;

TEST(StringTest, FormatPositiveIntShouldMatchExpected) {
  const size_t kBufferLen = 32;
  char buffer[kBufferLen];
  const char golden[] = "Int: 55";
  auto bytes_written = ETSnprintf(buffer, kBufferLen, "Int: %d", 55);
  EXPECT_EQ(sizeof(golden), bytes_written);
  EXPECT_STREQ(golden, buffer);
}

TEST(StringTest, FormatNegativeIntShouldMatchExpected) {
  const size_t kBufferLen = 32;
  char buffer[kBufferLen];
  const char golden[] = "Int: -55";
  auto bytes_written = ETSnprintf(buffer, kBufferLen, "Int: %d", -55);
  EXPECT_EQ(sizeof(golden), bytes_written);
  EXPECT_STREQ(golden, buffer);
}

TEST(StringTest, FormatUnsignedIntShouldMatchExpected) {
  const size_t kBufferLen = 32;
  char buffer[kBufferLen];
  const char golden[] = "UInt: 12345";
  auto bytes_written = ETSnprintf(buffer, kBufferLen, "UInt: %u", 12345);
  EXPECT_EQ(sizeof(golden), bytes_written);
  EXPECT_STREQ(golden, buffer);
}

TEST(StringTest, FormatHexShouldMatchExpected) {
  const size_t kBufferLen = 32;
  char buffer[kBufferLen];
  const char golden[] = "Hex: 0x12345";
  auto bytes_written = ETSnprintf(buffer, kBufferLen, "Hex: %x", 0x12345);
  EXPECT_EQ(sizeof(golden), bytes_written);
  EXPECT_STREQ(golden, buffer);
}

TEST(StringTest, FormatFloatShouldMatchExpected) {
  const size_t kBufferLen = 32;
  char buffer[kBufferLen];
  const char golden[] = "Float: 1.0*2^4";
  auto bytes_written = ETSnprintf(buffer, kBufferLen, "Float: %f", 16.);
  EXPECT_EQ(sizeof(golden), bytes_written);
  EXPECT_STREQ(golden, buffer);
}

TEST(StringTest, BadlyFormattedStringShouldProduceReasonableString) {
  const size_t kBufferLen = 32;
  char buffer[kBufferLen];
  const char golden[] = "Test Badly % formated % string";
  auto bytes_written =
      ETSnprintf(buffer, kBufferLen, "Test Badly %% formated %% string%");
  EXPECT_EQ(sizeof(golden), bytes_written);
  EXPECT_STREQ(golden, buffer);
}

TEST(StringTest, IntFormatOverrunShouldTruncate) {
  const size_t kBufferLen = 8;
  char buffer[kBufferLen];
  const char golden[] = "Int: ";
  auto bytes_written = ETSnprintf(buffer, kBufferLen, "Int: %d", 12345);
  EXPECT_EQ(sizeof(golden), bytes_written);
  EXPECT_STREQ(golden, buffer);
}

TEST(StringTest, UnsignedIntFormatOverrunShouldTruncate) {
  const size_t kBufferLen = 8;
  char buffer[kBufferLen];
  const char golden[] = "UInt: ";
  auto bytes_written = ETSnprintf(buffer, kBufferLen, "UInt: %u", 12345);
  EXPECT_EQ(sizeof(golden), bytes_written);
  EXPECT_STREQ(golden, buffer);
}

TEST(StringTest, HexFormatOverrunShouldTruncate) {
  const size_t kBufferLen = 8;
  char buffer[kBufferLen];
  const char golden[] = "Hex: ";
  auto bytes_written = ETSnprintf(buffer, kBufferLen, "Hex: %x", 0x12345);
  EXPECT_EQ(sizeof(golden), bytes_written);
  EXPECT_STREQ(golden, buffer);
}

TEST(StringTest, FloatFormatOverrunShouldTruncate) {
  const size_t kBufferLen = 12;
  char buffer[kBufferLen];
  const char golden[] = "Float: ";
  auto bytes_written = ETSnprintf(buffer, kBufferLen, "Float: %x", 12345.);
  EXPECT_EQ(sizeof(golden), bytes_written);
  EXPECT_STREQ(golden, buffer);
}

TEST(StringTest, FloatFormatShouldPrintFractionCorrectly) {
  const size_t kBufferLen = 24;
  char buffer[kBufferLen];
  const char golden[] = "Float: 1.0625*2^0";
  // Add small offset to float value to account for float rounding error.
  auto bytes_written = ETSnprintf(buffer, kBufferLen, "Float: %f", 1.0625001);
  EXPECT_EQ(sizeof(golden), bytes_written);
  EXPECT_STREQ(golden, buffer);
}

TEST(StringTest, FloatFormatShouldPrintFractionCorrectlyNoLeadingZeros) {
  const size_t kBufferLen = 24;
  char buffer[kBufferLen];
  const char golden[] = "Float: 1.6332993*2^-1";
  auto bytes_written = ETSnprintf(buffer, kBufferLen, "Float: %f", 0.816650);
  EXPECT_EQ(sizeof(golden), bytes_written);
  EXPECT_STREQ(golden, buffer);
}

TEST(StringTest, StringFormatOverrunShouldTruncate) {
  const size_t kBufferLen = 10;
  char buffer[kBufferLen];
  const char golden[] = "String: h";
  auto bytes_written =
      ETSnprintf(buffer, kBufferLen, "String: %s", "hello world");
  EXPECT_EQ(sizeof(golden), bytes_written);
  EXPECT_STREQ(golden, buffer);
}

TEST(StringTest, StringFormatWithExactOutputSizeOverrunShouldTruncate) {
  const size_t kBufferLen = 10;
  char buffer[kBufferLen];
  const char golden[] = "format st";
  auto bytes_written = ETSnprintf(buffer, kBufferLen, "format str");
  EXPECT_EQ(sizeof(golden), bytes_written);
  EXPECT_STREQ(golden, buffer);
}
