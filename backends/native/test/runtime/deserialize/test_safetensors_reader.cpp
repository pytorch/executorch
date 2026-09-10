// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/deserialize/SafeTensorsReader.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

namespace ptn {
namespace {

std::vector<uint8_t> make_safetensors(
    std::string_view header,
    std::string_view data) {
  const uint64_t header_size = header.size();
  std::vector<uint8_t> bytes(sizeof(header_size) + header.size() + data.size());
  std::memcpy(bytes.data(), &header_size, sizeof(header_size));
  std::memcpy(bytes.data() + sizeof(header_size), header.data(), header.size());
  std::memcpy(
      bytes.data() + sizeof(header_size) + header.size(),
      data.data(),
      data.size());
  return bytes;
}

// cppcheck-suppress-begin syntaxError
TEST(SafeTensorsReaderTest, ReadsIndexAndPayloadsInHeaderOrder) {
  const std::vector<uint8_t> bytes = make_safetensors(
      R"({"second":{"dtype":"U8","shape":[2],"data_offsets":[4,6]},"__metadata__":{},"first":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})",
      "abcdXY");

  const SafeTensorsReader reader = SafeTensorsReader::open(bytes);

  EXPECT_EQ(reader.names(), (std::vector<std::string>{"second", "first"}));
  ASSERT_NE(reader.find("first"), nullptr);
  EXPECT_EQ(reader.find("first")->dtype, kFloat);
  EXPECT_EQ(reader.find("first")->sizes, (std::vector<int64_t>{1}));
  EXPECT_EQ(reader.find("first")->offset, 0);
  EXPECT_EQ(reader.find("first")->nbytes, 4);
  EXPECT_EQ(reader.bytes(*reader.find("second"))[0], 'X');
  EXPECT_EQ(reader.total_bytes(), 6);
  EXPECT_EQ(reader.find("missing"), nullptr);
}

TEST(SafeTensorsReaderTest, RejectsNonEmptyMetadata) {
  EXPECT_THROW(
      SafeTensorsReader::open(
          make_safetensors(R"({"__metadata__":{"source":"test"}})", "")),
      std::runtime_error);
}

TEST(SafeTensorsReaderTest, RejectsInvalidMetadata) {
  EXPECT_THROW(
      SafeTensorsReader::open(make_safetensors("[]", "")), std::runtime_error);
  EXPECT_THROW(
      SafeTensorsReader::open(make_safetensors(
          R"({"x":{"dtype":"U8","shape":[1.0],"data_offsets":[0,1]}})", "x")),
      std::runtime_error);
  EXPECT_THROW(
      SafeTensorsReader::open(make_safetensors(
          R"({"x":{"dtype":"U8","shape":[1],"data_offsets":[0,2]}})", "x")),
      std::runtime_error);
}

TEST(SafeTensorsReaderTest, RejectsByteSizeOverflow) {
  EXPECT_THROW(
      SafeTensorsReader::open(make_safetensors(
          R"({"x":{"dtype":"F64","shape":[2305843009213693952],"data_offsets":[0,0]}})",
          "")),
      std::runtime_error);
}
// cppcheck-suppress-end syntaxError

} // namespace
} // namespace ptn
