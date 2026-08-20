/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/flat_tensor/serialize/flat_tensor_header.h>

#include <gtest/gtest.h>

#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/runtime.h>

using namespace ::testing;
using executorch::extension::FlatTensorHeader;
using executorch::runtime::Error;
using executorch::runtime::Result;

class FlatTensorHeaderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    executorch::runtime::runtime_init();
  }
};

/**
 * An example, valid flat_tensor header.
 *
 * This data is intentionally fragile. If the header layout or magic changes,
 * this test data must change too. The layout of the header is a contract, not
 * an implementation detail.
 */
// clang-format off
// @lint-ignore CLANGTIDY facebook-hte-CArray
constexpr char kExampleHeaderData[] = {
  // Magic bytes
  'F', 'H', '0', '1',
  // uint32_t header size (little endian)
  0x28, 0x00, 0x00, 0x00,
  // uint64_t flatbuffer_offset
  0x71, 0x61, 0x51, 0x41, 0x31, 0x21, 0x11, 0x01,
  // uint64_t flatbuffer_size
  0x72, 0x62, 0x52, 0x42, 0x32, 0x22, 0x12, 0x02,
  // uint64_t segment_base_offset
  0x73, 0x63, 0x53, 0x43, 0x33, 0x23, 0x13, 0x03,
  // uint64_t segment_data_size
  0x74, 0x64, 0x54, 0x44, 0x34, 0x24, 0x14, 0x04,
};

constexpr uint64_t kExampleFlatbufferOffset = 0x0111213141516171;
constexpr uint64_t kExampleFlatbufferSize = 0x0212223242526272;
constexpr uint64_t kExampleSegmentBaseOffset = 0x0313233343536373; 
constexpr uint64_t kExampleSegmentDataSize = 0x0414243444546474;

/**
 * Returns fake serialized FlatTensor data that contains kExampleHeaderData at
 * the expected offset.
 */
std::vector<uint8_t> CreateExampleFlatTensorHeader() {
  // Allocate memory representing the FlatTensor header.
  std::vector<uint8_t> ret(FlatTensorHeader::kNumHeadBytes);
  // Write non-zeros into it to make it more obvious if we read outside the
  // header.
  memset(ret.data(), 0x55, ret.size());
  // Copy the example header into the right offset.
  memcpy(
      ret.data() + FlatTensorHeader::kHeaderOffset,
      kExampleHeaderData,
      sizeof(kExampleHeaderData));
  return ret;
}

TEST_F(FlatTensorHeaderTest, ValidHeaderParsesCorrectly) {
  std::vector<uint8_t> flat_tensor = CreateExampleFlatTensorHeader();

  Result<FlatTensorHeader> header = FlatTensorHeader::Parse(flat_tensor.data(), flat_tensor.size());

  // The header should be present.
  ASSERT_EQ(header.error(), Error::Ok);  

  // Since each byte of these fields is unique, success demonstrates that the
  // endian-to-int conversion is correct and looks at the expected bytes of the
  // header.
  EXPECT_EQ(header->flatbuffer_offset, kExampleFlatbufferOffset);
  EXPECT_EQ(header->flatbuffer_size, kExampleFlatbufferSize);
  EXPECT_EQ(header->segment_base_offset, kExampleSegmentBaseOffset);
  EXPECT_EQ(header->segment_data_size, kExampleSegmentDataSize);
}
