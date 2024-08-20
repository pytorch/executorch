/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/schema/extended_header.h>

#include <gtest/gtest.h>

#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/runtime.h>

using namespace ::testing;
using executorch::runtime::Error;
using executorch::runtime::ExtendedHeader;
using executorch::runtime::Result;

class ExtendedHeaderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    executorch::runtime::runtime_init();
  }
};

/**
 * An example, valid extended header.
 *
 * This data is intentionally fragile. If the header layout or magic changes,
 * this test data must change too. The layout of the header is a contract, not
 * an implementation detail.
 */
// clang-format off
constexpr char kExampleHeaderData[] = {
  // Magic bytes
  'e', 'h', '0', '0',
  // uint32_t header size (little endian)
  0x18, 0x00, 0x00, 0x00,
  // uint64_t program size
  0x71, 0x61, 0x51, 0x41, 0x31, 0x21, 0x11, 0x01,
  // uint64_t segment base offset
  0x72, 0x62, 0x52, 0x42, 0x32, 0x22, 0x12, 0x02,
};
// clang-format on

/// The program_size field encoded in kExampleHeaderData. Each byte is unique
/// within the header data.
constexpr uint64_t kExampleProgramSize = 0x0111213141516171;

/// The segment_base_offset field encoded in kExampleHeaderData. Each byte is
/// unique within the header data.
constexpr uint64_t kExampleSegmentBaseOffset = 0x0212223242526272;

/// The offset to the header's length field, which is in the 4 bytes after the
/// magic.
constexpr size_t kHeaderLengthOffset =
    ExtendedHeader::kHeaderOffset + ExtendedHeader::kMagicSize;

/**
 * Returns fake serialized Program head data that contains kExampleHeaderData at
 * the expected offset.
 */
std::vector<uint8_t> CreateExampleProgramHead() {
  // Allocate memory representing the head of the serialized Program.
  std::vector<uint8_t> ret(ExtendedHeader::kNumHeadBytes);
  // Write non-zeros into it to make it more obvious if we read outside the
  // header.
  memset(ret.data(), 0x55, ret.size());
  // Copy the example header into the right offset.
  memcpy(
      ret.data() + ExtendedHeader::kHeaderOffset,
      kExampleHeaderData,
      sizeof(kExampleHeaderData));
  return ret;
}

TEST_F(ExtendedHeaderTest, ValidHeaderParsesCorrectly) {
  std::vector<uint8_t> program = CreateExampleProgramHead();

  Result<ExtendedHeader> header =
      ExtendedHeader::Parse(program.data(), program.size());

  // The header should be present.
  ASSERT_EQ(header.error(), Error::Ok);

  // Since each byte of these fields is unique, success demonstrates that the
  // endian-to-int conversion is correct and looks at the expected bytes of the
  // header.
  EXPECT_EQ(header->program_size, kExampleProgramSize);
  EXPECT_EQ(header->segment_base_offset, kExampleSegmentBaseOffset);
}

TEST_F(ExtendedHeaderTest, ShortDataFails) {
  std::vector<uint8_t> program = CreateExampleProgramHead();

  // Try parsing a smaller-than-required part of the data.
  ASSERT_GE(program.size(), ExtendedHeader::kNumHeadBytes);
  Result<ExtendedHeader> header =
      ExtendedHeader::Parse(program.data(), ExtendedHeader::kNumHeadBytes - 1);

  // Should have been rejected.
  EXPECT_EQ(header.error(), Error::InvalidArgument);
}

TEST_F(ExtendedHeaderTest, MissingHeaderNotFound) {
  // Program head data without the extended header magic bytes.
  std::vector<uint8_t> program(ExtendedHeader::kNumHeadBytes);
  memset(program.data(), 0x55, program.size());

  // The header should not be found.
  Result<ExtendedHeader> header =
      ExtendedHeader::Parse(program.data(), program.size());
  EXPECT_EQ(header.error(), Error::NotFound);
}

TEST_F(ExtendedHeaderTest, BadMagicTreatedAsMissing) {
  // Get a valid header.
  std::vector<uint8_t> program = CreateExampleProgramHead();

  // Should be present.
  {
    Result<ExtendedHeader> header =
        ExtendedHeader::Parse(program.data(), program.size());
    ASSERT_EQ(header.error(), Error::Ok);
  }

  // Change a character in the magic.
  program[ExtendedHeader::kHeaderOffset] = 'x';

  // No longer present.
  {
    Result<ExtendedHeader> header =
        ExtendedHeader::Parse(program.data(), program.size());
    EXPECT_EQ(header.error(), Error::NotFound);
  }
}

TEST_F(ExtendedHeaderTest, ShorterHeaderLengthFails) {
  // Get a valid header.
  std::vector<uint8_t> program = CreateExampleProgramHead();

  // Should be present.
  {
    Result<ExtendedHeader> header =
        ExtendedHeader::Parse(program.data(), program.size());
    ASSERT_EQ(header.error(), Error::Ok);
  }

  // Make the header length smaller.
  // First demonstrate that we're looking in the right place.
  EXPECT_EQ(program[kHeaderLengthOffset], 0x18);
  program[kHeaderLengthOffset] = 0x10;

  // Program now considered invalid.
  {
    Result<ExtendedHeader> header =
        ExtendedHeader::Parse(program.data(), program.size());
    EXPECT_EQ(header.error(), Error::InvalidProgram);
  }
}

TEST_F(ExtendedHeaderTest, LongerHeaderLengthSucceeds) {
  // Get a valid header.
  std::vector<uint8_t> program = CreateExampleProgramHead();

  // Make the header length larger.
  // First demonstrate that we're looking in the right place.
  EXPECT_EQ(program[kHeaderLengthOffset], 0x18);
  program[kHeaderLengthOffset] = 0x20;

  // Should still be present and contain the expected values.
  {
    Result<ExtendedHeader> header =
        ExtendedHeader::Parse(program.data(), program.size());
    ASSERT_EQ(header.error(), Error::Ok);
    EXPECT_EQ(header->program_size, kExampleProgramSize);
    EXPECT_EQ(header->segment_base_offset, kExampleSegmentBaseOffset);
  }
}
