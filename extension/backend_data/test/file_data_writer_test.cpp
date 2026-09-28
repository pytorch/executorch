/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/backend_data/buffer_data_writer.h>
#include <executorch/extension/backend_data/file_data_writer.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <executorch/extension/backend_data/data_writer.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/platform/runtime.h>

using executorch::extension::BufferDataWriter;
using executorch::extension::DataWriter;
using executorch::extension::FileDataWriter;
using executorch::runtime::Error;
using executorch::runtime::Span;

namespace {

Span<const uint8_t> bytes(const std::string& value) {
  return Span<const uint8_t>(
      reinterpret_cast<const uint8_t*>(value.data()), value.size());
}

std::string read_file(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  return std::string(
      std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

class TestFile final {
 public:
  explicit TestFile(const std::string& contents) {
    static std::atomic<size_t> next_id{0};
    path_ = ::testing::TempDir() + "executorch-file-data-writer-" +
        std::to_string(next_id.fetch_add(1));
    std::ofstream output(path_, std::ios::binary | std::ios::trunc);
    output.write(contents.data(), contents.size());
  }

  ~TestFile() {
    std::remove(path_.c_str());
  }

  const std::string& path() const {
    return path_;
  }

 private:
  std::string path_;
};

class DataWriterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    executorch::runtime::runtime_init();
  }
};

void write_test_plan(DataWriter& writer) {
  // Writes are deliberately not ordered by output offset. This also models
  // delegate callbacks arriving in an order unrelated to metadata tables.
  EXPECT_EQ(writer.write(bytes(std::string("BBBB")), 12), Error::Ok);
  EXPECT_EQ(writer.write(bytes(std::string("HEAD")), 0), Error::Ok);
  EXPECT_EQ(writer.write(bytes(std::string("copy")), 4), Error::Ok);
  EXPECT_EQ(writer.write(bytes(std::string("AAAA")), 8), Error::Ok);
}

TEST_F(DataWriterTest, FileWriterRunsRandomAccessPlan) {
  TestFile destination("old contents");
  FileDataWriter writer(destination.path());
  write_test_plan(writer);

  EXPECT_EQ(read_file(destination.path()), "old contents");
  EXPECT_EQ(writer.publish(), Error::Ok);
  EXPECT_EQ(read_file(destination.path()), "HEADcopyAAAABBBB");
}

TEST_F(DataWriterTest, BufferWriterRunsSameRandomAccessPlan) {
  const std::string original = "old contents";
  std::vector<uint8_t> output(original.begin(), original.end());
  BufferDataWriter writer(&output);
  write_test_plan(writer);

  EXPECT_EQ(std::string(output.begin(), output.end()), original);
  EXPECT_EQ(writer.publish(), Error::Ok);
  EXPECT_EQ(std::string(output.begin(), output.end()), "HEADcopyAAAABBBB");
}

TEST_F(DataWriterTest, FileWriterZeroFillsSparseRangesAndTracksEmptyWrites) {
  TestFile destination("old contents");
  FileDataWriter writer(destination.path());
  EXPECT_EQ(writer.write(bytes(std::string("data")), 0), Error::Ok);
  EXPECT_EQ(writer.write(Span<const uint8_t>(), 8), Error::Ok);
  EXPECT_EQ(writer.publish(), Error::Ok);

  const std::string output = read_file(destination.path());
  ASSERT_EQ(output.size(), 8);
  EXPECT_EQ(output.substr(0, 4), "data");
  EXPECT_EQ(output.substr(4), std::string(4, '\0'));
}

TEST_F(DataWriterTest, BufferWriterZeroFillsSparseRangesAndTracksEmptyWrites) {
  std::vector<uint8_t> output;
  BufferDataWriter writer(&output);
  EXPECT_EQ(writer.write(bytes(std::string("data")), 0), Error::Ok);
  EXPECT_EQ(writer.write(Span<const uint8_t>(), 8), Error::Ok);
  EXPECT_EQ(writer.publish(), Error::Ok);

  ASSERT_EQ(output.size(), 8);
  EXPECT_EQ(std::string(output.begin(), output.begin() + 4), "data");
  EXPECT_EQ(
      std::string(output.begin() + 4, output.end()), std::string(4, '\0'));
}

TEST_F(DataWriterTest, LaterWritesReplaceOverlappingBytes) {
  std::vector<uint8_t> output;
  BufferDataWriter writer(&output);
  EXPECT_EQ(writer.write(bytes(std::string("abcdefgh")), 0), Error::Ok);
  EXPECT_EQ(writer.write(bytes(std::string("XYZ")), 2), Error::Ok);
  EXPECT_EQ(writer.publish(), Error::Ok);
  EXPECT_EQ(std::string(output.begin(), output.end()), "abXYZfgh");
}

TEST_F(DataWriterTest, DestructionDiscardsUnpublishedOutputs) {
  TestFile destination("original");
  {
    FileDataWriter writer(destination.path());
    EXPECT_EQ(writer.write(bytes(std::string("replacement")), 0), Error::Ok);
  }
  EXPECT_EQ(read_file(destination.path()), "original");

  const std::string original = "original";
  std::vector<uint8_t> output(original.begin(), original.end());
  {
    BufferDataWriter writer(&output);
    EXPECT_EQ(writer.write(bytes(std::string("replacement")), 0), Error::Ok);
  }
  EXPECT_EQ(std::string(output.begin(), output.end()), original);
}

TEST_F(DataWriterTest, PublishWithoutWritesCreatesEmptyOutput) {
  TestFile destination("original");
  FileDataWriter writer(destination.path());
  EXPECT_EQ(writer.publish(), Error::Ok);
  EXPECT_TRUE(read_file(destination.path()).empty());

  std::vector<uint8_t> output = {1, 2, 3};
  BufferDataWriter buffer_writer(&output);
  EXPECT_EQ(buffer_writer.publish(), Error::Ok);
  EXPECT_TRUE(output.empty());
}

TEST_F(DataWriterTest, RejectsUseAfterPublicationAndOffsetOverflow) {
  std::vector<uint8_t> output;
  BufferDataWriter writer(&output);
  EXPECT_EQ(
      writer.write(bytes(std::string("overflow")), SIZE_MAX),
      Error::InvalidArgument);
  EXPECT_EQ(writer.publish(), Error::Ok);
  EXPECT_EQ(writer.publish(), Error::InvalidState);
  EXPECT_EQ(writer.write(bytes(std::string("late")), 0), Error::InvalidState);
}

} // namespace
