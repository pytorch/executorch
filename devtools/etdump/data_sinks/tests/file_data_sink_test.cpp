/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/devtools/etdump/data_sinks/file_data_sink.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>
#include <stdio.h> // tmpnam(), remove()
#include <fstream>

using namespace ::testing;
using ::executorch::etdump::FileDataSink;
using ::executorch::runtime::Error;
using ::executorch::runtime::Result;

class FileDataSinkTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize the runtime environment
    torch::executor::runtime_init();

    // Define the file path for testing
    std::array<char, L_tmpnam> buf;
    const char* ret = std::tmpnam(buf.data());
    ASSERT_NE(ret, nullptr) << "Could not generate temp file";
    buf[L_tmpnam - 1] = '\0';
    file_path_ = std::string(buf.data()) + "-executorch-testing";
  }

  void TearDown() override {
    // Remove the test file
    std::remove(file_path_.c_str());
  }

  std::string file_path_;
};

TEST_F(FileDataSinkTest, CreationExpectFail) {
  // Create a FileDataSink instance with a valid file path
  Result<FileDataSink> success = FileDataSink::create(file_path_.c_str());
  ASSERT_TRUE(success.ok());

  // Try to create another FileDataSink instance with an invalid file path
  Result<FileDataSink> fail_with_invalid_file_path = FileDataSink::create("");
  ASSERT_EQ(fail_with_invalid_file_path.error(), Error::AccessFailed);
}

TEST_F(FileDataSinkTest, WriteDataToFile) {
  const char* data = "Hello, World!";
  size_t data_size = strlen(data);

  // Create a FileDataSink instance
  Result<FileDataSink> result = FileDataSink::create(file_path_.c_str());
  ASSERT_TRUE(result.ok());

  FileDataSink* data_sink = &result.get();

  // Write data to the file
  Result<size_t> write_result = data_sink->write(data, data_size);
  ASSERT_TRUE(write_result.ok());

  size_t used_bytes = data_sink->get_used_bytes();
  EXPECT_EQ(used_bytes, data_size);

  data_sink->close();

  // Expect fail if write again after close
  Result<size_t> write_result_after_close = data_sink->write(data, data_size);
  ASSERT_EQ(write_result_after_close.error(), Error::AccessFailed);

  // Verify the file contents
  std::ifstream file(file_path_, std::ios::binary);
  file.seekg(0, std::ios::end);
  size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);
  EXPECT_EQ(file_size, used_bytes);

  // Read the file content and verify it matches the original data
  std::vector<char> file_content(file_size);
  file.read(file_content.data(), file_size);
  file.close();

  EXPECT_EQ(std::memcmp(file_content.data(), data, data_size), 0);
}

TEST_F(FileDataSinkTest, WriteMultipleDataAndCheckOffsets) {
  const char* data1 = "Accelerate";
  const char* data2 = "Core";
  const char* data3 = "Experience";
  size_t data1_size = strlen(data1);
  size_t data2_size = strlen(data2);
  size_t data3_size = strlen(data3);

  // Create a FileDataSink instance
  Result<FileDataSink> result = FileDataSink::create(file_path_.c_str());
  ASSERT_TRUE(result.ok());
  FileDataSink* data_sink = &result.get();

  // Write multiple data chunks and check offsets
  Result<size_t> offset1 = data_sink->write(data1, data1_size);
  ASSERT_TRUE(offset1.ok());
  EXPECT_EQ(offset1.get(), 0);

  Result<size_t> offset2 = data_sink->write(data2, data2_size);
  ASSERT_TRUE(offset2.ok());
  EXPECT_EQ(offset2.get(), data1_size);

  Result<size_t> offset3 = data_sink->write(data3, data3_size);
  ASSERT_TRUE(offset3.ok());
  EXPECT_EQ(offset3.get(), data1_size + data2_size);
  size_t used_bytes = data_sink->get_used_bytes();
  EXPECT_EQ(used_bytes, data1_size + data2_size + data3_size);

  data_sink->close();

  // Verify the file contents
  std::ifstream file(file_path_, std::ios::binary);
  file.seekg(0, std::ios::end);
  size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);
  EXPECT_EQ(file_size, used_bytes);

  // Read the file content
  std::vector<char> file_content(file_size);
  file.read(file_content.data(), file_size);
  file.close();

  // Verify each data chunk in the file using offsets
  EXPECT_EQ(
      std::memcmp(file_content.data() + offset1.get(), data1, data1_size), 0);
  EXPECT_EQ(
      std::memcmp(file_content.data() + offset2.get(), data2, data2_size), 0);
  EXPECT_EQ(
      std::memcmp(file_content.data() + offset3.get(), data3, data3_size), 0);
}

TEST_F(FileDataSinkTest, WriteInPlaceTensorZeroFill) {
  // Test writing with nullptr ptr and non-zero length (in-place tensor case)
  // This should zero-fill the file
  size_t length = 16;

  // Create a FileDataSink instance
  Result<FileDataSink> result = FileDataSink::create(file_path_.c_str());
  ASSERT_TRUE(result.ok());
  FileDataSink* data_sink = &result.get();

  // Write nullptr with non-zero length
  Result<size_t> write_result = data_sink->write(nullptr, length);
  ASSERT_TRUE(write_result.ok());
  EXPECT_EQ(write_result.get(), 0);

  size_t used_bytes = data_sink->get_used_bytes();
  EXPECT_EQ(used_bytes, length);

  data_sink->close();

  // Verify the file contents are zero-filled
  std::ifstream file(file_path_, std::ios::binary);
  file.seekg(0, std::ios::end);
  size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);
  EXPECT_EQ(file_size, length);

  // Read the file content and verify it's all zeros
  std::vector<uint8_t> file_content(file_size);
  file.read(reinterpret_cast<char*>(file_content.data()), file_size);
  file.close();

  for (size_t i = 0; i < length; ++i) {
    EXPECT_EQ(file_content[i], 0);
  }
}

TEST_F(FileDataSinkTest, WriteZeroLengthReturnsCurrentOffset) {
  // Create a FileDataSink instance
  Result<FileDataSink> result = FileDataSink::create(file_path_.c_str());
  ASSERT_TRUE(result.ok());
  FileDataSink* data_sink = &result.get();

  // Zero length write should return current offset (0)
  Result<size_t> ret = data_sink->write(nullptr, 0);
  ASSERT_TRUE(ret.ok());
  EXPECT_EQ(ret.get(), 0);

  // Write some data first
  const char* data = "Hello";
  size_t data_size = strlen(data);
  Result<size_t> write_ret = data_sink->write(data, data_size);
  ASSERT_TRUE(write_ret.ok());

  // Zero length write should return current offset
  size_t current_used = data_sink->get_used_bytes();
  Result<size_t> ret2 = data_sink->write(nullptr, 0);
  ASSERT_TRUE(ret2.ok());
  EXPECT_EQ(ret2.get(), current_used);

  data_sink->close();
}

TEST_F(FileDataSinkTest, WriteNullptrWithZeroLengthReturnsCurrentOffset) {
  // Create a FileDataSink instance
  Result<FileDataSink> result = FileDataSink::create(file_path_.c_str());
  ASSERT_TRUE(result.ok());
  FileDataSink* data_sink = &result.get();

  // Write some data first to advance the offset
  const char* data = "Test data";
  size_t data_size = strlen(data);
  Result<size_t> write_ret = data_sink->write(data, data_size);
  ASSERT_TRUE(write_ret.ok());

  size_t current_used = data_sink->get_used_bytes();

  // Writing nullptr with zero length should return current offset
  Result<size_t> ret = data_sink->write(nullptr, 0);
  ASSERT_TRUE(ret.ok());
  EXPECT_EQ(ret.get(), current_used);

  data_sink->close();
}
