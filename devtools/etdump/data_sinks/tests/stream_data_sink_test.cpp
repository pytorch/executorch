/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/devtools/etdump/data_sinks/stream_data_sink.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>
#include <stdio.h> // tmpnam(), remove()
#include <fstream>

using namespace ::testing;
using ::executorch::etdump::StreamDataSink;
using ::executorch::runtime::Error;
using ::executorch::runtime::Result;
using ::executorch::runtime::Span;

class StreamDataSinkTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize the runtime environment
    torch::executor::runtime_init();

    // Allocate a small buffer for testing
    buffer_size_ = 128; // Small buffer size for testing
    buffer_ptr_ = malloc(buffer_size_);

    // Define the file path for testing
    std::array<char, L_tmpnam> buf;
    const char* ret = std::tmpnam(buf.data());
    ASSERT_NE(ret, nullptr) << "Coult not generate temp file";
    buf[L_tmpnam - 1] = '\0';
    file_path_ = std::string(buf.data()) + "-executorch-testing";
  }

  void TearDown() override {
    // Free the allocated buffer
    free(buffer_ptr_);

    // Remove the test file
    std::remove(file_path_.c_str());
  }

  size_t buffer_size_;
  void* buffer_ptr_;
  std::string file_path_;
};

TEST_F(StreamDataSinkTest, CreationExpectFail) {
  // Create a StreamDataSink instance with legal arguments
  Result<StreamDataSink> success =
      StreamDataSink::create(buffer_ptr_, buffer_size_, file_path_.c_str());
  ASSERT_TRUE(success.ok());

  // Try to create another StreamDataSink instanc with illegal file path
  Result<StreamDataSink> fail_with_invalid_file_path =
      StreamDataSink::create(buffer_ptr_, buffer_size_, "");
  ASSERT_EQ(fail_with_invalid_file_path.error(), Error::AccessFailed);

  // Try to create another StreamDataSink instanc with illegal file path
  Result<StreamDataSink> fail_with_invalid_alignment =
      StreamDataSink::create(buffer_ptr_, buffer_size_, file_path_.c_str(), 24);
  ASSERT_EQ(fail_with_invalid_alignment.error(), Error::InvalidArgument);
}

TEST_F(StreamDataSinkTest, WriteSmallDataToBuffer) {
  // Create a StreamDataSink instance
  Result<StreamDataSink> result =
      StreamDataSink::create(buffer_ptr_, buffer_size_, file_path_.c_str());
  ASSERT_TRUE(result.ok());

  StreamDataSink* data_sink = &result.get();
  const char* data = "Hello, World!";
  size_t data_size = strlen(data);

  // Write a small amount of data and check the buffer
  Result<size_t> write_result = data_sink->write(data, data_size);
  ASSERT_TRUE(write_result.ok());

  // Calculate the expected offset and verify the data in the buffer
  size_t offset = write_result.get();
  const uint8_t* buffer_data = static_cast<const uint8_t*>(buffer_ptr_);
  EXPECT_EQ(std::memcmp(buffer_data + offset, data, data_size), 0);
  EXPECT_EQ(data_sink->get_used_bytes(), data_size + offset);
}

TEST_F(StreamDataSinkTest, WriteAndFlushSmallData) {
  // Create a StreamDataSink instance
  Result<StreamDataSink> result =
      StreamDataSink::create(buffer_ptr_, buffer_size_, file_path_.c_str());
  ASSERT_TRUE(result.ok());

  StreamDataSink* data_sink = &result.get();
  const char* data = "Hello, World!";
  size_t data_size = strlen(data);

  // Write a small amount of data
  Result<size_t> write_result = data_sink->write(data, data_size);
  ASSERT_TRUE(write_result.ok());
  size_t offset = write_result.get();

  // Flush the buffer and verify the file contents
  Result<bool> flush_result = data_sink->flush();
  ASSERT_TRUE(flush_result.ok());

  size_t used_bytes = data_sink->get_used_bytes();

  // delete data_sink for flushing the data to file and closing the file
  data_sink->~StreamDataSink();

  std::ifstream file(file_path_, std::ios::binary);
  file.seekg(0, std::ios::end);
  size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);
  EXPECT_EQ(file_size, used_bytes);

  // Read the file content and verify it matches the original data
  std::vector<char> file_content(file_size);
  file.read(file_content.data(), file_size);
  file.close();

  // Verify that the file content matches the original data, considering
  // alignment
  EXPECT_EQ(std::memcmp(file_content.data() + offset, data, data_size), 0);
}

TEST_F(StreamDataSinkTest, WriteMultipleSmallDataWithAutoFlush) {
  // Create a StreamDataSink instance
  Result<StreamDataSink> result =
      StreamDataSink::create(buffer_ptr_, buffer_size_, file_path_.c_str());
  ASSERT_TRUE(result.ok());

  StreamDataSink* data_sink = &result.get();
  const char* data1 = "Acceleration";
  const char* data2 = "Core";
  const char* data3 = "Experience";
  size_t data1_size = strlen(data1);
  size_t data2_size = strlen(data2);
  size_t data3_size = strlen(data3);

  // Write multiple small data chunks
  Result<size_t> offset1 = data_sink->write(data1, data1_size);
  ASSERT_TRUE(offset1.ok());

  Result<size_t> offset2 = data_sink->write(data2, data2_size);
  ASSERT_TRUE(offset2.ok());

  Result<size_t> offset3 = data_sink->write(data3, data3_size);
  ASSERT_TRUE(offset3.ok());

  size_t used_bytes = data_sink->get_used_bytes();
  // delete data_sink for flushing the data to file and closing the file
  data_sink->~StreamDataSink();

  // Verify that the data is flushed automatically
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
