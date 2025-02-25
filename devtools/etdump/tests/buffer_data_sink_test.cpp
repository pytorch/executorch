/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/devtools/etdump/buffer_data_sink.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/test/utils/DeathTest.h>
#include <gtest/gtest.h>

using namespace ::testing;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::etdump::BufferDataSink;
using ::executorch::runtime::Error;
using ::executorch::runtime::Result;
using ::executorch::runtime::Span;
using torch::executor::testing::TensorFactory;

class BufferDataSinkTest : public ::testing::Test {
 protected:
  void SetUp() override {
    torch::executor::runtime_init();
    // Allocate a small buffer for testing
    buffer_size_ = 128; // Small size for testing
    buffer_ptr_ = malloc(buffer_size_);
    buffer_ = Span<uint8_t>(static_cast<uint8_t*>(buffer_ptr_), buffer_size_);
  }

  void TearDown() override {
    free(buffer_ptr_);
  }

  size_t buffer_size_;
  void* buffer_ptr_;
  Span<uint8_t> buffer_;
};

TEST_F(BufferDataSinkTest, StorageSizeCheck) {
  Result<BufferDataSink> buffer_data_sink = BufferDataSink::create(buffer_);
  ASSERT_EQ(buffer_data_sink.error(), Error::Ok);

  Result<size_t> ret = buffer_data_sink.get().get_storage_size();
  ASSERT_EQ(ret.error(), Error::Ok);

  size_t storage_size = ret.get();
  EXPECT_EQ(storage_size, buffer_size_);
}

TEST_F(BufferDataSinkTest, WriteOneTensorAndCheckData) {
  TensorFactory<ScalarType::Float> tf;
  Tensor tensor = tf.make({1, 4}, {1.0, 2.0, 3.0, 4.0});

  Result<BufferDataSink> buffer_data_sink = BufferDataSink::create(buffer_);

  Result<size_t> ret =
      buffer_data_sink->write(tensor.const_data_ptr(), tensor.nbytes());
  ASSERT_EQ(ret.error(), Error::Ok);

  size_t offset = ret.get();

  EXPECT_NE(offset, static_cast<size_t>(-1));

  // Check that the data in the buffer matches the tensor data
  const float* buffer_data =
      reinterpret_cast<const float*>(buffer_.data() + offset);
  for (size_t i = 0; i < tensor.numel(); ++i) {
    EXPECT_EQ(buffer_data[i], tensor.const_data_ptr<float>()[i]);
  }
}

TEST_F(BufferDataSinkTest, WriteMultiTensorsAndCheckData) {
  TensorFactory<ScalarType::Float> tf;
  std::vector<Tensor> tensors = {
      tf.make({1, 4}, {1.0, 2.0, 3.0, 4.0}),
      tf.make({1, 4}, {5.0, 6.0, 7.0, 8.0})};
  Result<BufferDataSink> buffer_data_sink = BufferDataSink::create(buffer_);

  for (const auto& tensor : tensors) {
    Result<size_t> ret =
        buffer_data_sink->write(tensor.const_data_ptr(), tensor.nbytes());
    ASSERT_EQ(ret.error(), Error::Ok);

    size_t offset = ret.get();
    EXPECT_NE(offset, static_cast<size_t>(-1));
    // Check that the data in the buffer matches the tensor data
    const float* buffer_data =
        reinterpret_cast<const float*>(buffer_.data() + offset);
    for (size_t i = 0; i < tensor.numel(); ++i) {
      EXPECT_EQ(buffer_data[i], tensor.const_data_ptr<float>()[i]);
    }
  }
}

TEST_F(BufferDataSinkTest, PointerAlignmentCheck) {
  TensorFactory<ScalarType::Float> tf;
  Tensor tensor = tf.make({1, 4}, {1.0, 2.0, 3.0, 4.0});
  Result<BufferDataSink> buffer_data_sink = BufferDataSink::create(buffer_);

  Result<size_t> ret =
      buffer_data_sink->write(tensor.const_data_ptr(), tensor.nbytes());
  ASSERT_EQ(ret.error(), Error::Ok);

  size_t offset = ret.get();
  EXPECT_NE(offset, static_cast<size_t>(-1));
  // Check that the offset pointer is 64-byte aligned
  const uint8_t* offset_ptr = buffer_.data() + offset;
  EXPECT_EQ(reinterpret_cast<uintptr_t>(offset_ptr) % 64, 0);
}

TEST_F(BufferDataSinkTest, WriteUntilOverflow) {
  TensorFactory<ScalarType::Float> tf;
  Tensor tensor = tf.zeros({1, 8}); // Large tensor to fill the buffer

  Result<BufferDataSink> buffer_data_sink = BufferDataSink::create(buffer_);

  // Write tensors until we run out of space
  for (size_t i = 0; i < 2; i++) {
    Result<size_t> ret =
        buffer_data_sink->write(tensor.const_data_ptr(), tensor.nbytes());
    ASSERT_EQ(ret.error(), Error::Ok);
  }

  // Attempting to write another tensor should raise an error
  Result<size_t> ret =
      buffer_data_sink->write(tensor.const_data_ptr(), tensor.nbytes());
  ASSERT_EQ(ret.error(), Error::OutOfResources);
}

TEST_F(BufferDataSinkTest, illegalAlignment) {
  for (size_t i = 1; i <= 128; i <<= 1) {
    // Create a buffer_data_sink with legal alignment that is a power of 2 and
    // greater than 0
    Result<BufferDataSink> buffer_data_sink =
        BufferDataSink::create(buffer_, i);
    ASSERT_EQ(buffer_data_sink.error(), Error::Ok);
  }

  // Create a buffer_data_sink with illegal alignment that is not a power of 2
  // or greater than 0

  std::vector<size_t> illegal_alignments = {0, 3, 5, 7, 100, 127};

  for (size_t i = 0; i < illegal_alignments.size(); i++) {
    Result<BufferDataSink> buffer_data_sink =
        BufferDataSink::create(buffer_, illegal_alignments[i]);
    ASSERT_EQ(buffer_data_sink.error(), Error::InvalidArgument);
  }
}
