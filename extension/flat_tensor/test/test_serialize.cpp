/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/flat_tensor/serialize/serialize.h>

#include <executorch/extension/flat_tensor/serialize/flat_tensor_header.h>
#include <executorch/extension/flat_tensor/serialize/scalar_type_generated.h>
#include <executorch/extension/flat_tensor/serialize/schema_generated.h>

#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>
#include <map>
#include <sstream>

using namespace ::testing;
using executorch::runtime::Error;
using executorch::runtime::Result;

class FlatTensorSerializeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    executorch::runtime::runtime_init();
  }
};

TEST_F(FlatTensorSerializeTest, ValidFlatTensorSerialized) {
  const size_t kTensorAlignment = 16;
  std::map<std::string, executorch::aten::Tensor> flat_tensor_map;

  float linear_weight = 3.14f;
  auto weight = executorch::extension::make_tensor_ptr({1}, &linear_weight);

  float linear_bias = 2.0f;
  auto bias = executorch::extension::make_tensor_ptr({1}, &linear_bias);

  flat_tensor_map.insert({"linear.weight", *weight.get()});
  flat_tensor_map.insert({"linear.bias", *bias.get()});

  std::ostringstream buf;
  auto err = executorch::extension::flat_tensor::save_ptd(
      buf, flat_tensor_map, kTensorAlignment);
  ASSERT_EQ(err, Error::Ok);
  auto x = buf.str();
  const char* byte_buffer = x.c_str();

  // First 4 bytes are an offset to the flatbuffer root table.

  // Check magic ids.
  EXPECT_EQ(byte_buffer[4], 'F');
  EXPECT_EQ(byte_buffer[5], 'T');
  ASSERT_EQ(byte_buffer[6], '0');
  ASSERT_EQ(byte_buffer[7], '1');

  ASSERT_EQ(byte_buffer[8], 'F');
  ASSERT_EQ(byte_buffer[9], 'H');
  EXPECT_EQ(byte_buffer[10], '0');
  EXPECT_EQ(byte_buffer[11], '1');

  // Check Header
  auto header_buffer = byte_buffer + 8;
  EXPECT_EQ( // Check expected length
      *(uint32_t*)(header_buffer + 4),
      executorch::extension::FlatTensorHeader::kHeaderExpectedLength);

  EXPECT_EQ(
      *(uint64_t*)(header_buffer + 8),
      48); // Flatbuffer offset, header is 40 bytes + 8 bytes of padding
           // today, and then the flatbuffer starts.

  EXPECT_EQ(
      *(uint64_t*)(header_buffer + 16),
      232); // Flatbuffer size. This is fragile, and depends on the schema,
            // the builder, and the padding needed.

  // Segment offset, depends on the padded header and flatbuffer sizes.
  const uint64_t segment_offset = 48 + 232 + 8; // 8 is padding.
  EXPECT_EQ(*(uint64_t*)(header_buffer + 24), segment_offset);

  EXPECT_EQ(
      *(uint64_t*)(header_buffer + 32),
      20); // Segment total size, 8 bytes of data (2 floats), 24 bytes of
           // padding.

  // Check Flatbuffer
  auto flat_tensor = ::flat_tensor_flatbuffer::GetFlatTensor(byte_buffer);

  EXPECT_EQ(
      flat_tensor->version(),
      executorch::extension::flat_tensor::kSchemaVersion);
  EXPECT_EQ(flat_tensor->tensor_alignment(), 16);
  EXPECT_EQ(flat_tensor->tensors()->size(), 2);
  EXPECT_EQ(flat_tensor->segments()->size(), 1);

  auto tensor0 = flat_tensor->tensors()->Get(0);
  EXPECT_EQ(strcmp(tensor0->fully_qualified_name()->c_str(), "linear.bias"), 0);
  EXPECT_EQ(tensor0->scalar_type(), executorch_flatbuffer::ScalarType::FLOAT);
  EXPECT_EQ(tensor0->sizes()->size(), 1);
  EXPECT_EQ(tensor0->segment_index(), 0);
  EXPECT_EQ(tensor0->offset(), 0);

  auto tensor1 = flat_tensor->tensors()->Get(1);
  EXPECT_EQ(
      strcmp(tensor1->fully_qualified_name()->c_str(), "linear.weight"), 0);
  EXPECT_EQ(tensor1->scalar_type(), executorch_flatbuffer::ScalarType::FLOAT);
  EXPECT_EQ(tensor1->sizes()->size(), 1);
  EXPECT_EQ(tensor1->segment_index(), 0);
  EXPECT_EQ(tensor1->offset(), 16);

  // Test Segments
  auto segment = flat_tensor->segments()->Get(0);

  EXPECT_EQ(segment->offset(), 0);
  EXPECT_EQ(segment->size(), 20);
  uint8_t* data = (uint8_t*)(byte_buffer + segment_offset);
  EXPECT_EQ(*(float*)(data + 0), linear_bias);
  EXPECT_EQ(*(float*)(data + 16), linear_weight);
}
