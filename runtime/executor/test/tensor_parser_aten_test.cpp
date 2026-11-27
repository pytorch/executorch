/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Unit tests for the ATen-mode tensor parser's device handling. These exercise
 * the validation and CPU-tagging paths that run without a real accelerator. The
 * end-to-end CUDA device-pointer path is covered separately by
 * tensor_parser_device_test (portable mode) with a mock allocator; ATen-mode
 * CUDA execution needs a real device and is out of scope here.
 */

#include <executorch/runtime/executor/tensor_parser.h>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/executor/test/managed_memory_manager.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/schema/program_generated.h>

#include <gtest/gtest.h>

#include <vector>

using executorch::aten::Tensor;
using executorch::runtime::Error;
using executorch::runtime::MemoryManager;
using executorch::runtime::Result;
using executorch::runtime::aten::deserialization::parseTensor;
using executorch::runtime::testing::ManagedMemoryManager;

namespace {

constexpr size_t kMemBytes = 32 * 1024U;

// Builds a standalone serialized Tensor with no data_buffer and no
// allocation_info, i.e. an input/placeholder. parseTensor then resolves a null
// data pointer, so these tests reach the device validation and tagging logic
// without needing a Program or any device memory.
const executorch_flatbuffer::Tensor* buildTensor(
    flatbuffers::FlatBufferBuilder& builder,
    executorch_flatbuffer::TensorShapeDynamism dynamism,
    bool with_extra_info,
    executorch_flatbuffer::DeviceType device_type,
    int8_t device_index) {
  flatbuffers::Offset<executorch_flatbuffer::ExtraTensorInfo> extra_info = 0;
  if (with_extra_info) {
    executorch_flatbuffer::ExtraTensorInfoBuilder eti(builder);
    eti.add_device_type(device_type);
    eti.add_device_index(device_index);
    extra_info = eti.Finish();
  }
  const std::vector<int32_t> sizes = {2, 2};
  const std::vector<uint8_t> dim_order = {0, 1};
  const auto sizes_offset = builder.CreateVector(sizes);
  const auto dim_order_offset = builder.CreateVector(dim_order);

  executorch_flatbuffer::TensorBuilder tb(builder);
  tb.add_scalar_type(executorch_flatbuffer::ScalarType::FLOAT);
  tb.add_storage_offset(0);
  tb.add_sizes(sizes_offset);
  tb.add_dim_order(dim_order_offset);
  tb.add_data_buffer_idx(0);
  tb.add_shape_dynamism(dynamism);
  if (with_extra_info) {
    tb.add_extra_tensor_info(extra_info);
  }
  builder.Finish(tb.Finish());
  return flatbuffers::GetRoot<executorch_flatbuffer::Tensor>(
      builder.GetBufferPointer());
}

} // namespace

class TensorParserAtenTest : public ::testing::Test {
 protected:
  void SetUp() override {
    executorch::runtime::runtime_init();
  }
};

// A CPU tensor must stay unindexed. An explicit cpu:0 would mismatch the
// graph's default cpu tensors and trip ATen's same-device check.
TEST_F(TensorParserAtenTest, CpuTensorIsUnindexed) {
  flatbuffers::FlatBufferBuilder builder;
  const auto* s_tensor = buildTensor(
      builder,
      executorch_flatbuffer::TensorShapeDynamism::STATIC,
      /*with_extra_info=*/true,
      executorch_flatbuffer::DeviceType::CPU,
      /*device_index=*/0);

  ManagedMemoryManager mmm(kMemBytes, kMemBytes);
  Result<Tensor> tensor = parseTensor(nullptr, &mmm.get(), s_tensor);
  ASSERT_EQ(tensor.error(), Error::Ok);
  EXPECT_TRUE(tensor->is_cpu());
  EXPECT_FALSE(tensor->device().has_index());
  EXPECT_EQ(tensor->scalar_type(), executorch::aten::ScalarType::Float);
}

// A tensor with no extra_tensor_info (older PTE files) defaults to CPU.
TEST_F(TensorParserAtenTest, MissingExtraInfoDefaultsToCpu) {
  flatbuffers::FlatBufferBuilder builder;
  const auto* s_tensor = buildTensor(
      builder,
      executorch_flatbuffer::TensorShapeDynamism::STATIC,
      /*with_extra_info=*/false,
      executorch_flatbuffer::DeviceType::CPU,
      /*device_index=*/0);

  ManagedMemoryManager mmm(kMemBytes, kMemBytes);
  Result<Tensor> tensor = parseTensor(nullptr, &mmm.get(), s_tensor);
  ASSERT_EQ(tensor.error(), Error::Ok);
  EXPECT_TRUE(tensor->is_cpu());
  EXPECT_FALSE(tensor->device().has_index());
}

// An unmapped DeviceType from the untrusted PTE is rejected before it can reach
// c10::Device as garbage.
TEST_F(TensorParserAtenTest, InvalidDeviceTypeIsRejected) {
  flatbuffers::FlatBufferBuilder builder;
  const auto* s_tensor = buildTensor(
      builder,
      executorch_flatbuffer::TensorShapeDynamism::STATIC,
      /*with_extra_info=*/true,
      static_cast<executorch_flatbuffer::DeviceType>(7),
      /*device_index=*/0);

  ManagedMemoryManager mmm(kMemBytes, kMemBytes);
  EXPECT_EQ(
      parseTensor(nullptr, &mmm.get(), s_tensor).error(),
      Error::InvalidProgram);
}

// A negative accelerator index is not a valid serialized placement.
TEST_F(TensorParserAtenTest, NegativeAcceleratorIndexIsRejected) {
  flatbuffers::FlatBufferBuilder builder;
  const auto* s_tensor = buildTensor(
      builder,
      executorch_flatbuffer::TensorShapeDynamism::STATIC,
      /*with_extra_info=*/true,
      executorch_flatbuffer::DeviceType::CUDA,
      /*device_index=*/-1);

  ManagedMemoryManager mmm(kMemBytes, kMemBytes);
  EXPECT_EQ(
      parseTensor(nullptr, &mmm.get(), s_tensor).error(),
      Error::InvalidProgram);
}

// A fully dynamic tensor gets a CPU-resizable allocator, so a non-CPU device
// cannot be honored and must be rejected rather than silently tagged CPU.
TEST_F(TensorParserAtenTest, DynamicUnboundCudaIsRejected) {
  flatbuffers::FlatBufferBuilder builder;
  const auto* s_tensor = buildTensor(
      builder,
      executorch_flatbuffer::TensorShapeDynamism::DYNAMIC_UNBOUND,
      /*with_extra_info=*/true,
      executorch_flatbuffer::DeviceType::CUDA,
      /*device_index=*/0);

  ManagedMemoryManager mmm(kMemBytes, kMemBytes);
  EXPECT_EQ(
      parseTensor(nullptr, &mmm.get(), s_tensor).error(), Error::NotSupported);
}

// The DYNAMIC_UNBOUND guard only rejects non-CPU; CPU resizable tensors still
// parse so the guard does not regress the common case.
TEST_F(TensorParserAtenTest, DynamicUnboundCpuIsAllowed) {
  flatbuffers::FlatBufferBuilder builder;
  const auto* s_tensor = buildTensor(
      builder,
      executorch_flatbuffer::TensorShapeDynamism::DYNAMIC_UNBOUND,
      /*with_extra_info=*/false,
      executorch_flatbuffer::DeviceType::CPU,
      /*device_index=*/0);

  ManagedMemoryManager mmm(kMemBytes, kMemBytes);
  Result<Tensor> tensor = parseTensor(nullptr, &mmm.get(), s_tensor);
  ASSERT_EQ(tensor.error(), Error::Ok);
  EXPECT_TRUE(tensor->is_cpu());
  EXPECT_FALSE(tensor->device().has_index());
}
