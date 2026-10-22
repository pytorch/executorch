/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// The allocator registry is a process wide static with no way to remove an
// entry, so the "nothing registered for this device" path needs a binary that
// never registers anything.

#include <gtest/gtest.h>
#include <cstdint>
#include <cstdlib>

#include <executorch/devtools/etdump/data_sinks/buffer_data_sink.h>
#include <executorch/devtools/etdump/etdump_flatcc.h>
#include <executorch/runtime/core/device_allocator.h>
#include <executorch/runtime/core/portable_type/tensor_impl.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/test/utils/DeathTest.h>

using namespace ::executorch::etdump;
using namespace ::executorch::runtime;
using namespace ::executorch::runtime::etensor;

namespace {

TEST(ETDumpNoDeviceAllocatorTest, LogTensorOnUnregisteredDeviceAborts) {
  runtime_init();
  ASSERT_EQ(get_device_allocator(DeviceType::CUDA), nullptr);

  // Host memory, so the test fails by not dying rather than by crashing if
  // ETDump ever reads the pointer instead of reporting the missing allocator.
  float data[] = {1.5f, 2.5f, 3.5f, 4.5f};
  int32_t sizes[] = {4};
  uint8_t dim_order[] = {0};
  int32_t strides[] = {1};
  TensorImpl impl(
      ScalarType::Float,
      1,
      sizes,
      data,
      dim_order,
      strides,
      TensorShapeDynamism::STATIC,
      DeviceType::CUDA,
      0);
  Tensor tensor(&impl);

  const size_t debug_buf_size = 2048;
  void* debug_buf = malloc(debug_buf_size);
  auto buffer_data_sink = BufferDataSink::create(debug_buf, debug_buf_size);
  ASSERT_TRUE(buffer_data_sink.ok());

  ETDumpGen etdump_gen;
  etdump_gen.create_event_block("test_block");
  etdump_gen.set_data_sink(&buffer_data_sink.get());

  ET_EXPECT_DEATH(
      etdump_gen.log_evalue(EValue(tensor)), "No device allocator registered");

  free(debug_buf);
}

} // namespace
