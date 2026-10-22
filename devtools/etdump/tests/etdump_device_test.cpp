/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <executorch/devtools/etdump/data_sinks/buffer_data_sink.h>
#include <executorch/devtools/etdump/etdump_flatcc.h>
#include <executorch/devtools/etdump/etdump_schema_flatcc_builder.h>
#include <executorch/devtools/etdump/etdump_schema_flatcc_reader.h>
#include <executorch/runtime/core/device_allocator.h>
#include <executorch/runtime/core/portable_type/tensor_impl.h>
#include <executorch/runtime/core/test/mock_cuda_allocator.h>
#include <executorch/runtime/platform/runtime.h>

using namespace ::executorch::etdump;
using namespace ::executorch::runtime;
using namespace ::executorch::runtime::etensor;
using namespace ::executorch::runtime::testing;

namespace {

// Backs its device memory with host memory, so the copy path runs without a
// GPU.
MockCudaAllocator g_mock_cuda;

class ETDumpDeviceTest : public ::testing::Test {
 protected:
  // The registry only accepts allocators with static lifetime and aborts on a
  // second registration for the same device type, so register once per binary.
  static void SetUpTestSuite() {
    register_device_allocator(&g_mock_cuda);
  }

  void SetUp() override {
    runtime_init();
    g_mock_cuda.reset();
    etdump_gen_ = new ETDumpGen();
    debug_buf_ = malloc(kDebugBufSize);
  }

  void TearDown() override {
    delete etdump_gen_;
    free(debug_buf_);
  }

  static constexpr size_t kDebugBufSize = 2048;

  ETDumpGen* etdump_gen_ = nullptr;
  void* debug_buf_ = nullptr;
  int32_t sizes_[1] = {4};
  uint8_t dim_order_[1] = {0};
  int32_t strides_[1] = {1};
};

TEST_F(ETDumpDeviceTest, LogTensorOnDeviceCopiesItBackToHost) {
  ASSERT_EQ(get_device_allocator(DeviceType::CUDA), &g_mock_cuda);

  float device_data[] = {1.5f, 2.5f, 3.5f, 4.5f};
  TensorImpl impl(
      ScalarType::Float,
      1,
      sizes_,
      device_data,
      dim_order_,
      strides_,
      TensorShapeDynamism::STATIC,
      DeviceType::CUDA,
      0);
  Tensor tensor(&impl);

  auto buffer_data_sink = BufferDataSink::create(debug_buf_, kDebugBufSize);
  ASSERT_TRUE(buffer_data_sink.ok());

  etdump_gen_->create_event_block("test_block");
  etdump_gen_->set_data_sink(&buffer_data_sink.get());
  etdump_gen_->log_evalue(EValue(tensor));

  EXPECT_EQ(g_mock_cuda.d2h_count_, 1);
  EXPECT_EQ(g_mock_cuda.last_d2h_size_, sizeof(device_data));
  EXPECT_EQ(g_mock_cuda.last_d2h_src_, device_data);
  EXPECT_EQ(g_mock_cuda.last_d2h_index_, 0);

  ETDumpResult result = etdump_gen_->get_etdump_data();
  ASSERT_TRUE(result.buf != nullptr);
  ASSERT_TRUE(result.size != 0);

  size_t size = 0;
  void* buf = flatbuffers_read_size_prefix(result.buf, &size);
  etdump_ETDump_table_t etdump =
      etdump_ETDump_as_root_with_identifier(buf, etdump_ETDump_file_identifier);
  etdump_RunData_vec_t run_data_vec = etdump_ETDump_run_data(etdump);
  ASSERT_EQ(etdump_RunData_vec_len(run_data_vec), 1);
  etdump_Event_vec_t events =
      etdump_RunData_events(etdump_RunData_vec_at(run_data_vec, 0));
  ASSERT_EQ(etdump_Event_vec_len(events), 1);
  etdump_Tensor_table_t logged =
      etdump_Value_tensor(etdump_DebugEvent_debug_entry(
          etdump_Event_debug_event(etdump_Event_vec_at(events, 0))));

  // The sink aligns each blob, so the bytes start at the recorded offset rather
  // than at the front of the buffer.
  const long offset = etdump_Tensor_offset(logged);
  ASSERT_GE(offset, 0);
  EXPECT_EQ(
      memcmp((uint8_t*)debug_buf_ + offset, device_data, sizeof(device_data)),
      0);

  free(result.buf);
}

TEST_F(ETDumpDeviceTest, LogTensorOnCpuDoesNotStageThroughTheAllocator) {
  float host_data[] = {1.5f, 2.5f, 3.5f, 4.5f};
  TensorImpl impl(
      ScalarType::Float,
      1,
      sizes_,
      host_data,
      dim_order_,
      strides_,
      TensorShapeDynamism::STATIC,
      DeviceType::CPU,
      0);
  Tensor tensor(&impl);

  auto buffer_data_sink = BufferDataSink::create(debug_buf_, kDebugBufSize);
  ASSERT_TRUE(buffer_data_sink.ok());

  etdump_gen_->create_event_block("test_block");
  etdump_gen_->set_data_sink(&buffer_data_sink.get());
  etdump_gen_->log_evalue(EValue(tensor));

  EXPECT_EQ(g_mock_cuda.d2h_count_, 0);

  ETDumpResult result = etdump_gen_->get_etdump_data();
  ASSERT_TRUE(result.buf != nullptr);
  ASSERT_TRUE(result.size != 0);
  free(result.buf);
}

} // namespace
