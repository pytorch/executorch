/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

#include <executorch/devtools/etdump/data_sinks/buffer_data_sink.h>
#include <executorch/devtools/etdump/data_sinks/data_sink_base.h>
#include <executorch/runtime/core/event_tracer.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/platform/platform.h>

#define ETDUMP_VERSION 0

struct flatcc_builder;

namespace executorch {
namespace etdump {

using ::executorch::runtime::DelegateDebugIntId;
using ::executorch::runtime::EventTracerFilterBase;
using ::executorch::runtime::Result;

namespace internal {
struct ETDumpStaticAllocator {
  ETDumpStaticAllocator() = default;

  void
  set_buffer(uint8_t* buffer, size_t total_buf_size, size_t alloc_buf_size) {
    data = buffer;
    data_size = alloc_buf_size;
    allocated = 0;
    out_size = total_buf_size - alloc_buf_size;
    // The front of the buffer is the end of the allocation buffer.
    // We start writing from the end of the allocation buffer, and
    // move backwards.
    front_cursor = &buffer[alloc_buf_size + out_size];
    front_left = out_size;
  }

  // Pointer to backing buffer to allocate from.
  uint8_t* data{nullptr};

  // Size of backing buffer.
  size_t data_size{0};

  // Current allocation offset.
  size_t allocated{0};

  // Size of build buffer.
  size_t out_size{0};

  // Pointer to front of build buffer.
  uint8_t* front_cursor{nullptr};

  // Bytes left in front of front_cursor.
  size_t front_left{0};
};
} // namespace internal

struct ETDumpResult {
  void* buf;
  size_t size;
};

class ETDumpGen : public ::executorch::runtime::EventTracer {
 public:
  ETDumpGen(::executorch::runtime::Span<uint8_t> buffer = {nullptr, (size_t)0});
  ~ETDumpGen() override;
  void clear_builder();

  void create_event_block(const char* name) override;
  virtual ::executorch::runtime::EventTracerEntry start_profiling(
      const char* name,
      ::executorch::runtime::ChainID chain_id = -1,
      ::executorch::runtime::DebugHandle debug_handle = 0) override;
  virtual void end_profiling(
      ::executorch::runtime::EventTracerEntry prof_entry) override;
  virtual ::executorch::runtime::EventTracerEntry start_profiling_delegate(
      const char* name,
      DelegateDebugIntId delegate_debug_index) override;
  virtual void end_profiling_delegate(
      ::executorch::runtime::EventTracerEntry prof_entry,
      const void* metadata,
      size_t metadata_len) override;
  virtual void log_profiling_delegate(
      const char* name,
      DelegateDebugIntId delegate_debug_index,
      et_timestamp_t start_time,
      et_timestamp_t end_time,
      const void* metadata,
      size_t metadata_len) override;
  virtual void track_allocation(
      ::executorch::runtime::AllocatorID id,
      size_t size) override;
  virtual ::executorch::runtime::AllocatorID track_allocator(
      const char* name) override;
  virtual Result<bool> log_evalue(
      const ::executorch::runtime::EValue& evalue,
      ::executorch::runtime::LoggedEValueType evalue_type =
          ::executorch::runtime::LoggedEValueType::kIntermediateOutput)
      override;
  /**
   * Log an intermediate tensor output from a delegate.
   */
  virtual Result<bool> log_intermediate_output_delegate(
      const char* name,
      DelegateDebugIntId delegate_debug_index,
      const executorch::aten::Tensor& output) override;

  /**
   * Log an intermediate tensor array output from a delegate.
   */
  virtual Result<bool> log_intermediate_output_delegate(
      const char* name,
      DelegateDebugIntId delegate_debug_index,
      const ::executorch::runtime::ArrayRef<executorch::aten::Tensor> output)
      override;

  /**
   * Log an intermediate int output from a delegate.
   */
  virtual Result<bool> log_intermediate_output_delegate(
      const char* name,
      DelegateDebugIntId delegate_debug_index,
      const int& output) override;

  /**
   * Log an intermediate bool output from a delegate.
   */
  virtual Result<bool> log_intermediate_output_delegate(
      const char* name,
      DelegateDebugIntId delegate_debug_index,
      const bool& output) override;

  /**
   * Log an intermediate double output from a delegate.
   */
  virtual Result<bool> log_intermediate_output_delegate(
      const char* name,
      DelegateDebugIntId delegate_debug_index,
      const double& output) override;

  /**
   * Set the filter of event tracer for delegation intermediate outputs.
   */
  virtual void set_delegation_intermediate_output_filter(
      EventTracerFilterBase* event_tracer_filter) override;

  Result<bool> set_debug_buffer(::executorch::runtime::Span<uint8_t> buffer);
  void set_data_sink(DataSinkBase* data_sink);
  ETDumpResult get_etdump_data();
  size_t get_num_blocks();
  DataSinkBase* get_data_sink();
  bool is_static_etdump();
  void reset();

 private:
  enum class State {
    Init,
    BlockCreated,
    AddingAllocators,
    AddingEvents,
    Done,
  };

  void check_ready_to_add_events();
  int64_t create_string_entry(const char* name);

  /**
   * Templated helper function used to log various types of intermediate output.
   * Supported types include tensor, tensor array, int, bool and double.
   */
  template <typename T>
  Result<bool> log_intermediate_output_delegate_helper(
      const char* name,
      DelegateDebugIntId delegate_debug_index,
      const T& output);

  Result<long> write_tensor_or_return_error(executorch::aten::Tensor tensor);

  struct flatcc_builder* builder_;
  size_t num_blocks_ = 0;
  DataSinkBase* data_sink_;

  // It is only for set_debug_buffer function.
  BufferDataSink buffer_data_sink_;

  int bundled_input_index_ = -1;
  State state_ = State::Init;
  struct internal::ETDumpStaticAllocator alloc_;

  EventTracerFilterBase* filter_ = nullptr;
};

} // namespace etdump
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using etdump_result = ::executorch::etdump::ETDumpResult;
using ::executorch::etdump::ETDumpGen;
} // namespace executor
} // namespace torch
