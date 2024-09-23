/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

#include <executorch/runtime/core/event_tracer.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/platform/platform.h>

#define ETDUMP_VERSION 0

struct flatcc_builder;

namespace executorch {
namespace etdump {

namespace internal {
struct ETDumpStaticAllocator {
  ETDumpStaticAllocator() = default;

  void
  set_buffer(uint8_t* buffer, size_t total_buf_size, size_t alloc_buf_size) {
    data = buffer;
    data_size = alloc_buf_size;
    allocated = 0;
    out_size = total_buf_size - alloc_buf_size;
    front_cursor = &buffer[alloc_buf_size];
    front_left = out_size / 2;
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
      ::executorch::runtime::DebugHandle delegate_debug_index) override;
  virtual void end_profiling_delegate(
      ::executorch::runtime::EventTracerEntry prof_entry,
      const void* metadata,
      size_t metadata_len) override;
  virtual void log_profiling_delegate(
      const char* name,
      ::executorch::runtime::DebugHandle delegate_debug_index,
      et_timestamp_t start_time,
      et_timestamp_t end_time,
      const void* metadata,
      size_t metadata_len) override;
  virtual void track_allocation(
      ::executorch::runtime::AllocatorID id,
      size_t size) override;
  virtual ::executorch::runtime::AllocatorID track_allocator(
      const char* name) override;
  virtual void log_evalue(
      const ::executorch::runtime::EValue& evalue,
      ::executorch::runtime::LoggedEValueType evalue_type =
          ::executorch::runtime::LoggedEValueType::kIntermediateOutput)
      override;
  /**
   * Log an intermediate tensor output from a delegate.
   */
  virtual void log_intermediate_output_delegate(
      const char* name,
      ::executorch::runtime::DebugHandle delegate_debug_index,
      const exec_aten::Tensor& output) override;

  /**
   * Log an intermediate tensor array output from a delegate.
   */
  virtual void log_intermediate_output_delegate(
      const char* name,
      ::executorch::runtime::DebugHandle delegate_debug_index,
      const ::executorch::runtime::ArrayRef<exec_aten::Tensor> output) override;

  /**
   * Log an intermediate int output from a delegate.
   */
  virtual void log_intermediate_output_delegate(
      const char* name,
      ::executorch::runtime::DebugHandle delegate_debug_index,
      const int& output) override;

  /**
   * Log an intermediate bool output from a delegate.
   */
  virtual void log_intermediate_output_delegate(
      const char* name,
      ::executorch::runtime::DebugHandle delegate_debug_index,
      const bool& output) override;

  /**
   * Log an intermediate double output from a delegate.
   */
  virtual void log_intermediate_output_delegate(
      const char* name,
      ::executorch::runtime::DebugHandle delegate_debug_index,
      const double& output) override;
  void set_debug_buffer(::executorch::runtime::Span<uint8_t> buffer);
  ETDumpResult get_etdump_data();
  size_t get_num_blocks();
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
  size_t copy_tensor_to_debug_buffer(exec_aten::Tensor tensor);

  /**
   * Templated helper function used to log various types of intermediate output.
   * Supported types include tensor, tensor array, int, bool and double.
   */
  template <typename T>
  void log_intermediate_output_delegate_helper(
      const char* name,
      ::executorch::runtime::DebugHandle delegate_debug_index,
      const T& output);

  struct flatcc_builder* builder_;
  size_t num_blocks_ = 0;
  ::executorch::runtime::Span<uint8_t> debug_buffer_;
  size_t debug_buffer_offset_ = 0;
  int bundled_input_index_ = -1;
  State state_ = State::Init;
  struct internal::ETDumpStaticAllocator alloc_;
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
