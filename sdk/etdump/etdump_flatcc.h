/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/span.h>
#include <cstdint>
#include "executorch/runtime/core/event_tracer.h"
#include "executorch/runtime/platform/platform.h"

#define ETDUMP_VERSION 0

struct flatcc_builder;

namespace torch {
namespace executor {

enum ETDumpGen_State {
  ETDumpGen_Init,
  ETDumpGen_Block_Created,
  ETDumpGen_Adding_Allocators,
  ETDumpGen_Adding_Events,
  ETDumpGen_Done,
};

struct etdump_result {
  void* buf;
  size_t size;
};

struct etdump_static_allocator {
  etdump_static_allocator() {}

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

class ETDumpGen : public EventTracer {
 public:
  ETDumpGen(Span<uint8_t> buffer = {nullptr, (size_t)0});
  ~ETDumpGen() override;
  void clear_builder();

  void create_event_block(const char* name) override;
  virtual EventTracerEntry start_profiling(
      const char* name,
      ChainID chain_id = -1,
      DebugHandle debug_handle = 0) override;
  virtual void end_profiling(EventTracerEntry prof_entry) override;
  virtual EventTracerEntry start_profiling_delegate(
      const char* name,
      DebugHandle delegate_debug_index) override;
  virtual void end_profiling_delegate(
      EventTracerEntry prof_entry,
      const void* metadata,
      size_t metadata_len) override;
  virtual void log_profiling_delegate(
      const char* name,
      DebugHandle delegate_debug_index,
      et_timestamp_t start_time,
      et_timestamp_t end_time,
      const void* metadata,
      size_t metadata_len) override;
  virtual void track_allocation(AllocatorID id, size_t size) override;
  virtual AllocatorID track_allocator(const char* name) override;
  virtual void log_evalue(
      const EValue& evalue,
      LoggedEValueType evalue_type =
          LoggedEValueType::kIntermediateOutput) override;
  void set_debug_buffer(Span<uint8_t> buffer);
  etdump_result get_etdump_data();
  size_t get_num_blocks();
  bool is_static_etdump();
  void reset();

 private:
  struct flatcc_builder* builder;
  size_t num_blocks = 0;
  Span<uint8_t> debug_buffer;
  size_t debug_buffer_offset = 0;
  int bundled_input_index = -1;
  ETDumpGen_State etdump_gen_state = ETDumpGen_Init;
  struct etdump_static_allocator alloc;

  void check_ready_to_add_events();
  int64_t create_string_entry(const char* name);
  size_t copy_tensor_to_debug_buffer(exec_aten::Tensor tensor);
};

} // namespace executor
} // namespace torch
