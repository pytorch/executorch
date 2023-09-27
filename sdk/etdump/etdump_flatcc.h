/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/sdk/etdump/etdump_schema_flatcc_builder.h>
#include <executorch/sdk/etdump/etdump_schema_flatcc_reader.h>
#include "executorch/runtime/core/event_tracer.h"
#include "executorch/runtime/platform/platform.h"

#define ETDUMP_VERSION 0

namespace torch {
namespace executor {

enum ETDumpGen_State {
  ETDumpGen_Init,
  ETDumpGen_Block_Created,
  ETDumpGen_Adding_Allocators,
  ETDumpGen_Adding_Events,
};

struct etdump_result {
  void* buf;
  size_t size;
};

class ETDumpGen : public EventTracer {
 public:
  ETDumpGen(void* buffer, size_t buf_size);

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
      const char* metadata) override;
  virtual void log_profiling_delegate(
      const char* name,
      DebugHandle delegate_debug_index,
      et_timestamp_t start_time,
      et_timestamp_t end_time,
      const char* metadata) override;
  virtual void track_allocation(AllocatorID id, size_t size) override;
  virtual AllocatorID track_allocator(const char* name) override;
  etdump_result get_etdump_data();
  size_t get_num_blocks();

 private:
  flatcc_builder_t builder;
  size_t num_blocks = 0;
  ETDumpGen_State etdump_gen_state = ETDumpGen_Init;

  void check_ready_to_add_events();
  int64_t create_string_entry(const char* name);
};

} // namespace executor
} // namespace torch
