/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/platform/profiler.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/sdk/etdump/etdump_gen.h>
#include "executorch/sdk/etdump/etdump_schema_generated.h"

using namespace etdump;

namespace torch {
namespace executor {

class ETDumpGenTest : public ::testing::Test {
 protected:
  void SetUp() override {
    torch::executor::runtime_init();
    EXECUTORCH_PROFILE_CREATE_BLOCK("default");
  }

  const etdump::ETDump* generate_etdump() {
    ETDumpGen et_dump_gen(allocator);
    prof_result_t prof_result;
    EXECUTORCH_DUMP_PROFILE_RESULTS(&prof_result);

    for (size_t i = 0; i < prof_result.num_blocks; i++) {
      et_dump_gen.CreateProfileBlockEntry((
          prof_header_t*)((uintptr_t)prof_result.prof_data + prof_buf_size * i));
    }

    et_dump_gen.generate_etdump();
    const uint8_t* et_dump_buf = et_dump_gen.get_etdump_data();
    return GetETDump(et_dump_buf);
  }

  uint8_t buf[10240];
  MemoryAllocator allocator{10240, buf};
};

TEST_F(ETDumpGenTest, SimpleDump) {
  ETDumpGen et_dump_gen(allocator);
  et_dump_gen.generate_etdump();
  const uint8_t* et_dump_buf = et_dump_gen.get_etdump_data();
  auto et_dump = GetETDump(et_dump_buf);
  auto identifier = flatbuffers::GetBufferIdentifier(et_dump_buf);
  et_dump->version();
  ASSERT_EQ(identifier[0], 'E');
  ASSERT_EQ(identifier[1], 'D');
}

TEST_F(ETDumpGenTest, SingleProfileEvent) {
  { EXECUTORCH_SCOPE_PROF("test_event"); }

  auto et_dump = generate_etdump();
  auto run_data = et_dump->run_data();
  ASSERT_EQ(run_data->size(), 1);

  auto profile_blocks = run_data->Get(0)->profile_blocks();
  ASSERT_EQ(profile_blocks->size(), 1);
  auto debug_blocks = run_data->Get(0)->debug_blocks();
  ASSERT_EQ(debug_blocks->size(), 0);

  ASSERT_EQ(profile_blocks->Get(0)->name()->str(), "default");
  ASSERT_EQ(profile_blocks->Get(0)->allocators()->size(), 0);

  ASSERT_EQ(profile_blocks->Get(0)->profile_events()->size(), 1);
  ASSERT_EQ(
      profile_blocks->Get(0)->profile_events()->Get(0)->name()->str(),
      "test_event");
}

TEST_F(ETDumpGenTest, SingleMemoryAllocationEvent) {
  uint8_t buf[256];
  MemoryAllocator allocator{256, buf};
  allocator.enable_profiling("test_allocator");

  allocator.allocate(64);

  auto et_dump = generate_etdump();
  auto run_data = et_dump->run_data();
  ASSERT_EQ(run_data->size(), 1);

  auto profile_blocks = run_data->Get(0)->profile_blocks();
  ASSERT_EQ(profile_blocks->size(), 1);

  ASSERT_EQ(profile_blocks->Get(0)->allocators()->size(), 1);
  ASSERT_EQ(
      profile_blocks->Get(0)->allocators()->Get(0)->name()->str(),
      "test_allocator");
  ASSERT_EQ(profile_blocks->Get(0)->allocation_events()->size(), 1);
  ASSERT_EQ(
      profile_blocks->Get(0)->allocation_events()->Get(0)->allocation_size(),
      64);

  ASSERT_EQ(profile_blocks->Get(0)->profile_events()->size(), 0);
}

TEST_F(ETDumpGenTest, MultipleProfileBlocks) {
  { EXECUTORCH_SCOPE_PROF("test_event"); }
  EXECUTORCH_PROFILE_CREATE_BLOCK("default_1");
  auto tok = EXECUTORCH_BEGIN_PROF("test_event_1");
  EXECUTORCH_END_PROF(tok);
  tok = EXECUTORCH_BEGIN_PROF("test_event_2");
  EXECUTORCH_END_PROF(tok);

  auto et_dump = generate_etdump();
  auto run_data = et_dump->run_data();
  ASSERT_EQ(run_data->size(), 1);

  auto profile_blocks = run_data->Get(0)->profile_blocks();
  ASSERT_EQ(profile_blocks->size(), 2);

  ASSERT_EQ(profile_blocks->Get(0)->name()->str(), "default");
  ASSERT_EQ(profile_blocks->Get(1)->name()->str(), "default_1");

  ASSERT_EQ(profile_blocks->Get(0)->profile_events()->size(), 1);
  ASSERT_EQ(profile_blocks->Get(1)->profile_events()->size(), 2);

  ASSERT_EQ(
      profile_blocks->Get(0)->profile_events()->Get(0)->name()->str(),
      "test_event");
  ASSERT_EQ(
      profile_blocks->Get(1)->profile_events()->Get(0)->name()->str(),
      "test_event_1");
  ASSERT_EQ(
      profile_blocks->Get(1)->profile_events()->Get(1)->name()->str(),
      "test_event_2");
}

} // namespace executor
} // namespace torch
