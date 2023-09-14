/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/runtime/platform/runtime.h>
#include <executorch/sdk/etdump/etdump_flatcc.h>
#include <executorch/test/utils/DeathTest.h>
#include <cstdint>
#include <cstring>
#include <memory>

namespace torch {
namespace executor {

class ProfilerETDumpTest : public ::testing::Test {
 protected:
  void SetUp() override {
    torch::executor::runtime_init();
    const size_t buf_size = 8192;
    buf = static_cast<uint8_t*>(malloc(buf_size * sizeof(uint8_t)));

    etdump_gen = new ETDumpGen(buf, buf_size);
  }

  void TearDown() override {
    free(buf);
    delete etdump_gen;
  }

  ETDumpGen* etdump_gen;
  uint8_t* buf = nullptr;
};

TEST_F(ProfilerETDumpTest, SingleProfileEvent) {
  etdump_gen->create_event_block("test_block");
  EventTracerEntry entry = etdump_gen->start_profiling("test_event", 0, 1);
  etdump_gen->end_profiling(entry);

  etdump_result result = etdump_gen->get_etdump_data();
  ASSERT_TRUE(result.buf != nullptr);
  ASSERT_TRUE(result.size != 0);

  etdump_ETDump_table_t etdump = etdump_ETDump_as_root_with_identifier(
      result.buf, etdump_ETDump_file_identifier);

  ASSERT_NE(etdump, nullptr);
  EXPECT_EQ(etdump_ETDump_version(etdump), ETDUMP_VERSION);

  etdump_RunData_vec_t run_data_vec = etdump_ETDump_run_data(etdump);
  EXPECT_EQ(etdump_gen->get_num_blocks(), etdump_RunData_vec_len(run_data_vec));

  etdump_RunData_table_t run_data_single_prof =
      etdump_RunData_vec_at(run_data_vec, 0);
  EXPECT_EQ(
      std::string(
          etdump_RunData_name(run_data_single_prof),
          strlen(etdump_RunData_name(run_data_single_prof))),
      "test_block");

  free(result.buf);
}

TEST_F(ProfilerETDumpTest, MultipleProfileEvent) {
  etdump_gen->create_event_block("test_block");

  // Create the profile events and then add the actual profile events in
  // reverse.
  EventTracerEntry entry_1 = etdump_gen->start_profiling("test_event_1", 0, 1);
  EventTracerEntry entry_2 = etdump_gen->start_profiling("test_event_2", 0, 2);

  etdump_gen->end_profiling(entry_2);
  etdump_gen->end_profiling(entry_1);
}

TEST_F(ProfilerETDumpTest, EmptyBlocks) {
  etdump_gen->create_event_block("test_block");
  etdump_gen->create_event_block("test_block_1");
  etdump_gen->create_event_block("test_block_2");

  EventTracerEntry entry = etdump_gen->start_profiling("test_event_1", 0, 1);
  etdump_gen->end_profiling(entry);

  etdump_result result = etdump_gen->get_etdump_data();
  ASSERT_TRUE(result.buf != nullptr);
  ASSERT_TRUE(result.size != 0);

  etdump_ETDump_table_t etdump = etdump_ETDump_as_root_with_identifier(
      result.buf, etdump_ETDump_file_identifier);

  etdump_RunData_vec_t run_data_vec = etdump_ETDump_run_data(etdump);
  ASSERT_EQ(etdump_RunData_vec_len(run_data_vec), 3);
  ASSERT_EQ(
      etdump_Event_vec_len(
          etdump_RunData_events(etdump_RunData_vec_at(run_data_vec, 0))),
      0);

  free(result.buf);
}

TEST_F(ProfilerETDumpTest, AddAllocators) {
  etdump_gen->create_event_block("test_block");
  AllocatorID allocator_id = etdump_gen->track_allocator("test_allocator");
  EXPECT_EQ(allocator_id, 1);
  allocator_id = etdump_gen->track_allocator("test_allocator_1");
  EXPECT_EQ(allocator_id, 2);

  // Add a profiling event and then try to add an allocator which should fail.
  EventTracerEntry entry = etdump_gen->start_profiling("test_event", 0, 1);
  etdump_gen->end_profiling(entry);
  ET_EXPECT_DEATH(etdump_gen->track_allocator("test_allocator"), "");
}

TEST_F(ProfilerETDumpTest, AllocationEvents) {
  etdump_gen->create_event_block("test_block");

  // Add allocation events.
  etdump_gen->track_allocation(1, 64);
  etdump_gen->track_allocation(2, 128);

  // Add a mix of performance and memory events.
  etdump_gen->track_allocation(1, 64);
  EventTracerEntry entry = etdump_gen->start_profiling("test_event", 0, 1);
  etdump_gen->end_profiling(entry);
  etdump_gen->track_allocation(2, 128);
}

TEST_F(ProfilerETDumpTest, MultipleBlocksWithEvents) {
  etdump_gen->create_event_block("test_block");

  AllocatorID allocator_id_0 = etdump_gen->track_allocator("test_allocator_0");
  AllocatorID allocator_id_1 = etdump_gen->track_allocator("test_allocator_1");
  etdump_gen->track_allocation(allocator_id_0, 64);
  etdump_gen->track_allocation(allocator_id_1, 128);

  EventTracerEntry entry = etdump_gen->start_profiling("test_event", 0, 1);
  etdump_gen->end_profiling(entry);
  etdump_gen->create_event_block("test_block_1");
  allocator_id_0 = etdump_gen->track_allocator("test_allocator_0");
  allocator_id_1 = etdump_gen->track_allocator("test_allocator_1");
  etdump_gen->track_allocation(allocator_id_0, 64);
  etdump_gen->track_allocation(allocator_id_0, 128);

  entry = etdump_gen->start_profiling("test_event", 0, 1);
  etdump_gen->end_profiling(entry);

  etdump_result result = etdump_gen->get_etdump_data();
  ASSERT_TRUE(result.buf != nullptr);
  ASSERT_TRUE(result.size != 0);

  etdump_ETDump_table_t etdump = etdump_ETDump_as_root_with_identifier(
      result.buf, etdump_ETDump_file_identifier);

  ASSERT_NE(etdump, nullptr);
  EXPECT_EQ(etdump_ETDump_version(etdump), ETDUMP_VERSION);

  etdump_RunData_vec_t run_data_vec = etdump_ETDump_run_data(etdump);
  ASSERT_EQ(etdump_gen->get_num_blocks(), etdump_RunData_vec_len(run_data_vec));

  etdump_RunData_table_t run_data_0 = etdump_RunData_vec_at(run_data_vec, 0);
  EXPECT_EQ(
      std::string(
          etdump_RunData_name(run_data_0),
          strlen(etdump_RunData_name(run_data_0))),
      "test_block");

  etdump_Allocator_vec_t allocator_vec_0 =
      etdump_RunData_allocators(run_data_0);
  ASSERT_EQ(etdump_Allocator_vec_len(allocator_vec_0), 2);

  etdump_Allocator_table_t allocator_0 =
      etdump_Allocator_vec_at(allocator_vec_0, 0);
  EXPECT_EQ(
      std::string(
          etdump_Allocator_name(allocator_0),
          strlen(etdump_Allocator_name(allocator_0))),
      "test_allocator_0");

  etdump_Event_vec_t event_vec = etdump_RunData_events(run_data_0);
  ASSERT_EQ(etdump_Event_vec_len(event_vec), 3);

  etdump_Event_table_t event_0 = etdump_Event_vec_at(event_vec, 0);
  EXPECT_EQ(
      etdump_AllocationEvent_allocation_size(
          etdump_Event_allocation_event(event_0)),
      64);

  etdump_Event_table_t event_2 = etdump_Event_vec_at(event_vec, 2);
  flatbuffers_string_t event_2_name =
      etdump_ProfileEvent_name(etdump_Event_profile_event(event_2));
  EXPECT_EQ(std::string(event_2_name, strlen(event_2_name)), "test_event");

  free(result.buf);
}

TEST_F(ProfilerETDumpTest, VerifyData) {
  etdump_gen->create_event_block("test_block");

  etdump_gen->track_allocator("single prof allocator");

  EventTracerEntry entry = etdump_gen->start_profiling("test_event", 0, 1);
  etdump_gen->end_profiling(entry);
  entry = etdump_gen->start_profiling("test_event2", 0, 1);
  etdump_gen->end_profiling(entry);

  etdump_result result = etdump_gen->get_etdump_data();
  ASSERT_TRUE(result.buf != nullptr);
  ASSERT_TRUE(result.size != 0);

  etdump_ETDump_table_t etdump = etdump_ETDump_as_root_with_identifier(
      result.buf, etdump_ETDump_file_identifier);

  ASSERT_NE(etdump, nullptr);
  EXPECT_EQ(etdump_ETDump_version(etdump), ETDUMP_VERSION);

  etdump_RunData_vec_t run_data_vec = etdump_ETDump_run_data(etdump);
  EXPECT_EQ(etdump_gen->get_num_blocks(), etdump_RunData_vec_len(run_data_vec));

  etdump_RunData_table_t run_data_single_prof =
      etdump_RunData_vec_at(run_data_vec, 0);
  EXPECT_EQ(
      std::string(
          etdump_RunData_name(run_data_single_prof),
          strlen(etdump_RunData_name(run_data_single_prof))),
      "test_block");

  etdump_Allocator_vec_t allocator_vec =
      etdump_RunData_allocators(run_data_single_prof);

  etdump_Event_table_t single_event =
      etdump_Event_vec_at(etdump_RunData_events(run_data_single_prof), 0);

  etdump_ProfileEvent_table_t single_prof_event =
      etdump_Event_profile_event(single_event);

  EXPECT_EQ(
      std::string(
          etdump_ProfileEvent_name(single_prof_event),
          strlen(etdump_ProfileEvent_name(single_prof_event))),
      "test_event");
  EXPECT_EQ(etdump_ProfileEvent_chain_id(single_prof_event), 0);

  flatbuffers_string_t allocator_name =
      etdump_Allocator_name(etdump_Allocator_vec_at(allocator_vec, 0));
  EXPECT_EQ(
      std::string(allocator_name, strlen(allocator_name)),
      "single prof allocator");

  free(result.buf);
}

} // namespace executor
} // namespace torch
