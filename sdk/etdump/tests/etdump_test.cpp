/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/span.h>
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

    etdump_gen = new ETDumpGen();
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

  size_t size = 0;
  void* buf = flatbuffers_read_size_prefix(result.buf, &size);
  etdump_ETDump_table_t etdump =
      etdump_ETDump_as_root_with_identifier(buf, etdump_ETDump_file_identifier);

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

  size_t size = 0;
  void* buf = flatbuffers_read_size_prefix(result.buf, &size);
  etdump_ETDump_table_t etdump =
      etdump_ETDump_as_root_with_identifier(buf, etdump_ETDump_file_identifier);

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

TEST_F(ProfilerETDumpTest, DebugEvent) {
  testing::TensorFactory<ScalarType::Float> tf;
  EValue evalue(tf.ones({3, 2}));

  etdump_gen->create_event_block("test_block");

  void* ptr = malloc(2048);
  Span<uint8_t> buffer((uint8_t*)ptr, 2048);

  etdump_gen->set_debug_buffer(buffer);
  etdump_gen->log_evalue(evalue);
  etdump_gen->log_evalue(evalue, LoggedEValueType::kProgramOutput);

  EValue evalue_int((int64_t)5);
  etdump_gen->log_evalue(evalue_int);

  EValue evalue_double((double)1.5);
  etdump_gen->log_evalue(evalue_double);

  EValue evalue_bool(true);
  etdump_gen->log_evalue(evalue_bool);

  etdump_gen->log_evalue(evalue_bool);

  free(ptr);
}

TEST_F(ProfilerETDumpTest, VerifyLogging) {
  testing::TensorFactory<ScalarType::Float> tf;
  EValue evalue(tf.ones({3, 2}));

  etdump_gen->create_event_block("test_block");

  void* ptr = malloc(2048);
  Span<uint8_t> buffer((uint8_t*)ptr, 2048);

  etdump_gen->set_debug_buffer(buffer);
  etdump_gen->log_evalue(evalue);
  etdump_gen->log_evalue(evalue, LoggedEValueType::kProgramOutput);

  etdump_result result = etdump_gen->get_etdump_data();
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
  ASSERT_EQ(etdump_Event_vec_len(events), 2);

  etdump_Event_table_t event = etdump_Event_vec_at(events, 0);

  etdump_DebugEvent_table_t single_debug_event =
      etdump_Event_debug_event(event);
  etdump_Value_table_t value =
      etdump_DebugEvent_debug_entry(single_debug_event);
  ASSERT_EQ(etdump_Value_tensor_is_present(value), true);
  ASSERT_EQ(etdump_Value_output_is_present(value), false);

  etdump_Tensor_table_t tensor = etdump_Value_tensor(value);
  executorch_flatbuffer_ScalarType_enum_t scalar_enum =
      etdump_Tensor_scalar_type(tensor);
  ASSERT_EQ(scalar_enum, executorch_flatbuffer_ScalarType_FLOAT);
  flatbuffers_int64_vec_t sizes = etdump_Tensor_sizes(tensor);
  ASSERT_EQ(flatbuffers_int64_vec_len(sizes), 2);
  ASSERT_EQ(flatbuffers_int64_vec_at(sizes, 0), 3);
  ASSERT_EQ(flatbuffers_int64_vec_at(sizes, 1), 2);

  event = etdump_Event_vec_at(events, 1);
  single_debug_event = etdump_Event_debug_event(event);
  value = etdump_DebugEvent_debug_entry(single_debug_event);
  ASSERT_EQ(etdump_Value_tensor_is_present(value), true);
  ASSERT_EQ(etdump_Value_output_is_present(value), true);
  etdump_Bool_table_t bool_val = etdump_Value_output_get(value);
  bool bool_val_from_table = etdump_Bool_bool_val(bool_val);
  ASSERT_EQ(bool_val_from_table, true);

  free(ptr);
  free(result.buf);
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

  size_t size = 0;
  void* buf = flatbuffers_read_size_prefix(result.buf, &size);
  etdump_ETDump_table_t etdump =
      etdump_ETDump_as_root_with_identifier(buf, etdump_ETDump_file_identifier);

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

  size_t size = 0;
  void* buf = flatbuffers_read_size_prefix(result.buf, &size);
  etdump_ETDump_table_t etdump =
      etdump_ETDump_as_root_with_identifier(buf, etdump_ETDump_file_identifier);

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
  EXPECT_EQ(etdump_ProfileEvent_chain_index(single_prof_event), 0);

  flatbuffers_string_t allocator_name =
      etdump_Allocator_name(etdump_Allocator_vec_at(allocator_vec, 0));
  EXPECT_EQ(
      std::string(allocator_name, strlen(allocator_name)),
      "single prof allocator");

  free(result.buf);
}

TEST_F(ProfilerETDumpTest, LogDelegateEvents) {
  etdump_gen->create_event_block("test_block");

  // Event 0
  etdump_gen->log_profiling_delegate(nullptr, 276, 1, 2, nullptr);
  // Event 1
  etdump_gen->log_profiling_delegate(nullptr, 278, 1, 2, "test_metadata");
  EventTracerEntry entry = etdump_gen->start_profiling_delegate(
      "test_event", static_cast<torch::executor::DebugHandle>(-1));
  EXPECT_NE(entry.delegate_event_id_type, DelegateDebugIdType::kNone);
  // Event 2
  etdump_gen->end_profiling_delegate(entry, "test_metadata");
  // Event 3
  etdump_gen->log_profiling_delegate(
      "test_event",
      static_cast<torch::executor::DebugHandle>(-1),
      1,
      2,
      nullptr);
  // Event 4
  etdump_gen->log_profiling_delegate(
      "test_event",
      static_cast<torch::executor::DebugHandle>(-1),
      1,
      2,
      "test_metadata");

  // Only a valid name or delegate debug index should be passed in. If valid
  // entries are passed in for both then the test should assert out.
  ET_EXPECT_DEATH(
      etdump_gen->start_profiling_delegate("test_event", 1),
      "Only name or delegate_debug_index can be valid. Check DelegateMappingBuilder documentation for more details.");
  ET_EXPECT_DEATH(
      etdump_gen->log_profiling_delegate("test_event", 1, 1, 2, nullptr),
      "Only name or delegate_debug_index can be valid. Check DelegateMappingBuilder documentation for more details.");
  ET_EXPECT_DEATH(
      etdump_gen->end_profiling(entry),
      "Delegate events must use end_profiling_delegate to mark the end of a delegate profiling event.");

  etdump_result result = etdump_gen->get_etdump_data();
  ASSERT_TRUE(result.buf != nullptr);
  ASSERT_TRUE(result.size != 0);

  // Run verification tests on the data that was just serialized.
  size_t size = 0;
  void* buf = flatbuffers_read_size_prefix(result.buf, &size);
  etdump_ETDump_table_t etdump =
      etdump_ETDump_as_root_with_identifier(buf, etdump_ETDump_file_identifier);
  etdump_RunData_vec_t run_data_vec = etdump_ETDump_run_data(etdump);

  // Event 0
  etdump_RunData_table_t run_data_0 = etdump_RunData_vec_at(run_data_vec, 0);
  etdump_Event_vec_t event_vec = etdump_RunData_events(run_data_0);
  ASSERT_EQ(etdump_Event_vec_len(event_vec), 5);
  etdump_Event_table_t event = etdump_Event_vec_at(event_vec, 0);

  flatbuffers_string_t delegate_debug_id_name =
      etdump_ProfileEvent_delegate_debug_id_str(
          etdump_Event_profile_event(event));

  // Event 0 should have a empty delegate_debug_id_str
  EXPECT_EQ(delegate_debug_id_name, nullptr);
  // Check for the correct delegate_debug_id_int
  EXPECT_EQ(
      etdump_ProfileEvent_delegate_debug_id_int(
          etdump_Event_profile_event(event)),
      276);
  flatbuffers_string_t debug_metadata_name =
      etdump_ProfileEvent_delegate_debug_metadata(
          etdump_Event_profile_event(event));
  // Event 0 should have an empty delegate_debug_metadata string.
  EXPECT_EQ(debug_metadata_name, nullptr);

  // Event 1
  event = etdump_Event_vec_at(event_vec, 1);
  // Check for the correct delegate_debug_id_int
  EXPECT_EQ(
      etdump_ProfileEvent_delegate_debug_id_int(
          etdump_Event_profile_event(event)),
      278);
  debug_metadata_name = etdump_ProfileEvent_delegate_debug_metadata(
      etdump_Event_profile_event(event));
  // Check for the correct delegate_debug_metadata string
  EXPECT_EQ(
      std::string(debug_metadata_name, strlen(debug_metadata_name)),
      "test_metadata");

  // Event 2
  event = etdump_Event_vec_at(event_vec, 2);
  delegate_debug_id_name = etdump_ProfileEvent_delegate_debug_id_str(
      etdump_Event_profile_event(event));
  // Check for the correct delegate_debug_id_str string.
  EXPECT_EQ(
      std::string(delegate_debug_id_name, strlen(delegate_debug_id_name)),
      "test_event");
  // Event 2 used a string delegate debug identifier, so delegate_debug_id_int
  // should be -1.
  EXPECT_EQ(
      etdump_ProfileEvent_delegate_debug_id_int(
          etdump_Event_profile_event(event)),
      -1);
  free(result.buf);
}

} // namespace executor
} // namespace torch
