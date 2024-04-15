/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "executorch/sdk/etdump/etdump_flatcc.h"
#include <executorch/sdk/etdump/etdump_schema_flatcc_builder.h>
#include <executorch/sdk/etdump/etdump_schema_flatcc_reader.h>
#include <flatcc/flatcc_types.h>
#include <stdio.h>
#include <string.h>
#include "executorch/runtime/core/exec_aten/exec_aten.h"
#include "executorch/runtime/core/exec_aten/util/scalar_type_util.h"
#include "executorch/runtime/platform/assert.h"
#include "executorch/sdk/etdump/emitter.h"

namespace torch {
namespace executor {

namespace {

executorch_flatbuffer_ScalarType_enum_t get_flatbuffer_scalar_type(
    exec_aten::ScalarType tensor_scalar_type) {
  switch (tensor_scalar_type) {
    case exec_aten::ScalarType::Byte:
      return executorch_flatbuffer_ScalarType_BYTE;
    case exec_aten::ScalarType::Char:
      return executorch_flatbuffer_ScalarType_CHAR;
    case exec_aten::ScalarType::Short:
      return executorch_flatbuffer_ScalarType_SHORT;
    case exec_aten::ScalarType::Float:
      return executorch_flatbuffer_ScalarType_FLOAT;
    case exec_aten::ScalarType::Int:
      return executorch_flatbuffer_ScalarType_INT;
    case exec_aten::ScalarType::Long:
      return executorch_flatbuffer_ScalarType_LONG;
    case exec_aten::ScalarType::Double:
      return executorch_flatbuffer_ScalarType_DOUBLE;
    case exec_aten::ScalarType::Bool:
      return executorch_flatbuffer_ScalarType_BOOL;
    default:
      ET_CHECK_MSG(
          0,
          "This ScalarType = %hhd is not yet supported in ETDump",
          static_cast<char>(tensor_scalar_type));
  }
}

etdump_Tensor_ref_t add_tensor_entry(
    flatcc_builder_t* builder,
    const exec_aten::Tensor& tensor,
    long offset) {
  etdump_Tensor_start(builder);

  etdump_Tensor_scalar_type_add(
      builder, get_flatbuffer_scalar_type(tensor.scalar_type()));
  etdump_Tensor_sizes_start(builder);

  for (auto dim : tensor.sizes()) {
    int64_t cast_dim = static_cast<int64_t>(dim);
    etdump_Tensor_sizes_push(builder, &cast_dim);
  }
  etdump_Tensor_sizes_end(builder);

  etdump_Tensor_strides_start(builder);
  for (auto dim : tensor.strides()) {
    int64_t cast_dim = static_cast<int64_t>(dim);
    etdump_Tensor_strides_push(builder, &cast_dim);
  }
  etdump_Tensor_strides_end(builder);
  etdump_Tensor_offset_add(builder, offset);

  return etdump_Tensor_end(builder);
}

static uint8_t* alignPointer(void* ptr, size_t alignment) {
  intptr_t addr = reinterpret_cast<intptr_t>(ptr);
  if ((addr & (alignment - 1)) == 0) {
    // Already aligned.
    return reinterpret_cast<uint8_t*>(ptr);
  }
  addr = (addr | (alignment - 1)) + 1;
  return reinterpret_cast<uint8_t*>(addr);
}

} // namespace

constexpr size_t max_alloc_buf_size = 128 * 1024;

// Constructor implementation
ETDumpGen::ETDumpGen(Span<uint8_t> buffer) {
  // Initialize the flatcc builder using the buffer and buffer size.

  if (buffer.data() != nullptr) {
    builder = (struct flatcc_builder*)alignPointer(buffer.data(), 64);
    uintptr_t buffer_with_builder =
        (uintptr_t)alignPointer(builder + sizeof(struct flatcc_builder), 64);
    size_t buffer_size = buffer.size() -
        (size_t)(buffer_with_builder - (uintptr_t)buffer.data());
    alloc.set_buffer(
        (uint8_t*)buffer_with_builder,
        buffer_size,
        (size_t)((buffer_size / 4 > max_alloc_buf_size) ? max_alloc_buf_size
                                                        : buffer_size / 4));
    et_flatcc_custom_init(builder, &alloc);
  } else {
    builder = (struct flatcc_builder*)malloc(sizeof(struct flatcc_builder));
    ET_CHECK_MSG(
        builder != nullptr, "Failed to allocate memory for flatcc builder.");
    flatcc_builder_init(builder);
  }
  flatbuffers_buffer_start(builder, etdump_ETDump_file_identifier);
  etdump_ETDump_start_as_root_with_size(builder);
  etdump_ETDump_version_add(builder, ETDUMP_VERSION);
  etdump_ETDump_run_data_start(builder);
  etdump_ETDump_run_data_push_start(builder);
}

ETDumpGen::~ETDumpGen() {
  flatcc_builder_clear(builder);
  if (!is_static_etdump()) {
    free(builder);
  }
}

void ETDumpGen::clear_builder() {
  flatcc_builder_clear(builder);
}

void ETDumpGen::create_event_block(const char* name) {
  if (etdump_gen_state == ETDumpGen_Adding_Events) {
    etdump_RunData_events_end(builder);
  }
  if (num_blocks > 0) {
    etdump_ETDump_run_data_push_end(builder);
    etdump_ETDump_run_data_push_start(builder);
  }
  ++num_blocks;
  etdump_RunData_name_create_strn(builder, name, strlen(name));
  if (bundled_input_index != -1) {
    etdump_RunData_bundled_input_index_add(builder, bundled_input_index);
  }
  etdump_gen_state = ETDumpGen_Block_Created;
}

int64_t ETDumpGen::create_string_entry(const char* name) {
  return flatbuffers_string_create_str(builder, name);
}

// ETDumpGen has the following possible states, ETDumpGen_Init,
// ETDumpGen_Block_Created, ETDumpGen_Adding_Allocators,
// ETDumpGen_Adding_Events. Right after boot-up the state of ETDump will be
// ETDumpGen_Init. At this point we have an option of adding allocators that
// we want to track. Once we've completed adding the allocators we want to track
// we will close the allocators table and move ETDumpGen to the
// ETDumpGen_Adding_Events state. After this point we can start adding events to
// ETDump as we wish.
// The reason we need to maintain this state machine inside of ETDumpGen is
// because, once a table of one type has been closed and another table of a
// different type is opened after it we cannot open another table of the first
// type again. In this case once we close the allocators table and start pushing
// to the events table we cannot push to the allocators table again.
void ETDumpGen::check_ready_to_add_events() {
  if (etdump_gen_state != ETDumpGen_Adding_Events) {
    ET_CHECK_MSG(
        (etdump_gen_state == ETDumpGen_Adding_Allocators ||
         etdump_gen_state == ETDumpGen_Block_Created),
        "ETDumpGen in an invalid state. Cannot add new events now.");
    if (etdump_gen_state == ETDumpGen_Adding_Allocators) {
      etdump_RunData_allocators_end(builder);
    }
    etdump_RunData_events_start(builder);
    etdump_gen_state = ETDumpGen_Adding_Events;
  }
}

EventTracerEntry ETDumpGen::start_profiling(
    const char* name,
    ChainID chain_id,
    DebugHandle debug_handle) {
  EventTracerEntry prof_entry;
  prof_entry.event_id = name != nullptr ? create_string_entry(name) : -1;
  prof_entry.delegate_event_id_type = DelegateDebugIdType::kNone;

  if (chain_id == -1) {
    prof_entry.chain_id = chain_id_;
    prof_entry.debug_handle = debug_handle_;
  } else {
    prof_entry.chain_id = chain_id;
    prof_entry.debug_handle = debug_handle;
  }
  prof_entry.start_time = et_pal_current_ticks();
  return prof_entry;
}

// TODO: Update all occurrences of the ProfileEvent calls once the
// EventTracerEntry struct is updated.
EventTracerEntry ETDumpGen::start_profiling_delegate(
    const char* name,
    DebugHandle delegate_debug_index) {
  ET_CHECK_MSG(
      (name == nullptr) ^ (delegate_debug_index == -1),
      "Only name or delegate_debug_index can be valid. Check DelegateMappingBuilder documentation for more details.");
  check_ready_to_add_events();
  EventTracerEntry prof_entry;
  DelegateDebugIdType delegate_event_id_type =
      name == nullptr ? DelegateDebugIdType::kInt : DelegateDebugIdType::kStr;
  prof_entry.delegate_event_id_type = delegate_event_id_type;
  prof_entry.chain_id = chain_id_;
  prof_entry.debug_handle = debug_handle_;
  prof_entry.event_id = delegate_debug_index == static_cast<unsigned int>(-1)
      ? create_string_entry(name)
      : delegate_debug_index;
  prof_entry.start_time = et_pal_current_ticks();
  return prof_entry;
}

void ETDumpGen::end_profiling_delegate(
    EventTracerEntry event_tracer_entry,
    const void* metadata,
    size_t metadata_len) {
  et_timestamp_t end_time = et_pal_current_ticks();
  check_ready_to_add_events();

  // Start building the ProfileEvent entry.
  etdump_ProfileEvent_start(builder);
  etdump_ProfileEvent_start_time_add(builder, event_tracer_entry.start_time);
  etdump_ProfileEvent_end_time_add(builder, end_time);
  etdump_ProfileEvent_chain_index_add(builder, chain_id_);
  etdump_ProfileEvent_instruction_id_add(builder, debug_handle_);
  // Delegate debug identifier can either be of a string type or an integer
  // type. If it's a string type then it's a value of type
  // flatbuffers_string_ref_t type, whereas if it's an integer type then we
  // write the integer value directly.
  if (event_tracer_entry.delegate_event_id_type == DelegateDebugIdType::kInt) {
    etdump_ProfileEvent_delegate_debug_id_int_add(
        builder, event_tracer_entry.event_id);
  } else {
    etdump_ProfileEvent_delegate_debug_id_str_add(
        builder, event_tracer_entry.event_id);
  }
  flatbuffers_uint8_vec_ref_t vec_ref = flatbuffers_uint8_vec_create_pe(
      builder, (const uint8_t*)metadata, metadata_len);
  etdump_ProfileEvent_delegate_debug_metadata_add(builder, vec_ref);
  etdump_ProfileEvent_ref_t id = etdump_ProfileEvent_end(builder);
  etdump_RunData_events_push_start(builder);
  etdump_Event_profile_event_add(builder, id);
  etdump_RunData_events_push_end(builder);
}

void ETDumpGen::log_profiling_delegate(
    const char* name,
    DebugHandle delegate_debug_index,
    et_timestamp_t start_time,
    et_timestamp_t end_time,
    const void* metadata,
    size_t metadata_len) {
  ET_CHECK_MSG(
      (name == nullptr) ^ (delegate_debug_index == -1),
      "Only name or delegate_debug_index can be valid. Check DelegateMappingBuilder documentation for more details.");
  check_ready_to_add_events();
  int64_t string_id = name != nullptr ? create_string_entry(name) : -1;
  etdump_ProfileEvent_start(builder);
  etdump_ProfileEvent_start_time_add(builder, start_time);
  etdump_ProfileEvent_end_time_add(builder, end_time);
  etdump_ProfileEvent_chain_index_add(builder, chain_id_);
  etdump_ProfileEvent_instruction_id_add(builder, debug_handle_);
  if (string_id == -1) {
    etdump_ProfileEvent_delegate_debug_id_int_add(
        builder, delegate_debug_index);
  } else {
    etdump_ProfileEvent_delegate_debug_id_str_add(builder, string_id);
  }
  flatbuffers_uint8_vec_ref_t vec_ref = flatbuffers_uint8_vec_create_pe(
      builder, (const uint8_t*)metadata, metadata_len);
  etdump_ProfileEvent_delegate_debug_metadata_add(builder, vec_ref);
  etdump_ProfileEvent_ref_t id = etdump_ProfileEvent_end(builder);
  etdump_RunData_events_push_start(builder);
  etdump_Event_profile_event_add(builder, id);
  etdump_RunData_events_push_end(builder);
}

void ETDumpGen::end_profiling(EventTracerEntry prof_entry) {
  et_timestamp_t end_time = et_pal_current_ticks();
  ET_CHECK_MSG(
      prof_entry.delegate_event_id_type == DelegateDebugIdType::kNone,
      "Delegate events must use end_profiling_delegate to mark the end of a delegate profiling event.");
  check_ready_to_add_events();

  etdump_ProfileEvent_start(builder);
  etdump_ProfileEvent_start_time_add(builder, prof_entry.start_time);
  etdump_ProfileEvent_end_time_add(builder, end_time);
  etdump_ProfileEvent_chain_index_add(builder, prof_entry.chain_id);
  etdump_ProfileEvent_instruction_id_add(builder, prof_entry.debug_handle);
  if (prof_entry.event_id != -1) {
    etdump_ProfileEvent_name_add(builder, prof_entry.event_id);
  }
  etdump_ProfileEvent_ref_t id = etdump_ProfileEvent_end(builder);
  etdump_RunData_events_push_start(builder);
  etdump_Event_profile_event_add(builder, id);
  etdump_RunData_events_push_end(builder);
}

AllocatorID ETDumpGen::track_allocator(const char* name) {
  ET_CHECK_MSG(
      (etdump_gen_state == ETDumpGen_Block_Created ||
       etdump_gen_state == ETDumpGen_Adding_Allocators),
      "Allocators can only be added immediately after a new block is created and before any events are added.");
  if (etdump_gen_state != ETDumpGen_Adding_Allocators) {
    etdump_RunData_allocators_start(builder);
    etdump_gen_state = ETDumpGen_Adding_Allocators;
  }
  flatbuffers_string_ref_t ref = create_string_entry(name);
  etdump_RunData_allocators_push_create(builder, ref);
  return etdump_RunData_allocators_reserved_len(builder);
}

void ETDumpGen::track_allocation(
    AllocatorID allocator_id,
    size_t allocation_size) {
  check_ready_to_add_events();

  etdump_RunData_events_push_start(builder);
  etdump_Event_allocation_event_create(builder, allocator_id, allocation_size);
  etdump_RunData_events_push_end(builder);
}

etdump_result ETDumpGen::get_etdump_data() {
  etdump_result result;
  if (etdump_gen_state == ETDumpGen_Adding_Events) {
    etdump_RunData_events_end(builder);
  } else if (etdump_gen_state == ETDumpGen_Adding_Allocators) {
    etdump_RunData_allocators_end(builder);
  } else if (etdump_gen_state == ETDumpGen_Init) {
    result.buf = nullptr;
    result.size = 0;
    return result;
  }
  etdump_ETDump_run_data_push_end(builder);
  etdump_ETDump_run_data_end(builder);
  etdump_ETDump_ref_t root = etdump_ETDump_end(builder);
  flatbuffers_buffer_end(builder, root);
  if (num_blocks == 0) {
    result = {nullptr, 0};
  } else {
    if (alloc.data) {
      result.buf = alloc.front_cursor;
      result.size = alloc.out_size - alloc.front_left;
    } else {
      result.buf =
          flatcc_builder_finalize_aligned_buffer(builder, &result.size);
    }
  }
  return result;
}

void ETDumpGen::set_debug_buffer(Span<uint8_t> buffer) {
  debug_buffer = buffer;
}

size_t ETDumpGen::copy_tensor_to_debug_buffer(exec_aten::Tensor tensor) {
  if (tensor.nbytes() == 0) {
    return static_cast<size_t>(-1);
  }
  uint8_t* offset_ptr =
      alignPointer(debug_buffer.data() + debug_buffer_offset, 64);
  debug_buffer_offset = (offset_ptr - debug_buffer.data()) + tensor.nbytes();
  ET_CHECK_MSG(
      debug_buffer_offset <= debug_buffer.size(),
      "Ran out of space to store intermediate outputs.");
  memcpy(offset_ptr, tensor.const_data_ptr(), tensor.nbytes());
  return (size_t)(offset_ptr - debug_buffer.data());
}

void ETDumpGen::log_evalue(const EValue& evalue, LoggedEValueType evalue_type) {
  if (debug_buffer.empty()) {
    return;
  }

  check_ready_to_add_events();

  etdump_DebugEvent_start(builder);

  etdump_DebugEvent_chain_index_add(builder, chain_id_);
  etdump_DebugEvent_instruction_id_add(builder, debug_handle_);

  switch (evalue.tag) {
    case Tag::Tensor: {
      exec_aten::Tensor tensor = evalue.toTensor();
      long offset = copy_tensor_to_debug_buffer(tensor);
      etdump_Tensor_ref_t tensor_ref =
          add_tensor_entry(builder, tensor, offset);

      etdump_Value_start(builder);
      etdump_Value_val_add(builder, etdump_ValueType_Tensor);
      etdump_Value_tensor_add(builder, tensor_ref);
      if (evalue_type == LoggedEValueType::kProgramOutput) {
        auto bool_ref = etdump_Bool_create(builder, FLATBUFFERS_TRUE);
        etdump_Value_output_add(builder, bool_ref);
      }
      auto value_ref = etdump_Value_end(builder);

      etdump_DebugEvent_debug_entry_add(builder, value_ref);
      break;
    }

    case Tag::ListTensor: {
      exec_aten::ArrayRef<exec_aten::Tensor> tensors = evalue.toTensorList();
      etdump_Tensor_vec_start(builder);
      for (size_t i = 0; i < tensors.size(); ++i) {
        long offset = copy_tensor_to_debug_buffer(tensors[i]);
        etdump_Tensor_vec_push(
            builder, add_tensor_entry(builder, tensors[i], offset));
      }
      etdump_Tensor_vec_ref_t tensor_vec_ref = etdump_Tensor_vec_end(builder);
      etdump_TensorList_ref_t tensor_list_ref =
          etdump_TensorList_create(builder, tensor_vec_ref);

      etdump_Value_start(builder);
      etdump_Value_val_add(builder, etdump_ValueType_TensorList);
      etdump_Value_tensor_list_add(builder, tensor_list_ref);
      if (evalue_type == LoggedEValueType::kProgramOutput) {
        auto bool_ref = etdump_Bool_create(builder, FLATBUFFERS_TRUE);
        etdump_Value_output_add(builder, bool_ref);
      }
      auto value_ref = etdump_Value_end(builder);

      etdump_DebugEvent_debug_entry_add(builder, value_ref);
      break;
    }

    case Tag::Int: {
      int64_t val = evalue.toInt();
      auto int_ref = etdump_Int_create(builder, val);

      etdump_Value_start(builder);
      etdump_Value_val_add(builder, etdump_ValueType_Int);
      etdump_Value_int_value_add(builder, int_ref);
      auto value_ref = etdump_Value_end(builder);
      etdump_DebugEvent_debug_entry_add(builder, value_ref);

      break;
    }

    case Tag::Double: {
      double val = evalue.toDouble();
      auto double_ref = etdump_Double_create(builder, val);

      etdump_Value_start(builder);
      etdump_Value_double_value_add(builder, double_ref);
      etdump_Value_val_add(builder, etdump_ValueType_Double);
      auto value_ref = etdump_Value_end(builder);
      etdump_DebugEvent_debug_entry_add(builder, value_ref);

      break;
    }

    case Tag::Bool: {
      flatbuffers_bool_t flatbuffer_bool_val =
          evalue.toBool() ? FLATBUFFERS_TRUE : FLATBUFFERS_FALSE;
      auto bool_ref = etdump_Bool_create(builder, flatbuffer_bool_val);

      etdump_Value_start(builder);
      etdump_Value_bool_value_add(builder, bool_ref);
      etdump_Value_val_add(builder, etdump_ValueType_Bool);
      auto value_ref = etdump_Value_end(builder);
      etdump_DebugEvent_debug_entry_add(builder, value_ref);

      break;
    }

    default:
      ET_CHECK_MSG(
          0,
          "This EValue type = %d is not yet supported for logging\n",
          static_cast<int>(evalue.tag));
      break;
  }

  etdump_DebugEvent_ref_t debug_event = etdump_DebugEvent_end(builder);

  etdump_RunData_events_push_start(builder);
  etdump_Event_debug_event_add(builder, debug_event);
  etdump_RunData_events_push_end(builder);
}

size_t ETDumpGen::get_num_blocks() {
  return num_blocks;
}

bool ETDumpGen::is_static_etdump() {
  return alloc.data != nullptr;
}

} // namespace executor
} // namespace torch
