/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/devtools/etdump/etdump_flatcc.h>

#include <cstring>

#include <executorch/devtools/etdump/emitter.h>
#include <executorch/devtools/etdump/etdump_schema_flatcc_builder.h>
#include <executorch/devtools/etdump/etdump_schema_flatcc_reader.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/assert.h>

#include <flatcc/flatcc_types.h>

using ::exec_aten::Tensor;
using ::executorch::runtime::AllocatorID;
using ::executorch::runtime::ArrayRef;
using ::executorch::runtime::ChainID;
using ::executorch::runtime::DebugHandle;
using ::executorch::runtime::DelegateDebugIdType;
using ::executorch::runtime::EValue;
using ::executorch::runtime::EventTracerEntry;
using ::executorch::runtime::LoggedEValueType;
using ::executorch::runtime::Span;
using ::executorch::runtime::Tag;

namespace executorch {
namespace etdump {

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
    flatcc_builder_t* builder_,
    const exec_aten::Tensor& tensor,
    long offset) {
  etdump_Tensor_start(builder_);

  etdump_Tensor_scalar_type_add(
      builder_, get_flatbuffer_scalar_type(tensor.scalar_type()));
  etdump_Tensor_sizes_start(builder_);

  for (auto dim : tensor.sizes()) {
    int64_t cast_dim = static_cast<int64_t>(dim);
    etdump_Tensor_sizes_push(builder_, &cast_dim);
  }
  etdump_Tensor_sizes_end(builder_);

  etdump_Tensor_strides_start(builder_);
  for (auto dim : tensor.strides()) {
    int64_t cast_dim = static_cast<int64_t>(dim);
    etdump_Tensor_strides_push(builder_, &cast_dim);
  }
  etdump_Tensor_strides_end(builder_);
  etdump_Tensor_offset_add(builder_, offset);

  return etdump_Tensor_end(builder_);
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

// Constructor implementation
ETDumpGen::ETDumpGen(Span<uint8_t> buffer) {
  constexpr size_t max_alloc_buf_size = 128 * 1024;

  // Initialize the flatcc builder_ using the buffer and buffer size.

  if (buffer.data() != nullptr) {
    builder_ = (struct flatcc_builder*)alignPointer(buffer.data(), 64);
    uintptr_t buffer_with_builder =
        (uintptr_t)alignPointer(builder_ + sizeof(struct flatcc_builder), 64);
    size_t buffer_size = buffer.size() -
        (size_t)(buffer_with_builder - (uintptr_t)buffer.data());
    alloc_.set_buffer(
        (uint8_t*)buffer_with_builder,
        buffer_size,
        (size_t)((buffer_size / 4 > max_alloc_buf_size) ? max_alloc_buf_size
                                                        : buffer_size / 4));
    internal::etdump_flatcc_custom_init(builder_, &alloc_);
  } else {
    builder_ = (struct flatcc_builder*)malloc(sizeof(struct flatcc_builder));
    ET_CHECK_MSG(
        builder_ != nullptr, "Failed to allocate memory for flatcc builder_.");
    flatcc_builder_init(builder_);
  }
  reset();
}

ETDumpGen::~ETDumpGen() {
  flatcc_builder_clear(builder_);
  if (!is_static_etdump()) {
    free(builder_);
  }
}

void ETDumpGen::reset() {
  state_ = State::Init;
  num_blocks_ = 0;
  flatcc_builder_reset(builder_);
  flatbuffers_buffer_start(builder_, etdump_ETDump_file_identifier);
  etdump_ETDump_start_as_root_with_size(builder_);
  etdump_ETDump_version_add(builder_, ETDUMP_VERSION);
  etdump_ETDump_run_data_start(builder_);
  etdump_ETDump_run_data_push_start(builder_);
}

void ETDumpGen::create_event_block(const char* name) {
  if (state_ == State::AddingEvents) {
    etdump_RunData_events_end(builder_);
  } else if (state_ == State::Done) {
    reset();
  }
  if (num_blocks_ > 0) {
    etdump_ETDump_run_data_push_end(builder_);
    etdump_ETDump_run_data_push_start(builder_);
  }
  ++num_blocks_;
  etdump_RunData_name_create_strn(builder_, name, strlen(name));
  if (bundled_input_index_ != -1) {
    etdump_RunData_bundled_input_index_add(builder_, bundled_input_index_);
  }
  state_ = State::BlockCreated;
}

int64_t ETDumpGen::create_string_entry(const char* name) {
  return flatbuffers_string_create_str(builder_, name);
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
  if (state_ != State::AddingEvents) {
    ET_CHECK_MSG(
        (state_ == State::AddingAllocators || state_ == State::BlockCreated),
        "ETDumpGen in an invalid state. Cannot add new events now.");
    if (state_ == State::AddingAllocators) {
      etdump_RunData_allocators_end(builder_);
    }
    etdump_RunData_events_start(builder_);
    state_ = State::AddingEvents;
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
  etdump_ProfileEvent_start(builder_);
  etdump_ProfileEvent_start_time_add(builder_, event_tracer_entry.start_time);
  etdump_ProfileEvent_end_time_add(builder_, end_time);
  etdump_ProfileEvent_chain_index_add(builder_, chain_id_);
  etdump_ProfileEvent_instruction_id_add(builder_, debug_handle_);
  // Delegate debug identifier can either be of a string type or an integer
  // type. If it's a string type then it's a value of type
  // flatbuffers_string_ref_t type, whereas if it's an integer type then we
  // write the integer value directly.
  if (event_tracer_entry.delegate_event_id_type == DelegateDebugIdType::kInt) {
    etdump_ProfileEvent_delegate_debug_id_int_add(
        builder_, event_tracer_entry.event_id);
  } else {
    etdump_ProfileEvent_delegate_debug_id_str_add(
        builder_, event_tracer_entry.event_id);
  }
  flatbuffers_uint8_vec_ref_t vec_ref = flatbuffers_uint8_vec_create_pe(
      builder_, (const uint8_t*)metadata, metadata_len);
  etdump_ProfileEvent_delegate_debug_metadata_add(builder_, vec_ref);
  etdump_ProfileEvent_ref_t id = etdump_ProfileEvent_end(builder_);
  etdump_RunData_events_push_start(builder_);
  etdump_Event_profile_event_add(builder_, id);
  etdump_RunData_events_push_end(builder_);
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
  etdump_ProfileEvent_start(builder_);
  etdump_ProfileEvent_start_time_add(builder_, start_time);
  etdump_ProfileEvent_end_time_add(builder_, end_time);
  etdump_ProfileEvent_chain_index_add(builder_, chain_id_);
  etdump_ProfileEvent_instruction_id_add(builder_, debug_handle_);
  if (string_id == -1) {
    etdump_ProfileEvent_delegate_debug_id_int_add(
        builder_, delegate_debug_index);
  } else {
    etdump_ProfileEvent_delegate_debug_id_str_add(builder_, string_id);
  }
  flatbuffers_uint8_vec_ref_t vec_ref = flatbuffers_uint8_vec_create_pe(
      builder_, (const uint8_t*)metadata, metadata_len);
  etdump_ProfileEvent_delegate_debug_metadata_add(builder_, vec_ref);
  etdump_ProfileEvent_ref_t id = etdump_ProfileEvent_end(builder_);
  etdump_RunData_events_push_start(builder_);
  etdump_Event_profile_event_add(builder_, id);
  etdump_RunData_events_push_end(builder_);
}

void ETDumpGen::log_intermediate_output_delegate(
    const char* name,
    DebugHandle delegate_debug_index,
    const Tensor& output) {
  log_intermediate_output_delegate_helper(name, delegate_debug_index, output);
}

void ETDumpGen::log_intermediate_output_delegate(
    const char* name,
    DebugHandle delegate_debug_index,
    const ArrayRef<Tensor> output) {
  log_intermediate_output_delegate_helper(name, delegate_debug_index, output);
}

void ETDumpGen::log_intermediate_output_delegate(
    const char* name,
    DebugHandle delegate_debug_index,
    const int& output) {
  log_intermediate_output_delegate_helper(name, delegate_debug_index, output);
}

void ETDumpGen::log_intermediate_output_delegate(
    const char* name,
    DebugHandle delegate_debug_index,
    const bool& output) {
  log_intermediate_output_delegate_helper(name, delegate_debug_index, output);
}

void ETDumpGen::log_intermediate_output_delegate(
    const char* name,
    DebugHandle delegate_debug_index,
    const double& output) {
  log_intermediate_output_delegate_helper(name, delegate_debug_index, output);
}

template <typename T>
void ETDumpGen::log_intermediate_output_delegate_helper(
    const char* name,
    DebugHandle delegate_debug_index,
    const T& output) {
  ET_CHECK_MSG(
      (name == nullptr) ^ (delegate_debug_index == -1),
      "Only name or delegate_debug_index can be valid. Check DelegateMappingBuilder documentation for more details.");
  if (debug_buffer_.empty()) {
    ET_CHECK_MSG(0, "Must pre-set debug buffer with set_debug_buffer()\n");
    return;
  }

  check_ready_to_add_events();
  int64_t string_id = name != nullptr ? create_string_entry(name) : -1;

  etdump_DebugEvent_start(builder_);

  etdump_DebugEvent_chain_index_add(builder_, chain_id_);
  etdump_DebugEvent_instruction_id_add(builder_, debug_handle_);
  if (string_id == -1) {
    etdump_DebugEvent_delegate_debug_id_int_add(builder_, delegate_debug_index);
  } else {
    etdump_DebugEvent_delegate_debug_id_str_add(builder_, string_id);
  }

  // Check the type of `output` then call the corresponding logging functions
  if constexpr (std::is_same<T, Tensor>::value) {
    long offset = copy_tensor_to_debug_buffer(output);
    etdump_Tensor_ref_t tensor_ref = add_tensor_entry(builder_, output, offset);

    etdump_Value_start(builder_);
    etdump_Value_val_add(builder_, etdump_ValueType_Tensor);
    etdump_Value_tensor_add(builder_, tensor_ref);

  } else if constexpr (std::is_same<T, ArrayRef<Tensor>>::value) {
    etdump_Tensor_vec_start(builder_);
    for (size_t i = 0; i < output.size(); ++i) {
      long offset = copy_tensor_to_debug_buffer(output[i]);
      etdump_Tensor_vec_push(
          builder_, add_tensor_entry(builder_, output[i], offset));
    }
    etdump_Tensor_vec_ref_t tensor_vec_ref = etdump_Tensor_vec_end(builder_);
    etdump_TensorList_ref_t tensor_list_ref =
        etdump_TensorList_create(builder_, tensor_vec_ref);

    etdump_Value_start(builder_);
    etdump_Value_val_add(builder_, etdump_ValueType_TensorList);
    etdump_Value_tensor_list_add(builder_, tensor_list_ref);
  } else if constexpr (std::is_same<T, int>::value) {
    auto int_ref = etdump_Int_create(builder_, output);

    etdump_Value_start(builder_);
    etdump_Value_val_add(builder_, etdump_ValueType_Int);
    etdump_Value_int_value_add(builder_, int_ref);
  } else if constexpr (std::is_same<T, double>::value) {
    auto double_ref = etdump_Double_create(builder_, output);

    etdump_Value_start(builder_);
    etdump_Value_double_value_add(builder_, double_ref);
    etdump_Value_val_add(builder_, etdump_ValueType_Double);
  } else if constexpr (std::is_same<T, bool>::value) {
    flatbuffers_bool_t flatbuffer_bool_val =
        output ? FLATBUFFERS_TRUE : FLATBUFFERS_FALSE;
    auto bool_ref = etdump_Bool_create(builder_, flatbuffer_bool_val);

    etdump_Value_start(builder_);
    etdump_Value_bool_value_add(builder_, bool_ref);
    etdump_Value_val_add(builder_, etdump_ValueType_Bool);
  } else {
    ET_CHECK_MSG(0, "Unsupported output type for intermediate logging\n");
  }

  auto value_ref = etdump_Value_end(builder_);
  etdump_DebugEvent_debug_entry_add(builder_, value_ref);

  etdump_DebugEvent_ref_t debug_event = etdump_DebugEvent_end(builder_);

  etdump_RunData_events_push_start(builder_);
  etdump_Event_debug_event_add(builder_, debug_event);
  etdump_RunData_events_push_end(builder_);
}

void ETDumpGen::end_profiling(EventTracerEntry prof_entry) {
  et_timestamp_t end_time = et_pal_current_ticks();
  ET_CHECK_MSG(
      prof_entry.delegate_event_id_type == DelegateDebugIdType::kNone,
      "Delegate events must use end_profiling_delegate to mark the end of a delegate profiling event.");
  check_ready_to_add_events();

  etdump_ProfileEvent_start(builder_);
  etdump_ProfileEvent_start_time_add(builder_, prof_entry.start_time);
  etdump_ProfileEvent_end_time_add(builder_, end_time);
  etdump_ProfileEvent_chain_index_add(builder_, prof_entry.chain_id);
  etdump_ProfileEvent_instruction_id_add(builder_, prof_entry.debug_handle);
  if (prof_entry.event_id != -1) {
    etdump_ProfileEvent_name_add(builder_, prof_entry.event_id);
  }
  etdump_ProfileEvent_ref_t id = etdump_ProfileEvent_end(builder_);
  etdump_RunData_events_push_start(builder_);
  etdump_Event_profile_event_add(builder_, id);
  etdump_RunData_events_push_end(builder_);
}

AllocatorID ETDumpGen::track_allocator(const char* name) {
  ET_CHECK_MSG(
      (state_ == State::BlockCreated || state_ == State::AddingAllocators),
      "Allocators can only be added immediately after a new block is created and before any events are added.");
  if (state_ != State::AddingAllocators) {
    etdump_RunData_allocators_start(builder_);
    state_ = State::AddingAllocators;
  }
  flatbuffers_string_ref_t ref = create_string_entry(name);
  etdump_RunData_allocators_push_create(builder_, ref);
  return etdump_RunData_allocators_reserved_len(builder_);
}

void ETDumpGen::track_allocation(
    AllocatorID allocator_id,
    size_t allocation_size) {
  check_ready_to_add_events();

  etdump_RunData_events_push_start(builder_);
  etdump_Event_allocation_event_create(builder_, allocator_id, allocation_size);
  etdump_RunData_events_push_end(builder_);
}

ETDumpResult ETDumpGen::get_etdump_data() {
  ETDumpResult result;
  if (state_ == State::AddingEvents) {
    etdump_RunData_events_end(builder_);
  } else if (state_ == State::AddingAllocators) {
    etdump_RunData_allocators_end(builder_);
  } else if (state_ == State::Init) {
    result.buf = nullptr;
    result.size = 0;
    return result;
  }
  etdump_ETDump_run_data_push_end(builder_);
  etdump_ETDump_run_data_end(builder_);
  etdump_ETDump_ref_t root = etdump_ETDump_end(builder_);
  flatbuffers_buffer_end(builder_, root);
  if (num_blocks_ == 0) {
    result = {nullptr, 0};
  } else {
    if (alloc_.data) {
      result.buf = alloc_.front_cursor;
      result.size = alloc_.out_size - alloc_.front_left;
    } else {
      result.buf =
          flatcc_builder_finalize_aligned_buffer(builder_, &result.size);
    }
  }
  state_ = State::Done;
  return result;
}

void ETDumpGen::set_debug_buffer(Span<uint8_t> buffer) {
  debug_buffer_ = buffer;
}

size_t ETDumpGen::copy_tensor_to_debug_buffer(exec_aten::Tensor tensor) {
  if (tensor.nbytes() == 0) {
    return static_cast<size_t>(-1);
  }
  uint8_t* offset_ptr =
      alignPointer(debug_buffer_.data() + debug_buffer_offset_, 64);
  debug_buffer_offset_ = (offset_ptr - debug_buffer_.data()) + tensor.nbytes();
  ET_CHECK_MSG(
      debug_buffer_offset_ <= debug_buffer_.size(),
      "Ran out of space to store intermediate outputs.");
  memcpy(offset_ptr, tensor.const_data_ptr(), tensor.nbytes());
  return (size_t)(offset_ptr - debug_buffer_.data());
}

void ETDumpGen::log_evalue(const EValue& evalue, LoggedEValueType evalue_type) {
  if (debug_buffer_.empty()) {
    return;
  }

  check_ready_to_add_events();

  etdump_DebugEvent_start(builder_);

  etdump_DebugEvent_chain_index_add(builder_, chain_id_);
  etdump_DebugEvent_instruction_id_add(builder_, debug_handle_);

  switch (evalue.tag) {
    case Tag::Tensor: {
      exec_aten::Tensor tensor = evalue.toTensor();
      long offset = copy_tensor_to_debug_buffer(tensor);
      etdump_Tensor_ref_t tensor_ref =
          add_tensor_entry(builder_, tensor, offset);

      etdump_Value_start(builder_);
      etdump_Value_val_add(builder_, etdump_ValueType_Tensor);
      etdump_Value_tensor_add(builder_, tensor_ref);
      if (evalue_type == LoggedEValueType::kProgramOutput) {
        auto bool_ref = etdump_Bool_create(builder_, FLATBUFFERS_TRUE);
        etdump_Value_output_add(builder_, bool_ref);
      }
      auto value_ref = etdump_Value_end(builder_);

      etdump_DebugEvent_debug_entry_add(builder_, value_ref);
      break;
    }

    case Tag::ListTensor: {
      exec_aten::ArrayRef<exec_aten::Tensor> tensors = evalue.toTensorList();
      etdump_Tensor_vec_start(builder_);
      for (size_t i = 0; i < tensors.size(); ++i) {
        long offset = copy_tensor_to_debug_buffer(tensors[i]);
        etdump_Tensor_vec_push(
            builder_, add_tensor_entry(builder_, tensors[i], offset));
      }
      etdump_Tensor_vec_ref_t tensor_vec_ref = etdump_Tensor_vec_end(builder_);
      etdump_TensorList_ref_t tensor_list_ref =
          etdump_TensorList_create(builder_, tensor_vec_ref);

      etdump_Value_start(builder_);
      etdump_Value_val_add(builder_, etdump_ValueType_TensorList);
      etdump_Value_tensor_list_add(builder_, tensor_list_ref);
      if (evalue_type == LoggedEValueType::kProgramOutput) {
        auto bool_ref = etdump_Bool_create(builder_, FLATBUFFERS_TRUE);
        etdump_Value_output_add(builder_, bool_ref);
      }
      auto value_ref = etdump_Value_end(builder_);

      etdump_DebugEvent_debug_entry_add(builder_, value_ref);
      break;
    }

    case Tag::Int: {
      int64_t val = evalue.toInt();
      auto int_ref = etdump_Int_create(builder_, val);

      etdump_Value_start(builder_);
      etdump_Value_val_add(builder_, etdump_ValueType_Int);
      etdump_Value_int_value_add(builder_, int_ref);
      auto value_ref = etdump_Value_end(builder_);
      etdump_DebugEvent_debug_entry_add(builder_, value_ref);

      break;
    }

    case Tag::Double: {
      double val = evalue.toDouble();
      auto double_ref = etdump_Double_create(builder_, val);

      etdump_Value_start(builder_);
      etdump_Value_double_value_add(builder_, double_ref);
      etdump_Value_val_add(builder_, etdump_ValueType_Double);
      auto value_ref = etdump_Value_end(builder_);
      etdump_DebugEvent_debug_entry_add(builder_, value_ref);

      break;
    }

    case Tag::Bool: {
      flatbuffers_bool_t flatbuffer_bool_val =
          evalue.toBool() ? FLATBUFFERS_TRUE : FLATBUFFERS_FALSE;
      auto bool_ref = etdump_Bool_create(builder_, flatbuffer_bool_val);

      etdump_Value_start(builder_);
      etdump_Value_bool_value_add(builder_, bool_ref);
      etdump_Value_val_add(builder_, etdump_ValueType_Bool);
      auto value_ref = etdump_Value_end(builder_);
      etdump_DebugEvent_debug_entry_add(builder_, value_ref);

      break;
    }

    default:
      ET_CHECK_MSG(
          0,
          "This EValue type = %d is not yet supported for logging\n",
          static_cast<int>(evalue.tag));
      break;
  }

  etdump_DebugEvent_ref_t debug_event = etdump_DebugEvent_end(builder_);

  etdump_RunData_events_push_start(builder_);
  etdump_Event_debug_event_add(builder_, debug_event);
  etdump_RunData_events_push_end(builder_);
}

size_t ETDumpGen::get_num_blocks() {
  return num_blocks_;
}

bool ETDumpGen::is_static_etdump() {
  return alloc_.data != nullptr;
}

} // namespace etdump
} // namespace executorch
