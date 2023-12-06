/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/executor/program.h>

#include <cstddef>
#include <cstdint>

#include <executorch/runtime/core/event_tracer_hooks.h>
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/platform/profiler.h>
#include <executorch/schema/extended_header.h>
#include <executorch/schema/program_generated.h>

/*
 * Program verification can increase code size by ~30k. Targets that need to
 * save this space can avoid building it by passing
 * -DET_ENABLE_PROGRAM_VERIFICATION=0 on the compile line.
 */
#ifndef ET_ENABLE_PROGRAM_VERIFICATION
#define ET_ENABLE_PROGRAM_VERIFICATION 1
#endif

#pragma clang diagnostic ignored "-Wshadow"

namespace torch {
namespace executor {

namespace {

/**
 * Program data must be aligned to this value to properly parse it. Must be a
 * power of 2. Note that max_align_t is the alignment that malloc() and new
 * guarantee.
 */
constexpr size_t kMinimumAlignment = alignof(std::max_align_t);

bool IsAligned(const void* data) {
  uintptr_t addr = reinterpret_cast<uintptr_t>(data);
  return addr % kMinimumAlignment == 0;
}

Result<executorch_flatbuffer::ExecutionPlan*> get_execution_plan(
    const executorch_flatbuffer::Program* program,
    const char* method_name) {
  auto execution_plans = program->execution_plan();
  for (size_t i = 0; i < execution_plans->size(); i++) {
    auto plan = execution_plans->GetMutableObject(i);
    if (std::strcmp(plan->name()->c_str(), method_name) == 0) {
      return plan;
    }
  }
  ET_LOG(Error, "No method named '%s' in program", method_name);
  return Error::InvalidArgument;
}

} // namespace

/* static */ Result<Program> Program::load(
    DataLoader* loader,
    Program::Verification verification) {
  EXECUTORCH_SCOPE_PROF("Program::load");

  // See if the program size is in the header.
  size_t program_size = 0;
  size_t segment_base_offset = 0;
  {
    EXECUTORCH_SCOPE_PROF("Program::check_header");
    Result<FreeableBuffer> header =
        loader->Load(/*offset=*/0, ExtendedHeader::kNumHeadBytes);
    if (!header.ok()) {
      return header.error();
    }
    Result<ExtendedHeader> eh =
        ExtendedHeader::Parse(header->data(), header->size());
    if (eh.ok()) {
      // The header has the program size.
      program_size = eh->program_size;
      segment_base_offset = eh->segment_base_offset;
    } else if (eh.error() == Error::NotFound) {
      // No header; the program consumes the whole file, and there are no
      // segments.
      program_size = ET_UNWRAP(loader->size());
    } else {
      ET_LOG(Error, "Extended header may be corrupt");
      return eh.error();
    }
  }

  // Load the flatbuffer data as a segment.
  uint32_t prof_tok = EXECUTORCH_BEGIN_PROF("Program::load_data");
  Result<FreeableBuffer> program_data =
      loader->Load(/*offset=*/0, program_size);
  if (!program_data.ok()) {
    return program_data.error();
  }
  EXECUTORCH_END_PROF(prof_tok);

  // Make sure the magic header matches the expected version.
  if (!executorch_flatbuffer::ProgramBufferHasIdentifier(
          program_data->data())) {
    ET_LOG(
        Error,
        "Program identifier '%.4s' != expected '%.4s'",
        flatbuffers::GetBufferIdentifier(program_data->data()),
        executorch_flatbuffer::ProgramIdentifier());
    return Error::InvalidProgram;
  }

  // Do extra verification if requested.
  if (verification == Verification::InternalConsistency) {
#if ET_ENABLE_PROGRAM_VERIFICATION
    EXECUTORCH_SCOPE_PROF("Program::verify_internal_consistency");
    flatbuffers::Verifier verifier(
        reinterpret_cast<const uint8_t*>(program_data->data()),
        program_data->size());
    bool ok = executorch_flatbuffer::VerifyProgramBuffer(verifier);
    ET_CHECK_OR_RETURN_ERROR(
        ok,
        InvalidProgram,
        "Verification failed; data may be truncated or corrupt");
#else
    ET_LOG(
        Info, "InternalConsistency verification requested but not available");
#endif
  }

  // The flatbuffer data must start at an aligned address to ensure internal
  // alignment of flatbuffer fields.
  ET_CHECK_OR_RETURN_ERROR(
      IsAligned(program_data->data()),
      InvalidArgument,
      "Program data 0x%p must be aligned to %zu",
      program_data->data(),
      kMinimumAlignment);

  // Get the pointer to the root flatbuffer table.
  const executorch_flatbuffer::Program* flatbuffer_program =
      executorch_flatbuffer::GetProgram(program_data->data());

  // The FreeableBuffer owns the data that flatbuffer_program points into. Also
  // keep a pointer to the loader so it can load more segments when necessary.
  return Program(
      loader,
      segment_base_offset,
      std::move(program_data.get()),
      flatbuffer_program);
}

size_t Program::num_methods() const {
  auto internal_program =
      static_cast<const executorch_flatbuffer::Program*>(internal_program_);
  return internal_program->execution_plan()->size();
}

Result<const char*> Program::get_method_name(size_t plan_index) const {
  if (plan_index >= this->num_methods()) {
    return Error::InvalidArgument;
  }
  auto internal_program =
      static_cast<const executorch_flatbuffer::Program*>(internal_program_);
  return internal_program->execution_plan()->Get(plan_index)->name()->c_str();
}

Result<Method> Program::load_method(
    const char* method_name,
    MemoryManager* memory_manager,
    EventTracer* event_tracer) const {
  EXECUTORCH_SCOPE_PROF("Program::load_method");
  internal::event_tracer_create_event_block(event_tracer, "Default");
  internal::EventTracerProfileScope event_tracer_scope =
      internal::EventTracerProfileScope(event_tracer, "Program::load_method");
  auto plan = get_execution_plan(internal_program_, method_name);
  if (!plan.ok()) {
    return plan.error();
  }
  return Method::load(plan.get(), this, memory_manager, event_tracer);
}

Result<MethodMeta> Program::method_meta(const char* method_name) const {
  auto plan = get_execution_plan(internal_program_, method_name);
  if (!plan.ok()) {
    return plan.error();
  }
  return MethodMeta(plan.get());
}

Result<const void*> Program::get_constant_buffer_data(
    size_t buffer_index) const {
  auto internal_program =
      static_cast<const executorch_flatbuffer::Program*>(internal_program_);
  size_t size = internal_program->constant_buffer()->size();
  ET_CHECK_OR_RETURN_ERROR(
      buffer_index < size,
      InvalidArgument,
      "Constant buffer %zu out of program buffer range %zu",
      buffer_index,
      size);

  const auto& constant_buffer = *internal_program->constant_buffer();

  return static_cast<const void*>(
      constant_buffer[buffer_index]->storage()->data());
}

Result<int64_t> Program::get_non_const_buffer_size(
    size_t buffer_index,
    const char* method_name) const {
  auto plan = get_execution_plan(internal_program_, method_name);
  if (!plan.ok()) {
    return plan.error();
  }
  auto non_const_buffer_sizes = plan.get()->non_const_buffer_sizes();
  if (buffer_index >= non_const_buffer_sizes->size()) {
    ET_LOG(
        Error,
        "invalid buffer index %zu for size %zu",
        buffer_index,
        (size_t)non_const_buffer_sizes->size());
    return Error::InvalidArgument;
  }
  return (*(plan.get()->non_const_buffer_sizes()))[buffer_index];
}

Result<size_t> Program::num_non_const_buffers(const char* method_name) const {
  auto plan = get_execution_plan(internal_program_, method_name);
  if (!plan.ok()) {
    return plan.error();
  }
  return plan.get()->non_const_buffer_sizes()->size();
}

Result<const char*> Program::get_output_flattening_encoding(
    const char* method_name) const {
  auto plan = get_execution_plan(internal_program_, method_name);
  if (!plan.ok()) {
    return plan.error();
  }
  return plan.get()->container_meta_type()->encoded_out_str()->c_str();
}

Error Program::get_backend_delegate_data(
    size_t index,
    const void** out_data,
    size_t* out_size) const {
  const auto* data_list =
      static_cast<const executorch_flatbuffer::Program*>(internal_program_)
          ->backend_delegate_data();
  ET_CHECK_OR_RETURN_ERROR(
      index < data_list->size(),
      NotFound,
      "index %zu >= list size %" PRIu32,
      index,
      data_list->size());
  auto data = data_list->Get(index)->data();
  *out_data = data->data();
  *out_size = data->size();
  return Error::Ok;
}

/* static */ Program::HeaderStatus Program::check_header(
    const void* data,
    size_t size) {
  if (size < kMinHeadBytes) {
    return HeaderStatus::ShortData;
  }
  if (executorch_flatbuffer::ProgramBufferHasIdentifier(data)) {
    // The data has the same file_identifier string as the schema.fbs file
    // that this runtime was built with.
    return HeaderStatus::CompatibleVersion;
  }
  const char* id = flatbuffers::GetBufferIdentifier(data);
  if (id[0] == 'E' && id[1] == 'T') {
    // It looks like an executorch file, but not the version we expect.
    return HeaderStatus::IncompatibleVersion;
  }
  return HeaderStatus::NotPresent;
}

Result<FreeableBuffer> Program::LoadSegment(size_t index) const {
  EXECUTORCH_SCOPE_PROF("Program::LoadSegment");
  if (loader_ == nullptr || segment_base_offset_ == 0) {
    ET_LOG(Error, "No segments in program: requested index %zu", index);
    return Error::NotFound;
  }
  size_t num_segments = internal_program_->segments()->size();
  if (index >= num_segments) {
    ET_LOG(
        Error, "Segment index %zu out of range (>= %zu)", index, num_segments);
    return Error::NotFound;
  }
  const executorch_flatbuffer::DataSegment* segment =
      internal_program_->segments()->Get(index);
  // Could fail if offset and size are out of bound for the data, or if this
  // is reading from a file and fails, or for many other reasons depending on
  // the implementation of the loader.
  return loader_->Load(
      segment_base_offset_ + segment->offset(), segment->size());
}

} // namespace executor
} // namespace torch
