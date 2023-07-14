#include <executorch/runtime/executor/program.h>

#include <cstddef>
#include <cstdint>

#include <executorch/runtime/platform/profiler.h>
#include <executorch/schema/extended_header.h>
#include <executorch/schema/schema_generated.h>

/*
 * Program verification can increase code size by ~30k. Targets that need to
 * save this space can avoid building it by passing
 * -DET_ENABLE_PROGRAM_VERIFICATION=0 on the compile line.
 */
#ifndef ET_ENABLE_PROGRAM_VERIFICATION
#define ET_ENABLE_PROGRAM_VERIFICATION 1
#endif

namespace torch {
namespace executor {

namespace {

bool IsAligned(const void* data, size_t alignment) {
  uintptr_t addr = reinterpret_cast<uintptr_t>(data);
  return addr % alignment == 0;
}

/**
 * Tries deserializing the data as a Program flabuffer file. Returns nullptr if
 * the file appears to be corrupt or incompatible.
 */
const executorch::Program* DeserializeFlatbufferData(const void* data) {
  if (Program::check_header(data, Program::kMinHeadBytes) !=
      Program::HeaderStatus::CompatibleVersion) {
    ET_LOG(
        Error,
        "Program identifier '%.4s' != expected '%.4s'",
        flatbuffers::GetBufferIdentifier(data),
        executorch::ProgramIdentifier());
    return nullptr;
  }

  // The provided pointer must start at an aligned address to ensure internal
  // alignment of flatbuffer fields.
  if (!IsAligned(data, FLATBUFFERS_MAX_ALIGNMENT)) {
    ET_LOG(
        Error,
        "Program data 0x%p must be aligned to %u",
        data,
        FLATBUFFERS_MAX_ALIGNMENT);
    return nullptr;
  }

  const executorch::Program* program = executorch::GetProgram(data);
  if (program->segments()->size() > 0) {
    ET_LOG(
        Error,
        "Program constructor does not support segments; use Program::Load()");
    return nullptr;
  }
  return program;
}

} // namespace

Program::Program(const void* serialized_content)
    : Program(
          /*loader=*/nullptr,
          /*segment_base_offset=*/0,
          FreeableBuffer(
              /*data=*/nullptr,
              /*size=*/0,
              /*free_fn=*/nullptr),
          DeserializeFlatbufferData(serialized_content)) {}

/* static */ Result<Program> Program::Load(
    DataLoader* loader,
    Program::Verification verification) {
  EXECUTORCH_SCOPE_PROF("Program::Load");

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
  if (!executorch::ProgramBufferHasIdentifier(program_data->data())) {
    ET_LOG(
        Error,
        "Program identifier '%.4s' != expected '%.4s'",
        flatbuffers::GetBufferIdentifier(program_data->data()),
        executorch::ProgramIdentifier());
    return Error::InvalidProgram;
  }

  // Do extra verification if requested.
  if (verification == Verification::InternalConsistency) {
#if ET_ENABLE_PROGRAM_VERIFICATION
    EXECUTORCH_SCOPE_PROF("Program::verify_internal_consistency");
    flatbuffers::Verifier verifier(
        reinterpret_cast<const uint8_t*>(program_data->data()),
        program_data->size());
    bool ok = executorch::VerifyProgramBuffer(verifier);
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
      IsAligned(program_data->data(), FLATBUFFERS_MAX_ALIGNMENT),
      InvalidArgument,
      "Program data 0x%p must be aligned to %u",
      program_data->data(),
      FLATBUFFERS_MAX_ALIGNMENT);

  // Get the pointer to the root flatbuffer table.
  const executorch::Program* flatbuffer_program =
      executorch::GetProgram(program_data->data());

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
      static_cast<const executorch::Program*>(internal_program_);
  return internal_program->execution_plan()->size();
}

Result<const char*> Program::get_method_name(size_t plan_idx) const {
  if (plan_idx >= this->num_methods()) {
    return Error::InvalidArgument;
  }
  auto internal_program =
      static_cast<const executorch::Program*>(internal_program_);
  return internal_program->execution_plan()->Get(plan_idx)->name()->c_str();
}

const void* Program::get_constant_buffer_data(size_t buffer_idx) const {
  ET_CHECK(is_valid());
  auto internal_program =
      static_cast<const executorch::Program*>(internal_program_);
  ET_CHECK_MSG(
      buffer_idx < constant_buffer_size(),
      "Constant buffer %zu out of program buffer range %zu",
      buffer_idx,
      constant_buffer_size());

  const auto& constant_buffer = *internal_program->constant_buffer();

  return static_cast<const void*>(
      constant_buffer[buffer_idx]->storage()->data());
}

size_t Program::constant_buffer_size() const {
  ET_CHECK(is_valid());
  auto internal_program =
      static_cast<const executorch::Program*>(internal_program_);
  return internal_program->constant_buffer()->size();
}

int64_t Program::get_non_const_buffer_size(
    size_t buffer_idx,
    size_t execution_plan_idx) const {
  ET_CHECK(is_valid());
  ET_CHECK_MSG(
      execution_plan_idx == Program::kForwardMethodIndex,
      "Unsupported plan index %zu != %zu",
      execution_plan_idx,
      Program::kForwardMethodIndex);
  auto res = this->get_non_const_buffer_size(buffer_idx, "forward");
  ET_CHECK(res.ok());
  return res.get();
}

Result<int64_t> Program::get_non_const_buffer_size(
    size_t buffer_idx,
    const char* method_name) const {
  auto internal_program =
      static_cast<const executorch::Program*>(internal_program_);
  for (auto plan : *internal_program->execution_plan()) {
    if (std::strcmp(plan->name()->c_str(), method_name) == 0) {
      auto non_const_buffer_sizes = plan->non_const_buffer_sizes();
      if (buffer_idx >= non_const_buffer_sizes->size()) {
        return Error::InvalidArgument;
      }
      return (*plan->non_const_buffer_sizes())[buffer_idx];
    }
  }
  return Error::InvalidArgument;
}

size_t Program::num_non_const_buffers(size_t execution_plan_idx) const {
  ET_CHECK(is_valid());
  ET_CHECK_MSG(
      execution_plan_idx == Program::kForwardMethodIndex,
      "Unsupported plan index %zu != %zu",
      execution_plan_idx,
      Program::kForwardMethodIndex);
  auto res = this->num_non_const_buffers("forward");
  ET_CHECK(res.ok());
  return res.get();
}

Result<size_t> Program::num_non_const_buffers(const char* method_name) const {
  auto internal_program =
      static_cast<const executorch::Program*>(internal_program_);
  for (auto plan : *internal_program->execution_plan()) {
    if (std::strcmp(plan->name()->c_str(), method_name) == 0) {
      return plan->non_const_buffer_sizes()->size();
    }
  }
  return Error::InvalidArgument;
}

const char* Program::get_output_flattening_encoding(
    size_t execution_plan_idx) const {
  ET_CHECK(is_valid());
  ET_CHECK_MSG(
      execution_plan_idx == Program::kForwardMethodIndex,
      "Executor only supports a single execution plan at this time, but received a query about plan #%zu",
      execution_plan_idx);
  auto res = this->get_output_flattening_encoding("forward");
  ET_CHECK(res.ok());
  return res.get();
}

Result<const char*> Program::get_output_flattening_encoding(
    const char* method_name) const {
  auto internal_program =
      static_cast<const executorch::Program*>(internal_program_);
  for (auto plan : *internal_program->execution_plan()) {
    if (std::strcmp(plan->name()->c_str(), method_name) == 0) {
      return plan->container_meta_type()->encoded_out_str()->c_str();
    }
  }
  return Error::InvalidArgument;
}

Error Program::get_backend_delegate_data(
    size_t index,
    const void** out_data,
    size_t* out_size) const {
  ET_CHECK(is_valid());
  const auto* data_list =
      static_cast<const executorch::Program*>(internal_program_)
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
  if (executorch::ProgramBufferHasIdentifier(data)) {
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
  const executorch::DataSegment* segment =
      internal_program_->segments()->Get(index);
  // Could fail if offset and size are out of bound for the data, or if this
  // is reading from a file and fails, or for many other reasons depending on
  // the implementation of the loader.
  return loader_->Load(
      segment_base_offset_ + segment->offset(), segment->size());
}

} // namespace executor
} // namespace torch
