/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "coreai_pte.h"
#include "coreai_backend_id.h"

#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/executor/program_validation.h>
#include <executorch/schema/extended_header.h>
#include <executorch/schema/program_generated.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

namespace executorch::backends::coreai {
namespace {
namespace fb = executorch_flatbuffer;
using runtime::DataLoader;
using runtime::Error;
using runtime::ExtendedHeader;
using runtime::FreeableBuffer;
using runtime::Result;
using SegmentInfo = DataLoader::SegmentInfo;

struct ProgramExtent {
  size_t file_size;
  size_t program_size;
  ExtendedHeader header{};
  bool has_header = false;
};

struct VerifiedProgram {
  FreeableBuffer storage;
  const fb::Program* program;
};

Result<const void*> cpu_data(const FreeableBuffer& buffer, size_t size) {
  auto data = buffer.data_safe();
  if (!data.ok()) {
    return data.error();
  }
  if ((data.get() == nullptr && size != 0) || buffer.size() < size) {
    return Error::InvalidProgram;
  }
  return data.get();
}

// Mirror Program::load's header and buffer checks without loading any methods.
Result<ProgramExtent> parse_header(DataLoader& loader) {
  auto source_size = loader.size();
  if (!source_size.ok()) {
    return source_size.error();
  }
  ProgramExtent extent{source_size.get(), source_size.get()};
  if (extent.file_size >= ExtendedHeader::kNumHeadBytes) {
    auto prefix = loader.load(
        0,
        ExtendedHeader::kNumHeadBytes,
        SegmentInfo(SegmentInfo::Type::Program));
    if (!prefix.ok()) {
      return prefix.error();
    }
    auto data = cpu_data(prefix.get(), ExtendedHeader::kNumHeadBytes);
    if (!data.ok()) {
      return data.error();
    }
    auto parsed =
        ExtendedHeader::Parse(data.get(), ExtendedHeader::kNumHeadBytes);
    if (parsed.ok()) {
      extent.header = parsed.get();
      extent.has_header = true;
      const auto& header = extent.header;
      if (header.program_size > std::numeric_limits<size_t>::max() ||
          header.segment_base_offset > std::numeric_limits<size_t>::max() ||
          header.segment_data_size > std::numeric_limits<size_t>::max() ||
          header.program_size < ExtendedHeader::kNumHeadBytes ||
          header.program_size > extent.file_size) {
        return Error::InvalidProgram;
      }
      extent.program_size = static_cast<size_t>(header.program_size);
      // Legacy headers do not specify the segment extent.
      if (header.segment_data_size != 0 &&
          (header.segment_base_offset < extent.program_size ||
           header.segment_base_offset > extent.file_size ||
           header.segment_data_size >
               extent.file_size - header.segment_base_offset)) {
        return Error::InvalidProgram;
      }
    } else if (parsed.error() != Error::NotFound) {
      return Error::InvalidProgram;
    }
  }
  if (extent.program_size == 0 ||
      extent.program_size >= FLATBUFFERS_MAX_BUFFER_SIZE) {
    return Error::InvalidProgram;
  }
  return extent;
}

Result<VerifiedProgram> load_program(
    DataLoader& loader,
    const ProgramExtent& extent) {
  auto loaded = loader.load(
      0, extent.program_size, SegmentInfo(SegmentInfo::Type::Program));
  if (!loaded.ok()) {
    return loaded.error();
  }
  auto data = cpu_data(loaded.get(), extent.program_size);
  if (!data.ok()) {
    return data.error();
  }
  if (reinterpret_cast<uintptr_t>(data.get()) % alignof(std::max_align_t) !=
      0) {
    return Error::InvalidArgument;
  }
  flatbuffers::Verifier verifier(
      static_cast<const uint8_t*>(data.get()), extent.program_size);
  if (!fb::VerifyProgramBuffer(verifier)) {
    return Error::InvalidProgram;
  }
  const auto* program = fb::GetProgram(data.get());
  if (program->version() > runtime::Program::kMaxSupportedSchemaVersion ||
      runtime::validate_program(program) != Error::Ok) {
    return Error::InvalidProgram;
  }
  return VerifiedProgram{std::move(loaded.get()), program};
}

Result<FreeableBuffer> load_inline(const fb::Program& program, uint32_t index) {
  const auto* buffers = program.backend_delegate_data();
  if (buffers == nullptr) {
    return Error::InvalidProgram;
  }
  if (index >= buffers->size()) {
    return Error::NotFound;
  }
  const auto* bytes = buffers->Get(index)->data();
  if (bytes == nullptr) {
    return Error::InvalidProgram;
  }
  return FreeableBuffer(bytes->data(), bytes->size(), nullptr, nullptr);
}

Result<FreeableBuffer> load_segment(
    DataLoader& loader,
    const fb::Program& program,
    const ProgramExtent& extent,
    uint32_t index,
    const char* backend_id) {
  const auto& header = extent.header;
  const auto* segments = program.segments();
  if (!extent.has_header || header.segment_base_offset == 0) {
    return Error::NotFound;
  }
  if (segments == nullptr) {
    return Error::InvalidProgram;
  }
  if (index >= segments->size()) {
    return Error::NotFound;
  }
  const auto* segment = segments->Get(index);
  if (segment->offset() > std::numeric_limits<size_t>::max() ||
      segment->size() > std::numeric_limits<size_t>::max()) {
    return Error::InvalidProgram;
  }
  const size_t base = static_cast<size_t>(header.segment_base_offset);
  const size_t offset = static_cast<size_t>(segment->offset());
  const size_t size = static_cast<size_t>(segment->size());
  if (base > extent.file_size || offset > extent.file_size - base ||
      size > extent.file_size - base - offset ||
      (header.segment_data_size != 0 &&
       (offset > header.segment_data_size ||
        size > header.segment_data_size - offset))) {
    return Error::InvalidProgram;
  }
  auto buffer = loader.load(
      base + offset,
      size,
      SegmentInfo(SegmentInfo::Type::Backend, index, backend_id));
  if (!buffer.ok()) {
    return buffer.error();
  }
  auto bytes = cpu_data(buffer.get(), size);
  if (!bytes.ok()) {
    return bytes.error();
  }
  // Keep the loader's owner while rejecting data beyond the declared extent.
  if (buffer->size() != size) {
    return Error::InvalidProgram;
  }
  return std::move(buffer.get());
}

Result<FreeableBuffer> load_processed(
    DataLoader& loader,
    const fb::Program& program,
    const ProgramExtent& extent,
    const fb::BackendDelegate& delegate) {
  const auto* reference = delegate.processed();
  if (reference == nullptr) {
    return Error::InvalidProgram;
  }
  switch (reference->location()) {
    case fb::DataLocation::INLINE:
      return load_inline(program, reference->index());
    case fb::DataLocation::SEGMENT:
      return load_segment(
          loader, program, extent, reference->index(), delegate.id()->c_str());
    default:
      return Error::InvalidProgram;
  }
}
} // namespace

Result<CoreAIPteData> inspect_coreai_pte(DataLoader& loader) {
  auto extent = parse_header(loader);
  if (!extent.ok()) {
    return extent.error();
  }
  auto loaded = load_program(loader, extent.get());
  if (!loaded.ok()) {
    return loaded.error();
  }
  const auto* program = loaded->program;
  CoreAIPteData result(std::move(loaded->storage));
  for (const auto* plan : *program->execution_plan()) {
    if (plan->delegates() == nullptr) {
      continue;
    }
    for (const auto* delegate : *plan->delegates()) {
      if (delegate->id() == nullptr) {
        continue;
      }
      const std::string_view id(
          delegate->id()->c_str(), delegate->id()->size());
      if (id != kCoreAIBackendId) {
        continue;
      }
      auto buffer = load_processed(loader, *program, extent.get(), *delegate);
      if (!buffer.ok()) {
        return buffer.error();
      }
      result.processed.emplace_back(std::move(buffer.get()));
    }
  }
  return result;
}

} // namespace executorch::backends::coreai
