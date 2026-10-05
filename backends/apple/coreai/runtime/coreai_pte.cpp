/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "coreai_pte.h"

#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/executor/program_validation.h>
#include <executorch/schema/extended_header.h>
#include <executorch/schema/program_generated.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <set>
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

Result<const void*> cpu_data(const FreeableBuffer& buffer, size_t size) {
  auto data = buffer.data_safe();
  if (!data.ok()) {
    return data.error();
  }
  if (data.get() == nullptr || buffer.size() < size) {
    return Error::InvalidProgram;
  }
  return data.get();
}
} // namespace

Result<CoreAIPteData> inspect_coreai_pte(DataLoader& loader) {
  auto source_size = loader.size();
  if (!source_size.ok()) {
    return source_size.error();
  }
  const size_t file_size = source_size.get();
  size_t program_size = file_size;
  ExtendedHeader header{};
  bool has_header = false;
  if (file_size >= ExtendedHeader::kNumHeadBytes) {
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
      header = parsed.get();
      has_header = true;
      if (header.program_size > std::numeric_limits<size_t>::max() ||
          header.segment_base_offset > std::numeric_limits<size_t>::max() ||
          header.segment_data_size > std::numeric_limits<size_t>::max() ||
          header.program_size < ExtendedHeader::kNumHeadBytes ||
          header.program_size > file_size) {
        return Error::InvalidProgram;
      }
      program_size = static_cast<size_t>(header.program_size);
      if (header.segment_base_offset != 0 &&
          (header.segment_base_offset < program_size ||
           header.segment_base_offset > file_size)) {
        return Error::InvalidProgram;
      }
      if (header.segment_data_size != 0 &&
          (header.segment_base_offset == 0 ||
           header.segment_data_size > file_size - header.segment_base_offset)) {
        return Error::InvalidProgram;
      }
    } else if (parsed.error() != Error::NotFound) {
      return Error::InvalidProgram;
    }
  }
  if (program_size == 0 || program_size >= FLATBUFFERS_MAX_BUFFER_SIZE) {
    return Error::InvalidProgram;
  }
  auto loaded =
      loader.load(0, program_size, SegmentInfo(SegmentInfo::Type::Program));
  if (!loaded.ok()) {
    return loaded.error();
  }
  auto data = cpu_data(loaded.get(), program_size);
  if (!data.ok()) {
    return data.error();
  }
  if (reinterpret_cast<uintptr_t>(data.get()) % alignof(std::max_align_t) !=
      0) {
    return Error::InvalidProgram;
  }
  if (program_size >= ExtendedHeader::kNumHeadBytes) {
    auto retained_header =
        ExtendedHeader::Parse(data.get(), ExtendedHeader::kNumHeadBytes);
    if (has_header) {
      if (!retained_header.ok() ||
          retained_header->program_size != header.program_size ||
          retained_header->segment_base_offset != header.segment_base_offset ||
          retained_header->segment_data_size != header.segment_data_size) {
        return Error::InvalidProgram;
      }
    } else if (
        retained_header.ok() || retained_header.error() != Error::NotFound) {
      return Error::InvalidProgram;
    }
  }
  flatbuffers::Verifier verifier(
      static_cast<const uint8_t*>(data.get()), program_size);
  if (!fb::VerifyProgramBuffer(verifier)) {
    return Error::InvalidProgram;
  }
  const auto* program = fb::GetProgram(data.get());
  if (program->version() > runtime::Program::kMaxSupportedSchemaVersion ||
      runtime::validate_program(program) != Error::Ok) {
    return Error::InvalidProgram;
  }

  CoreAIPteData result(std::move(loaded.get()));
  std::set<std::string_view> method_names;
  for (const auto* plan : *program->execution_plan()) {
    if (plan->name() == nullptr || plan->non_const_buffer_sizes() == nullptr ||
        plan->inputs() == nullptr || plan->outputs() == nullptr ||
        !method_names.emplace(plan->name()->c_str(), plan->name()->size())
             .second) {
      return Error::InvalidProgram;
    }
    if (plan->delegates() == nullptr) {
      continue;
    }
    for (const auto* delegate : *plan->delegates()) {
      if (delegate == nullptr || delegate->id() == nullptr) {
        return Error::InvalidProgram;
      }
      constexpr std::string_view backend = "CoreAIBackend";
      const std::string_view id(
          delegate->id()->c_str(), delegate->id()->size());
      if (id != backend) {
        continue;
      }
      const auto* reference = delegate->processed();
      if (reference == nullptr) {
        return Error::InvalidProgram;
      }
      const uint32_t index = reference->index();
      if (reference->location() == fb::DataLocation::INLINE) {
        const auto* buffers = program->backend_delegate_data();
        if (buffers == nullptr || index >= buffers->size() ||
            buffers->Get(index) == nullptr) {
          return Error::InvalidProgram;
        }
        const auto* bytes = buffers->Get(index)->data();
        if (bytes == nullptr || bytes->size() == 0) {
          return Error::InvalidProgram;
        }
        result.processed.emplace_back(
            bytes->data(), bytes->size(), nullptr, nullptr);
      } else if (reference->location() == fb::DataLocation::SEGMENT) {
        const auto* segments = program->segments();
        if (!has_header || header.segment_base_offset == 0 ||
            segments == nullptr || index >= segments->size() ||
            segments->Get(index) == nullptr) {
          return Error::InvalidProgram;
        }
        const auto* segment = segments->Get(index);
        if (segment->offset() > std::numeric_limits<size_t>::max() ||
            segment->size() > std::numeric_limits<size_t>::max() ||
            segment->size() == 0) {
          return Error::InvalidProgram;
        }
        const size_t base = static_cast<size_t>(header.segment_base_offset);
        const size_t offset = static_cast<size_t>(segment->offset());
        const size_t size = static_cast<size_t>(segment->size());
        if (offset > file_size - base || size > file_size - base - offset ||
            (header.segment_data_size != 0 &&
             (offset > header.segment_data_size ||
              size > header.segment_data_size - offset))) {
          return Error::InvalidProgram;
        }
        auto buffer = loader.load(
            base + offset,
            size,
            SegmentInfo(
                SegmentInfo::Type::Backend, index, delegate->id()->c_str()));
        if (!buffer.ok()) {
          return buffer.error();
        }
        auto bytes = cpu_data(buffer.get(), size);
        if (!bytes.ok()) {
          return bytes.error();
        }
        // Preserve the original owner without exposing bytes beyond the
        // requested extent.
        if (buffer->size() != size) {
          return Error::InvalidProgram;
        }
        result.processed.emplace_back(std::move(buffer.get()));
      } else {
        return Error::InvalidProgram;
      }
    }
  }
  return result;
}

} // namespace executorch::backends::coreai
