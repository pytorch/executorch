/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "coreai_pte_fixture.h"

#include <executorch/runtime/executor/program.h>
#include <executorch/schema/extended_header.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace executorch::backends::coreai::testing {
namespace fb = executorch_flatbuffer;
using executorch::runtime::Error;
using executorch::runtime::ExtendedHeader;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::Result;

FBSyntheticPTE::FBSyntheticPTE(const std::vector<CacheMethodSpec>& methods, bool extended_header) {
  flatbuffers::FlatBufferBuilder builder;
  builder.ForceDefaults(true);
  // Malformed-field tests must mutate only the selected table's vtable.
  builder.DedupVtables(false);
  std::vector<flatbuffers::Offset<fb::ExecutionPlan>> plans;
  std::vector<flatbuffers::Offset<fb::BackendDelegateInlineData>> inline_data;
  std::vector<flatbuffers::Offset<fb::DataSegment>> segments;
  std::vector<uint8_t> segment_bytes;
  auto append_segment = [&](NSData* data) {
    while (segment_bytes.size() % 16 != 0) {
      segment_bytes.push_back(0);
    }
    const size_t index = segments.size();
    segments.push_back(fb::CreateDataSegment(builder, segment_bytes.size(), data.length));
    const auto* begin = static_cast<const uint8_t*>(data.bytes);
    if (data.length != 0) {
      segment_bytes.insert(segment_bytes.end(), begin, begin + data.length);
    }
    return static_cast<uint32_t>(index);
  };
  for (const auto& method : methods) {
    std::vector<flatbuffers::Offset<fb::BackendDelegate>> delegates;
    for (const auto& delegate : method.delegates) {
      uint32_t index;
      auto location = fb::DataLocation::INLINE;
      if (delegate.segmented) {
        index = append_segment(delegate.data);
        location = fb::DataLocation::SEGMENT;
      } else {
        index = static_cast<uint32_t>(inline_data.size());
        inline_data.push_back(fb::CreateBackendDelegateInlineData(
            builder, builder.CreateVector(static_cast<const uint8_t*>(delegate.data.bytes),
                                          delegate.data.length)));
      }
      if (delegate.index_override >= 0) {
        index = static_cast<uint32_t>(delegate.index_override);
      }
      auto reference = delegate.missing_processed
                           ? flatbuffers::Offset<fb::BackendDelegateDataReference>{}
                           : fb::CreateBackendDelegateDataReference(builder, location, index);
      auto name = delegate.backend == nullptr
                      ? flatbuffers::Offset<flatbuffers::String>{}
                      : builder.CreateString(delegate.backend, delegate.backend_length.value_or(
                                                                   std::strlen(delegate.backend)));
      delegates.push_back(fb::CreateBackendDelegate(builder, name, reference));
    }
    auto name = method.name == nullptr ? flatbuffers::Offset<flatbuffers::String>{}
                                       : builder.CreateString(method.name);
    auto values = builder.CreateVector(std::vector<flatbuffers::Offset<fb::EValue>>{});
    auto inputs = builder.CreateVector(std::vector<int32_t>{});
    auto outputs = builder.CreateVector(std::vector<int32_t>{});
    auto backends = builder.CreateVector(delegates);
    auto sizes = builder.CreateVector(std::vector<int64_t>{0});
    fb::ExecutionPlanBuilder plan(builder);
    plan.add_name(name);
    plan.add_values(values);
    plan.add_inputs(inputs);
    plan.add_outputs(outputs);
    plan.add_delegates(backends);
    plan.add_non_const_buffer_sizes(sizes);
    plans.push_back(plan.Finish());
  }
  // Delegate metadata access must not request named asset payloads.
  uint32_t asset_segment =
      append_segment([@"unread named asset" dataUsingEncoding:NSUTF8StringEncoding]);
  auto named = fb::CreateNamedData(
      builder, builder.CreateString("coreai/ab/model.aimodel/graph.bin"), asset_segment);
  auto names = builder.CreateVector(std::vector<flatbuffers::Offset<fb::NamedData>>{named});
  auto constants =
      fb::CreateSubsegmentOffsets(builder, 0, builder.CreateVector(std::vector<uint64_t>{0}));
  auto plan_vector = builder.CreateVector(plans);
  auto inline_vector = builder.CreateVector(inline_data);
  auto segment_vector = builder.CreateVector(segments);
  fb::ProgramBuilder program(builder);
  program.add_version(executorch::runtime::Program::kMaxSupportedSchemaVersion);
  program.add_execution_plan(plan_vector);
  program.add_backend_delegate_data(inline_vector);
  program.add_segments(segment_vector);
  program.add_constant_segment(constants);
  program.add_named_data(names);
  fb::FinishProgramBuffer(builder, program.Finish());
  if (!extended_header) {
    bytes.assign(builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize());
    return;
  }
  // Preserve flatbuffer alignment when inserting the extended header.
  constexpr size_t extra = 32;
  const size_t program_size = builder.GetSize() + extra;
  const size_t segment_base = (program_size + 15) & ~size_t(15);
  bytes.resize(segment_base + segment_bytes.size());
  std::memcpy(bytes.data(), builder.GetBufferPointer(), ExtendedHeader::kHeaderOffset);
  std::memcpy(bytes.data() + ExtendedHeader::kHeaderOffset + extra,
              builder.GetBufferPointer() + ExtendedHeader::kHeaderOffset,
              builder.GetSize() - ExtendedHeader::kHeaderOffset);
  auto root_offset = flatbuffers::ReadScalar<flatbuffers::uoffset_t>(bytes.data());
  flatbuffers::WriteScalar<flatbuffers::uoffset_t>(bytes.data(), root_offset + extra);
  std::memcpy(bytes.data() + 8, ExtendedHeader::kMagic, ExtendedHeader::kMagicSize);
  flatbuffers::WriteScalar<uint32_t>(bytes.data() + 12, extra);
  flatbuffers::WriteScalar<uint64_t>(bytes.data() + 16, program_size);
  flatbuffers::WriteScalar<uint64_t>(bytes.data() + 24, segment_base);
  flatbuffers::WriteScalar<uint64_t>(bytes.data() + 32, segment_bytes.size());
  std::memcpy(bytes.data() + segment_base, segment_bytes.data(), segment_bytes.size());
}

Result<size_t> FBSyntheticPTE::size() const {
  if (size_error != Error::Ok) {
    return size_error;
  }
  return bytes.size();
}

Result<FreeableBuffer> FBSyntheticPTE::load(size_t offset, size_t size,
                                            const SegmentInfo& info) const {
  const size_t call = requests.size();
  requests.push_back({offset, size, info.segment_type, info.segment_index,
                      info.descriptor == nullptr ? "" : info.descriptor});
  const bool fault = static_cast<int>(call) == fault_call;
  if (fault && load_fault == LoadFault::Error) {
    return load_error;
  }
  if (offset > bytes.size() || size > bytes.size() - offset) {
    return Error::AccessFailed;
  }
  if (info.segment_type != SegmentInfo::Type::Program &&
      info.segment_type != SegmentInfo::Type::Backend) {
    ADD_FAILURE() << "Unexpected segment type: " << static_cast<int>(info.segment_type);
    return Error::InvalidArgument;
  }
  const bool backend = info.segment_type == SegmentInfo::Type::Backend;
  if (backend) {
    if (info.descriptor == nullptr) {
      ADD_FAILURE() << "Backend segment has no descriptor";
      return Error::InvalidArgument;
    }
    backend_indices.push_back(info.segment_index);
    backend_descriptors.emplace_back(info.descriptor);
    if (static_cast<int>(info.segment_index) == fail_backend_index) {
      return Error::AccessFailed;
    }
  } else if (info.descriptor != nullptr) {
    ADD_FAILURE() << "Program segment has a backend descriptor";
    return Error::InvalidArgument;
  }
  auto* loads = backend ? &backend_loads : &program_loads;
  void* allocation = std::malloc(std::max(size, size_t(1)));
  if (allocation == nullptr) {
    ADD_FAILURE() << "Could not allocate " << size << " bytes";
    return Error::MemoryAllocationFailed;
  }
  auto* data = static_cast<uint8_t*>(allocation);
  if (size != 0) {
    std::memcpy(data, bytes.data() + offset, size);
  }
  struct Allocation {
    const FBSyntheticPTE* loader;
    void* base;
    bool backend;
  };
  auto* context = new Allocation{this, allocation, backend};
  requests.back().data = data;
  ++*loads;
  return FreeableBuffer(
      data, size,
      [](void* opaque, void*, size_t) {
        auto* context = static_cast<Allocation*>(opaque);
        const auto* loader = context->loader;
        if (context->backend) {
          loader->program_alive_at_backend_release &=
              loader->program_loads > loader->program_releases;
          ++loader->backend_releases;
        } else {
          ++loader->program_releases;
        }
        loader->release_order.push_back(context->backend ? SegmentInfo::Type::Backend
                                                         : SegmentInfo::Type::Program);
        std::free(context->base);
        delete context;
      },
      context);
}

::testing::AssertionResult FBSyntheticPTE::check_released() const {
  if (program_loads == program_releases && backend_loads == backend_releases) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure()
         << "Program loads/releases: " << program_loads.load() << "/" << program_releases.load()
         << "; backend loads/releases: " << backend_loads.load() << "/" << backend_releases.load();
}

}  // namespace executorch::backends::coreai::testing
