/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cstddef>
#include <cstdint>

#include <executorch/runtime/core/error.h>

namespace executorch {
namespace runtime {
class MemoryAllocator;
} // namespace runtime
namespace backends {
namespace vgf {

/**
 * CPU mapping of one VGF-owned model-boundary allocation.
 *
 * Returned mappings remain valid until Method destruction or a terminal
 * execution error, whichever occurs first. After an execution error, discard
 * all borrowed mappings and bound tensor pointers without accessing data.
 * Never free/unmap the VGF-owned pointer.
 *
 * Host accesses must be serialized with Method::execute() and Method
 * destruction. The backend mutex protects this lookup only, not subsequent
 * use of the returned pointer. The backend handles non-coherent memory cache
 * maintenance during execute().
 *
 * This API assumes each simultaneously loaded Method has its OWN method
 * MemoryAllocator. Pass exactly the allocator supplied to Program::load_method
 * by the Method's MemoryManager::method_allocator(). The backend receives the
 * same pointer through BackendInitContext::get_runtime_allocator(). If two
 * delegates share that pointer, the query rejects the ambiguous key.
 *
 * Direct binding further requires a 1:1 Method-to-VGF-delegate boundary
 * mapping, e.g. full delegation with graph IO memory planning disabled.
 */
struct VgfMappedBoundaryBuffer {
  void* data = nullptr;
  size_t logical_nbytes = 0;
  size_t capacity_nbytes = 0;
  bool host_coherent = false;
  uint64_t generation = 0;
};

/**
 * Return the mapped boundary for a Method with a unique method allocator.
 *
 * Rejects duplicate allocator registrations (including multiple VGF
 * partitions), missing/invalid delegates, and ineligible IO allocations.
 * The build must enable EXECUTORCH_VGF_ZERO_COPY_IO and runtime zero-copy must
 * not be disabled via ET_VGF_ZERO_COPY=0.
 */
executorch::runtime::Error get_vgf_mapped_boundary_buffer(
    const executorch::runtime::MemoryAllocator* method_allocator,
    bool input,
    size_t argument_index,
    VgfMappedBoundaryBuffer* buffer) noexcept;

/** Exact-match predicate used by the backend fast path and unit tests. */
inline bool vgf_owned_mapped_buffer_matches(
    const VgfMappedBoundaryBuffer& buffer,
    const void* pointer,
    size_t nbytes) noexcept {
  return buffer.data != nullptr && pointer == buffer.data &&
      nbytes == buffer.logical_nbytes;
}

} // namespace vgf
} // namespace backends
} // namespace executorch
