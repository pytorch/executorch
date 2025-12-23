/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/compiler.h>

using executorch::runtime::Error;
using executorch::runtime::Result;
namespace executorch::extension::utils {

// Util to get alighment adjusted allocation size
inline Result<size_t> get_aligned_size(size_t size, size_t alignment) {
  // The minimum alignment that malloc() is guaranteed to provide.
  static constexpr size_t kMallocAlignment = alignof(std::max_align_t);
  if (alignment > kMallocAlignment) {
    // To get higher alignments, allocate extra and then align the returned
    // pointer. This will waste an extra `alignment - 1` bytes every time, but
    // this is the only portable way to get aligned memory from the heap.
    const size_t extra = alignment - 1;
    if ET_UNLIKELY (extra >= SIZE_MAX - size) {
      ET_LOG(Error, "Malloc size overflow: size=%zu + extra=%zu", size, extra);
      return Result<size_t>(Error::InvalidArgument);
    }
    size += extra;
  }
  return size;
}

} // namespace executorch::extension::utils
