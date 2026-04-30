/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

namespace executorch {
namespace backends {
namespace native {

/**
 * Why a buffer is being allocated. Orthogonal to MemoryKind, which says
 * WHERE the buffer's storage lives. BufferRole tells the engine WHAT
 * the buffer's lifecycle role is.
 *
 *   Internal  : Intermediate value. Lifetime fully managed by the
 *               executor; engine allocates real backing storage.
 *   Input     : Graph input. Caller binds caller-owned storage at
 *               execute time via bind_io. Engine sets up a "shell"
 *               allocation (TensorImpl with no real bytes; data_ptr
 *               re-aliased per execute).
 *   Output    : Graph output. Symmetric to Input.
 *   Constant  : Weight or NDM-loaded value. Initial bytes are provided
 *               once and immutable thereafter (or treated as such for
 *               read-only access).
 *
 * Replaces the historical `mem_obj_id == -1` sentinel for "graph IO,
 * defer to bind_io" with an explicit, self-documenting field. The
 * `mem_obj_id` field continues to mean only "AOT memory-planner slot
 * id"; its sentinel use is gone.
 */
enum class BufferRole : uint8_t {
  Internal = 0,
  Input,
  Output,
  Constant,
};

inline const char* to_string(BufferRole r) {
  switch (r) {
    case BufferRole::Internal:
      return "Internal";
    case BufferRole::Input:
      return "Input";
    case BufferRole::Output:
      return "Output";
    case BufferRole::Constant:
      return "Constant";
  }
  return "?";
}

} // namespace native
} // namespace backends
} // namespace executorch
