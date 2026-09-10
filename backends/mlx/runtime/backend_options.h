/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Shared option keys for the MLX backend. Included by the backend itself
// (to read the per-model runtime spec) and by callers/runners (to set it via
// a LoadBackendOptionsMap), keeping the string literals in one place.

#pragma once

namespace executorch {
namespace backends {
namespace mlx {

// Backend id under which the MLX backend registers (see MLXBackend.cpp).
inline constexpr char kMLXBackendId[] = "MLXBackend";

// Per-model runtime-spec key. Value N means: call mlx::core::clear_cache()
// every N execute() calls to release MLX's cached buffer pool; 0/unset
// disables.
//
// NOTE on granularity: MLX's buffer cache is a process-global singleton, so the
// flush is global even though this key is read per delegate handle. The counter
// is per-handle, so with a single MLX handle (the common case — gemma's
// prefill/decode share one "forward" method) the cadence is exactly "every N
// forwards"; if a process loads multiple MLX handles, the effective cadence is
// the aggregate of their executes and any handle's flush frees the shared pool
// for all. This bounds resident-*average*: between flushes the cache can still
// grow to MLX's default ceiling, and each flush is followed by a cold-cache
// realloc. A future set_cache_limit-style key could complement this by bounding
// peak footprint continuously.
inline constexpr char kClearCacheIntervalKey[] = "clear_cache_interval";

// Per-model runtime-spec key (bool). When true, the handle does not allocate
// its own default mutable buffers at init() — per-session buffers are managed
// by mlx_mutable_state.h instead. Only valid for multi-session loads, and only
// when the program's init chain does not reference mutable buffers (init()
// errors otherwise). Saves one full mutable-buffer (KV-cache) copy per handle.
inline constexpr char kSkipMutableBufferInitKey[] = "skip_mutable_buffer_init";

} // namespace mlx
} // namespace backends
} // namespace executorch
