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

// Per-model runtime-spec key. Value N means: while running a method, evaluate
// the live per-execution tensors once the intermediates produced since the last
// evaluation exceed N bytes. 0/unset disables the mechanism entirely and is the
// default.
//
// WHY: MLX is lazy. Interpreter::dispatch only builds graph nodes, and nothing
// is materialized until MLXBackend::execute calls async_eval on the method
// outputs, so for a long instruction chain every intermediate in the method is
// live at the same instant. Whisper-small's 495-instruction encode peaks at
// 1105 MB of MLX allocation against 95 MB of steady-state active memory, which
// is what makes the model unusable on an iPhone (pytorch/executorch#22513).
// Each evaluation costs a GPU sync, so the cost tracks the NUMBER of
// evaluations; budgeting bytes rather than counting instructions puts them only
// in the methods that actually allocate.
//
// NOTE that this is a THRESHOLD, not a hard memory limit. It is best-effort
// evaluation scheduling, and peak footprint can exceed it:
//   - A long SCAN or IF branch accumulates across its whole body and is only
//     checked once control returns to the enclosing chain, so it can overshoot
//     by the size of that body.
//   - The per-instruction estimate is the largest tensor the instruction
//     touches, which can overcount (an op that only reads a large tensor is
//     charged for it) and so can trigger evaluation earlier than the true
//     pending bytes warrant.
//   - Ops that evaluate internally reduce the real pending work without
//     reducing the running estimate.
// Treat it as a knob to trade GPU syncs against peak memory, and tune it
// against measurements rather than expecting the value to bound RSS.
inline constexpr char kEvalThresholdBytesKey[] = "eval_threshold_bytes";

} // namespace mlx
} // namespace backends
} // namespace executorch
