/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

// CUDA-PRIVATE per-session mutable-state management. This is intentionally NOT
// a generic ExecuTorch (Module/Method/BackendInterface) API: it is the
// CUDA/AOTI implementation of "one loaded model, many logical contexts" and is
// consumed only by CUDA-specific LLM engines (e.g. Qwen35MoEEngine). The public
// serving abstraction stays LLMEngine/LLMSession.
//
// State is keyed by a CONTEXT (one per loaded model/engine), NOT
// process-global, so multiple models (e.g. Qwen + Gemma) and repeated engine
// lifecycles in one process stay isolated. An engine: creates a context, scopes
// its model load (begin/end) so the backend associates each delegate handle
// with the context, registers the model's mutable FQNs, creates sessions,
// selects an active session before each execute, and destroys the context on
// teardown.

namespace executorch {
namespace backends {
namespace cuda {

struct CudaDelegateHandle; // defined in cuda_delegate_handle.h

// Opaque per-engine context id (0 = invalid).
using MutableStateContext = int;
constexpr MutableStateContext kInvalidMutableContext = 0;

// Active-session sentinel: execute() rebinds nothing (single-session / legacy).
constexpr int kNoMutableSession = -1;

// --- Engine-facing API (call from the CUDA-specific LLM engine) -------------

// Create / destroy a context. destroy frees all of the context's sessions,
// templates, descriptors, and handle associations (safe to call once at engine
// teardown; sessions destroyed afterward become no-ops).
MutableStateContext mutable_state_create_context();
void mutable_state_destroy_context(MutableStateContext ctx);

// Scope a model load to a context: call begin BEFORE load_method and end AFTER,
// so the delegate handles initialized during the load are associated with
// `ctx`. Nesting is not supported (one load at a time per thread).
void mutable_state_begin_load(MutableStateContext ctx);
void mutable_state_end_load();

// Declare the context's per-session mutable-state FQNs (from the model's
// get_mutable_buffer_metadata). Call before begin_load/load_method.
void mutable_state_register_fqns(
    MutableStateContext ctx,
    const std::vector<std::string>& fqns);

// True if the context's loaded delegate(s) expose the AOTI constant-management
// symbols required for per-session rebinding. If false, the caller MUST run
// single-session.
bool mutable_state_available(MutableStateContext ctx);

// Bytes one session adds (sum of mutable-buffer sizes), 0 if not yet known.
int64_t mutable_state_bytes_per_session(MutableStateContext ctx);

// Validate every declared FQN was discovered in some loaded method's constants.
// Call after loading all methods; non-Ok must abort multi-session serving.
::executorch::runtime::Error mutable_state_validate_coverage(
    MutableStateContext ctx);

// Create / destroy a logical session within a context. create returns a token
// (>= 0); buffers are allocated lazily on the session's first execute.
::executorch::runtime::Result<int> mutable_state_create_session(
    MutableStateContext ctx);
void mutable_state_destroy_session(MutableStateContext ctx, int token);

// Select the active (context, session) for subsequent Module::execute calls ON
// THIS THREAD. Set before execute, reset token to kNoMutableSession after; the
// engine must hold its serialization lock across set + execute + read-out.
void mutable_state_set_active(MutableStateContext ctx, int token);

// --- CudaBackend-internal hooks (called from cuda_backend.cpp) ---------------

// From CudaBackend::init: associate this handle with the context currently
// being loaded (begin_load), record symbol availability, and build the
// descriptor table + capture initial templates from the still-initial
// constants.
void mutable_state_note_handle(CudaDelegateHandle* handle);

// From CudaBackend::execute, before running: if a session is active on this
// thread for this handle's context, rebind the container's mutable constants to
// the session's buffers. No-op (Ok) when no session is active.
::executorch::runtime::Error mutable_state_rebind_for_execute(
    CudaDelegateHandle* handle);

} // namespace cuda
} // namespace backends
} // namespace executorch
