/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Unit test for the MLX per-session mutable-state manager
// (backends/mlx/runtime/mlx_mutable_state.{h,cpp}).
//
// Verifies that two sessions created on one loaded program get independent
// mutable buffers: writing into session A's buffer does not leak into session
// B's, and A's value persists across a rebind to B and back. This is the MLX
// analogue of the CUDA "no-bleed" guarantee, exercised directly on the manager
// (no model or tokenizer needed).

#include "MLXExecutor.h"
#include "MLXLoader.h"
#include "mlx_mutable_state.h"

#include <mlx/mlx.h>

#include <cstdio>

using namespace ::executorch::backends::mlx;

namespace {

int g_failures = 0;

#define CHECK(cond)                                         \
  do {                                                      \
    if (!(cond)) {                                          \
      std::printf("FAIL: %s (line %d)\n", #cond, __LINE__); \
      ++g_failures;                                         \
    }                                                       \
  } while (0)

// Build a minimal program with a single 1-element float mutable buffer at tid
// 0.
MLXProgram make_program() {
  MLXProgram program;
  program.num_mutable_buffer_tensors = 1;
  program.mutable_buffer_map.push_back(SlotVariant{0, SlotType::TensorSlot});
  TensorMeta meta;
  meta.shape.push_back(ShapeDim{/*value=*/1});
  meta.scalar_type = ScalarType::Float;
  program.tensor_meta.resize(1);
  program.tensor_meta[0] = meta;
  return program;
}

float read0(const MutableBufferData& bufs) {
  auto arr = bufs.get(Tid{0});
  ::mlx::core::eval(arr);
  return arr.item<float>();
}

} // namespace

int main() {
  MLXProgram program = make_program();

  // Handle's default (init-time) mutable buffers.
  MutableBufferData default_bufs;
  load_mutable_buffers(program, default_bufs);

  int dummy = 0;
  const void* handle = &dummy;

  MutableStateContextOwner owner;
  CHECK(static_cast<bool>(owner));

  // Associate the handle with the context (as MLXBackend::init would).
  owner.with_load_scope(
      [&]() { mutable_state_note_handle(handle, &program, &default_bufs); });

  CHECK(owner.available());
  CHECK(owner.bytes_per_session() == static_cast<int64_t>(sizeof(float)));

  auto tokA = owner.create_session();
  auto tokB = owner.create_session();
  CHECK(tokA.ok());
  CHECK(tokB.ok());
  CHECK(tokA.get() != tokB.get());

  ExecutionState state;

  // Session A: rebind, then write a marker (7.0) into its buffer.
  owner.with_active_session(tokA.get(), [&]() {
    auto err = mutable_state_rebind_for_execute(handle, state);
    CHECK(err == ::executorch::runtime::Error::Ok);
    state.mutable_buffers->set(
        Tid{0}, ::mlx::core::full({1}, 7.0f, ::mlx::core::float32));
    return err;
  });

  // Session B: a fresh rebind must see zeros, not A's marker.
  owner.with_active_session(tokB.get(), [&]() {
    auto err = mutable_state_rebind_for_execute(handle, state);
    CHECK(err == ::executorch::runtime::Error::Ok);
    CHECK(read0(*state.mutable_buffers) == 0.0f);
    return err;
  });

  // Back to session A: the marker must persist (isolation, no bleed).
  owner.with_active_session(tokA.get(), [&]() {
    auto err = mutable_state_rebind_for_execute(handle, state);
    CHECK(err == ::executorch::runtime::Error::Ok);
    CHECK(read0(*state.mutable_buffers) == 7.0f);
    return err;
  });

  // With sessions present, executing without an active session is refused
  // (prevents running against unmanaged/shared state).
  {
    auto err = mutable_state_rebind_for_execute(handle, state);
    CHECK(err == ::executorch::runtime::Error::InvalidState);
  }

  owner.destroy_session(tokA.get());
  owner.destroy_session(tokB.get());
  mutable_state_forget_handle(handle);

  if (g_failures == 0) {
    std::printf("OK: mlx_mutable_state isolation test passed\n");
    return 0;
  }
  std::printf("FAILED: %d checks\n", g_failures);
  return 1;
}
