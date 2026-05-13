/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/native/NativeBackendInternal.h>

#include <executorch/backends/native/runtimes/cpu/CpuRuntime.h>
#include <executorch/backends/native/runtimes/host_pool/HostPoolRuntime.h>

#ifdef ET_NATIVE_HAS_METAL
#include <executorch/backends/native/runtimes/metal/MetalRuntime.h>
#endif

#include <string>
#include <unordered_set>
#include <utility>

namespace executorch {
namespace backends {
namespace native {

namespace {

// Build the default ProviderSet. Only the host pool is registered
// eagerly — it's cheap, has no device init, and is needed by every
// program (it owns graph IO and cross-runtime intermediates).
//
// Compute providers (CPU, Metal, fake_accel) are constructed LAZILY by
// the per-init lazy_*_runtime() singletons below, and only when the
// load-time `compute_unit` option allows them. This avoids paying
// device init costs (MTLDevice creation, etc.) for loads that don't
// use that compute unit.
std::vector<std::unique_ptr<Runtime>> make_default_runtimes() {
  std::vector<std::unique_ptr<Runtime>> ps;
  ps.push_back(std::make_unique<HostPoolRuntime>());
  return ps;
}

} // namespace

// Lazy CPU singleton. Always available; the only conditioning is the
// load-time `compute_unit` option (caller decides whether to invoke).
Runtime* lazy_cpu_runtime() {
  static auto cpu = std::make_unique<CpuRuntime>();
  return cpu.get();
}

// Lazy "fake_accel" singleton — a CpuRuntime restricted to add/mul
// for routing tests. Opted in via compute_unit="fake_accel".
//
// Uses accept_io_directly=false so the router emits explicit IO
// TransferSteps around fake_accel segments. This more faithfully
// simulates a real accelerator (which would have its own memory and
// require copies) and exercises the cross-engine IO path in tests.
Runtime* lazy_fake_accel_runtime() {
  static auto fake = []() {
    std::unordered_set<std::string> allow = {
        "aten::add",
        "aten::add.Tensor",
        "aten::mul",
        "aten::mul.Tensor",
    };
    auto r = std::make_unique<CpuRuntime>(
        "fake_accel", std::move(allow), /*accept_io_directly=*/false);
    ET_LOG(
        Info,
        "NativeBackend: registering CpuRuntime(\"fake_accel\") (lazy; routing-test scaffolding)");
    return r;
  }();
  return fake.get();
}

#ifdef ET_NATIVE_HAS_METAL
// Lazy Metal singleton. Constructed (and the MTLDevice initialized) on
// the first init() that allows Metal via compute_unit. Returns nullptr
// if Metal compiled in but its stream couldn't initialize (no Metal-
// capable device).
//
// Function-local static defers construction until the first call
// (Meyers singleton; thread-safe in C++11+). Once constructed, the
// MetalRuntime lives until process exit.
Runtime* lazy_metal_runtime() {
  static std::unique_ptr<MetalRuntime> metal =
      []() -> std::unique_ptr<MetalRuntime> {
    auto m = std::make_unique<MetalRuntime>();
    if (!m->stream_ready()) {
      ET_LOG(Info, "NativeBackend: MetalRuntime unavailable");
      return nullptr;
    }
    ET_LOG(Info, "NativeBackend: registering MetalRuntime (lazy)");
    return m;
  }();
  return metal.get();
}
#endif

// Process-wide RuntimeRegistry. Constructed once on first call (Meyers
// singleton; thread-safe in C++11+); lives until process exit.
//
// The Runtimes it owns — including any per-Runtime shared state they
// hold directly (Metal stream, future kernel caches / JIT artifacts) —
// are therefore shared across every DelegateInstance in the process.
// Per-program state still lives on each DelegateInstance's
// `owned_instances` (the Engines).
//
// Contract for non-trivial per-Runtime shared state added later: must
// be safe for concurrent reads + occasional writes from multiple
// delegate threads (see threading model in design doc).
RuntimeRegistry& native_runtime_registry() {
  static RuntimeRegistry registry(make_default_runtimes());
  return registry;
}

} // namespace native
} // namespace backends
} // namespace executorch
