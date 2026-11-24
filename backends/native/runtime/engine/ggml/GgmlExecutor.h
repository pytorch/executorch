// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <ggml-backend.h>
#include <ggml-cpp.h>

#include <executorch/backends/native/runtime/engine/Engine.h>
#include <executorch/backends/native/runtime/engine/ggml/GgmlCache.h>
#include <executorch/backends/native/runtime/engine/ggml/GgmlEngine.h>

namespace ptn {

struct Method;
class Package;

// Returns the options.device_index-th GPU device in ggml's registry. Throws if
// the linked ggml variant registers no such device.
ggml_backend_dev_t select_ggml_gpu_device(const GgmlEngineOptions& options);

struct GgmlLoadPlan;

namespace ggml_detail {
struct GraphInstance;
} // namespace ggml_detail

// Storage the executables of one context share. `keys` are the data keys that
// any method of its program writes; `storage` holds each one's buffer. Program
// validation has already required every binding of a key to have the same
// metadata. `constants` holds the device tensors of immutable constants, keyed
// by their load plans: the package bytes they load, where, and every
// transform that decides their device bytes. The context owns `storage` from
// the first successful compile that binds each key, so releasing executables
// keeps its value; a constant lives while an executable holds it.
struct GgmlContextState {
  struct Constant {
    std::weak_ptr<ggml_tensor> tensor;
    // How the tensor was loaded, which every later binder must match.
    std::shared_ptr<const GgmlLoadPlan> plan;
  };

  std::unordered_set<std::string> keys;
  std::unordered_map<std::string, std::shared_ptr<ggml_tensor>> storage;
  // Every method of the program, which a key's storage type must suit.
  std::vector<const Method*> methods;
  std::unordered_map<std::string, Constant> constants;

  // The backends every executable of the context runs on, and the one
  // scheduler whose compute buffers all their graphs allocate from. Each
  // compile reserves the buffers for its method's worst-case graph, which
  // `reserved` keeps for the scheduler's next growth.
  std::vector<ggml_backend_ptr> backends;
  ggml_backend_t primary_backend = nullptr;
  ggml_backend_sched_ptr scheduler;
  size_t scheduler_size = 0;
  std::vector<std::weak_ptr<const ggml_detail::GraphInstance>> reserved;
  // The graph the scheduler holds an allocation for, or null.
  const ggml_cgraph* allocated = nullptr;
  // Bytes of the scheduler's compute buffers, and how often they grew.
  size_t compute_buffer_bytes = 0;
  size_t compute_buffer_allocations = 0;
};

// `kv_cache`, which must outlive the executable, serves the method's
// kvcache::update_and_attend calls; a method with such calls and no cache
// does not compile.
std::unique_ptr<EngineExecutable> make_ggml_executable(
    const Method& method,
    const Package& package,
    const GgmlEngineOptions& options,
    size_t* materialized_constant_bytes,
    size_t* heap_copied_constant_bytes,
    size_t* f16_kv_cache_bytes,
    GgmlContextState& state,
    GgmlKvCache kv_cache = {});

} // namespace ptn
