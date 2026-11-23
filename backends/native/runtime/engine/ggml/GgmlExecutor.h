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

#include <executorch/backends/native/runtime/engine/Engine.h>
#include <executorch/backends/native/runtime/engine/ggml/GgmlCache.h>
#include <executorch/backends/native/runtime/engine/ggml/GgmlEngine.h>

namespace ptn {

struct Method;
class Package;

// Returns the options.device_index-th GPU device in ggml's registry. Throws if
// the linked ggml variant registers no such device.
ggml_backend_dev_t select_ggml_gpu_device(const GgmlEngineOptions& options);

// The mutable state of one context: the data keys that any method of its
// program writes, and the storage of each, which every executable of the
// context binding the key shares. The context owns the storage from the first
// successful compile that binds the key, so releasing executables keeps its
// value. Program validation has already required every binding of a key to
// have the same metadata.
struct GgmlContextState {
  std::unordered_set<std::string> keys;
  std::unordered_map<std::string, std::shared_ptr<ggml_tensor>> storage;
  // Every method of the program, which a key's storage type must suit.
  std::vector<const Method*> methods;
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
