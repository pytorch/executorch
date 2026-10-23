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

#include <ggml-backend.h>

#include <executorch/backends/native/runtime/engine/Engine.h>
#include <executorch/backends/native/runtime/engine/ggml/GgmlEngine.h>

namespace ptn {

struct Method;
class Package;

// Returns the options.device_index-th GPU device in ggml's registry. Throws if
// the linked ggml variant registers no such device.
ggml_backend_dev_t select_ggml_gpu_device(const GgmlEngineOptions& options);

// Mutable buffer key -> method whose live executable binds it.
using GgmlMutableKeyOwners = std::unordered_map<std::string, std::string>;

// The executable holds the method's mutable buffer keys in
// `mutable_key_owners` until it is destroyed.
std::unique_ptr<EngineExecutable> make_ggml_executable(
    const Method& method,
    const Package& package,
    const GgmlEngineOptions& options,
    size_t* materialized_constant_bytes,
    size_t* heap_copied_constant_bytes,
    GgmlMutableKeyOwners& mutable_key_owners);

} // namespace ptn
