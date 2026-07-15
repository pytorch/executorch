/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/gpu_shared/runtime/export.h>
#include <executorch/runtime/backend/backend_init_context.h>
#include <executorch/runtime/core/result.h>

#include <cstdint>
#include <string>

namespace executorch {
namespace backends {
namespace gpu_shared {

inline constexpr char kSharedContextTokenOption[] = "gpu_shared_context_token";
inline constexpr char kSharedContextModeOption[] = "gpu_shared_context_mode";
inline constexpr char kSharedGroupIdOption[] = "gpu_shared_group_id";

enum class SharedContextMode : uint8_t {
  kDisabled = 0,
  kLookupOnly = 1,
  kLookupOrCreate = 2,
  kCreateOnly = 3,
};

// Load-time configuration shared by the VGF and Vulkan delegates. These values
// are RuntimeSpec options: context selection is a deployment concern and must
// not be serialized into a backend CompileSpec or a .pte file.
struct SharedGpuRuntimeConfig final {
  std::string token = "default";
  int group_id = 0;
  SharedContextMode context_mode = SharedContextMode::kLookupOrCreate;

  bool enabled() const {
    return context_mode != SharedContextMode::kDisabled;
  }

  bool lookup_only() const {
    return context_mode == SharedContextMode::kLookupOnly;
  }

  bool lookup_or_create() const {
    return context_mode == SharedContextMode::kLookupOrCreate;
  }

  bool create_only() const {
    return context_mode == SharedContextMode::kCreateOnly;
  }
};

EXECUTORCH_GPU_SHARED_API runtime::Result<SharedGpuRuntimeConfig>
parse_shared_gpu_runtime_config(const runtime::BackendInitContext& context);

} // namespace gpu_shared
} // namespace backends
} // namespace executorch
