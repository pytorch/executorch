/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <webgpu/webgpu.h>

#ifdef WGPU_BACKEND_ENABLE_PROFILING
#include <executorch/backends/webgpu/runtime/WebGPUQueryPool.h>

#include <memory>
#endif // WGPU_BACKEND_ENABLE_PROFILING

namespace executorch {
namespace backends {
namespace webgpu {

struct WebGPUContext {
  WGPUInstance instance = nullptr;
  WGPUAdapter adapter = nullptr;
  WGPUDevice device = nullptr;
  WGPUQueue queue = nullptr;
  // True if the device was created with the ShaderF16 feature; reserved for a
  // future fp16 storage/compute path (fp32 is used when false or unset).
  bool shader_f16_supported = false;
#ifdef WGPU_BACKEND_ENABLE_PROFILING
  // True if the device was created with the TimestampQuery feature (bench).
  bool timestamp_supported = false;
  // Bench-only: timestamp-query pool, lazily created in execute() (env-gated).
  std::unique_ptr<WebGPUQueryPool> querypool;
#endif // WGPU_BACKEND_ENABLE_PROFILING
};

WebGPUContext create_webgpu_context();
void destroy_webgpu_context(WebGPUContext& ctx);

// Global context used by WebGPUGraph::build() when no device is pre-set.
void set_default_webgpu_context(WebGPUContext* ctx);
WebGPUContext* get_default_webgpu_context();

} // namespace webgpu
} // namespace backends
} // namespace executorch
