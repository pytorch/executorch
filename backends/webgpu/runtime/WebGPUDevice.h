/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <webgpu/webgpu.h>

namespace executorch {
namespace backends {
namespace webgpu {

struct WebGPUContext {
  WGPUInstance instance = nullptr;
  WGPUAdapter adapter = nullptr;
  WGPUDevice device = nullptr;
  WGPUQueue queue = nullptr;
};

WebGPUContext create_webgpu_context();
void destroy_webgpu_context(WebGPUContext& ctx);

// Global context used by WebGPUGraph::build() when no device is pre-set.
void set_default_webgpu_context(WebGPUContext* ctx);
WebGPUContext* get_default_webgpu_context();

} // namespace webgpu
} // namespace backends
} // namespace executorch
