/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/api/Context.h>
#include <executorch/backends/vulkan/runtime/api/containers/ParamsBuffer.h>

namespace vkcompute {

/*
 * Represents a symbolic integer whose value can be variable. It is implemented
 * as a thin wrapper around a `ParamsBuffer` object that holds the value of the
 * integer. The `ParamsBuffer` object allows the value of the symbolic integer
 * to be changed from the CPU and have those changes be visible to all shaders
 * that use the symbolic integer; it also allows the value of the symbolic
 * integer to be the result of a compute shader.
 *
 * Regular scalar types represented by `TypeTag::INT` cannot be used for
 * symbolic integers because their value is assumed to be constant; therefore
 * the `Value` instance holding the value of the scalar does not contain
 * any reference to the GPU buffers used to pass its value into compute shaders.
 * Therefore, updating the value of the scalar does not impact the value seen
 * by compute shaders.
 */
struct SymInt final {
  api::ParamsBuffer gpu_buffer;

  explicit SymInt(api::Context* context_p, const int32_t val);

  void set(const int32_t val);

  void operator=(const int32_t val);
};

} // namespace vkcompute
