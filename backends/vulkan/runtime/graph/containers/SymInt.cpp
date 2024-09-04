/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/containers/SymInt.h>

namespace vkcompute {

SymInt::SymInt(api::Context* context_p, const int32_t val)
    : gpu_buffer(context_p, val){};

void SymInt::set(const int32_t val) {
  gpu_buffer.update(val);
}

void SymInt::operator=(const int32_t val) {
  gpu_buffer.update(val);
}

} // namespace vkcompute
