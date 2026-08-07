/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <webgpu/webgpu.h>

#include <cstdint>
#include <vector>

namespace executorch::backends::webgpu {

class WebGPUGraph;

void q4gsw_linear_impl_with_input_buffer(
    WebGPUGraph& graph,
    const std::vector<int>& args,
    WGPUBuffer input_buffer,
    uint64_t input_nbytes);

} // namespace executorch::backends::webgpu
