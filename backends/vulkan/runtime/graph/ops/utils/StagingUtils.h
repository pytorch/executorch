/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

namespace vkcompute {

//
// Functions to copy data into and out of a staging buffer
//

void copy_ptr_to_staging(
    const void* src,
    api::StorageBuffer& staging,
    const size_t nbytes);
void copy_staging_to_ptr(
    api::StorageBuffer& staging,
    void* dst,
    const size_t nbytes);

void set_staging_zeros(api::StorageBuffer& staging, const size_t nbytes);

//
// Functions to get shaders
//

api::ShaderInfo get_nchw_to_image_shader(const vTensor& v_dst);
api::ShaderInfo get_image_to_nchw_shader(const vTensor& v_src);

} // namespace vkcompute
