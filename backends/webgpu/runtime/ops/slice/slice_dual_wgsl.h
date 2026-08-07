/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

namespace executorch::backends::webgpu {

// @generated from slice_dual.wgsl - DO NOT EDIT.
// wgsl-sha256: 845ec79985822d5621fd16688a45af8f363310f949ea4ac14c77bf0e84f9c9c0
inline constexpr const char* kSliceDualWGSL = R"(
// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct TensorMeta {
  ndim: u32,
  numel: u32,
  sizes: array<vec4<u32>, 2>,
  strides: array<vec4<u32>, 2>,
}
@group(0) @binding(2) var<uniform> out_meta: TensorMeta;
@group(0) @binding(3) var<uniform> in_meta: TensorMeta;

struct Params {
  dim: u32,
  start: u32,
  step: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

// Second destination for a chained aten.slice_copy.Tensor whose range is the
// whole of `output` (start == 0, step == 1, end >= size on its dim), i.e. a
// pure elementwise copy of `output`. Same flat index, so one gather feeds both
// stores. `output` is still written, so no consumer of it is assumed dead.
@group(0) @binding(5) var<storage, read_write> output2: array<f32>;

// Fixed workgroup size (Apple scalar ALU: no override loop bounds). 64 is the
// value clamp_workgroup_size already yields for slice.wgsl.
@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    // 2D-spill: numel can exceed the 65535 per-dim grid cap. Stride is the
    // literal 64 above, which is the real threads-per-group either way.
    let out_bufi = gid.x + gid.y * (num_workgroups.x * 64u);
    if (out_bufi >= out_meta.numel) {
        return;
    }

    // Gather: out_bufi -> in_bufi, sliced dim coord = start + coord*step.
    var rem = out_bufi;
    var in_bufi: u32 = 0u;
    for (var d: u32 = 0u; d < out_meta.ndim; d = d + 1u) {
        let coord = rem / out_meta.strides[d >> 2u][d & 3u];
        rem = rem % out_meta.strides[d >> 2u][d & 3u];
        var in_coord = coord;
        if (d == params.dim) {
            in_coord = params.start + coord * params.step;
        }
        in_bufi = in_bufi + in_coord * in_meta.strides[d >> 2u][d & 3u];
    }
    let v = input[in_bufi];
    output[out_bufi] = v;
    output2[out_bufi] = v;
}
// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
)";

inline constexpr uint32_t kSliceDualWorkgroupSizeX = 64;
inline constexpr uint32_t kSliceDualWorkgroupSizeY = 1;
inline constexpr uint32_t kSliceDualWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
