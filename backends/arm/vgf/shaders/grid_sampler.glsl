// Copyright 2026 Arm Limited and/or its affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#version 450
#extension GL_ARM_tensors : require

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Input0 {
    float input0[];
};

layout(set = 0, binding = 1) uniform tensorARM<float, 4> grid;

layout(set = 0, binding = 2) writeonly buffer Output0 {
    float output0[];
};

void main() {
    uvec3 gid = gl_GlobalInvocationID.xyz;
    uint output_batch = tensorSizeARM(grid, 0);
    uint output_height = tensorSizeARM(grid, 1);
    uint output_width = tensorSizeARM(grid, 2);
    if (gid.x >= output_width || gid.y >= output_height || gid.z >= output_batch) {
        return;
    }

    uint output_spatial_size = output_width * output_height;
    if (output_spatial_size == 0u) {
        return;
    }

    uint channels = output0.length() / (output_spatial_size * output_batch);
    uint base =
        ((gid.z * output_spatial_size) + (gid.y * output_width) + gid.x) * channels;
    for (uint channel = 0u; channel < channels; ++channel) {
        output0[base + channel] = 0.0;
    }
}
