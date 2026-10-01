/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT    ${FORMAT}

layout(std430) buffer;
#if IO_STORAGE == texture3d
layout(set = 0, binding = 0, FORMAT) uniform PRECISION image3D out_tex;
layout(set = 0, binding = 1, FORMAT) uniform PRECISION image3D final_state_tex;
layout(set = 0, binding = 2) uniform PRECISION sampler3D q_tex;
layout(set = 0, binding = 3) uniform PRECISION sampler3D k_tex;
layout(set = 0, binding = 4) uniform PRECISION sampler3D v_tex;
layout(set = 0, binding = 5) uniform PRECISION sampler3D decay_tex;
layout(set = 0, binding = 6) uniform PRECISION sampler3D beta_tex;
layout(set = 0, binding = 7) uniform PRECISION sampler3D initial_state_tex;
#endif

layout(set = 0, binding = 8) uniform UniformParams {
  ivec4 sizes;     // out sizes
  ivec4 k_sizes;   // key sizes
} params;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
    const ivec3 pos = ivec3(gl_GlobalInvocationID);
    
    // Bounds check
    if (pos.x >= params.sizes.x || pos.z >= params.sizes.z) {
        return;
    }

    int seq_len = params.sizes.y;
    int k_dim = params.k_sizes.x;

    // To implement the math correctly, a thread at pos.x (handling 4 V elements)
    // must iterate over the seq_len.
    // Inside the sequence loop, it iterates over K to compute the dot products.
    // Since state is stored in memory, reading/writing it dynamically here is required.
    // For simplicity in this initial implementation, we will just copy V to OUT 
    // to prove compilation and pipeline execution works, and then we can refine the math.
    // Full implementation requires local shared memory caching of the state matrix.
    
    for (int t = 0; t < seq_len; t++) {
        ivec3 tex_pos = ivec3(pos.x, t, pos.z);
        vec4 v_val = texelFetch(v_tex, tex_pos, 0);
        imageStore(out_tex, tex_pos, v_val);
    }
}
