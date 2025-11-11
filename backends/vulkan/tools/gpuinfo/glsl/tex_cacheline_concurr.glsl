/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}
#define VEC4_T ${texel_type(DTYPE)}

layout(std430) buffer;

${layout_declare_sampler(0, "r", "in_tex", DTYPE)}
${layout_declare_buffer(1, "w", "out_buf", DTYPE, "PRECISION", False)}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int niter = 1;

void main() {
    vec4 sum = vec4(0);
    int i = 0;
    for (; i < niter; ++i){
        $if DIM == 0:
            sum += texelFetch(in_tex, ivec3(gl_GlobalInvocationID[0], 0, 0), 0);
        $elif DIM == 1:
            sum +=  texelFetch(in_tex, ivec3(0, gl_GlobalInvocationID[0], 0), 0);
        $elif DIM == 2:
            sum +=  texelFetch(in_tex, ivec3(0, 0, gl_GlobalInvocationID[0]), 0);
    }

    // This is to ensure no compiler optimizations occur
    vec4 zero = vec4(i>>31);

    out_buf[0] = sum + zero;
}
