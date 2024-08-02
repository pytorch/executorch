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

${layout_declare_sampler(0, "r", "A", DTYPE)}
${layout_declare_buffer(1, "w", "B", DTYPE, "PRECISION", False)}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int niter = 1;
layout(constant_id = 4) const int nvec = 1;
layout(constant_id = 5) const int local_group_size = 1;
// The address mask works as a modulo because x % 2^n == x & (2^n - 1).
// This will help us limit address accessing to a specific set of unique
// addresses depending on the access size we want to measure.
layout(constant_id = 6) const int addr_mask = 1;
layout(constant_id = 7) const int workgroup_width = 1;

void main() {
    vec4 sum = vec4(0);
    uint offset = (gl_WorkGroupID[0] * workgroup_width  + gl_LocalInvocationID[0]) & addr_mask;

    int i = 0;
    for (; i < niter; ++i){
      VEC4_T in_texel;
      $for j in range(int(NUNROLL)):
        $if DIM == 0:
            in_texel = texelFetch(A, ivec3(offset, 0, 0), 0);
        $elif DIM == 1:
            in_texel = texelFetch(A, ivec3(0, offset, 0), 0);
        $elif DIM == 2:
            in_texel = texelFetch(A, ivec3(0, 0, offset), 0);

        sum *= in_texel;

        // On each unroll, a new unique address will be accessed through the offset,
        // limited by the address mask to a specific set of unique addresses
        offset = (offset + local_group_size) & addr_mask;
    }

    // This is to ensure no compiler optimizations occur
    vec4 zero = vec4(i>>31);

    B[gl_LocalInvocationID[0]] = sum + zero;
}
