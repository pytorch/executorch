/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

layout(std430) buffer;

${layout_declare_buffer(0, "w", "out_buff", DTYPE)}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

$if METHOD == "scheduler":
    shared int shared_counter;
$elif METHOD == "physical":
    layout(constant_id = 3) const int NITER = 1;
$else:
    $raise Exception("Unsupported value for warp_size")

void main() {

    $if METHOD == "scheduler":
        shared_counter = 0;
        memoryBarrierShared();
        int i = atomicAdd(shared_counter, 1);
        memoryBarrierShared();
        out_buff[gl_GlobalInvocationID[0]] = i;
    $else:
        int sum = 0;
        for (int j = 0; j < NITER; ++j) {
            // Integer division is an exemplary multi-cycle instruction that can
            // hardly be optimized, thus reducing the impact of latency hiding.
            sum += j / 3;
            barrier();
        }
        out_buff[gl_GlobalInvocationID[0]] = sum;
}
