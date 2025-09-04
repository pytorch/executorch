/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define T ${buffer_scalar_type(DTYPE)}

${define_active_storage_type("buffer")}
${define_required_extensions(DTYPE)}

layout(std430) buffer;

${layout_declare_tensor(B, "w", "concat_offset", DTYPE, "buffer")}

${layout_declare_ubo(B, "int", "concat_dim")}

$for i in range(NUM_INPUTS):
  ${layout_declare_ubo(B, "ivec4", "in" + str(i+1) + "_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  // Only one thread needs to update the offset
  if (gl_GlobalInvocationID.x != 0) {
    return;
  }

  // Sum up the sizes along the concat dimension for all input tensors
  int total_size = 0;
  $for i in range(NUM_INPUTS):
    total_size += in${i+1}_sizes[concat_dim];

  // Add to the current offset
  concat_offset[0] += T(total_size);
}
