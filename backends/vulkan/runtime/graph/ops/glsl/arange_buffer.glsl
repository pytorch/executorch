/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions("buffer", DTYPE)}

#define PRECISION ${PRECISION}

#define T ${buffer_scalar_type(DTYPE)}

${define_active_storage_type("buffer")}

layout(std430) buffer;

#include "indexing.glslh"

${layout_declare_tensor(B, "w", "t_out", DTYPE, "buffer")}

${layout_declare_ubo(B, "BufferMetadata", "outp")}
${layout_declare_ubo(B, "uint", "start")}
${layout_declare_ubo(B, "uint", "step")}

layout(push_constant) uniform restrict Block {
  ivec2 params_are_int;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#include "dispatch.glslh"

float decode_param(const uint value, const int is_int) {
  return is_int != 0 ? float(int(value)) : uintBitsToFloat(value);
}

void main() {
  const uint out_bufi = linear_idx_from_gid();
  if (out_of_bounds(out_bufi, outp)) {
    return;
  }

  const float start_val = decode_param(start, params_are_int.x);
  const float step_val = decode_param(step, params_are_int.y);
  t_out[out_bufi] = T(start_val + out_bufi * step_val);
}
