/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}
#define UBO_PARAMS ${UBO_PARAMS}

#define VEC4_T ${texel_type(DTYPE)}
#define T ${buffer_scalar_type(DTYPE)}

${define_active_storage_type("buffer")}
${define_required_extensions(DTYPE)}

layout(std430) buffer;

#include "indexing_utils.h"
${layout_declare_tensor(B, "w", "t_out", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_in", DTYPE, "buffer")}

$if UBO_PARAMS:
  $if OP_NAME == "slice":
    ${layout_declare_ubo(B, "int", "start")}
    ${layout_declare_ubo(B, "int", "step")}

  $if OP_NAME == "select":
    ${layout_declare_ubo(B, "int", "index")}

layout(push_constant) uniform restrict Block {
  ivec4 in_sizes;
  ivec4 out_strides;
  ivec4 in_strides;
  int out_numel;
  int selected_dim;
  $if not UBO_PARAMS:
    $if OP_NAME == "slice":
      int start;
      int step;

    $if OP_NAME == "select":
      int index;
};

${layout_declare_spec_const(C, "int", "out_layout", "DEFAULT_LAYOUT")}
${layout_declare_spec_const(C, "int", "in_layout", "DEFAULT_LAYOUT")}

const lowp ivec4 out_dim_order = unhash_dim_order(out_layout);

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#include "${OP_NAME}.glslh"

void main() {
  const int out_bufi = ivec3(gl_GlobalInvocationID).x;
  if (out_bufi >= out_numel) {
    return;
  }

  const ivec4 out_tidx = bufi_to_tidx(out_bufi, out_strides, out_dim_order);
  ivec4 in_tidx = out_tidx_to_in_tidx(out_tidx);

  const int in_bufi = tidx_to_bufi(in_tidx, in_strides);
  t_out[out_bufi] = t_in[in_bufi];
}
