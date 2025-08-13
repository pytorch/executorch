/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

// Binary comparison ops require that the output is boolean and not the same as input.
$IS_COMPARISON_OP = (any([name in VARIANT_NAME for name in ["binary_eq",  "binary_lt", "binary_le", "binary_gt", "binary_ge"]]))

#define NAME ${VARIANT_NAME}

#define VEC4_T ${texel_type(DTYPE)}
$if IS_COMPARISON_OP:
  #define T ${buffer_scalar_type("uint8")}
  #define VEC4_OUT_T ${texel_type("uint8")}
$else:
  #define T ${buffer_scalar_type(DTYPE)}
  #define VEC4_OUT_T VEC4_T

#define op(X, Y, A) ${OPERATOR}

${define_active_storage_type(STORAGE)}
${define_required_extensions(DTYPE)}


$if IS_COMPARISON_OP:
  ${define_required_extensions("uint8")}

layout(std430) buffer;

$if IS_COMPARISON_OP:
  ${layout_declare_tensor(B, "w", "t_out", "uint8", STORAGE)}
$else:
  ${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE)}

${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_other", DTYPE, STORAGE)}

$if STORAGE == "buffer":
  layout(push_constant) uniform restrict Block {
    ivec4 in_sizes;
    ivec4 other_sizes;
    ivec4 out_strides;
    ivec4 in_strides;
    ivec4 other_strides;
    int out_numel;
    float alpha;
  };
$else:
  layout(push_constant) uniform restrict Block {
    ivec4 out_sizes;
    ivec4 in_sizes;
    ivec4 other_sizes;
    ivec2 broadcast_params;
    float alpha;
  };

#include "broadcasting_utils.h"
#include "indexing_utils.h"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "out_layout", "DEFAULT_LAYOUT")}
${layout_declare_spec_const(C, "int", "in_layout", "DEFAULT_LAYOUT")}
${layout_declare_spec_const(C, "int", "other_layout", "DEFAULT_LAYOUT")}

$if STORAGE == "buffer":
  const lowp ivec4 out_dim_order = unhash_dim_order(out_layout);
$else:
  const lowp ivec4 out_axis_map = unhash_axis_map(out_layout);
  const lowp int packed_dim = unhash_packed_dim(out_layout);

  const lowp ivec4 in_axis_map = unhash_axis_map(in_layout);

  const lowp ivec4 other_axis_map = unhash_axis_map(other_layout);

#ifdef USING_BUFFER

void main() {
  const int out_bufi = ivec3(gl_GlobalInvocationID).x;
  if (out_bufi >= out_numel) {
    return;
  }

  // Simple case; no broadcasting
  if (in_sizes == other_sizes) {
    t_out[out_bufi] = T(op(t_in[out_bufi], t_other[out_bufi], T(alpha)));
    return;
  }

  const ivec4 out_tidx = bufi_to_tidx(out_bufi, out_strides, out_dim_order);
  const ivec4 in_tidx = min(out_tidx, in_sizes - 1);
  const ivec4 other_tidx = min(out_tidx, other_sizes - 1);

  const int in_bufi = tidx_to_bufi(in_tidx, in_strides);
  const int other_bufi = tidx_to_bufi(other_tidx, other_strides);

  t_out[out_bufi] = T(op(t_in[in_bufi], t_other[other_bufi], T(alpha)));
}

#else // USING_TEXTURE

void main() {
  const ivec3 lpos = ivec3(gl_GlobalInvocationID);
  const ivec4 tidx = lpos_to_tidx(lpos, out_sizes, out_axis_map.w, packed_dim);

  if (any(greaterThanEqual(tidx, out_sizes))) {
    return;
  }

  // broadcast on logical sizes
  ivec4 in_idx = broadcast_indices(tidx, in_sizes);
  VEC4_T in_texel = VEC4_T(load_texel(
    t_in,
    // read axis mapped texel
    tidx_to_pos(in_idx, in_sizes, in_axis_map, packed_dim)));

  // broadcast on logical sizes
  ivec4 other_idx = broadcast_indices(tidx, other_sizes);
  VEC4_T other_texel = VEC4_T(load_texel(
    t_other,
    // read axis mapped texel
    tidx_to_pos(other_idx, other_sizes, other_axis_map, packed_dim)));

  // Check boolean broadcast flags; we use ivec2 instead of bvec2 for alignment.
  if (broadcast_params.x > 0) {
    in_texel = in_texel.xxxx;
  }
  if (broadcast_params.y > 0) {
    other_texel = other_texel.xxxx;
  }

  write_texel_lpos(
    t_out,
    lpos,
    VEC4_OUT_T(op(in_texel, other_texel, alpha)),
    out_axis_map);
}

#endif
