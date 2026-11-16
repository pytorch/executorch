/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions(STORAGE, DTYPE)}

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_load_type(DTYPE, STORAGE)}

// Runtime bounds of int32 clamps stay int so that large values remain exact.
$if DYNAMIC_PARAMS:
  #define PARAM_T ${texel_load_component_type(DTYPE, "texture3d")}
$else:
  #define PARAM_T float
#define IN_T ${texel_type(DTYPE)}
#define OUT_T IN_T
$if DTYPE in ["half", "float"]:
  #define UNARY_OP_IS_FLOAT
$elif DTYPE == "uint8":
  #define UNARY_OP_IS_BITWISE

${define_active_storage_type(STORAGE)}

#include "indexing.glslh"

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE, is_scalar_array=False)}

$if STORAGE == "buffer":
  ${layout_declare_ubo(B, "BufferMetadata", "outp")}
$else:
  ${layout_declare_ubo(B, "TextureMetadata", "outp")}
$if DYNAMIC_PARAMS:
  ${layout_declare_ubo(B, "uint", "minimum")}
  ${layout_declare_ubo(B, "uint", "maximum")}

layout(push_constant) uniform restrict Block {
$if DYNAMIC_PARAMS:
  ivec2 bounds_are_int;
$else:
  float minimum;
  float maximum;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "op_type", "0")}
${layout_declare_spec_const(C, "int", "out_layout", "CONTIG_LAYOUT_INT")}
const int out_packed_dim = get_packed_dim(out_layout);

#include "dispatch.glslh"
#include "unary_op.glslh"

$if DYNAMIC_PARAMS:
  float decode_bound(const uint value, const int is_int) {
    return is_int != 0 ? float(int(value)) : uintBitsToFloat(value);
  }

  #define PARAM_A PARAM_T(decode_bound(minimum, bounds_are_int.x))
  #define PARAM_B PARAM_T(decode_bound(maximum, bounds_are_int.y))
$else:
  #define PARAM_A minimum
  #define PARAM_B maximum

void main() {
#ifdef USING_BUFFER
  const uint idx = linear_idx_from_gid();
  if (idx >= div_4(numel(outp))) {
    return;
  }
  const IN_T in_texel = IN_T(t_in[idx]);
#else
  const ivec3 idx = ivec3(gl_GlobalInvocationID);
  if (out_of_bounds(idx, outp)) {
    return;
  }
  const IN_T in_texel = IN_T(texelFetch(t_in, idx, 0));
#endif

  OUT_T out_texel = apply_unary_op(op_type, in_texel, PARAM_A, PARAM_B);
  // Input padding lanes are assumed zero, so only ops that may map zero to a
  // nonzero value need to mask them. The op is not evaluated at zero to skip
  // the mask: the Mali-G715 compiler folds min(max(0, a), b) to 0 when b < 0.
  if (!unary_op_preserves_zero(op_type)) {
    out_texel = mask_padding_lanes(out_texel, idx, outp, out_packed_dim);
  }

#ifdef USING_BUFFER
  t_out[idx] = VEC4_T(out_texel);
#else
  imageStore(t_out, idx, VEC4_T(out_texel));
#endif
}
