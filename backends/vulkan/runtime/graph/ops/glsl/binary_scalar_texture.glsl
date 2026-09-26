/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Binary comparison ops write a bool/uint8 output dtype, which differs from
// the input dtype. IS_COMPARISON_OP is set explicitly per shader variant in the
// .yaml.

#version 450 core

$PROMOTED_DTYPE = get_higher_precision_dtype(DTYPE, SCALAR_VALUE_TYPE)

${define_required_extensions(STORAGE, DTYPE)}
${define_required_extensions(STORAGE, PROMOTED_DTYPE)}
${define_explicit_type_extensions(SCALAR_VALUE_TYPE)}
${define_explicit_type_extensions(PROMOTED_DTYPE)}
$if IS_COMPARISON_OP:
  ${define_required_extensions(STORAGE, "uint8")}

#define PRECISION ${PRECISION}

#define NAME ${VARIANT_NAME}

#define VEC4_T ${texel_load_type(DTYPE, STORAGE)}
#define T ${texel_load_component_type(DTYPE, STORAGE)}
#define SCALAR_T ${buffer_scalar_type(SCALAR_VALUE_TYPE)}
#define COMPUTE_VEC4_T ${texel_load_type(PROMOTED_DTYPE, STORAGE)}
#define COMPUTE_T ${texel_load_component_type(PROMOTED_DTYPE, STORAGE)}
$if IS_COMPARISON_OP:
  #define VEC4_OUT_T ${texel_load_type("uint8", STORAGE)}
$else:
  #define VEC4_OUT_T VEC4_T

#define op(X, Y) ${OPERATOR}

${define_active_storage_type(STORAGE)}

layout(std430) buffer;

#include "indexing.glslh"

$if IS_COMPARISON_OP:
  ${layout_declare_tensor(B, "w", "t_out", "uint8", STORAGE)}
$else:
  ${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE)}

${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE)}

${layout_declare_ubo(B, "TextureMetadata", "outp")}
${layout_declare_ubo(B, "TextureMetadata", "inp")}

layout(push_constant) uniform restrict Block {
  SCALAR_T scalar_value;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

$if not IS_COMPARISON_OP:
  #include "binary_op_defs.glslh"

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (out_of_bounds(pos, outp)) {
    return;
  }

  VEC4_T in_texel = texelFetch(t_in, pos, 0);
  VEC4_OUT_T out_texel = VEC4_OUT_T(
      op(COMPUTE_VEC4_T(in_texel), COMPUTE_VEC4_T(scalar_value)));

  imageStore(t_out, pos, out_texel);
}
