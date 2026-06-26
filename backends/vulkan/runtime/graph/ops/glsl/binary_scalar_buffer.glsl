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

${define_required_extensions(STORAGE, DTYPE)}
${define_explicit_type_extensions(SCALAR_VALUE_TYPE)}
$if IS_COMPARISON_OP:
  ${define_required_extensions(STORAGE, "uint8")}

#define PRECISION ${PRECISION}

#define NAME ${VARIANT_NAME}

#define T ${buffer_scalar_type(DTYPE)}
#define SCALAR_T ${buffer_scalar_type(SCALAR_VALUE_TYPE)}
$if IS_COMPARISON_OP:
  #define OUT_T ${buffer_scalar_type("uint8")}
$else:
  #define OUT_T ${buffer_scalar_type(DTYPE)}

#define op(X, Y) ${OPERATOR}

${define_active_storage_type(STORAGE)}

layout(std430) buffer;

#include "indexing.glslh"

$if IS_COMPARISON_OP:
  ${layout_declare_tensor(B, "w", "t_out", "uint8", STORAGE)}
$else:
  ${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE)}

${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE)}

${layout_declare_ubo(B, "BufferMetadata", "outp")}
${layout_declare_ubo(B, "BufferMetadata", "inp")}

layout(push_constant) uniform restrict Block {
  SCALAR_T scalar_value;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

$if not IS_COMPARISON_OP:
  #include "binary_op_defs.glslh"

void main() {
  const uint out_bufi = gl_GlobalInvocationID.x;
  if (out_of_bounds(out_bufi, outp)) {
    return;
  }

  t_out[out_bufi] = OUT_T(op(t_in[out_bufi], T(scalar_value)));
}
