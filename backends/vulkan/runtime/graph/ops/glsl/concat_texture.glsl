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
#define T ${buffer_scalar_type(DTYPE)}

#define USING_TEXTURE3D

layout(std430) buffer;

#include "indexing_utils.h"

${layout_declare_tensor(B, "rw", "t_out", DTYPE, "texture3d")}

$for i in range(NUM_INPUTS):
  ${layout_declare_tensor(B, "r", "t_inp" + str(i), DTYPE, "texture3d")}

${layout_declare_tensor(B, "r", "t_concat_offset", "int", "buffer")}

${layout_declare_ubo(B, "int", "concat_dim")}

$in_metadata = ""
$for i in range(NUM_INPUTS):
  $in_metadata += "ivec4 inp" + str(i) + "_sizes;\n"

layout(push_constant) uniform restrict Block {
  ivec4 out_sizes;
  ${in_metadata}
};

${layout_declare_spec_const(C, "int", "out_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 out_axis_map = unhash_axis_map(out_layout);
const lowp int out_packed_dim = unhash_packed_dim(out_layout);

$for i in range(NUM_INPUTS):
  ${layout_declare_spec_const(C, "int", "inp" + str(i) + "_layout", "DEFAULT_LAYOUT")}
  const lowp ivec4 inp${i}_axis_map = unhash_axis_map(inp${i}_layout);
  const lowp int inp${i}_packed_dim = unhash_packed_dim(inp${i}_layout);

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#define NUM_INPUTS ${NUM_INPUTS}

#include "concat_utils.glslh"

/*
 * This shader template concatenates up to NUM_INPUT input tensors to the
 * output tensor along the concat_dim. Elements from the input tensor will
 * be inserted along the output's concat_dim starting at concat_offset.
 *
 * Each thread is responsible for writing out one output texel. The data
 * required for the output texel may be read from multiple input texels of one
 * input tensor.
 */
void main() {
  const int tid = ivec3(gl_GlobalInvocationID).x;

  // Sum of the sizes of all input tensors along the concat_dim
  const int concat_numel = total_concat_dim_numel();

  // The 1-3 input tensors are interpreted as one concatenated tensor ("volume")
  // along the concat_dim for the purposes of tensor indexing. Each thread is
  // responsible for writing out 4 elements along the packed dim of the output
  // tensor by reading the source data from the input tensor(s).
  ivec4 inp_volume_sizes = out_sizes;
  inp_volume_sizes[concat_dim] = total_concat_dim_numel();

  // Reconstruct inp_volume_texel_sizes from Concat.cpp
  ivec4 inp_volume_texel_sizes = inp_volume_sizes;
  inp_volume_texel_sizes[out_packed_dim] = DIV_UP_4(
      inp_volume_texel_sizes[out_packed_dim]
  ) + 1;

  // tensor index of the first element that will be read from the input volume
  ivec4 inp_volume_start_tidx = nchwi_to_tidx(tid, inp_volume_texel_sizes);
  inp_volume_start_tidx[out_packed_dim] = MUL_4(
      inp_volume_start_tidx[out_packed_dim]
  );

  int concat_offset = t_concat_offset[0];

  // tensor index of the first element that will be written to the output tensor
  ivec4 out_write_start_tidx = inp_volume_start_tidx;
  out_write_start_tidx[concat_dim] += concat_offset;

  // To write to the the desired output element, we will need to load the texel
  // to which the element belongs. Calculate the tensor index of the first
  // element of that texel.
  ivec4 out_read_start_tidx = out_write_start_tidx;
  out_read_start_tidx[out_packed_dim] = ALIGN_DOWN_4(
      out_write_start_tidx[out_packed_dim]);

  // bounds check
  if (any(greaterThanEqual(out_read_start_tidx, out_sizes))) {
    return;
  }

  ivec3 out_pos = tidx_to_pos(
      out_read_start_tidx,
      out_sizes,
      out_axis_map,
      out_packed_dim
  );

  VEC4_T out_texel = imageLoad(t_out, out_pos);

  VEC4_T test_texel = VEC4_T(-1.0);

  for (int comp = 0; comp < 4; ++comp) {
    ivec4 out_tidx = out_read_start_tidx;
    out_tidx[out_packed_dim] += comp;


    // It's possible that the current texel element has been written to as part
    // of the previous input batch; if so, then don't overwrite this texel
    // element
    if (out_tidx[concat_dim] < concat_offset) {
      test_texel[comp] = -5.0;
      continue;
    }

    // Calculate the tidx of the input volume that corresponds to this output
    // element
    ivec4 inp_volume_tidx = out_tidx;
    inp_volume_tidx[concat_dim] -= concat_offset;

    // go through the list of input tensors, and figure out which input this
    // output element should be read from.
    $for i in range(NUM_INPUTS):
      if (inp_volume_tidx[concat_dim] < inp${i}_sizes[concat_dim]) {
        // Special fast path case if, for the first output texel element, the
        // corresponding input element is at the start of the texel it belongs
        // to. In this case, the input texel can be written as-is to the output
        // texel. Also require that The entire input texel is valid and does not
        // contain any padding elements.
        if (comp == 0 &&
            out_tidx[out_packed_dim] % 4 == 0 &&
            inp_volume_tidx[inp${i}_packed_dim] % 4 == 0 &&
            inp_volume_tidx[inp${i}_packed_dim] + 3 < inp${i}_sizes[inp${i}_packed_dim]) {
          const ivec3 in_pos = tidx_to_pos(
              inp_volume_tidx,
              inp${i}_sizes,
              inp${i}_axis_map,
              inp${i}_packed_dim);

          out_texel = texelFetch(t_inp${i}, in_pos, 0);
          break;
        }

        // Otherwise, locate the specific input element required
        const ivec4 in_posi = tidx_to_posi(
            inp_volume_tidx,
            inp${i}_sizes,
            inp${i}_axis_map,
            inp${i}_packed_dim);

        out_texel[comp] = texelFetch(t_inp${i}, in_posi.xyz, 0)[in_posi.w];
        test_texel[comp] = out_texel[comp];
        continue;
      }
      else {
        inp_volume_tidx[concat_dim] -= inp${i}_sizes[concat_dim];
      }
  }

  imageStore(t_out, out_pos, out_texel);
}
