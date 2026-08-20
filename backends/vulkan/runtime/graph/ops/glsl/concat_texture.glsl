/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions("texture3d", DTYPE)}

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_type(DTYPE)}
#define T ${buffer_scalar_type(DTYPE)}

${define_active_storage_type("texture3d")}

layout(std430) buffer;

#include "indexing.glslh"

${layout_declare_tensor(B, "rw", "t_out", DTYPE, "texture3d")}

$for i in range(NUM_INPUTS):
  ${layout_declare_tensor(B, "r", "t_inp" + str(i), DTYPE, "texture3d")}

${layout_declare_tensor(B, "r", "t_concat_offset", "int", "buffer")}

${layout_declare_ubo(B, "TextureMetadata", "outp")}

$for i in range(NUM_INPUTS):
  ${layout_declare_ubo(B, "TextureMetadata", "inp" + str(i) + "p")}

${layout_declare_spec_const(C, "int", "concat_dim", "0")}
${layout_declare_spec_const(C, "int", "out_layout", "CONTIG_LAYOUT_INT")}

$for i in range(NUM_INPUTS):
  ${layout_declare_spec_const(C, "int", "inp" + str(i) + "_layout", "CONTIG_LAYOUT_INT")}

const int out_packed_dim = get_packed_dim(out_layout);

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#define NUM_INPUTS ${NUM_INPUTS}

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
  const int tid = int(gl_GlobalInvocationID.x);

  // Compute inp_volume_sizes from output sizes, replacing concat_dim with the
  // sum of all input sizes along that dimension.
  ivec4 inp_volume_sizes = outp.sizes;
  int total_concat_dim = 0;
  $for i in range(NUM_INPUTS):
    total_concat_dim += safe_idx(inp${i}p.sizes, concat_dim);
  safe_set(inp_volume_sizes, concat_dim, total_concat_dim);

  // Reconstruct inp_volume_texel_sizes from Concat.cpp
  ivec4 inp_volume_texel_sizes = inp_volume_sizes;
  {
    int packed_size = inp_volume_texel_sizes[out_packed_dim];
    inp_volume_texel_sizes[out_packed_dim] = div_up_4(packed_size) + 1;
  }

  // Decompose flat index into 4D tensor index (contiguous WHCN layout)
  TensorIndex4D inp_volume_start_tidx;
  {
    int remaining = tid;
    const int div_x = remaining / inp_volume_texel_sizes.x;
    const int div_xy = div_x / inp_volume_texel_sizes.y;
    inp_volume_start_tidx.data = ivec4(
        remaining % inp_volume_texel_sizes.x,
        div_x % inp_volume_texel_sizes.y,
        div_xy % inp_volume_texel_sizes.z,
        div_xy / inp_volume_texel_sizes.z);
  }
  inp_volume_start_tidx.data[out_packed_dim] =
      inp_volume_start_tidx.data[out_packed_dim] * 4;

  int concat_offset = t_concat_offset[0];

  // tensor index of the first element that will be written to the output tensor
  TensorIndex4D out_write_start_tidx = inp_volume_start_tidx;
  out_write_start_tidx.data[concat_dim] += concat_offset;

  // To write to the desired output element, we will need to load the texel
  // to which the element belongs. Calculate the tensor index of the first
  // element of that texel.
  TensorIndex4D out_read_start_tidx = out_write_start_tidx;
  out_read_start_tidx.data[out_packed_dim] =
      out_write_start_tidx.data[out_packed_dim] & ~3;

  // bounds check
  if (any(greaterThanEqual(out_read_start_tidx.data, outp.sizes))) {
    return;
  }

  ivec3 out_pos = tensor4d_idx_to_texel_pos_simple(
      outp, out_read_start_tidx, out_layout);

  VEC4_T out_texel = imageLoad(t_out, out_pos);

  for (int comp = 0; comp < 4; ++comp) {
    TensorIndex4D out_tidx = out_read_start_tidx;
    out_tidx.data[out_packed_dim] += comp;

    // It's possible that the current texel element has been written to as part
    // of the previous input batch; if so, then don't overwrite this texel
    // element
    if (out_tidx.data[concat_dim] < concat_offset) {
      continue;
    }

    // Calculate the tidx of the input volume that corresponds to this output
    // element
    TensorIndex4D inp_volume_tidx = out_tidx;
    inp_volume_tidx.data[concat_dim] -= concat_offset;

    // go through the list of input tensors, and figure out which input this
    // output element should be read from.
    $for i in range(NUM_INPUTS):
      if (inp_volume_tidx.data[concat_dim] < safe_idx(inp${i}p.sizes, concat_dim)) {
        // Special fast path case if, for the first output texel element, the
        // corresponding input element is at the start of the texel it belongs
        // to. In this case, the input texel can be written as-is to the output
        // texel. Also require that the entire input texel is valid and does not
        // contain any padding elements.
        const int inp${i}_packed_dim = get_packed_dim(inp${i}_layout);
        if (comp == 0 &&
            out_tidx.data[out_packed_dim] % 4 == 0 &&
            inp_volume_tidx.data[inp${i}_packed_dim] % 4 == 0 &&
            inp_volume_tidx.data[inp${i}_packed_dim] + 3 < safe_idx(inp${i}p.sizes, inp${i}_packed_dim)) {
          const ivec3 in_pos = tensor4d_idx_to_texel_pos_simple(
              inp${i}p, inp_volume_tidx, inp${i}_layout);

          out_texel = texelFetch(t_inp${i}, in_pos, 0);
          break;
        }

        // Otherwise, locate the specific input element required
        const TextureElementIndex in_elem = tensor4d_idx_to_texture_element_idx_simple(
            inp${i}p, inp_volume_tidx, inp${i}_layout);

        out_texel[comp] = texelFetch(t_inp${i}, in_elem.pos, 0)[in_elem.comp];
        continue;
      }
      else {
        inp_volume_tidx.data[concat_dim] -= safe_idx(inp${i}p.sizes, concat_dim);
      }
  }

  imageStore(t_out, out_pos, out_texel);
}
