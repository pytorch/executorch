/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions(STORAGE, DTYPE)}
${define_required_extensions("buffer", BUF_DTYPE)}

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_load_type(DTYPE, STORAGE)}
#define SCALAR_T ${texel_load_component_type(DTYPE, STORAGE)}

${define_active_storage_type(STORAGE)}

layout(std430) buffer;

#include "indexing.glslh"

${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_buffer(B, "r", "buf_in", BUF_DTYPE)}

${layout_declare_ubo(B, "TextureMetadata", "outp")}

$if not FROM_STAGING:
  ${layout_declare_ubo(B, "BufferMetadata", "buf_meta")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "out_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "transpose_hw", "0")}

$if not FROM_STAGING:
  ${layout_declare_spec_const(C, "int", "buf_layout", "CONTIG_LAYOUT_INT")}

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (out_of_bounds(pos, outp)) {
    return;
  }

  TensorIndex4D tidx = texture_pos_to_tensor4d_idx_simple(outp, pos, out_layout);

  const int packed_dim = get_packed_dim(out_layout);
  int packed_dim_val;
  if (packed_dim == 0) {
    packed_dim_val = tidx.data.x;
  } else if (packed_dim == 1) {
    packed_dim_val = tidx.data.y;
  } else if (packed_dim == 2) {
    packed_dim_val = tidx.data.z;
  } else {
    packed_dim_val = tidx.data.w;
  }

  int packed_dim_size;
  if (packed_dim == 0) {
    packed_dim_size = outp.sizes.x;
  } else if (packed_dim == 1) {
    packed_dim_size = outp.sizes.y;
  } else if (packed_dim == 2) {
    packed_dim_size = outp.sizes.z;
  } else {
    packed_dim_size = outp.sizes.w;
  }

  VEC4_T texel = VEC4_T(0);
  int limit = min(4, packed_dim_size - packed_dim_val);

  for (int comp = 0; comp < limit; comp++) {
    TensorIndex4D buf_tidx = tidx;
    if (transpose_hw == 1) {
      int tmp = buf_tidx.data.x;
      buf_tidx.data.x = buf_tidx.data.y;
      buf_tidx.data.y = tmp;
    }

    $if FROM_STAGING:
      // Compute contiguous NCHW index
      ivec4 s = outp.sizes;
      if (transpose_hw == 1) {
        s.xy = s.yx;
      }
      int nchwi = buf_tidx.data.x
                + buf_tidx.data.y * s.x
                + buf_tidx.data.z * s.x * s.y
                + buf_tidx.data.w * s.x * s.y * s.z;
      texel[comp] = SCALAR_T(buf_in[nchwi]);
    $else:
      int bufi = tensor4d_idx_to_buf_idx(buf_meta, buf_tidx, buf_layout);
      texel[comp] = SCALAR_T(buf_in[bufi]);

    if (packed_dim == 0) {
      tidx.data.x++;
    } else if (packed_dim == 1) {
      tidx.data.y++;
    } else if (packed_dim == 2) {
      tidx.data.z++;
    } else {
      tidx.data.w++;
    }
  }

#ifdef USING_TEXTURE2D
  imageStore(t_out, pos.xy, texel);
#else
  imageStore(t_out, pos, texel);
#endif
}
