#version 450 core

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_type(DTYPE)}

layout(std430) buffer;

#include "indexing_utils.h"

${layout_declare_tensor(0, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(1, "r", "t_in", DTYPE, STORAGE)}
${layout_declare_ubo(2, "ivec4", "out_sizes")}
${layout_declare_ubo(3, "ivec4", "in_sizes")}
${layout_declare_ubo(4, "int", "pad_left", "int", "pad_top", "int", "pad_front")}
${layout_declare_ubo(5, "float", "fill_value")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int packed_dim = C_DIM;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const ivec4 idx = to_tensor_idx(pos, out_sizes, packed_dim);

  if (pos_out_of_bounds(pos, out_sizes, packed_dim)) {
    return;
  }

  VEC4_T outtex = VEC4_T(fill_value);
  // mask_z/y/x is used to determine whether need to fecth data from input tensor
  bool mask_z = (idx.z + 3) < pad_front || idx.z > (pad_front + in_sizes.z - 1);
  bool mask_y = idx.y >= pad_top && idx.y <= pad_top + in_sizes.y - 1;
  bool mask_x = idx.x >= pad_left && idx.x <= pad_left + in_sizes.x - 1;

  if (!mask_z && mask_y && mask_x) {
    // channel_mask is to determine the situation that when padding channel dimension,
    // in one texel, some elements are filled vaule and some value are from input tensor
    ivec4 c_ind = ivec4(idx.z) + ivec4(0, 1, 2, 3);
    ivec4 channel_mask = ivec4(lessThan(c_ind, ivec4(pad_front))) + ivec4(greaterThan(c_ind, ivec4(pad_front + in_sizes.z - 1)));

    ivec4 in_idx = idx;
    in_idx.x -= pad_left;
    in_idx.y -= pad_top;
    in_idx.z -= divup4(pad_front) * 4;
    const int shift = pad_front % 4;
    VEC4_T cur_in_texel = texelFetch(t_in, to_texture_pos(in_idx, in_sizes, packed_dim), 0);
    VEC4_T next_in_texel;
    // When shift is not 0, we need to read 2 texels from input tensor to write into output
    // for example:
    // input texel is [[1 2 3 4], [5 6 x x]] and front_pad = 2
    // output texel is [[p p 1 2], [3 4 5 6]], where p is the filled value then need to fetch 2 texels to fill [3 4 5 6].
    if (shift != 0) {
      in_idx.z += 4;
      next_in_texel = texelFetch(t_in, to_texture_pos(in_idx, in_sizes, packed_dim), 0);
    } else {
      next_in_texel = cur_in_texel;
    }

    VEC4_T inter_texel;
    for (int i = 0; i < 4; i++) {
      if (i < shift) {
        inter_texel[i] = cur_in_texel[4-shift+i];
      } else {
        inter_texel[i] = next_in_texel[i-shift];
      }
    }
    outtex = inter_texel * (VEC4_T(1) - channel_mask) + outtex * channel_mask;
  }

  int packed_idx = idx[packed_dim];
  const int packed_dim_size = out_sizes[packed_dim];
  if (packed_idx + 3 >= packed_dim_size) {
    ivec4 packed_ind = ivec4(packed_idx) + ivec4(0, 1, 2, 3);
    VEC4_T valid_idx = VEC4_T(lessThan(packed_ind, ivec4(packed_dim_size)));
    outtex = outtex * valid_idx;
  }

  imageStore(t_out, pos, outtex);
}
