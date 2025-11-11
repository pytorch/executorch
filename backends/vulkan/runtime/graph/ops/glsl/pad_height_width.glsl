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

  bool mask_height = idx.y >= pad_top && idx.y <= pad_top + in_sizes.y - 1;
  bool mask_width = idx.x >= pad_left && idx.x <= pad_left + in_sizes.x - 1;

  VEC4_T outtex = VEC4_T(fill_value);
  if (mask_height && mask_width) {
    ivec4 in_idx = idx;
    in_idx.x -= pad_left;
    in_idx.y -= pad_top;
    outtex = texelFetch(t_in, to_texture_pos(in_idx, in_sizes, packed_dim), 0);
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
