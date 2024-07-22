#version 450 core

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_type(DTYPE)}

layout(std430) buffer;

#include "indexing_utils.h"

${layout_declare_tensor(0, "w", "t_out_1", DTYPE, STORAGE)}
${layout_declare_tensor(1, "w", "t_out_2", DTYPE, STORAGE)}
${layout_declare_tensor(2, "r", "t_in_1", DTYPE, STORAGE)}
${layout_declare_tensor(3, "r", "t_in_2", DTYPE, STORAGE)}
${layout_declare_ubo(4, "ivec4", "out_sizes")}
${layout_declare_ubo(5, "ivec4", "in_sizes_1")}
${layout_declare_ubo(6, "ivec4", "in_sizes_2")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int packed_dim = C_DIM;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const ivec4 idx = to_tensor_idx(pos, out_sizes, packed_dim);

  if (pos_out_of_bounds(pos, out_sizes, packed_dim)) {
    return;
  }

  ivec4 in_idx_1 = idx;
  in_idx_1.x = idx.y;
  in_idx_1.y = 0;
  imageStore(t_out_1, pos, texelFetch(t_in_1, to_texture_pos(in_idx_1, in_sizes_1, packed_dim), 0));

  ivec4 in_idx_2 = idx;
  in_idx_2.y = 0;
  imageStore(t_out_2, pos, texelFetch(t_in_2, to_texture_pos(in_idx_2, in_sizes_2, packed_dim), 0));
}
