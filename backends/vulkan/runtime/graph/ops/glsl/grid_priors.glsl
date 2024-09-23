#version 450 core

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_type(DTYPE)}

layout(std430) buffer;

#include "indexing_utils.h"

${layout_declare_tensor(0, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_ubo(1, "ivec4", "in_sizes")}
${layout_declare_ubo(2, "ivec4", "out_sizes")}
${layout_declare_ubo(3, "int", "stride", "float", "offset")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int packed_dim = C_DIM;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const ivec4 idx = to_tensor_idx(pos, out_sizes, packed_dim);

  if (pos_out_of_bounds(pos, out_sizes, packed_dim)) {
    return;
  }
  int width = in_sizes.x;
  VEC4_T outtex;
  if (pos.x == 0) {
    float value = (pos.y % width + offset) * stride;
    outtex = VEC4_T(value, 0, 0, 0);
  } else if (pos.x == 1) {
    float value = (pos.y / width + offset) * stride;
    outtex = VEC4_T(value, 0, 0, 0);
  }

  imageStore(t_out, pos, outtex);
}
