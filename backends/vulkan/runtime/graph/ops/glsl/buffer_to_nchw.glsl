#version 450 core

#define PRECISION ${PRECISION}

#define T ${buffer_scalar_type(DTYPE)}

#include "indexing_utils.h"

${define_required_extensions(DTYPE)}

layout(std430) buffer;

${layout_declare_tensor(0, "w", "nchw_buf", DTYPE, STORAGE)}
${layout_declare_tensor(1, "r", "t_in", DTYPE, STORAGE)}
${layout_declare_ubo(2, "ivec4", "in_sizes")}
${layout_declare_ubo(3, "ivec4", "in_strides")}
${layout_declare_ubo(4, "int", "numel")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// This constant is unused in this shader but is kept so that the signature is
// consistent with image_to_nchw.
layout(constant_id = 3) const int UNUSED_packed_dim = W_DIM;

void main() {
  int out_id = int(gl_GlobalInvocationID.x);
  if (out_id >= numel) {
    return;
  }

  ivec4 t_in_idx = from_nchw_buffer_i(out_id, in_sizes);
  const int in_id = to_buffer_id(t_in_idx, in_strides);

  nchw_buf[out_id] = t_in[in_id];
}
