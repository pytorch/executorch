#version 450 core

#define PRECISION ${PRECISION}

#define T ${buffer_scalar_type(DTYPE)}

#include "indexing_utils.h"

${define_required_extensions(DTYPE)}

layout(std430) buffer;

${layout_declare_tensor(0, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(1, "r", "nchw_in", DTYPE, STORAGE)}
${layout_declare_ubo(2, "ivec4", "out_sizes")}
${layout_declare_ubo(3, "ivec4", "out_strides")}
${layout_declare_ubo(4, "int", "numel")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// This constant is unused in this shader but is kept so that the signature is
// consistent with nchw_to_image.
layout(constant_id = 3) const int UNUSED_packed_dim = W_DIM;

void main() {
  int out_id = int(gl_GlobalInvocationID.x);
  if (out_id >= numel) {
    return;
  }

  ivec4 out_idx = to_tensor_idx(out_id, out_strides);
  const int in_id = to_nchw_buffer_i(out_idx, out_sizes);

  t_out[out_id] = nchw_in[in_id];
}
