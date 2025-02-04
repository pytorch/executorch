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
${layout_declare_spec_const(C, "int", "UNUSED_layout", "0")}

void main() {
  int out_bufi = int(gl_GlobalInvocationID.x);
  if (out_bufi >= numel) {
    return;
  }

  ivec4 out_tidx = bufi_to_tidx(out_bufi, out_strides);
  const int in_nchwi = tidx_to_nchwi(out_tidx, out_sizes);

  t_out[out_bufi] = nchw_in[in_nchwi];
}
