#version 450 core

${define_required_extensions(STORAGE, DTYPE)}
${define_required_extensions("buffer", BUF_DTYPE)}

#define PRECISION ${PRECISION}

#define T ${buffer_scalar_type(DTYPE)}
#define DST_T ${buffer_scalar_type(BUF_DTYPE)}

layout(std430) buffer;

#include "indexing.glslh"

${layout_declare_tensor(B, "w", "nchw_buf", BUF_DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_inp", DTYPE, STORAGE)}

${layout_declare_ubo(B, "BufferMetadata", "inp")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#include "dispatch.glslh"

// This constant is unused in this shader but is kept so that the signature is
// consistent with image_to_nchw.
${layout_declare_spec_const(C, "int", "unused", "0")}

void main() {
  const uint nchwi = linear_idx_from_gid();
  const TensorIndex inp_tidx = contiguous_idx_to_tensor_idx(inp, nchwi);
  if (out_of_bounds(inp_tidx, inp)) {
    return;
  }

  const uint inp_bufi = tensor_idx_to_linear_idx(inp, inp_tidx);

  nchw_buf[nchwi] = DST_T(t_inp[inp_bufi]);
}
