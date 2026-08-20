#version 450 core

${define_required_extensions(STORAGE, DTYPE)}
${define_required_extensions("buffer", BUF_DTYPE)}

#define PRECISION ${PRECISION}

#define T ${buffer_scalar_type(DTYPE)}

layout(std430) buffer;

#include "indexing.glslh"

${layout_declare_tensor(B, "w", "t_outp", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "nchw_in", BUF_DTYPE, STORAGE)}

${layout_declare_ubo(B, "BufferMetadata", "outp")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// This constant is unused in this shader but is kept so that the signature is
// consistent with nchw_to_image.
${layout_declare_spec_const(C, "int", "unused", "0")}
${layout_declare_spec_const(C, "int", "transpose_hw", "0")}

void main() {
  const uint outp_bufi = int(gl_GlobalInvocationID.x);
  if (outp_bufi >= numel(outp)) {
    return;
  }

  TensorIndex outp_tidx = linear_idx_to_tensor_idx(outp, outp_bufi);
  uint nchwi;

  if (transpose_hw == 1) {
    BufferMetadata transposed_meta = outp;
    transposed_meta.sizes[0].xy = transposed_meta.sizes[0].yx;
    outp_tidx.data[0].xy = outp_tidx.data[0].yx;
    nchwi = tensor_idx_to_contiguous_idx(transposed_meta, outp_tidx);
  }
  // Normal case
  else {
    nchwi = tensor_idx_to_contiguous_idx(outp, outp_tidx);
  }

  t_outp[outp_bufi] = T(nchw_in[nchwi]);
}
