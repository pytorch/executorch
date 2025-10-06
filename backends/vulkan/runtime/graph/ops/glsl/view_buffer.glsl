#version 450 core

#define PRECISION ${PRECISION}

#define T ${buffer_scalar_type(DTYPE)}

${define_required_extensions(DTYPE)}

layout(std430) buffer;

#include "indexing.glslh"

${layout_declare_tensor(B, "w", "t_outp", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_inp", DTYPE, STORAGE)}

${layout_declare_ubo(B, "BufferMetadata", "outp")}
${layout_declare_ubo(B, "BufferMetadata", "inp")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * The insight behind the view operation is that the contiguous index of each
 * tensor element in the input and output tensors are the same.
 */
void main() {
  const uint outp_bufi = gl_GlobalInvocationID.x;
  if (outp_bufi >= numel(outp)) {
    return;
  }

  TensorIndex outp_tidx;
  linear_idx_to_tensor_idx(outp, outp_bufi, outp_tidx);

  // To map the output to the input, find the input element that has the same
  // contiguous index as the output element.
  const uint contig_idx = tensor_idx_to_contiguous_idx(outp, outp_tidx);

  TensorIndex inp_tidx;
  contiguous_idx_to_tensor_idx(inp, contig_idx, inp_tidx);

  const uint inp_bufi = tensor_idx_to_linear_idx(inp, inp_tidx);

  t_outp[outp_bufi] = t_inp[inp_bufi];
}
