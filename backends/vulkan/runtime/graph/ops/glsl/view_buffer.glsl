#version 450 core

#define PRECISION ${PRECISION}

#define T ${buffer_scalar_type(DTYPE)}

${define_required_extensions(STORAGE, DTYPE)}

layout(std430) buffer;

#include "indexing.glslh"

${layout_declare_tensor(B, "w", "t_outp", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_inp", DTYPE, STORAGE)}

${layout_declare_ubo(B, "BufferMetadata", "outp")}
${layout_declare_ubo(B, "BufferMetadata", "inp")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#include "dispatch.glslh"

${layout_declare_spec_const(C, "int", "outp_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "inp_layout", "CONTIG_LAYOUT_INT")}

/*
 * The insight behind the view operation is that the contiguous index of each
 * tensor element in the input and output tensors are the same.
 */
void main() {
  const uint outp_bufi = linear_idx_from_gid();
  if (outp_bufi >= numel(outp)) {
    return;
  }

  if (has_same_physical_element_order(outp, inp, outp_layout, inp_layout)) {
    t_outp[outp_bufi] = t_inp[outp_bufi];
    return;
  }

  const TensorIndex outp_tidx = linear_idx_to_tensor_idx(outp, outp_bufi);
  if (out_of_bounds(outp_tidx, outp)) {
    t_outp[outp_bufi] = T(0);
    return;
  }

  const uint contig_idx = tensor_idx_to_contiguous_idx(outp, outp_tidx);
  const TensorIndex inp_tidx = contiguous_idx_to_tensor_idx(inp, contig_idx);
  const uint inp_bufi = tensor_idx_to_linear_idx(inp, inp_tidx);

  t_outp[outp_bufi] = t_inp[inp_bufi];
}
