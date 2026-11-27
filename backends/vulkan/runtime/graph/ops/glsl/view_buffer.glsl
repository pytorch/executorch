#version 450 core

#define PRECISION ${PRECISION}

#define T ${buffer_scalar_type(DTYPE)}
#define VEC4_T ${texel_load_type(DTYPE, STORAGE)}

${define_required_extensions(STORAGE, DTYPE)}

$if VEC4:
  #define VEC4_OUTPUT

layout(std430) buffer;

#include "indexing.glslh"

$if VEC4:
  ${layout_declare_tensor(B, "w", "t_outp", DTYPE, STORAGE, is_scalar_array=False)}
$else:
  ${layout_declare_tensor(B, "w", "t_outp", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_inp", DTYPE, STORAGE)}

${layout_declare_ubo(B, "BufferMetadata", "outp")}
${layout_declare_ubo(B, "BufferMetadata", "inp")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#include "dispatch.glslh"

${layout_declare_spec_const(C, "int", "outp_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "inp_layout", "CONTIG_LAYOUT_INT")}

/*
 * Advances tidx to the next element in contiguous order and moves bufi to that
 * element's buffer index.
 */
void advance_contiguous(
    inout TensorIndex tidx,
    inout uint bufi,
    const BufferMetadata meta) {
  [[unroll]] for (int d = 0; d < DIMLIMIT; ++d) {
    tidx.data[div_4(d)][mod_4(d)]++;
    bufi += stride_at(meta, d);
    if (idx_at(tidx, d) < size_at(meta, d)) {
      return;
    }
    bufi -= size_at(meta, d) * stride_at(meta, d);
    tidx.data[div_4(d)][mod_4(d)] = 0;
  }
}

/*
 * The insight behind the view operation is that the contiguous index of each
 * tensor element in the input and output tensors are the same.
 */
void main() {
#ifdef VEC4_OUTPUT
  // Each thread writes one texel of four consecutive elements along the
  // output's packed dim, which must have stride 1.
  const int outp_packed_dim = get_packed_dim(outp_layout);
  const uint texel_idx = linear_idx_from_gid();
  if (texel_idx >= div_4(numel(outp))) {
    return;
  }

  const uint outp_bufi = mul_4(texel_idx);
  if (has_same_physical_element_order(outp, inp, outp_layout, inp_layout)) {
    t_outp[texel_idx] = VEC4_T(
        t_inp[outp_bufi],
        t_inp[outp_bufi + 1],
        t_inp[outp_bufi + 2],
        t_inp[outp_bufi + 3]);
    return;
  }

  VEC4_T texel = VEC4_T(0);
  const TensorIndex outp_tidx = linear_idx_to_tensor_idx(outp, outp_bufi);
  if (!out_of_bounds(outp_tidx, outp)) {
    const uint contig_idx = tensor_idx_to_contiguous_idx(outp, outp_tidx);
    uint contig_lane_stride = 1;
    [[unroll]] for (int d = 0; d < DIMLIMIT; ++d) {
      if (d < outp_packed_dim) {
        contig_lane_stride *= size_at(outp, d);
      }
    }
    const int valid_lanes = num_valid_lanes(outp_tidx, outp, outp_packed_dim);

    TensorIndex inp_tidx = contiguous_idx_to_tensor_idx(inp, contig_idx);
    uint inp_bufi = tensor_idx_to_linear_idx(inp, inp_tidx);
    texel[0] = t_inp[inp_bufi];
    [[unroll]] for (int i = 1; i < 4; ++i) {
      if (i < valid_lanes) {
        // When every dim below the packed dim has size 1, consecutive lanes
        // are consecutive in contiguous order, so the input index can be
        // advanced instead of recomputed.
        if (contig_lane_stride == 1) {
          advance_contiguous(inp_tidx, inp_bufi, inp);
        } else {
          inp_tidx = contiguous_idx_to_tensor_idx(
              inp, contig_idx + i * contig_lane_stride);
          inp_bufi = tensor_idx_to_linear_idx(inp, inp_tidx);
        }
        texel[i] = t_inp[inp_bufi];
      }
    }
  }
  t_outp[texel_idx] = texel;
#else
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
#endif
}
