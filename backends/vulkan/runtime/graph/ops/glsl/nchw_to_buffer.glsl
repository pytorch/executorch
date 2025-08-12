#version 450 core

#define PRECISION ${PRECISION}

#define T ${buffer_scalar_type(DTYPE)}

#include "indexing_utils.h"

${define_required_extensions(DTYPE)}

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "nchw_in", DTYPE, STORAGE)}

$if USE_PUSH_CONST:
  layout(push_constant) uniform restrict Block {
    ivec4 out_sizes;
    ivec4 out_strides;
    int numel;
  };
$else:
  ${layout_declare_ubo(B, "ivec4", "out_sizes")}
  ${layout_declare_ubo(B, "ivec4", "out_strides")}
  ${layout_declare_ubo(B, "int", "numel")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "out_layout", "DEFAULT_DIM_ORDER")}
const lowp ivec4 out_dim_order = unhash_dim_order(out_layout);
${layout_declare_spec_const(C, "int", "transpose_hw", "0")}

void main() {
  int out_bufi = int(gl_GlobalInvocationID.x);
  if (out_bufi >= numel) {
    return;
  }

  ivec4 out_tidx = bufi_to_tidx(out_bufi, out_strides, out_dim_order);

  ivec4 sizes = out_sizes;
  if (transpose_hw == 1) {
    sizes.xy = sizes.yx;
    out_tidx.xy = out_tidx.yx;
  }
  const int in_nchwi = tidx_to_nchwi(out_tidx, sizes);

  t_out[out_bufi] = nchw_in[in_nchwi];
}
