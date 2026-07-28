@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> inp: array<f32>;

// Up to 4D. The handler right-aligns dims into [4] (leading entries = 1, left/
// right pad = 0 for unpadded/leading dims), so the shader is rank-agnostic and
// always iterates 4 dims. in_dims[d] = input extent, left[d] = that dim's
// left-pad, out_dims[d] = in_dims[d] + left[d] + right[d].
struct Params {
  out_dims: vec4<u32>,
  in_dims: vec4<u32>,
  left: vec4<u32>,
  out_numel: u32,
  value: f32,
  _p0: u32,
  _p1: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 256;
override stride_x: u32 = 4294967295u; // = count_x * wg_size; set by host for 2D-spill

// constant_pad_nd, gather form, NCHW row-major fp32. One thread per OUTPUT
// element: decode its 4D coords, subtract each dim's left-pad to get the input
// coord; if ALL input coords are in-bounds -> copy inp[flat_in], else write
// `value`. Pure copy/fill -> bit-exact. (CPU-derisked == torch at 0.)
@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * stride_x + gid.x;
  if (i >= params.out_numel) {
    return;
  }

  // decode out coords (last dim fastest)
  var rem = i;
  let o3 = rem % params.out_dims.w;
  rem = rem / params.out_dims.w;
  let o2 = rem % params.out_dims.z;
  rem = rem / params.out_dims.z;
  let o1 = rem % params.out_dims.y;
  rem = rem / params.out_dims.y;
  let o0 = rem % params.out_dims.x;

  // subtract left pad -> input coord (wrapping subtract; check via < in_dim on
  // the unsigned result catches negatives because they wrap to huge values)
  let c0 = o0 - params.left.x;
  let c1 = o1 - params.left.y;
  let c2 = o2 - params.left.z;
  let c3 = o3 - params.left.w;

  let in0 = o0 >= params.left.x && c0 < params.in_dims.x;
  let in1 = o1 >= params.left.y && c1 < params.in_dims.y;
  let in2 = o2 >= params.left.z && c2 < params.in_dims.z;
  let in3 = o3 >= params.left.w && c3 < params.in_dims.w;

  if (in0 && in1 && in2 && in3) {
    let in_idx =
        ((c0 * params.in_dims.y + c1) * params.in_dims.z + c2) * params.in_dims.w
        + c3;
    out[i] = inp[in_idx];
  } else {
    out[i] = params.value;
  }
}
