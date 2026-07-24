@group(0) @binding(0) var<storage, read_write> t_out: array<f32>;
@group(0) @binding(1) var<storage, read> t_x: array<f32>;
@group(0) @binding(2) var<storage, read> t_weight: array<f32>;
@group(0) @binding(3) var<storage, read> t_bias: array<f32>;

struct Params {
  N: u32,
  IC: u32,
  H_in: u32,
  W_in: u32,
  OC: u32,
  H_out: u32,
  W_out: u32,
  Kh: u32,
  Kw: u32,
  stride_h: u32,
  stride_w: u32,
  pad_h: u32,
  pad_w: u32,
  dil_h: u32,
  dil_w: u32,
  has_bias: u32,
  numel: u32,
  groups: u32,
  ic_per_group: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
  output_min: f32,
  output_max: f32,
}
@group(0) @binding(4) var<uniform> params: Params;

override wg_size: u32 = 64u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  // 2D-folded flat index (lifts the 65535 1D-dispatch cap for large numel).
  let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
  if (idx >= params.numel) {
    return;
  }
  // Unravel the NCHW output index -> (n, oc, oh, ow).
  let ow = idx % params.W_out;
  var r = idx / params.W_out;
  let oh = r % params.H_out;
  r = r / params.H_out;
  let oc = r % params.OC;
  let n = r / params.OC;

  var acc: f32 = 0.0;
  if (params.has_bias != 0u) {
    acc = t_bias[oc];
  }
  // Grouped conv: output channel oc belongs to group g and dots only that
  // group's ic_per_group input channels. weight is [OC, IC/groups, Kh, Kw];
  // groups==1 (ic_per_group==IC, g==0) is the general dense case.
  let oc_per_group = params.OC / params.groups;
  let g = oc / oc_per_group;
  let ic_base = g * params.ic_per_group;
  for (var ic_local: u32 = 0u; ic_local < params.ic_per_group; ic_local = ic_local + 1u) {
    let ic = ic_base + ic_local;
    for (var kh: u32 = 0u; kh < params.Kh; kh = kh + 1u) {
      let ih = i32(oh) * i32(params.stride_h) - i32(params.pad_h) +
          i32(kh) * i32(params.dil_h);
      if (ih < 0 || ih >= i32(params.H_in)) {
        continue;
      }
      let in_row = ((n * params.IC + ic) * params.H_in + u32(ih)) * params.W_in;
      let w_row =
          ((oc * params.ic_per_group + ic_local) * params.Kh + kh) * params.Kw;
      for (var kw: u32 = 0u; kw < params.Kw; kw = kw + 1u) {
        let iw = i32(ow) * i32(params.stride_w) - i32(params.pad_w) +
            i32(kw) * i32(params.dil_w);
        if (iw < 0 || iw >= i32(params.W_in)) {
          continue;
        }
        acc = acc + t_x[in_row + u32(iw)] * t_weight[w_row + kw];
      }
    }
  }
  t_out[idx] = clamp(acc, params.output_min, params.output_max);
}
