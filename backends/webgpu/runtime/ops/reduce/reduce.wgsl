struct Params {
  outer_: u32,
  r_: u32,
  inner_: u32,
  is_mean: u32,
};

@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 256;

@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  let outs = params.outer_ * params.inner_;
  if (t >= outs) {
    return;
  }
  let oo = t / params.inner_;
  let ii = t % params.inner_;
  let base = oo * params.r_ * params.inner_ + ii;
  var acc: f32 = 0.0;
  for (var r: u32 = 0u; r < params.r_; r = r + 1u) {
    acc = acc + inp[base + r * params.inner_];
  }
  if (params.is_mean == 1u) {
    acc = acc / f32(params.r_);
  }
  out[t] = acc;
}
