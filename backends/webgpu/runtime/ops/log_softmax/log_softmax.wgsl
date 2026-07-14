struct Params {
  outer_: u32,
  r_: u32,
  inner_: u32,
  pad_: u32,
};

@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 256;

@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  let lines = params.outer_ * params.inner_;
  if (t >= lines) {
    return;
  }
  let oo = t / params.inner_;
  let ii = t % params.inner_;
  let base = oo * params.r_ * params.inner_ + ii;
  // Online flash-style softmax; -3.4e38 seeds max (Tint overflows -FLT_MAX).
  var mx: f32 = -3.4e38;
  var ssum: f32 = 0.0;
  for (var r: u32 = 0u; r < params.r_; r = r + 1u) {
    let v = inp[base + r * params.inner_];
    if (v > mx) {
      ssum = ssum * exp(mx - v);
      mx = v;
    }
    ssum = ssum + exp(v - mx);
  }
  let log_sum = log(ssum);
  for (var r: u32 = 0u; r < params.r_; r = r + 1u) {
    let idx = base + r * params.inner_;
    out[idx] = (inp[idx] - mx) - log_sum;
  }
}
