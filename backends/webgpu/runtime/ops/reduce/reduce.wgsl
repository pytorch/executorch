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

// Cooperative shared-memory reduction, one workgroup per output element: each
// thread sums a strided slice of the reduced dim into a shared partial, then
// thread 0 folds the partials. Same one-workgroup-per-row shared-memory shape as
// Vulkan's reduce_per_row_buffer.glsl. Fixed 256 upper bound >= any clamped
// wg_size; only [0, wg_size) is used.
var<workgroup> partials: array<f32, 256>;

@compute @workgroup_size(wg_size)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  // One workgroup per output; 2D-fold lifts the 65535 grid cap. `t` is uniform
  // across the workgroup, so the early return keeps the barrier in uniform flow.
  let t = wid.x + wid.y * num_workgroups.x;
  let outs = params.outer_ * params.inner_;
  if (t >= outs) {
    return;
  }
  let oo = t / params.inner_;
  let ii = t % params.inner_;
  let base = oo * params.r_ * params.inner_ + ii;

  var acc: f32 = 0.0;
  var k: u32 = lid.x;
  while (k < params.r_) {
    acc = acc + inp[base + k * params.inner_];
    k = k + wg_size;
  }
  partials[lid.x] = acc;
  workgroupBarrier();

  if (lid.x == 0u) {
    var s: f32 = partials[0];
    for (var i: u32 = 1u; i < wg_size; i = i + 1u) {
      s = s + partials[i];
    }
    if (params.is_mean == 1u) {
      s = s / f32(params.r_);
    }
    out[t] = s;
  }
}
