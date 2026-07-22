@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> mean: array<f32>;
@group(0) @binding(2) var<storage, read_write> rstd: array<f32>;

struct Params {
  n_channels: u32,
  hxw: u32,
  num_groups: u32,
  chans_per_group: u32,
  numel: u32,
  mean_numel: u32,
  group_size: u32,
  eps: f32,
}
@group(0) @binding(3) var<uniform> params: Params;

override wg_size: u32 = 64u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    // One thread per (n, group) -> mean/rstd of shape [N, G].
    let mg = gid.x + gid.y * (num_workgroups.x * wg_size);
    if (mg >= params.mean_numel) {
        return;
    }

    // A group's D channels x HxW form a contiguous NCHW block.
    let n = mg / params.num_groups;
    let g = mg % params.num_groups;
    let base = n * params.n_channels * params.hxw
        + g * params.chans_per_group * params.hxw;

    var s = 0.0;
    var ss = 0.0;
    for (var i = 0u; i < params.group_size; i = i + 1u) {
        let v = input[base + i];
        s = s + v;
        ss = ss + v * v;
    }
    let count = f32(params.group_size);
    let m = s / count;
    let variance = ss / count - m * m;
    mean[mg] = m;
    rstd[mg] = inverseSqrt(variance + params.eps);
}
