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

// Cooperative shared-memory reduction; mirrors Vulkan group_norm_reduce. Threads
// co-operate per (n, group) to accumulate sum and sum-of-squares.
// Fixed upper bound (>= any clamped wg_size); only [0, wg_size) is used.
var<workgroup> psum: array<f32, 256>;
var<workgroup> pss: array<f32, 256>;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    // One workgroup per (n, group) -> mean/rstd of shape [N, G].
    let mg = wid.x + wid.y * num_workgroups.x;
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
    var i = lid.x;
    while (i < params.group_size) {
        let v = input[base + i];
        s = s + v;
        ss = ss + v * v;
        i = i + wg_size;
    }
    psum[lid.x] = s;
    pss[lid.x] = ss;
    workgroupBarrier();

    if (lid.x == 0u) {
        var ts = psum[0];
        var tss = pss[0];
        for (var t = 1u; t < wg_size; t = t + 1u) {
            ts = ts + psum[t];
            tss = tss + pss[t];
        }
        let count = f32(params.group_size);
        let m = ts / count;
        let variance = tss / count - m * m;
        mean[mg] = m;
        rstd[mg] = inverseSqrt(variance + params.eps);
    }
}
