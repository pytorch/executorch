@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> freqs: array<f32>;

struct Params {
  seq: u32,
  width: u32,
  numel: u32,
  pad0: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

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

    // Pair-interleaved rope (Vulkan glsl): last dim [r,i], freqs [cos,sin].
    let c = idx % params.width;
    let n = (idx / params.width) % params.seq;
    let ce = c & ~1u;
    let base = n * params.width + ce;
    let cos_v = freqs[base];
    let sin_v = freqs[base + 1u];
    if ((c & 1u) == 0u) {
        output[idx] = input[idx] * cos_v - input[idx + 1u] * sin_v;
    } else {
        output[idx] = input[idx - 1u] * sin_v + input[idx] * cos_v;
    }
}
