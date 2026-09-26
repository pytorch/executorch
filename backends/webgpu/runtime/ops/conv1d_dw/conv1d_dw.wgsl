@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;

struct Params {
  kernel_size: u32,
  stride: u32,
  padding: u32,
  dilation: u32,
  channels: u32,
  in_len: u32,
  out_len: u32,
  numel: u32,
  has_bias: u32,
  pad0: u32,
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
    let oi = gid.x + gid.y * (num_workgroups.x * wg_size);
    if (oi >= params.numel) {
        return;
    }

    // Depthwise: each channel convolves with its own [K] filter (Vulkan glsl).
    let l_out = oi % params.out_len;
    let c = (oi / params.out_len) % params.channels;
    let n = oi / (params.out_len * params.channels);

    let w_base = c * params.kernel_size;
    let in_base = (n * params.channels + c) * params.in_len;
    var s = 0.0;
    for (var k: u32 = 0u; k < params.kernel_size; k = k + 1u) {
        let l_in = i32(l_out) * i32(params.stride) - i32(params.padding)
            + i32(k) * i32(params.dilation);
        if (l_in >= 0 && l_in < i32(params.in_len)) {
            s = s + weight[w_base + k] * input[in_base + u32(l_in)];
        }
    }
    if (params.has_bias != 0u) {
        s = s + bias[c];
    }
    output[oi] = clamp(s, params.output_min, params.output_max);
}
