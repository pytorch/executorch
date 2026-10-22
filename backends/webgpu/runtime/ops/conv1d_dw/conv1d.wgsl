override wg_size: u32 = 64u;

struct Params {
  in_channels: u32,
  out_channels: u32,
  in_len: u32,
  out_len: u32,
  kernel_size: u32,
  stride: u32,
  padding: u32,
  dilation: u32,
  numel: u32,
  has_bias: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
  if (idx >= params.numel) {
    return;
  }

  let out_t = idx % params.out_len;
  let out_c = (idx / params.out_len) % params.out_channels;
  let batch = idx / (params.out_channels * params.out_len);
  var sum = 0.0;

  for (var in_c = 0u; in_c < params.in_channels; in_c = in_c + 1u) {
    for (var k = 0u; k < params.kernel_size; k = k + 1u) {
      let in_t = i32(out_t * params.stride + k * params.dilation) -
          i32(params.padding);
      if (in_t >= 0 && in_t < i32(params.in_len)) {
        let input_idx =
            (batch * params.in_channels + in_c) * params.in_len + u32(in_t);
        let weight_idx =
            (out_c * params.in_channels + in_c) * params.kernel_size + k;
        sum = fma(input[input_idx], weight[weight_idx], sum);
      }
    }
  }
  if (params.has_bias != 0u) {
    sum = sum + bias[out_c];
  }
  output[idx] = sum;
}
