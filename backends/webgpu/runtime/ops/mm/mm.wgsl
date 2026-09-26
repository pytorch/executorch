struct Params {
  M: u32,
  N: u32,
  K: u32,
  pad_: u32,
};

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

override wg_size: u32 = 256;

@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.M * params.N;
  if (idx >= total) {
    return;
  }
  let m = idx / params.N;
  let n = idx % params.N;
  var acc: f32 = 0.0;
  for (var k: u32 = 0u; k < params.K; k = k + 1u) {
    acc = acc + a[m * params.K + k] * b[k * params.N + n];
  }
  out[idx] = acc;
}
