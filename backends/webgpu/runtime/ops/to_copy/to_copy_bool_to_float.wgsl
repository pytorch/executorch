override wg_size: u32 = 256u;

struct Params {
  num_elements: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.num_elements) {
    return;
  }
  let word = input[idx / 4u];
  let byte_shift = (idx % 4u) * 8u;
  let value = (word >> byte_shift) & 0xffu;
  output[idx] = select(0.0, 1.0, value != 0u);
}
