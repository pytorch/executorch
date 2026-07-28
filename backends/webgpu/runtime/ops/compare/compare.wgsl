@group(0) @binding(0) var<storage, read_write> t_out: array<u32>;
@group(0) @binding(1) var<storage, read> input1: array<f32>;
@group(0) @binding(2) var<storage, read> input2: array<f32>;

struct Params {
  num_elements: u32,
  op: u32,
  pad0: u32,
  pad1: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

override wg_size: u32 = 64u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  // One thread per output word = 4 bool bytes; num_elements%4==0 (host).
  let widx = gid.x + gid.y * (num_workgroups.x * wg_size);
  let words = (params.num_elements + 3u) / 4u;
  if (widx >= words) {
    return;
  }
  var packed: u32 = 0u;
  for (var j: u32 = 0u; j < 4u; j = j + 1u) {
    let i = widx * 4u + j;
    let a = input1[i];
    let b = input2[i];
    var r: bool;
    switch params.op {
      case 0u: { r = a == b; }  // eq
      case 1u: { r = a < b; }   // lt
      case 2u: { r = a <= b; }  // le
      case 3u: { r = a > b; }   // gt
      default: { r = a >= b; }  // ge
    }
    if (r) {
      packed = packed | (1u << (j * 8u));
    }
  }
  t_out[widx] = packed;
}
