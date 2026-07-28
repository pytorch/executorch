@group(0) @binding(0) var<storage, read_write> t_out: array<u32>;
@group(0) @binding(1) var<storage, read> t_a: array<u32>;
@group(0) @binding(2) var<storage, read> t_b: array<u32>;

struct Params {
  num_words: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

override wg_size: u32 = 64u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  // bool packed 4/word; canonical 0/1 bytes -> word-wise AND == per-byte AND.
  let w = gid.x + gid.y * (num_workgroups.x * wg_size);
  if (w >= params.num_words) {
    return;
  }
  t_out[w] = t_a[w] & t_b[w];
}
