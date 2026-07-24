@group(0) @binding(0) var<storage, read> t_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> t_out: array<u32>;

struct Params {
  num_rows: u32,
  reduce_size: u32,
  is_argmin: u32,
  pad0: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 64u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  let row = gid.x + gid.y * (num_workgroups.x * wg_size);
  if (row >= params.num_rows) {
    return;
  }
  let base = row * params.reduce_size;
  // Strict compare keeps the FIRST extremum = torch argmax/argmin tie-break.
  var best = t_in[base];
  var best_idx: u32 = 0u;
  for (var k: u32 = 1u; k < params.reduce_size; k = k + 1u) {
    let v = t_in[base + k];
    if (params.is_argmin != 0u) {
      if (v < best) {
        best = v;
        best_idx = k;
      }
    } else {
      if (v > best) {
        best = v;
        best_idx = k;
      }
    }
  }
  t_out[row] = best_idx;
}
