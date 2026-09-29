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

// Cooperative shared-memory arg-reduction; mirrors Vulkan reduce.glsl. Each
// thread scans a strided slice for its local extremum (strict compare -> first
// index within the slice), then thread 0 aggregates the partials, breaking ties
// by lowest index = torch argmax/argmin semantics.
var<workgroup> part_val: array<f32, 256>;
var<workgroup> part_idx: array<u32, 256>;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  // One workgroup per reduction row; 2D-fold lifts the 65535 grid cap.
  let row = wid.x + wid.y * num_workgroups.x;
  if (row >= params.num_rows) {
    return;
  }
  let base = row * params.reduce_size;

  // Seed with element 0 (real value, lowest index); threads owning no element
  // keep it, which is harmless since index 0 is a valid candidate.
  var best = t_in[base];
  var best_idx: u32 = 0u;
  var k = lid.x;
  while (k < params.reduce_size) {
    let v = t_in[base + k];
    if (params.is_argmin != 0u) {
      if (v < best) { best = v; best_idx = k; }
    } else {
      if (v > best) { best = v; best_idx = k; }
    }
    k = k + wg_size;
  }
  part_val[lid.x] = best;
  part_idx[lid.x] = best_idx;
  workgroupBarrier();

  if (lid.x == 0u) {
    var bv = part_val[0];
    var bi = part_idx[0];
    for (var t: u32 = 1u; t < wg_size; t = t + 1u) {
      let v = part_val[t];
      let idx = part_idx[t];
      if (params.is_argmin != 0u) {
        if (v < bv || (v == bv && idx < bi)) { bv = v; bi = idx; }
      } else {
        if (v > bv || (v == bv && idx < bi)) { bv = v; bi = idx; }
      }
    }
    t_out[row] = bi;
  }
}
