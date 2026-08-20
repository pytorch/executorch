struct Params {
  B: u32,
  M: u32,
  N: u32,
  K: u32,
};

@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> b: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: Params;

const TILE: u32 = 32u;
const TILE4: u32 = 8u;
const RPT: u32 = 4u;

var<workgroup> a_sub: array<array<vec4<f32>, 8>, 32>;
var<workgroup> b_sub: array<array<vec4<f32>, 8>, 32>;

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
  let tiles_per_m = (params.M + TILE - 1u) / TILE;
  let bn = wg_id.y / tiles_per_m;
  let row_tile = wg_id.y % tiles_per_m;
  let tile_row0 = row_tile * TILE;
  let tile_col0_4 = wg_id.x * TILE4;
  let tile_row = local_id.y * RPT;
  let tile_col4 = local_id.x;

  let k4 = params.K / 4u;
  let n4 = params.N / 4u;
  let a_base = bn * params.M * k4;
  let b_base = bn * params.K * n4;
  let out_base = bn * params.M * n4;

  var acc: array<vec4<f32>, 4>;
  for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
    acc[ir] = vec4<f32>(0.0);
  }

  let num_tiles = (params.K + TILE - 1u) / TILE;
  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
    let k4_start = t * TILE4;
    for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
      let arow = local_id.y * RPT + ir;
      let akcol = k4_start + local_id.x;
      if (tile_row0 + arow < params.M && akcol < k4) {
        a_sub[arow][local_id.x] = a[a_base + (tile_row0 + arow) * k4 + akcol];
      } else {
        a_sub[arow][local_id.x] = vec4<f32>(0.0);
      }
      let brow = t * TILE + arow;
      let bncol = tile_col0_4 + local_id.x;
      if (brow < params.K && bncol < n4) {
        b_sub[arow][local_id.x] = b[b_base + brow * n4 + bncol];
      } else {
        b_sub[arow][local_id.x] = vec4<f32>(0.0);
      }
    }
    workgroupBarrier();

    for (var kk: u32 = 0u; kk < TILE4; kk = kk + 1u) {
      let b0 = b_sub[kk * 4u + 0u][tile_col4];
      let b1 = b_sub[kk * 4u + 1u][tile_col4];
      let b2 = b_sub[kk * 4u + 2u][tile_col4];
      let b3 = b_sub[kk * 4u + 3u][tile_col4];
      for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
        let ac = a_sub[tile_row + ir][kk];
        acc[ir] = acc[ir] + b0 * ac.x + b1 * ac.y + b2 * ac.z + b3 * ac.w;
      }
    }
    workgroupBarrier();
  }

  for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
    let r = tile_row0 + tile_row + ir;
    let c4 = tile_col0_4 + tile_col4;
    if (r < params.M && c4 < n4) {
      out[out_base + r * n4 + c4] = acc[ir];
    }
  }
}
