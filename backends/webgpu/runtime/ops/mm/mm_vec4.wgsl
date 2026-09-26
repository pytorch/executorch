struct Params {
  M: u32,
  N: u32,
  K: u32,
  pad_: u32,
};

@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> b: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: Params;

const TILE: u32 = 32u;
const RPT: u32 = 4u;
const TILE4: u32 = 8u;

var<workgroup> a_sub: array<array<vec4<f32>, 8>, 32>;
var<workgroup> b_sub: array<array<vec4<f32>, 8>, 32>;

fn read_a4(row: u32, k4: u32) -> vec4<f32> {
  if (row < params.M && k4 * 4u < params.K) {
    return a[row * (params.K / 4u) + k4];
  }
  return vec4<f32>(0.0);
}

fn read_b4(krow: u32, n4: u32) -> vec4<f32> {
  if (krow < params.K && n4 * 4u < params.N) {
    return b[krow * (params.N / 4u) + n4];
  }
  return vec4<f32>(0.0);
}

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
  let tile_row0 = wg_id.y * TILE;
  let tile_col0_4 = wg_id.x * TILE4;
  let tile_row = local_id.y * RPT;
  let tile_col4 = local_id.x;

  var acc: array<vec4<f32>, 4>;
  for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
    acc[ir] = vec4<f32>(0.0);
  }

  let num_tiles = (params.K + TILE - 1u) / TILE;
  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
    let k4_start = t * TILE4;
    for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
      let arow = local_id.y * RPT + ir;
      a_sub[arow][local_id.x] = read_a4(tile_row0 + arow, k4_start + local_id.x);
      b_sub[arow][local_id.x] =
          read_b4(t * TILE + arow, tile_col0_4 + local_id.x);
    }
    workgroupBarrier();

    for (var k4: u32 = 0u; k4 < TILE4; k4 = k4 + 1u) {
      let b0 = b_sub[k4 * 4u + 0u][tile_col4];
      let b1 = b_sub[k4 * 4u + 1u][tile_col4];
      let b2 = b_sub[k4 * 4u + 2u][tile_col4];
      let b3 = b_sub[k4 * 4u + 3u][tile_col4];
      for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
        let ac = a_sub[tile_row + ir][k4];
        acc[ir] = acc[ir] + b0 * ac.x + b1 * ac.y + b2 * ac.z + b3 * ac.w;
      }
    }
    workgroupBarrier();
  }

  for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
    let r = tile_row0 + tile_row + ir;
    let c4 = tile_col0_4 + tile_col4;
    if (r < params.M && c4 * 4u < params.N) {
      out[r * (params.N / 4u) + c4] = acc[ir];
    }
  }
}
