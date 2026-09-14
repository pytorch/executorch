struct Params {
  M: u32,
  N: u32,
  K: u32,
  self_2d: u32,
  beta: f32,
  alpha: f32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> t_out: array<f32>;
@group(0) @binding(1) var<storage, read> t_self: array<f32>;
@group(0) @binding(2) var<storage, read> t_mat1: array<f32>;
@group(0) @binding(3) var<storage, read> t_mat2: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

// Shared-memory tiled addmm (same skeleton as the sibling `linear` op's
// linear_tiled.wgsl), re-derived for mat2's actual [K,N] layout (NOT [N,K]
// like linear's weight — mat2[k,n] is contiguous over n, so the tile-load's
// per-thread-row reads are cacheline-coalesced, unlike linear's weight; still
// RPT=4-float-strided across threads within a row, not 1-float-strided).
// out = beta*self + alpha*(mat1 @ mat2); mat1 [M,K], mat2 [K,N].
const TILE: u32 = 32u;
const RPT: u32 = 4u;

var<workgroup> a_sub: array<array<f32, 32>, 32>;
var<workgroup> b_sub: array<array<f32, 32>, 32>;

fn read_a(row: u32, col: u32) -> f32 {
  if (row < params.M && col < params.K) {
    return t_mat1[row * params.K + col];
  }
  return 0.0;
}

fn read_b(krow: u32, col: u32) -> f32 {
  if (krow < params.K && col < params.N) {
    return t_mat2[krow * params.N + col];
  }
  return 0.0;
}

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
  let tile_row0 = wg_id.y * TILE;
  let tile_col0 = wg_id.x * TILE;
  let tile_row = local_id.y * RPT;
  let tile_col = local_id.x * RPT;

  var acc: array<array<f32, 4>, 4>;
  for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
    for (var ic: u32 = 0u; ic < RPT; ic = ic + 1u) {
      acc[ir][ic] = 0.0;
    }
  }

  let num_tiles = (params.K + TILE - 1u) / TILE;
  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
    let k_start = t * TILE;
    for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
      let arow = local_id.y * RPT + ir;
      for (var kk: u32 = 0u; kk < RPT; kk = kk + 1u) {
        let col = local_id.x * RPT + kk;
        a_sub[arow][col] = read_a(tile_row0 + arow, k_start + col);
        b_sub[arow][col] = read_b(k_start + arow, tile_col0 + col);
      }
    }
    workgroupBarrier();

    for (var k: u32 = 0u; k < TILE; k = k + 1u) {
      for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
        let aval = a_sub[tile_row + ir][k];
        for (var ic: u32 = 0u; ic < RPT; ic = ic + 1u) {
          acc[ir][ic] = acc[ir][ic] + aval * b_sub[k][tile_col + ic];
        }
      }
    }
    workgroupBarrier();
  }

  for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
    for (var ic: u32 = 0u; ic < RPT; ic = ic + 1u) {
      let r = tile_row0 + tile_row + ir;
      let c = tile_col0 + tile_col + ic;
      if (r < params.M && c < params.N) {
        var self_val: f32;
        if (params.self_2d == 1u) {
          self_val = t_self[r * params.N + c];
        } else {
          self_val = t_self[c];
        }
        t_out[r * params.N + c] = params.beta * self_val + params.alpha * acc[ir][ic];
      }
    }
  }
}
