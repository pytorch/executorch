// STE weight-gradient d_W[N,K] = sum_m d_out[m,N]*x[m,K] (operands f32).

@group(0) @binding(0) var<storage, read_write> t_dw: array<f32>;    // [N, K]
@group(0) @binding(1) var<storage, read> t_dout: array<f32>;         // [M, N]
@group(0) @binding(2) var<storage, read> t_x: array<f32>;            // [M, K]

struct Params {
  M: u32,
  N: u32,
  K: u32,
  _pad: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

override wg_size: u32 = 64u;

const TN: u32 = 4u;
const TK: u32 = 4u;
const TILE_ELEMS: u32 = TN * TK;

@compute @workgroup_size(wg_size, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let nnt = (params.N + TN - 1u) / TN;
  let nkt = (params.K + TK - 1u) / TK;
  let tiles = nnt * nkt;
  if (gid.x >= tiles) {
    return;
  }
  let row_tile = gid.x / nkt;
  let col_tile = gid.x % nkt;
  let n0 = row_tile * TN;
  let k0 = col_tile * TK;

  var acc: array<f32, TILE_ELEMS>;
  for (var i: u32 = 0u; i < TILE_ELEMS; i = i + 1u) {
    acc[i] = 0.0;
  }

  var m: u32 = 0u;
  loop {
    if (m >= params.M) {
      break;
    }
    // Load the TN d_out values for row m once; reused across all TK k columns.
    var dout_reg: array<f32, TN>;
    for (var nl: u32 = 0u; nl < TN; nl = nl + 1u) {
      let n_eff = min(n0 + nl, params.N - 1u);
      dout_reg[nl] = t_dout[m * params.N + n_eff];
    }
    for (var kl: u32 = 0u; kl < TK; kl = kl + 1u) {
      let k_eff = min(k0 + kl, params.K - 1u);
      let xv = t_x[m * params.K + k_eff];
      for (var nl: u32 = 0u; nl < TN; nl = nl + 1u) {
        acc[nl * TK + kl] = acc[nl * TK + kl] + dout_reg[nl] * xv;
      }
    }
    m = m + 1u;
  }

  for (var nl: u32 = 0u; nl < TN; nl = nl + 1u) {
    let n = n0 + nl;
    for (var kl: u32 = 0u; kl < TK; kl = kl + 1u) {
      let k = k0 + kl;
      if (n < params.N && k < params.K) {
        t_dw[n * params.K + k] = acc[nl * TK + kl];
      }
    }
  }
}
