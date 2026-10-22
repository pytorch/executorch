// Mirrors q4gsw_linear.wgsl dequant; contracts over N (forward over K).

@group(0) @binding(0) var<storage, read_write> t_dx: array<f32>;    // [M, K]
@group(0) @binding(1) var<storage, read> t_dout: array<f32>;         // [M, N]
@group(0) @binding(2) var<storage, read> t_weight: array<u32>;
@group(0) @binding(3) var<storage, read> t_scales: array<f32>;

struct Params {
  M: u32,
  N: u32,
  K: u32,
  K_packed: u32,
  group_size: u32,
  padded_N: u32,
  has_bias: u32,
  _pad: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

override wg_size: u32 = 64u;

const TM: u32 = 4u;
const TK: u32 = 4u;
const TILE_ELEMS: u32 = TM * TK;

@compute @workgroup_size(wg_size, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let nrt = (params.M + TM - 1u) / TM;
  let nkt = (params.K + TK - 1u) / TK;
  let tiles = nrt * nkt;
  if (gid.x >= tiles) {
    return;
  }
  let row_tile = gid.x / nkt;
  let col_tile = gid.x % nkt;
  let m0 = row_tile * TM;
  let k0 = col_tile * TK;

  var acc: array<f32, TILE_ELEMS>;
  for (var i: u32 = 0u; i < TILE_ELEMS; i = i + 1u) {
    acc[i] = 0.0;
  }

  var n: u32 = 0u;
  loop {
    if (n >= params.N) {
      break;
    }
    // Load TM d_out for column n once; reused across TK k columns.
    var dout_reg: array<f32, TM>;
    for (var ml: u32 = 0u; ml < TM; ml = ml + 1u) {
      let m_eff = min(m0 + ml, params.M - 1u);
      dout_reg[ml] = t_dout[m_eff * params.N + n];
    }
    for (var kl: u32 = 0u; kl < TK; kl = kl + 1u) {
      let k_eff = min(k0 + kl, params.K - 1u);
      let byte_idx = n * params.K_packed + (k_eff >> 1u);
      let word = t_weight[byte_idx >> 2u];
      let b = (word >> ((byte_idx & 3u) * 8u)) & 0xFFu;
      var nib: u32;
      if ((k_eff & 1u) == 0u) {
        nib = b & 0x0Fu;
      } else {
        nib = (b >> 4u) & 0x0Fu;
      }
      let q = f32(i32(nib) - 8);
      let dq = q * t_scales[(k_eff / params.group_size) * params.padded_N + n];
      for (var ml: u32 = 0u; ml < TM; ml = ml + 1u) {
        acc[ml * TK + kl] = acc[ml * TK + kl] + dout_reg[ml] * dq;
      }
    }
    n = n + 1u;
  }

  for (var ml: u32 = 0u; ml < TM; ml = ml + 1u) {
    let m = m0 + ml;
    for (var kl: u32 = 0u; kl < TK; kl = kl + 1u) {
      let k = k0 + kl;
      if (m < params.M && k < params.K) {
        t_dx[m * params.K + k] = acc[ml * TK + kl];
      }
    }
  }
}
