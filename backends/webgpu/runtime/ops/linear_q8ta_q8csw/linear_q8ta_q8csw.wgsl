@group(0) @binding(0) var<storage, read_write> t_out: array<f32>;
@group(0) @binding(1) var<storage, read> t_x: array<u32>;
@group(0) @binding(2) var<storage, read> t_weight: array<u32>;
@group(0) @binding(3) var<storage, read> t_scales: array<f32>;
@group(0) @binding(4) var<storage, read> t_bias: array<f32>;

struct Params {
  M: u32,
  N: u32,
  K: u32,
  input_zero_point: i32,
  input_scale: f32,
  has_bias: u32,
  pad0: u32,
  pad1: u32,
}
@group(0) @binding(5) var<uniform> params: Params;

override wg_size: u32 = 64u;

const TM: u32 = 4u;
const TN: u32 = 4u;
const TILE_ELEMS: u32 = TM * TN;

fn unpack_i8(buf_idx: u32, arr_word: u32) -> i32 {
  // arr_word is the value at t[buf_idx>>2]; extract the (buf_idx&3) byte.
  return i32(((arr_word >> ((buf_idx & 3u) * 8u)) & 0xFFu) ^ 0x80u) - 128;
}

@compute @workgroup_size(wg_size, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  // 2D-folded tile index (lifts the 65535 1D-dispatch cap for large M*N).
  let tile = gid.x + gid.y * (num_workgroups.x * wg_size);
  let nct = (params.N + TN - 1u) / TN;
  if (tile >= ((params.M + TM - 1u) / TM) * nct) {
    return;
  }
  let m0 = (tile / nct) * TM;
  let n0 = (tile % nct) * TN;

  var acc: array<i32, TILE_ELEMS>;
  for (var i: u32 = 0u; i < TILE_ELEMS; i = i + 1u) {
    acc[i] = 0;
  }

  var k: u32 = 0u;
  loop {
    if (k >= params.K) {
      break;
    }
    var xq: array<i32, TM>;
    for (var ml: u32 = 0u; ml < TM; ml = ml + 1u) {
      let m_eff = min(m0 + ml, params.M - 1u);
      let bi = m_eff * params.K + k;
      xq[ml] = unpack_i8(bi, t_x[bi >> 2u]) - params.input_zero_point;
    }
    for (var nl: u32 = 0u; nl < TN; nl = nl + 1u) {
      let n_eff = min(n0 + nl, params.N - 1u);
      let bi = n_eff * params.K + k;
      let w = unpack_i8(bi, t_weight[bi >> 2u]);
      for (var ml: u32 = 0u; ml < TM; ml = ml + 1u) {
        acc[ml * TN + nl] = acc[ml * TN + nl] + xq[ml] * w;
      }
    }
    k = k + 1u;
  }

  // Dequantize to fp32 (no output requant); guard n<N (TN tile may overhang).
  for (var ml: u32 = 0u; ml < TM; ml = ml + 1u) {
    let m = m0 + ml;
    if (m >= params.M) {
      continue;
    }
    for (var nl: u32 = 0u; nl < TN; nl = nl + 1u) {
      let n = n0 + nl;
      if (n >= params.N) {
        continue;
      }
      var v = f32(acc[ml * TN + nl]) * params.input_scale * t_scales[n];
      if (params.has_bias != 0u) {
        v = v + t_bias[n];
      }
      t_out[m * params.N + n] = v;
    }
  }
}
