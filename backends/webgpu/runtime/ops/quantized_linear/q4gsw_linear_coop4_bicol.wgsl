@group(0) @binding(0) var<storage, read_write> t_out: array<f32>;
@group(0) @binding(1) var<storage, read> t_input: array<f32>;
@group(0) @binding(2) var<storage, read> t_weight: array<u32>;
@group(0) @binding(3) var<storage, read> t_scales: array<f32>;
@group(0) @binding(4) var<storage, read> t_bias: array<f32>;

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
@group(0) @binding(5) var<uniform> params: Params;

// Cooperative-over-K GEMV, 2 output columns/workgroup; input read once, reused.
const WG: u32 = 64u;
var<workgroup> partial: array<vec2<f32>, WG>;

@compute @workgroup_size(WG, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) ngrp: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>) {
  let num_pairs = (params.N + 1u) >> 1u;   // ceil(N/2) pairs (M == 1 decode)
  let stride = ngrp.x;
  let num_words = params.K >> 3u;          // K / 8 words per row
  let row_words = params.K_packed >> 2u;   // u32s per weight row (= K/8)
  var p = wid.x;
  loop {
    if (p >= num_pairs) {
      break;
    }
    let col0 = p << 1u;
    let col1 = col0 + 1u;
    let has1 = col1 < params.N;
    let wbase0 = col0 * row_words;
    let wbase1 = col1 * row_words;          // load guarded by has1

    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    var w: u32 = lid.x;
    loop {
      if (w >= num_words) {
        break;
      }
      let k0 = w << 3u;
      let ib = k0;                          // in_base = 0 (m == 0 for M == 1 decode)
      let sidx = (k0 / params.group_size) * params.padded_N;
      let word0 = t_weight[wbase0 + w];
      let scale0 = t_scales[sidx + col0];
      var word1: u32 = 0u;
      var scale1: f32 = 0.0;
      if (has1) {
        word1 = t_weight[wbase1 + w];
        scale1 = t_scales[sidx + col1];
      }
      for (var bi: u32 = 0u; bi < 4u; bi = bi + 1u) {
        let kk = bi << 1u;
        let in0 = t_input[ib + kk];         // shared across both columns
        let in1 = t_input[ib + kk + 1u];
        let b0 = (word0 >> (bi * 8u)) & 0xFFu;
        acc0 = acc0 + in0 * f32(i32(b0 & 0x0Fu) - 8) * scale0;
        acc0 = acc0 + in1 * f32(i32((b0 >> 4u) & 0x0Fu) - 8) * scale0;
        let b1 = (word1 >> (bi * 8u)) & 0xFFu;
        acc1 = acc1 + in0 * f32(i32(b1 & 0x0Fu) - 8) * scale1;
        acc1 = acc1 + in1 * f32(i32((b1 >> 4u) & 0x0Fu) - 8) * scale1;
      }
      w = w + WG;
    }

    partial[lid.x] = vec2<f32>(acc0, acc1);
    workgroupBarrier();
    var s: u32 = WG >> 1u;
    loop {
      if (s == 0u) {
        break;
      }
      if (lid.x < s) {
        partial[lid.x] = partial[lid.x] + partial[lid.x + s];
      }
      workgroupBarrier();
      s = s >> 1u;
    }
    if (lid.x == 0u) {
      var o0 = partial[0].x;
      var o1 = partial[0].y;
      if (params.has_bias != 0u) {
        o0 = o0 + t_bias[col0];
        if (has1) {
          o1 = o1 + t_bias[col1];
        }
      }
      t_out[col0] = o0;
      if (has1) {
        t_out[col1] = o1;
      }
    }
    workgroupBarrier();
    p = p + stride;
  }
}
