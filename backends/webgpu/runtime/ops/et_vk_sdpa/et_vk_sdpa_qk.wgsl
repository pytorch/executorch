@group(0) @binding(0) var<storage, read_write> attn: array<f32>;
@group(0) @binding(1) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> k: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> mask: array<f32>;

struct Params {
  B: u32,
  Hq: u32,
  Hkv: u32,
  S_q: u32,
  S_kv: u32,
  D: u32,
  g: u32,
  has_mask: u32,
  mask_mode: u32,
  tensor_layout: u32,
  _pad0: u32,
  scale: f32,
  bh_lo: u32,
  bh_count: u32,
  _pad1: u32,
  _pad2: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

override wg_size: u32 = 64;

// TMxTN register tile; mirrors sdpa/sdpa_compute_attn_weights.wgsl (not a knob).
const TM: u32 = 8u;
const TN: u32 = 4u;

fn q_row4(b: u32, h: u32, s: u32, d4_count: u32) -> u32 {
  if (params.tensor_layout == 0u) {
    return ((b * params.Hq + h) * params.S_q + s) * d4_count;
  }
  return ((b * params.S_q + s) * params.Hq + h) * d4_count;
}

fn k_row4(b: u32, h: u32, s: u32, d4_count: u32) -> u32 {
  if (params.tensor_layout == 0u) {
    return ((b * params.Hkv + h) * params.S_kv + s) * d4_count;
  }
  return ((b * params.S_kv + s) * params.Hkv + h) * d4_count;
}

fn load_q4(b: u32, h: u32, s: u32, d4: u32, d4_count: u32) -> vec4<f32> {
  if (s >= params.S_q) {
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
  }
  return q[q_row4(b, h, s, d4_count) + d4];
}

fn load_k4(b: u32, h: u32, c: u32, d4: u32, d4_count: u32) -> vec4<f32> {
  if (c >= params.S_kv) {
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
  }
  return k[k_row4(b, h, c, d4_count) + d4];
}

fn store_qk(
    scratch_row: u32,
    absolute_row: u32,
    s: u32,
    c: u32,
    raw: f32) {
  if (c >= params.S_kv) {
    return;
  }
  var val = raw * params.scale;
  if (params.has_mask != 0u) {
    if (params.mask_mode == 1u) {
      val = val + mask[params.S_kv * s + c];
    } else {
      val = val + mask[absolute_row + c];
    }
  }
  attn[scratch_row + c] = val;
}

@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let nrt = (params.S_q + TM - 1u) / TM;
  let nct = (params.S_kv + TN - 1u) / TN;
  let tiles = nrt * nct;
  let total = tiles * params.bh_count;
  // 2D dispatch fold: recover the linear tile index across x/y.
  let idx = gid.x + gid.y * (nwg.x * wg_size);
  if (idx >= total) {
    return;
  }

  // Tile within one (b, h) so a tile never straddles a head boundary.
  let local_bh = idx / tiles;
  let bh = params.bh_lo + local_bh;
  let rem = idx % tiles;
  let h = bh % params.Hq;
  let b = bh / params.Hq;
  let s0 = (rem / nct) * TM;
  let c0 = (rem % nct) * TN;
  let kv_h = h / params.g;
  let d4_count = params.D / 4u;

  var acc: array<vec4<f32>, TM>;
  for (var i: u32 = 0u; i < TM; i = i + 1u) {
    acc[i] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  }

  for (var d4: u32 = 0u; d4 < d4_count; d4 = d4 + 1u) {
    var qv: array<vec4<f32>, TM>;
    var kv: array<vec4<f32>, TN>;
    for (var i: u32 = 0u; i < TM; i = i + 1u) {
      qv[i] = load_q4(b, h, s0 + i, d4, d4_count);
    }
    for (var j: u32 = 0u; j < TN; j = j + 1u) {
      kv[j] = load_k4(b, kv_h, c0 + j, d4, d4_count);
    }
    for (var i: u32 = 0u; i < TM; i = i + 1u) {
      acc[i] = acc[i] +
          vec4<f32>(
              dot(qv[i], kv[0]),
              dot(qv[i], kv[1]),
              dot(qv[i], kv[2]),
              dot(qv[i], kv[3]));
    }
  }

  for (var i: u32 = 0u; i < TM; i = i + 1u) {
    let s = s0 + i;
    if (s < params.S_q) {
      let absolute_row = (bh * params.S_q + s) * params.S_kv;
      let scratch_row = (local_bh * params.S_q + s) * params.S_kv;
      let av = acc[i];
      store_qk(scratch_row, absolute_row, s, c0 + 0u, av.x);
      store_qk(scratch_row, absolute_row, s, c0 + 1u, av.y);
      store_qk(scratch_row, absolute_row, s, c0 + 2u, av.z);
      store_qk(scratch_row, absolute_row, s, c0 + 3u, av.w);
    }
  }
}
