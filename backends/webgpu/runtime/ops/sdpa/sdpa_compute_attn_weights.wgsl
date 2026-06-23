@group(0) @binding(0) var<storage, read_write> t_attn_weights: array<f32>;
@group(0) @binding(1) var<storage, read> t_q: array<f32>;
@group(0) @binding(2) var<storage, read> t_k_cache: array<f32>;

struct Params {
  S: u32,
  Hq: u32,
  Hkv: u32,
  D: u32,
  context_len: u32,
  input_pos: u32,
  g: u32,
  scale: f32,
}
@group(0) @binding(3) var<uniform> params: Params;

// WGSL forbids literal -inf; large finite negative is a WGSL-safe stand-in.
const NEG_INF: f32 = -1.0e30;

override wg_size: u32 = 64;

const TM: u32 = 4u;
const TN: u32 = 4u;

fn load_q_vec4(s: u32, h: u32, d4: u32) -> vec4<f32> {
  var r = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  if (s >= params.S) {
    return r;
  }
  let base = s * params.Hq * params.D + h * params.D;
  if (d4 + 0u < params.D) { r.x = t_q[base + d4 + 0u]; }
  if (d4 + 1u < params.D) { r.y = t_q[base + d4 + 1u]; }
  if (d4 + 2u < params.D) { r.z = t_q[base + d4 + 2u]; }
  if (d4 + 3u < params.D) { r.w = t_q[base + d4 + 3u]; }
  return r;
}

fn load_k_vec4(c: u32, kvh: u32, d4: u32) -> vec4<f32> {
  var r = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  if (c >= params.context_len) {
    return r;
  }
  let base = c * params.Hkv * params.D + kvh * params.D;
  if (d4 + 0u < params.D) { r.x = t_k_cache[base + d4 + 0u]; }
  if (d4 + 1u < params.D) { r.y = t_k_cache[base + d4 + 1u]; }
  if (d4 + 2u < params.D) { r.z = t_k_cache[base + d4 + 2u]; }
  if (d4 + 3u < params.D) { r.w = t_k_cache[base + d4 + 3u]; }
  return r;
}

fn store_qk(s: u32, c: u32, h: u32, raw: f32) {
  if (s >= params.S || c >= params.context_len) {
    return;
  }
  var val = raw * params.scale;
  // Causal mask: position c may not attend beyond s + input_pos.
  if (c > s + params.input_pos) {
    val = NEG_INF;
  }
  let idx = h * params.S * params.context_len + s * params.context_len + c;
  t_attn_weights[idx] = val;
}

@compute @workgroup_size(wg_size, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let nrt = (params.S + TM - 1u) / TM;
  let nct = (params.context_len + TN - 1u) / TN;
  let tiles = nrt * nct;
  let total = tiles * params.Hq;
  if (gid.x >= total) {
    return;
  }

  let h = gid.x / tiles;
  let rem = gid.x % tiles;
  let row_tile = rem / nct;
  let col_tile = rem % nct;
  let kvh = h / params.g;
  let s0 = row_tile * TM;
  let c0 = col_tile * TN;

  var acc: array<vec4<f32>, 4>;
  acc[0] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  acc[1] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  acc[2] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  acc[3] = vec4<f32>(0.0, 0.0, 0.0, 0.0);

  var d4: u32 = 0u;
  loop {
    if (d4 >= params.D) {
      break;
    }
    let q0 = load_q_vec4(s0 + 0u, h, d4);
    let q1 = load_q_vec4(s0 + 1u, h, d4);
    let q2 = load_q_vec4(s0 + 2u, h, d4);
    let q3 = load_q_vec4(s0 + 3u, h, d4);
    let k0 = load_k_vec4(c0 + 0u, kvh, d4);
    let k1 = load_k_vec4(c0 + 1u, kvh, d4);
    let k2 = load_k_vec4(c0 + 2u, kvh, d4);
    let k3 = load_k_vec4(c0 + 3u, kvh, d4);
    acc[0] += vec4<f32>(dot(q0, k0), dot(q0, k1), dot(q0, k2), dot(q0, k3));
    acc[1] += vec4<f32>(dot(q1, k0), dot(q1, k1), dot(q1, k2), dot(q1, k3));
    acc[2] += vec4<f32>(dot(q2, k0), dot(q2, k1), dot(q2, k2), dot(q2, k3));
    acc[3] += vec4<f32>(dot(q3, k0), dot(q3, k1), dot(q3, k2), dot(q3, k3));
    d4 = d4 + 4u;
  }

  var m: u32 = 0u;
  loop {
    if (m >= TM) {
      break;
    }
    let av = acc[m];
    store_qk(s0 + m, c0 + 0u, h, av.x);
    store_qk(s0 + m, c0 + 1u, h, av.y);
    store_qk(s0 + m, c0 + 2u, h, av.z);
    store_qk(s0 + m, c0 + 3u, h, av.w);
    m = m + 1u;
  }
}
