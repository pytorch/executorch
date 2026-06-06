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

// WGSL forbids literal -inf; large finite negative stands in (mirrors Vulkan).
const NEG_INF: f32 = -1.0e30;

override wg_size: u32 = 64;

@compute @workgroup_size(wg_size, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = params.Hq * params.S * params.context_len;
  let idx = gid.x;
  if (idx >= total) {
    return;
  }
  let c = idx % params.context_len;
  let s = (idx / params.context_len) % params.S;
  let h = idx / (params.context_len * params.S);

  let kvh = h / params.g;

  let q_base = s * params.Hq * params.D + h * params.D;
  let k_base = c * params.Hkv * params.D + kvh * params.D;

  var acc: f32 = 0.0;
  var d: u32 = 0u;
  loop {
    if (d >= params.D) {
      break;
    }
    acc = acc + t_q[q_base + d] * t_k_cache[k_base + d];
    d = d + 1u;
  }
  acc = acc * params.scale;

  // Causal mask: position c may not attend beyond s + input_pos.
  if (c > s + params.input_pos) {
    acc = NEG_INF;
  }

  t_attn_weights[idx] = acc;
}
