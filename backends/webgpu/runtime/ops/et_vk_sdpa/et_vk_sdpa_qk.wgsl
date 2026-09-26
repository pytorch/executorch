@group(0) @binding(0) var<storage, read_write> attn: array<f32>;
@group(0) @binding(1) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> k: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> mask: array<f32>;

struct Params {
  B: u32,
  H: u32,
  S_q: u32,
  S_kv: u32,
  D: u32,
  has_mask: u32,
  _pad0: u32,
  scale: f32,
}
@group(0) @binding(4) var<uniform> params: Params;

override wg_size: u32 = 64;

// Non-causal fused SDPA, QK phase. DSHB layout, row-major: q [B, H, S_q, D],
// k [B, H, S_kv, D]; q/k viewed as vec4<f32> over D (caller guarantees
// D % 4 == 0 for every model in scope). Supports asymmetric seq (S_q != S_kv,
// e.g. Hiera pooled query); when S_q == S_kv this reduces to plain
// self-attention. ONE thread per (b, h, s) ROW of attn_weights
// [B, H, S_q, S_kv]; the thread loops over c (0..S_kv) and d4 (0..D/4). Row
// count = B*H*S_q stays well under the 65535 1D dispatch limit for any ViT.
@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let num_rows = params.B * params.H * params.S_q;
  let row = gid.x;
  if (row >= num_rows) {
    return;
  }
  let s = row % params.S_q;
  let h = (row / params.S_q) % params.H;
  let b = row / (params.S_q * params.H);

  let d4_count = params.D / 4u;
  let qbase4 = ((b * params.H + h) * params.S_q + s) * d4_count;
  let kblock4 = (b * params.H + h) * params.S_kv; // first K row of this (b, h)
  let arow = ((b * params.H + h) * params.S_q + s) * params.S_kv;

  for (var c: u32 = 0u; c < params.S_kv; c = c + 1u) {
    let kbase4 = (kblock4 + c) * d4_count;
    var acc: f32 = 0.0;
    for (var d4: u32 = 0u; d4 < d4_count; d4 = d4 + 1u) {
      acc = acc + dot(q[qbase4 + d4], k[kbase4 + d4]);
    }
    acc = acc * params.scale;
    if (params.has_mask != 0u) {
      acc = acc + mask[arow + c];
    }
    attn[arow + c] = acc;
  }
}
