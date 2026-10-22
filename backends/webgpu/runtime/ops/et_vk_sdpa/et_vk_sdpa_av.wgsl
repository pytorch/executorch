@group(0) @binding(0) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> sm: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<vec4<f32>>;

struct Params {
  B: u32,
  H: u32,
  S_q: u32,
  S_kv: u32,
  D: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

override wg_size: u32 = 64;

// Non-causal fused SDPA, AV phase. out[b,h,s,d]=sum_c sm[b,h,s,c]*v[b,h,c,d].
// DSHB layout, row-major: out/sm rows over S_q, v rows over S_kv; v/out viewed
// as vec4<f32> over D (caller guarantees D % 4 == 0 for every model in scope).
// Supports asymmetric seq (S_q != S_kv); reduces to self-attention when
// S_q == S_kv. ONE thread per (b, h, s, d4) computing 4 output elements; the
// thread contracts over c (0..S_kv, scalar — S_kv isn't guaranteed % 4 == 0).
@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let d4_count = params.D / 4u;
  let total = params.B * params.H * params.S_q * d4_count;
  let i = gid.x;
  if (i >= total) {
    return;
  }
  let d4 = i % d4_count;
  let s = (i / d4_count) % params.S_q;
  let h = (i / (d4_count * params.S_q)) % params.H;
  let b = i / (d4_count * params.S_q * params.H);

  let smbase = ((b * params.H + h) * params.S_q + s) * params.S_kv;
  let vblock4 = (b * params.H + h) * params.S_kv; // first V row of this (b, h)

  var acc: vec4<f32> = vec4<f32>(0.0);
  for (var c: u32 = 0u; c < params.S_kv; c = c + 1u) {
    acc = acc + sm[smbase + c] * v[(vblock4 + c) * d4_count + d4];
  }
  out[i] = acc;
}
