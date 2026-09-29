@group(0) @binding(0) var<storage, read_write> t_out: array<f32>;
@group(0) @binding(1) var<storage, read> t_in: array<f32>;
@group(0) @binding(2) var<storage, read> t_freqs_cos: array<f32>;
@group(0) @binding(3) var<storage, read> t_freqs_sin: array<f32>;

struct Params {
  n_heads: u32,
  seq: u32,
  head_dim: u32,
  half_dim: u32,
  num_pairs: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

override wg_size: u32 = 64u;

// One thread per (even,odd) pair; interleaved Llama RoPE, shared xq/xk shader.
@compute @workgroup_size(wg_size, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pair = gid.x;
  if (pair >= params.num_pairs) {
    return;
  }
  let half_dim = params.half_dim;
  let pair_i = pair % half_dim;
  let t1 = pair / half_dim;
  let head = t1 % params.n_heads;
  let t2 = t1 / params.n_heads;
  let s = t2 % params.seq;
  let b = t2 / params.seq;

  let base =
      (((b * params.seq + s) * params.n_heads + head) * params.head_dim) +
      2u * pair_i;
  let freqs_idx = s * half_dim + pair_i;

  let c = t_freqs_cos[freqs_idx];
  let si = t_freqs_sin[freqs_idx];
  let x_r = t_in[base];
  let x_i = t_in[base + 1u];
  t_out[base] = x_r * c - x_i * si;
  t_out[base + 1u] = x_r * si + x_i * c;
}
