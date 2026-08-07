@group(0) @binding(0) var<storage, read_write> attn: array<f32>;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
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

fn q_row(b: u32, h: u32, s: u32) -> u32 {
  if (params.tensor_layout == 0u) {
    return ((b * params.Hq + h) * params.S_q + s) * params.D;
  }
  return ((b * params.S_q + s) * params.Hq + h) * params.D;
}

fn k_row(b: u32, h: u32, s: u32) -> u32 {
  if (params.tensor_layout == 0u) {
    return ((b * params.Hkv + h) * params.S_kv + s) * params.D;
  }
  return ((b * params.S_kv + s) * params.Hkv + h) * params.D;
}

@compute @workgroup_size(wg_size)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>) {
  let chunk_numel = params.bh_count * params.S_q * params.S_kv;
  let tid = gid.x + gid.y * (nwg.x * wg_size);
  if (tid >= chunk_numel) {
    return;
  }
  let idx = params.bh_lo * params.S_q * params.S_kv + tid;
  let c = idx % params.S_kv;
  let row = idx / params.S_kv;
  let s = row % params.S_q;
  let h = (row / params.S_q) % params.Hq;
  let b = row / (params.S_q * params.Hq);
  let kv_h = h / params.g;
  let qbase = q_row(b, h, s);
  let kbase = k_row(b, kv_h, c);

  var acc: f32 = 0.0;
  for (var d: u32 = 0u; d < params.D; d = d + 1u) {
    acc = acc + q[qbase + d] * k[kbase + d];
  }
  acc = acc * params.scale;
  if (params.has_mask != 0u) {
    if (params.mask_mode == 1u) {
      acc = acc + mask[params.S_kv * s + c];
    } else {
      acc = acc + mask[idx];
    }
  }
  attn[tid] = acc;
}
