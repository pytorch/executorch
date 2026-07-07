enable f16;

// f16 KV-cache variant of update_cache.wgsl: cache stored as f16, source f32.
@group(0) @binding(0) var<storage, read_write> t_cache: array<f16>;
@group(0) @binding(1) var<storage, read> t_value: array<f32>;

struct Params {
  numel: u32,
  dst_offset: u32,
  cache_numel: u32,
  _pad0: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 256;

@compute @workgroup_size(wg_size, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.numel) {
    return;
  }
  if (params.dst_offset + i >= params.cache_numel) {
    return;
  }
  t_cache[params.dst_offset + i] = f16(t_value[i]);
}
