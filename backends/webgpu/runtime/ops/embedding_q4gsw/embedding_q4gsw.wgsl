@group(0) @binding(0) var<storage, read_write> t_out: array<f32>;
@group(0) @binding(1) var<storage, read> t_indices: array<i32>;
@group(0) @binding(2) var<storage, read> t_weight: array<u32>;
@group(0) @binding(3) var<storage, read> t_scales: array<f32>;

struct Params {
  embed_dim: u32,
  blocks_per_row: u32,
  num_indices: u32,
  group_size: u32,
  groups_per_row: u32,
  bytes_per_row: u32,
  total_blocks: u32,
  _pad: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

override wg_size: u32 = 64u;

// One thread per 32-dim block of one gathered row (flat-buffer weight path).
@compute @workgroup_size(wg_size, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let block = gid.x;
  if (block >= params.total_blocks) {
    return;
  }
  let indices_idx = block / params.blocks_per_row;
  let base_dim = (block % params.blocks_per_row) * 32u;

  // token assumed in-range (mirrors Vulkan; no vocab clamp).
  let token = u32(t_indices[indices_idx]);
  let row_byte_base = token * params.bytes_per_row;
  let out_base = indices_idx * params.embed_dim + base_dim;

  for (var t: u32 = 0u; t < 32u; t = t + 1u) {
    let dim = base_dim + t;
    let byte_idx = row_byte_base + (dim >> 1u);
    let word = t_weight[byte_idx >> 2u];
    let b = (word >> ((byte_idx & 3u) * 8u)) & 0xFFu;
    var nib: u32;
    if ((dim & 1u) == 0u) {
      nib = (b >> 4u) & 0x0Fu;  // even dim -> high nibble
    } else {
      nib = b & 0x0Fu;          // odd dim -> low nibble
    }
    let q = f32(i32(nib) - 8); // +8-shifted on pack; recover signed [-8,7]
    let scale = t_scales[token * params.groups_per_row + dim / params.group_size];
    t_out[out_base + t] = q * scale;
  }
}
