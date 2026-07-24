@group(0) @binding(0) var<storage, read_write> t_out: array<f32>;

struct Params {
  numel: u32,
  width: u32,
  stride: f32,
  offset: f32,
}
@group(0) @binding(1) var<uniform> params: Params;

override wg_size: u32 = 64u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  // 2D-folded flat index (lifts the 65535 1D-dispatch cap for large numel).
  let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
  if (idx >= params.numel) {
    return;
  }
  // out[r,0]=(r%W+offset)*stride; out[r,1]=(r/W+offset)*stride (r=row=idx/2).
  let row = idx / 2u;
  var coord: u32;
  if ((idx & 1u) == 0u) {
    coord = row % params.width;
  } else {
    coord = row / params.width;
  }
  t_out[idx] = (f32(coord) + params.offset) * params.stride;
}
