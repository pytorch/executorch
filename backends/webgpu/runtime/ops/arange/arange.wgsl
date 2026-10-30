@group(0) @binding(0) var<storage, read_write> output: array<f32>;

struct Params {
  num_elements: u32,
  start: f32,
  step: f32,
  _pad: u32,
}
@group(0) @binding(1) var<uniform> params: Params;

override wg_size: u32 = 256;

@compute @workgroup_size(wg_size)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
    if (idx >= params.num_elements) {
        return;
    }
    output[idx] = params.start + f32(idx) * params.step;
}
