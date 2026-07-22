@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
  num_rows: u32,
  reduce_size: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 256u;

@compute @workgroup_size(wg_size)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let row = gid.x + gid.y * (num_workgroups.x * wg_size);
    if (row >= params.num_rows) {
        return;
    }
    let base = row * params.reduce_size;
    var acc = input[base];
    for (var j = 1u; j < params.reduce_size; j = j + 1u) {
        acc = max(acc, input[base + j]);
    }
    output[row] = acc;
}
