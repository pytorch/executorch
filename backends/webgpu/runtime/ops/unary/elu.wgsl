@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
  num_elements: u32,
  alpha: f32,
  scale: f32,
  input_scale: f32,
}
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 256u;

// aten.elu: scale * alpha * (exp(x * input_scale) - 1) for x <= 0, else
// scale * x, so NaN stays NaN.
@compute @workgroup_size(wg_size)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
    if (idx >= params.num_elements) {
        return;
    }
    let x = input[idx];
    let negative = params.alpha * (exp(min(x, 0.0) * params.input_scale) - 1.0);
    output[idx] = params.scale * select(x, negative, x <= 0.0);
}
