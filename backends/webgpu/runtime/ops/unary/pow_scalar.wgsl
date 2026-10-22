@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
  num_elements: u32,
  minimum: f32,
}
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 256u;

@compute @workgroup_size(wg_size)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
    if (idx >= params.num_elements) {
        return;
    }
    // WGSL pow(x, y) = exp2(y*log2(x)) is undefined for x < 0. Fix the common
    // integer-exponent case to match PyTorch (|x|^y, sign-preserved for odd y);
    // the rare non-integer negative case keeps the prior WGSL behavior.
    let x = input[idx];
    let e = params.minimum;
    var r: f32;
    if (x >= 0.0) {
        r = pow(x, e);
    } else if (fract(e) == 0.0) {
        r = pow(abs(x), e);
        if (fract(e * 0.5) != 0.0) {
            r = -r;
        }
    } else {
        r = pow(x, e);
    }
    output[idx] = r;
}
