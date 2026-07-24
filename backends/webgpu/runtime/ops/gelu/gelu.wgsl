@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
  num_elements: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 64u;

// Abramowitz & Stegun 7.1.26 erf approximation (max abs err ~1.5e-7).
fn erf_approx4(x: vec4<f32>) -> vec4<f32> {
    let s = sign(x);
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
    return s * y;
}

fn gelu_tanh4(x: vec4<f32>) -> vec4<f32> {
    let inner = clamp(0.7978845608028654 * (x + 0.044715 * x * x * x), vec4<f32>(-10.0), vec4<f32>(10.0));
    return 0.5 * x * (1.0 + tanh(inner));
}

fn gelu_erf4(x: vec4<f32>) -> vec4<f32> {
    return 0.5 * x * (1.0 + erf_approx4(x * 0.7071067811865476));
}

// Vec4 body + scalar-tail idiom (num_elements isn't guaranteed % 4 == 0 for
// arbitrary FFN activation shapes): each thread gathers up to 4 elements via
// select()-guarded scalar loads (the out-of-bounds lanes read a WebGPU-spec-
// clamped, NOT zero, value — but that value is discarded by the same select()
// before use), computes GELU as one vec4 op, then scatters back only the
// in-bounds lanes.
@compute @workgroup_size(wg_size, 1, 1)
fn main_tanh(@builtin(global_invocation_id) gid: vec3<u32>) {
    let base = gid.x * 4u;
    if (base >= params.num_elements) {
        return;
    }
    let x1 = select(0.0, input[base + 1u], base + 1u < params.num_elements);
    let x2 = select(0.0, input[base + 2u], base + 2u < params.num_elements);
    let x3 = select(0.0, input[base + 3u], base + 3u < params.num_elements);
    let y = gelu_tanh4(vec4<f32>(input[base], x1, x2, x3));
    output[base] = y.x;
    if (base + 1u < params.num_elements) { output[base + 1u] = y.y; }
    if (base + 2u < params.num_elements) { output[base + 2u] = y.z; }
    if (base + 3u < params.num_elements) { output[base + 3u] = y.w; }
}

@compute @workgroup_size(wg_size, 1, 1)
fn main_erf(@builtin(global_invocation_id) gid: vec3<u32>) {
    let base = gid.x * 4u;
    if (base >= params.num_elements) {
        return;
    }
    let x1 = select(0.0, input[base + 1u], base + 1u < params.num_elements);
    let x2 = select(0.0, input[base + 2u], base + 2u < params.num_elements);
    let x3 = select(0.0, input[base + 3u], base + 3u < params.num_elements);
    let y = gelu_erf4(vec4<f32>(input[base], x1, x2, x3));
    output[base] = y.x;
    if (base + 1u < params.num_elements) { output[base + 1u] = y.y; }
    if (base + 2u < params.num_elements) { output[base + 2u] = y.z; }
    if (base + 3u < params.num_elements) { output[base + 3u] = y.w; }
}
