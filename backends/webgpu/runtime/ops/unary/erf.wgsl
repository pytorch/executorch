@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
  num_elements: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 256u;

// erf(x) as an odd/even rational fit on [-4, 4] (Eigen's fast erf for float),
// within 5e-7 of the exact value with small relative error near 0. Past |x| = 4
// erf rounds to +-1 in fp32. NaN is passed through explicitly: clamp() of NaN
// is implementation-defined, and WGSL may assume float compares never see NaN,
// so the check reads the bits.
fn erf_rational(x: f32) -> f32 {
    let c = clamp(x, -4.0, 4.0);
    let c2 = c * c;
    var p = -2.72614225801306e-10;
    p = p * c2 + 2.77068142495902e-08;
    p = p * c2 - 2.10102402082508e-06;
    p = p * c2 - 5.69250639462346e-05;
    p = p * c2 - 7.34990630326855e-04;
    p = p * c2 - 2.95459980854025e-03;
    p = p * c2 - 1.60960333262415e-02;
    var q = -1.45660718464996e-05;
    q = q * c2 - 2.13374055278905e-04;
    q = q * c2 - 1.68282697438203e-03;
    q = q * c2 - 7.37332916720468e-03;
    q = q * c2 - 1.42647390514189e-02;
    let is_nan = (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
    return select(c * p / q, x, is_nan);
}

@compute @workgroup_size(wg_size)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
    if (idx >= params.num_elements) {
        return;
    }
    output[idx] = erf_rational(input[idx]);
}
