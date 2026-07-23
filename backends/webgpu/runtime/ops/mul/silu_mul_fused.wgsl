@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params {
  num_elements: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

// Fused SwiGLU activation: output = (g * sigmoid(g)) * up, folding the separate
// sigmoid(gate) -> mul(gate,sig)=silu -> mul(silu,up) triple into one dispatch.
// sigmoid + silu are computed in registers (never written to memory), so gate + up
// are read once and one output is written. The sigmoid form (1/(1+exp(-x))) and the
// multiply order match the original ops -> bit-exact.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.num_elements) {
        return;
    }
    let g = gate[idx];
    let sig = 1.0 / (1.0 + exp(-g));
    output[idx] = (g * sig) * up[idx];
}
