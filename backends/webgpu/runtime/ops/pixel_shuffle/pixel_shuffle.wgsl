@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
  r: u32,
  out_c: u32,
  out_h: u32,
  out_w: u32,
  in_c: u32,
  in_h: u32,
  in_w: u32,
  numel: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 64u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    // 2D-folded flat index (lifts the 65535 1D-dispatch cap for large numel).
    let oi = gid.x + gid.y * (num_workgroups.x * wg_size);
    if (oi >= params.numel) {
        return;
    }

    // (N, C*r*r, H, W) -> (N, C, H*r, W*r); leading dims collapse into b.
    let w_out = oi % params.out_w;
    let h_out = (oi / params.out_w) % params.out_h;
    let c_out = (oi / (params.out_w * params.out_h)) % params.out_c;
    let b = oi / (params.out_w * params.out_h * params.out_c);

    let w_in = w_out / params.r;
    let h_in = h_out / params.r;
    let c_in = c_out * params.r * params.r
        + (h_out % params.r) * params.r + (w_out % params.r);

    let in_bufi =
        ((b * params.in_c + c_in) * params.in_h + h_in) * params.in_w + w_in;
    output[oi] = input[in_bufi];
}
