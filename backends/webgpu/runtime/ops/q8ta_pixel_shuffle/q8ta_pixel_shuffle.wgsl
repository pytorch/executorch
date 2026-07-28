@group(0) @binding(0) var<storage, read> t_in: array<u32>;
@group(0) @binding(1) var<storage, read_write> t_out: array<u32>;

struct Params {
  r: u32,
  out_c: u32,
  out_h: u32,
  out_w: u32,
  in_c: u32,
  in_h: u32,
  in_w: u32,
  numel: u32,
  input_scale: f32,
  inv_output_scale: f32,
  input_zero_point: i32,
  output_zero_point: i32,
}
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 64u;

fn unpack_i8(word: u32, byte_idx: u32) -> i32 {
  return i32(((word >> (byte_idx * 8u)) & 0xFFu) ^ 0x80u) - 128;
}

@compute @workgroup_size(wg_size, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    // One thread per packed u32 word (4 int8 outputs); 2D-fold lifts 65535.
    let widx = gid.x + gid.y * (num_workgroups.x * wg_size);
    if (widx >= params.numel / 4u) {
        return;
    }
    var packed: u32 = 0u;
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        // (N, C*r*r, H, W) -> (N, C, H*r, W*r) gather; leading dims collapse to b.
        let oi = widx * 4u + k;
        let w_out = oi % params.out_w;
        let h_out = (oi / params.out_w) % params.out_h;
        let c_out = (oi / (params.out_w * params.out_h)) % params.out_c;
        let b = oi / (params.out_w * params.out_h * params.out_c);
        let w_in = w_out / params.r;
        let h_in = h_out / params.r;
        let c_in = c_out * params.r * params.r
            + (h_out % params.r) * params.r + (w_out % params.r);
        let in_flat =
            ((b * params.in_c + c_in) * params.in_h + h_in) * params.in_w + w_in;
        let q_in = unpack_i8(t_in[in_flat / 4u], in_flat % 4u);
        let deq = params.input_scale * f32(q_in - params.input_zero_point);
        var q =
            i32(round(deq * params.inv_output_scale)) + params.output_zero_point;
        q = clamp(q, -128, 127);
        packed = packed | ((bitcast<u32>(q) & 0xFFu) << (k * 8u));
    }
    t_out[widx] = packed;
}
