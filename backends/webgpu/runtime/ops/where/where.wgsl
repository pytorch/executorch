@group(0) @binding(0) var<storage, read> cond: array<u32>;
@group(0) @binding(1) var<storage, read> input_a: array<f32>;
@group(0) @binding(2) var<storage, read> input_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

struct TensorMeta {
  ndim: u32,
  numel: u32,
  sizes: vec4<u32>,
  strides: vec4<u32>,
}
@group(0) @binding(4) var<uniform> out_meta: TensorMeta;
@group(0) @binding(5) var<uniform> cond_meta: TensorMeta;
@group(0) @binding(6) var<uniform> a_meta: TensorMeta;
@group(0) @binding(7) var<uniform> b_meta: TensorMeta;

override wg_size: u32 = 64u;

// 1-byte bool packed 4-per-u32; extract byte i (mirrors q4gsw weight unpack).
fn cond_is_true(i: u32) -> bool {
    let word = cond[i >> 2u];
    return ((word >> ((i & 3u) * 8u)) & 0xFFu) != 0u;
}

@compute @workgroup_size(wg_size, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= out_meta.numel) {
        return;
    }

    var rem = idx;
    var lc: u32 = 0u;
    var la: u32 = 0u;
    var lb: u32 = 0u;
    for (var d: u32 = 0u; d < out_meta.ndim; d = d + 1u) {
        let coord = rem / out_meta.strides[d];
        rem = rem % out_meta.strides[d];
        lc = lc + min(coord, cond_meta.sizes[d] - 1u) * cond_meta.strides[d];
        la = la + min(coord, a_meta.sizes[d] - 1u) * a_meta.strides[d];
        lb = lb + min(coord, b_meta.sizes[d] - 1u) * b_meta.strides[d];
    }

    if (cond_is_true(lc)) {
        output[idx] = input_a[la];
    } else {
        output[idx] = input_b[lb];
    }
}
