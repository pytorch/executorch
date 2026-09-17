@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> index: array<i32>;

struct TensorMeta {
  ndim: u32,
  numel: u32,
  sizes: array<vec4<u32>, 2>,
  strides: array<vec4<u32>, 2>,
}
@group(0) @binding(3) var<uniform> out_meta: TensorMeta;
@group(0) @binding(4) var<uniform> in_meta: TensorMeta;

struct Params {
  info: vec4<u32>,
}
@group(0) @binding(5) var<uniform> params: Params;

override wg_size: u32 = 64u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    // 2D-folded flat index (lifts the 65535 1D-dispatch cap for large numel).
    let out_bufi = gid.x + gid.y * (num_workgroups.x * wg_size);
    if (out_bufi >= out_meta.numel) {
        return;
    }

    // Gather: in_coord = out_coord, but the dim coord is remapped by index[].
    let dim = params.info.x;
    var rem = out_bufi;
    var in_bufi: u32 = 0u;
    for (var d: u32 = 0u; d < out_meta.ndim; d = d + 1u) {
        let coord = rem / out_meta.strides[d >> 2u][d & 3u];
        rem = rem % out_meta.strides[d >> 2u][d & 3u];
        var in_coord = coord;
        if (d == dim) {
            in_coord = u32(index[coord]);
        }
        in_bufi = in_bufi + in_coord * in_meta.strides[d >> 2u][d & 3u];
    }
    output[out_bufi] = input[in_bufi];
}
