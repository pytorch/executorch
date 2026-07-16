@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct TensorMeta {
  ndim: u32,
  numel: u32,
  sizes: vec4<u32>,
  strides: vec4<u32>,
}
@group(0) @binding(2) var<uniform> out_meta: TensorMeta;
@group(0) @binding(3) var<uniform> in_meta: TensorMeta;

struct Params {
  dim: u32,
  index: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

override wg_size: u32 = 64u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_bufi = gid.x;
    if (out_bufi >= out_meta.numel) {
        return;
    }

    // Gather: out dim od -> in dim (od if od < dim else od+1); sel dim = index.
    var rem = out_bufi;
    var in_bufi: u32 = params.index * in_meta.strides[params.dim];
    for (var od: u32 = 0u; od < out_meta.ndim; od = od + 1u) {
        let coord = rem / out_meta.strides[od];
        rem = rem % out_meta.strides[od];
        var id = od;
        if (od >= params.dim) {
            id = od + 1u;
        }
        in_bufi = in_bufi + coord * in_meta.strides[id];
    }
    output[out_bufi] = input[in_bufi];
}
