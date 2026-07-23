@group(0) @binding(0) var<storage, read> self_: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct TensorMeta {
  ndim: u32,
  numel: u32,
  sizes: vec4<u32>,
  strides: vec4<u32>,
}
@group(0) @binding(3) var<uniform> out_meta: TensorMeta;
@group(0) @binding(4) var<uniform> self_meta: TensorMeta;

struct GatherParams {
  dim: u32,
}
@group(0) @binding(5) var<uniform> params: GatherParams;

override wg_size: u32 = 256;

@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x;
    if (o >= out_meta.numel) {
        return;
    }
    var rem = o;
    var self_idx: u32 = 0u;
    for (var d: u32 = 0u; d < out_meta.ndim; d = d + 1u) {
        let c = rem / out_meta.strides[d];
        rem = rem % out_meta.strides[d];
        var coord = c;
        if (d == params.dim) {
            coord = u32(indices[o]);
        }
        self_idx = self_idx + coord * self_meta.strides[d];
    }
    output[o] = self_[self_idx];
}
