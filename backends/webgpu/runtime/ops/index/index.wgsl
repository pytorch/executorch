@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> index: array<i32>;

struct Params {
  numel: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

override wg_size: u32 = 64;

@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_bufi = gid.x;
    if (out_bufi >= params.numel) {
        return;
    }

    // 1D-self gather out[i]=self[index[i]] (mirrors Vulkan index_tensor_buffer.glsl).
    let i = index[out_bufi];
    output[out_bufi] = input[u32(i)];
}
