@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

struct Params {
  num_elements: u32,
  mode: u32,
  scalar: f32,
  _pad: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 64u;

// One thread per output u32 word packs 4 bool bytes -> no inter-thread race.
@compute @workgroup_size(wg_size, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let word_idx = gid.x;
    let n_words = (params.num_elements + 3u) / 4u;
    if (word_idx >= n_words) {
        return;
    }
    var packed: u32 = 0u;
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        let i = word_idx * 4u + j;
        if (i < params.num_elements) {
            let v = input[i];
            var r: bool;
            if (params.mode == 0u) {
                r = v == params.scalar;
            } else if (params.mode == 1u) {
                r = v != params.scalar;
            } else if (params.mode == 2u) {
                r = v <= params.scalar;
            } else if (params.mode == 3u) {
                r = v >= params.scalar;
            } else {
                r = v < params.scalar;
            }
            if (r) {
                packed = packed | (1u << (j * 8u));
            }
        }
    }
    output[word_idx] = packed;
}
