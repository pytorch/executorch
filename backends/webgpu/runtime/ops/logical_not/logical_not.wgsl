@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

struct Params {
  num_elements: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 64u;

// One thread per u32 word inverts 4 bool bytes -> no inter-thread race.
@compute @workgroup_size(wg_size, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let word_idx = gid.x;
    let n_words = (params.num_elements + 3u) / 4u;
    if (word_idx >= n_words) {
        return;
    }
    let in_word = input[word_idx];
    var out_word: u32 = 0u;
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        let i = word_idx * 4u + j;
        if (i < params.num_elements) {
            let b = (in_word >> (j * 8u)) & 0xFFu;
            if (b == 0u) {
                out_word = out_word | (1u << (j * 8u));
            }
        }
    }
    output[word_idx] = out_word;
}
