#version 450 core

#define PRECISION ${PRECISION}

#define IN_VEC4_T ${texel_load_type(DTYPE, INPUT_STORAGE)}
#define T ${buffer_scalar_type(DTYPE)}

$if OUTPUT_STORAGE == "buffer":
  #define OUTPUT_BUFFER
$if INPUT_STORAGE == "buffer":
  #define INPUT_BUFFER

${define_required_extensions(DTYPE)}

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_cache", DTYPE, OUTPUT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_projected", DTYPE, INPUT_STORAGE, is_scalar_array=False)}

${layout_declare_ubo(B, "ivec4", "cache_sizes")}
${layout_declare_ubo(B, "ivec4", "projected_sizes")}
${layout_declare_ubo(B, "int", "input_pos")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * t_cache will have sizes of (batches, n_heads, max_context_len, head_dim).
 * t_projected will have sizes of (batches, seq_len, n_heads, head_dim).
 *
 * Note that the cache tensor swaps the order of the n_heads and seq len
 * dimensions. This is to faciliate more optimal memory access patterns when
 * using the caches to compute matrix multiplications.
 *
 * The cache update inserts the values of t_projected into t_cache at the index
 * specified by input_pos at the seq_len dimension. It is equivalent to calling

 * t_cache = t_cache.slice_scatter(
 *     t_projected, dim=1, start=input_pos, end=input_pos+seq_len)
 *
 * Note that this shader is implemented assuming that max_batch_size is 1.
 */

IN_VEC4_T read_projected_d4(
    const int d4,
    const int h,
    const int s,
    const int D4,
    const int H,
    const int S) {
#ifdef INPUT_BUFFER
  return t_projected[(s * H * D4) + (h * D4) + d4];
#else
  return texelFetch(t_projected, ivec3(d4, h, s), 0);
#endif
}

void write_cache_d4(
    const IN_VEC4_T texel,
    const int d4,
    const int c,
    const int h,
    const int D4,
    const int C,
    const int H) {
#ifdef OUTPUT_BUFFER
  t_cache[(c * H * D4) + (h * D4) + d4] = texel;
#else
  imageStore(t_cache, ivec3(d4, h, c), texel);
#endif
}

void main() {
  const int d4 = int(gl_GlobalInvocationID.x); // idx along the head_dim dim
  const int s = int(gl_GlobalInvocationID.y); // idx along the seq_len dim
  const int h = int(gl_GlobalInvocationID.z); // idx along the n_heads dim

  const int D4 = div_up_4(projected_sizes.x);
  const int S = projected_sizes.z;
  const int H = projected_sizes.y;

  const int c = s + input_pos; // idx along max_context_len dim
  const int C = cache_sizes.z;

  if (d4 >= D4 || c >= C || h >= H) {
    return;
  }

  IN_VEC4_T in_texel = IN_VEC4_T(0.0);
  if (s < S) {
    in_texel = read_projected_d4(d4, h, s, D4, H, S);
  }

  write_cache_d4(in_texel, d4, c, h, D4, C, H);
}
