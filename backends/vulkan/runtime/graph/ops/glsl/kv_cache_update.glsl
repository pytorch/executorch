#version 450 core

#define PRECISION ${PRECISION}

#define T ${buffer_scalar_type(DTYPE)}

${define_active_storage_type(STORAGE)}
${define_required_extensions(DTYPE)}

layout(std430) buffer;

#include "indexing_utils.h"

${layout_declare_tensor(B, "w", "cache", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "projected", DTYPE, STORAGE)}
$if STORAGE == "buffer":
  ${layout_declare_ubo(B, "int", "projected_numel")}
  ${layout_declare_ubo(B, "ivec4", "cache_strides")}
  ${layout_declare_ubo(B, "int", "input_pos")}
$else:
  ${layout_declare_ubo(B, "ivec3", "projected_limits")}
  ${layout_declare_ubo(B, "int", "input_pos")}


layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * t_cache will have sizes of (max_batch_size, max_seq_len, n_heads, head_dim).
 * t_projected will have sizes of (batch_size, seq_len, n_heads, head_dim).
 *
 * The cache update inserts the values of t_projected into t_cache at the index
 * specified by input_pos at the seq_len dimension. It is equivalent to calling

 * t_cache = t_cache.slice_scatter(
 *     t_projected, dim=1, start=input_pos, end=input_pos+seq_len)
 *
 * Note that this shader is implemented assuming that max_batch_size is 1.
 */

#ifdef USING_BUFFER

/***************************
 ** Buffer Implementation **
 ***************************/

void main() {
  int projected_bufi = int(gl_GlobalInvocationID.x);
  // Bump cache index forward by input_pos elements along the seq_len dimension.
  // cache_strides contains the strides of the cache tensor.
  int cache_bufi = input_pos * cache_strides.z + projected_bufi;
  if (projected_bufi >= projected_numel) {
    return;
  }
  cache[cache_bufi] = projected[projected_bufi];
}

#else

/****************************
 ** Texture Implementation **
 ****************************/

// Note that this shader assumes the that tensors are width packed, i.e.
// packed_dim = 0
void main() {
  const ivec3 projected_pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(projected_pos, projected_limits))) {
    return;
  }

  const ivec3 cache_pos = ivec3(
      projected_pos.x,
      projected_pos.y,
      projected_pos.z + input_pos);

  write_texel(cache, cache_pos, load_texel(projected, projected_pos));
}

#endif // USING_BUFFER
