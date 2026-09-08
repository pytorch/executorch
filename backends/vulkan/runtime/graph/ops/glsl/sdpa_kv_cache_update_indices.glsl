#version 450 core

#define PRECISION ${PRECISION}

#define IN_VEC4_T ${texel_load_type(DTYPE, INPUT_STORAGE)}
#define T ${buffer_scalar_type(DTYPE)}

$if OUTPUT_STORAGE == "buffer":
  #define OUTPUT_BUFFER
$if INPUT_STORAGE == "buffer":
  #define INPUT_BUFFER

${define_required_extensions(INPUT_STORAGE, DTYPE)}
$if OUTPUT_STORAGE == "buffer" and INPUT_STORAGE != "buffer":
  ${define_required_extensions(OUTPUT_STORAGE, DTYPE)}

layout(std430) buffer;

#include "common.glslh"

$if OUTPUT_STORAGE == "buffer":
  ${layout_declare_tensor(B, "w", "t_cache", DTYPE, OUTPUT_STORAGE)}
$else:
  ${layout_declare_tensor(B, "w", "t_cache", DTYPE, OUTPUT_STORAGE, is_scalar_array=False)}
$if INPUT_STORAGE == "buffer":
  ${layout_declare_tensor(B, "r", "t_projected", DTYPE, INPUT_STORAGE)}
$else:
  ${layout_declare_tensor(B, "r", "t_projected", DTYPE, INPUT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_indices", "int32", "buffer")}

${layout_declare_ubo(B, "ivec4", "cache_sizes")}
${layout_declare_ubo(B, "ivec4", "projected_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

IN_VEC4_T read_projected_d4(
    const int d4,
    const int h,
    const int s,
    const int D,
    const int H) {
#ifdef INPUT_BUFFER
  const int d = d4 * 4;
  const int base = (s * H * D) + (h * D) + d;
  IN_VEC4_T texel = IN_VEC4_T(0.0);
  if (d < D) {
    texel.x = T(t_projected[base]);
  }
  if (d + 1 < D) {
    texel.y = T(t_projected[base + 1]);
  }
  if (d + 2 < D) {
    texel.z = T(t_projected[base + 2]);
  }
  if (d + 3 < D) {
    texel.w = T(t_projected[base + 3]);
  }
  return texel;
#else
  return texelFetch(t_projected, ivec3(d4, h, s), 0);
#endif
}

void write_cache_d4(
    const IN_VEC4_T texel,
    const int d4,
    const int c,
    const int h,
    const int D,
    const int H) {
#ifdef OUTPUT_BUFFER
  const int d = d4 * 4;
  const int base = (c * H * D) + (h * D) + d;
  if (d < D) {
    t_cache[base] = T(texel.x);
  }
  if (d + 1 < D) {
    t_cache[base + 1] = T(texel.y);
  }
  if (d + 2 < D) {
    t_cache[base + 2] = T(texel.z);
  }
  if (d + 3 < D) {
    t_cache[base + 3] = T(texel.w);
  }
#else
  imageStore(t_cache, ivec3(d4, h, c), texel);
#endif
}

void main() {
  const int d4 = int(gl_GlobalInvocationID.x);
  const int s = int(gl_GlobalInvocationID.y);
  const int h = int(gl_GlobalInvocationID.z);

  const int D = projected_sizes.x;
  const int D4 = div_up_4(D);
  const int H = projected_sizes.y;
  const int S = projected_sizes.z;
  const int C = cache_sizes.z;

  if (d4 >= D4 || s >= S || h >= H) {
    return;
  }

  const int c = t_indices[s];
  if (c < 0 || c >= C) {
    return;
  }

  write_cache_d4(
      read_projected_d4(d4, h, s, D, H), d4, c, h, D, H);
}
