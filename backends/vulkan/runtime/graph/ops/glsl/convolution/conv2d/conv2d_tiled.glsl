/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

// Half tensors are stored as Float on devices without float16 arithmetic or
// 16-bit storage, so the half variants only run where float16 is supported.
$if DTYPE == "half":
  #extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : require
#define VEC4_T ${buffer_gvec_type(DTYPE, 4)}
#define SCALAR_T ${buffer_scalar_type(DTYPE)}
#define TILE_X ${TILE_X}
#define TILE_Y ${TILE_Y}
#define TILE_PIXELS (TILE_X * TILE_Y)

$if WEIGHT_BUFFER:
  ${define_required_extensions("buffer", DTYPE)}

layout(std430) buffer;

${layout_declare_tensor(0, "w", "t_out", DTYPE, "texture3d")}
${layout_declare_tensor(1, "r", "t_in", DTYPE, "texture3d", precision="mediump" if DTYPE == "half" else "PRECISION")}
$if WEIGHT_BUFFER:
  ${layout_declare_tensor(2, "r", "t_kernel", DTYPE, "buffer", is_scalar_array=False)}
$else:
  ${layout_declare_tensor(2, "r", "t_kernel", DTYPE, "texture2d", precision="mediump" if DTYPE == "half" else "PRECISION")}
${layout_declare_tensor(3, "r", "t_bias", DTYPE, "texture2d")}
${layout_declare_ubo(4, "ivec3", "out_limits")}
${layout_declare_ubo(5, "ivec4", "in_sizes")}
${layout_declare_ubo(6, "ivec2", "kernel_size", "ivec2", "stride", "ivec2", "padding", "ivec2", "dilation")}
${layout_declare_ubo(7, "ivec2", "overlay_region", "int", "in_group_size")}
${layout_declare_ubo(8, "float", "out_min", "float", "out_max")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "ngroups", "1")}
$if SPECIALIZED_BOUNDS:
  ${layout_declare_spec_const(4, "int", "input_quads", "1")}
  ${layout_declare_spec_const(5, "int", "kernel_extent", "3")}
  #define IN_GROUP_SIZE (4 * input_quads)
  #define KERNEL_SIZE ivec2(kernel_extent)
$else:
  #define IN_GROUP_SIZE in_group_size
  #define KERNEL_SIZE kernel_size

#define OUT_N4 ${OUT_N4}
$if CLAMP_TAPS:
  #define CLAMP_TAPS
$if ZERO_BORDER:
  #define ZERO_BORDER
$if WEIGHT_BUFFER:
  #define WEIGHT_BUFFER
$if CONST_TEXTURE_OFFSETS:
  #define CONST_TEXTURE_OFFSETS
${layout_declare_spec_const(8, "int", "in_group_channels", "1")}
${layout_declare_spec_const(9, "int", "out_group_channels", "1")}
$if GATHER_GROUPS:
  #define GATHER_GROUPS
${layout_declare_spec_const(6, "bool", "clamp_out", "false")}
${layout_declare_spec_const(7, "bool", "has_bias", "false")}
#define OUTPUT_VALUE(v) (clamp_out ? clamp(vec4(v), out_min, out_max) : vec4(v))

#ifdef GATHER_GROUPS
SCALAR_T group_input(ivec2 p, int ci, int co) {
  if (co >= out_group_channels * ngroups) {
    return SCALAR_T(0);
  }
  const int channel = (co / out_group_channels) * in_group_channels + ci;
  const vec4 fetched = texelFetch(t_in, ivec3(p, channel / 4), 0);
  const int lane = channel % 4;
  return SCALAR_T(lane == 0 ? fetched.x
      : lane == 1 ? fetched.y : lane == 2 ? fetched.z : fetched.w);
}
#endif

VEC4_T sums[TILE_PIXELS][OUT_N4];
VEC4_T weights[OUT_N4][4];
VEC4_T inputs[TILE_PIXELS];

void kernel_range(ivec2 origin, out ivec2 kstart, out ivec2 kend) {
  kstart = ivec2(0);
  kend = KERNEL_SIZE;
#ifdef CLAMP_TAPS
#if TILE_X != 1 || TILE_Y != 1
#error CLAMP_TAPS requires a single output pixel
#endif
  if (any(lessThan(origin, ivec2(0))) ||
      any(greaterThan(origin + overlay_region, in_sizes.xy))) {
    kstart = max(ivec2(0), (dilation - ivec2(1) - origin) / dilation);
    kend = min(KERNEL_SIZE, (in_sizes.xy + dilation - ivec2(1) - origin) / dilation);
  }
#endif
}

highp vec3 input_scale() {
#ifdef ZERO_BORDER
  const highp ivec3 input_extent = textureSize(t_in, 0);
  const highp vec3 input_rcp = vec3(1.0) / vec3(input_extent);
  return input_rcp;
#else
  return vec3(0);
#endif
}

highp float input_start(int group_offset, highp vec3 input_rcp) {
#ifdef ZERO_BORDER
  return (float(group_offset) + 0.5) * input_rcp.z;
#else
  return 0.0;
#endif
}

void input_step(inout highp float input_z, highp vec3 input_rcp) {
#ifdef ZERO_BORDER
  input_z += input_rcp.z;
#endif
}

void load_weights(ivec3 pos, int z4, int kx, int ky) {
  const int weight_x = (z4 * KERNEL_SIZE.x + kx) * 4;
  [[unroll]] for (int slice = 0; slice < OUT_N4; ++slice) {
    [[unroll]] for (int lane = 0; lane < 4; ++lane) {
      const int weight_y = (pos.z + slice) * KERNEL_SIZE.y + ky;
#ifdef WEIGHT_BUFFER
      weights[slice][lane] =
          t_kernel[weight_y * (IN_GROUP_SIZE * KERNEL_SIZE.x) + weight_x + lane];
#else
      weights[slice][lane] =
          VEC4_T(texelFetch(t_kernel, ivec2(weight_x + lane, weight_y), 0));
#endif
    }
  }
}

void load_inputs(
    ivec3 pos, ivec2 tap, int z4, int group_offset,
    highp vec3 input_rcp, highp float input_z) {
#ifdef CONST_TEXTURE_OFFSETS
  // Constant texture offsets require unit stride.
  const highp vec3 tap_uv =
      vec3((vec2(tap) + vec2(0.5)) * input_rcp.xy, input_z);
  const highp vec4 fetched0 =
      textureLodOffset(t_in, tap_uv, 0.0, ivec3(0, 0, 0));
  const highp vec4 fetched1 =
      textureLodOffset(t_in, tap_uv, 0.0, ivec3(1, 0, 0));
  const highp vec4 fetched2 =
      textureLodOffset(t_in, tap_uv, 0.0, ivec3(0, 1, 0));
  const highp vec4 fetched3 =
      textureLodOffset(t_in, tap_uv, 0.0, ivec3(1, 1, 0));
  inputs[0] = VEC4_T(fetched0);
  inputs[1] = VEC4_T(fetched1);
  inputs[2] = VEC4_T(fetched2);
  inputs[3] = VEC4_T(fetched3);
#else
  [[unroll]] for (int i = 0; i < TILE_PIXELS; ++i) {
    const ivec2 offset = ivec2(i % TILE_X, i / TILE_X);
    const ivec2 p = tap + offset * stride;
#ifdef CLAMP_TAPS
    const vec4 fetched = texelFetch(t_in, ivec3(p, z4 + group_offset), 0);
    inputs[i] = VEC4_T(fetched);
#elif defined(ZERO_BORDER)
    const highp vec4 fetched = textureLod(
        t_in, vec3((vec2(p) + vec2(0.5)) * input_rcp.xy, input_z), 0.0);
    inputs[i] = VEC4_T(fetched);
#else
    const bool valid = pos.x + offset.x < out_limits.x &&
        pos.y + offset.y < out_limits.y &&
        all(greaterThanEqual(p, ivec2(0))) &&
        all(lessThan(p, in_sizes.xy));
    const vec4 fetched = texelFetch(
        t_in, ivec3(clamp(p, ivec2(0), in_sizes.xy - ivec2(1)), z4 + group_offset), 0);
    inputs[i] = VEC4_T(mix(vec4(0), fetched, bvec4(valid)));
#endif
  }
#endif
}

// Branches once, outside the unrolled loops. Implementations that do not fold
// specialization constants (e.g. SwiftShader) would otherwise keep a branch
// per multiply-add, and SwiftShader's pipeline creation time grows
// exponentially with the number of branches in a loop.
void accumulate() {
  if (ngroups > 1) {
    [[unroll]] for (int lane = 0; lane < 4; ++lane) {
      [[unroll]] for (int i = 0; i < TILE_PIXELS; ++i) {
        [[unroll]] for (int s = 0; s < OUT_N4; ++s) {
          sums[i][s] = fma(weights[s][lane], VEC4_T(inputs[i][lane]), sums[i][s]);
        }
      }
    }
  } else {
    [[unroll]] for (int lane = 0; lane < 4; ++lane) {
      [[unroll]] for (int i = 0; i < TILE_PIXELS; ++i) {
        [[unroll]] for (int s = 0; s < OUT_N4; ++s) {
          sums[i][s] += weights[s][lane] * inputs[i][lane];
        }
      }
    }
  }
}

void store_tile(ivec3 pos) {
#if OUT_N4 == 2
  const bool second_slice = pos.z + 1 < out_limits.z;
#endif
  [[unroll]] for (int i = 0; i < TILE_PIXELS; ++i) {
    const ivec2 offset = ivec2(i % TILE_X, i / TILE_X);
    if (pos.x + offset.x < out_limits.x && pos.y + offset.y < out_limits.y) {
      imageStore(t_out, pos + ivec3(offset, 0), OUTPUT_VALUE(sums[i][0]));
#if OUT_N4 == 2
      if (second_slice) {
        imageStore(t_out, pos + ivec3(offset, 1), OUTPUT_VALUE(sums[i][1]));
      }
#endif
    }
  }
}

#ifdef GATHER_GROUPS
void gather_accumulate(ivec3 pos, ivec2 tap, int z4) {
  for (int lane = 0;
       lane < 4 && z4 * 4 + lane < in_group_channels; ++lane) {
    [[unroll]] for (int i = 0; i < TILE_PIXELS; ++i) {
      const ivec2 offset = ivec2(i % TILE_X, i / TILE_X);
      const ivec2 p = tap + offset * stride;
      if (all(lessThan(pos.xy + offset, out_limits.xy)) &&
          all(greaterThanEqual(p, ivec2(0))) && all(lessThan(p, in_sizes.xy))) {
        const int co = pos.z * 4;
        const int ci = z4 * 4 + lane;
        const VEC4_T x = VEC4_T(
            group_input(p, ci, co), group_input(p, ci, co + 1),
            group_input(p, ci, co + 2), group_input(p, ci, co + 3));
        sums[i][0] = fma(weights[0][lane], x, sums[i][0]);
      }
    }
  }
}
#endif

int kernel_end_x(ivec2 kend) {
#ifdef CLAMP_TAPS
  return kend.x;
#else
  return KERNEL_SIZE.x;
#endif
}

int kernel_end_y(ivec2 kend) {
#ifdef CLAMP_TAPS
  return kend.y;
#else
  return KERNEL_SIZE.y;
#endif
}

/*
 * Direct 2D convolution with activation-dtype accumulation. Each invocation
 * computes a spatial output tile for OUT_N4 consecutive texel slices (4 output
 * channels per slice), reusing each input texel across the tile.
 *
 * ZERO_BORDER variants read padding taps through the input sampler, which
 * returns zero outside the image, instead of bounds checking every tap. They
 * require the input image extents to match the logical input sizes exactly.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID) * ivec3(TILE_X, TILE_Y, OUT_N4);
  if (any(greaterThanEqual(pos, out_limits))) {
    return;
  }

  const int group_offset = ngroups == 1 ? 0
      : (pos.z * 4 / out_group_channels) * (in_group_channels / 4);

  const highp vec3 input_rcp = input_scale();

  VEC4_T bias[OUT_N4];
  [[unroll]] for (int s = 0; s < OUT_N4; ++s) {
    bias[s] = has_bias
        ? VEC4_T(texelFetch(t_bias, ivec2(pos.z + s, 0), 0)) : VEC4_T(0);
  }
  [[unroll]] for (int i = 0; i < TILE_PIXELS; ++i) {
    [[unroll]] for (int s = 0; s < OUT_N4; ++s) {
      sums[i][s] = bias[s];
    }
  }

  const ivec2 origin = pos.xy * stride - padding;
  ivec2 kstart, kend;
  kernel_range(origin, kstart, kend);
  for (int ky = kstart.y; ky < kernel_end_y(kend); ++ky) {
    for (int kx = kstart.x; kx < kernel_end_x(kend); ++kx) {
      highp float input_z = input_start(group_offset, input_rcp);
      int z4 = 0;
      do {
        const ivec2 tap = origin + ivec2(kx, ky) * dilation;
        load_weights(pos, z4, kx, ky);

#ifdef GATHER_GROUPS
        gather_accumulate(pos, tap, z4);
#else
        load_inputs(pos, tap, z4, group_offset, input_rcp, input_z);
        accumulate();
#endif
        input_step(input_z, input_rcp);
      } while (++z4 < IN_GROUP_SIZE / 4);
    }
  }

  store_tile(pos);
}
