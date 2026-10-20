// Copyright 2026 Arm Limited and/or its affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#version 450
#extension GL_ARM_tensors : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

layout(set=0, binding=0) uniform sampler2D image0;
layout(set=0, binding=1) uniform sampler2D image1;
layout(set=0, binding=2) readonly buffer PreviousFlow { uint previousFlow[]; };
layout(set=0, binding=3) uniform tensorARM<int8_t, 4> tailPhase;
layout(set=0, binding=4, rgba8_snorm) uniform writeonly image2D warped0;
layout(set=0, binding=5, rgba8_snorm) uniform writeonly image2D warped1;
layout(set=0, binding=6) uniform tensorARM<int8_t, 4> mask;
layout(local_size_x=8, local_size_y=8, local_size_z=1) in;

const int SIGMOID_TABLE[256] = int[](@SIGMOID_TABLE@);
@RESCALES@

vec2 uv(ivec2 p, vec2 size, vec2 offset) {
  vec2 baseGrid = vec2(p) * vec2(2.0) / (size - vec2(1.0)) - vec2(1.0);
  vec2 texel = (baseGrid + offset + vec2(1.0)) * vec2(0.5) * (size - vec2(1.0));
  return (texel + vec2(0.5)) / size;
}

vec4 quantizeSample(vec4 raw, float inputScale, int inputZeroPoint, float outputScale, int outputZeroPoint) {
  vec4 value = (raw - vec4(float(inputZeroPoint))) * inputScale;
  vec4 code = roundEven(value / outputScale) + vec4(float(outputZeroPoint));
  return clamp(code, vec4(-127.0), vec4(127.0)) / 127.0;
}

void main() {
  ivec2 p = ivec2(gl_GlobalInvocationID.xy);
  ivec2 size = textureSize(image0, 0);
  if (p.x >= size.x || p.y >= size.y) return;

  uint phase = uint((p.y & 1) * 2 + (p.x & 1));
  uint coords[4] = uint[](0u, uint(p.y >> 1), uint(p.x >> 1), phase * 5u);
  int8_t state[5];
  tensorReadARM(tailPhase, coords, state);
  uint previous = previousFlow[uint(p.y * size.x + p.x) * 8u + 6u];
  int flowCodes[4];
  for (uint c=0u; c<4u; ++c) {
    int code = int(previous << ((3u-c)*8u)) >> 24;
    flowCodes[c] = clamp(rescaleSum(rescalePrevious(code) + rescaleDelta(int(state[c]))), -128, 127);
  }
  vec2 flow0 = (vec2(flowCodes[0], flowCodes[1]) - vec2(@FLOW_ZERO_POINT@)) * @FLOW_SCALE@;
  vec2 flow1 = (vec2(flowCodes[2], flowCodes[3]) - vec2(@FLOW_ZERO_POINT@)) * @FLOW_SCALE@;
  imageStore(warped0, p, quantizeSample(texture(image0, uv(p, vec2(size), flow0)) * 127.0,
      @INPUT0_SCALE@, @INPUT0_ZERO_POINT@, @OUTPUT0_SCALE@, @OUTPUT0_ZERO_POINT@));
  imageStore(warped1, p, quantizeSample(texture(image1, uv(p, vec2(size), flow1)) * 127.0,
      @INPUT1_SCALE@, @INPUT1_ZERO_POINT@, @OUTPUT1_SCALE@, @OUTPUT1_ZERO_POINT@));
  uint maskCoords[4] = uint[](0u, 0u, uint(p.y), uint(p.x));
  tensorWriteARM(mask, maskCoords, int8_t(SIGMOID_TABLE[int(state[4]) + 128]));
}
