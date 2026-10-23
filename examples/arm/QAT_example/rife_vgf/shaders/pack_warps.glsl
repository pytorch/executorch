// Copyright 2026 Arm Limited and/or its affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#version 450
#extension GL_ARM_tensors : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

layout(set=0, binding=0) uniform sampler2D image0;
layout(set=0, binding=1) uniform sampler2D image1;
layout(set=0, binding=2) uniform sampler2D image2;
layout(set=0, binding=3) uniform sampler2D image3;
layout(set=0, binding=4) uniform tensorARM<int8_t, 4> mask;
layout(set=0, binding=5) uniform tensorARM<int8_t, 4> features;
layout(set=0, binding=6) uniform tensorARM<int8_t, 4> flow;
layout(set=0, binding=7) writeonly buffer Output { uint output0[]; };
layout(local_size_x=8, local_size_y=8, local_size_z=1) in;

const float FLOW_SCALE = @FLOW_SCALE@;
const int FLOW_ZERO_POINT = @FLOW_ZERO_POINT@;
const float INPUT_SCALE = @INPUT_SCALE@;
const int INPUT_ZERO_POINT = @INPUT_ZERO_POINT@;
const float OUTPUT_SCALE = @OUTPUT_SCALE@;
const int OUTPUT_ZERO_POINT = @OUTPUT_ZERO_POINT@;

vec2 uv(ivec2 p, vec2 size, vec2 offset) {
  vec2 baseGrid = (vec2(p) * vec2(2.0) / (size - vec2(1.0))) - vec2(1.0);
  vec2 texel = (baseGrid + offset + vec2(1.0)) * vec2(0.5) * (size - vec2(1.0));
  return (texel + vec2(0.5)) / size;
}

ivec4 quantizeSample(vec4 raw) {
  vec4 value = (raw - vec4(float(INPUT_ZERO_POINT))) * INPUT_SCALE;
  vec4 code = roundEven(value / OUTPUT_SCALE) + vec4(float(OUTPUT_ZERO_POINT));
  return ivec4(clamp(code, vec4(-127.0), vec4(127.0)));
}

void main() {
  ivec2 p = ivec2(gl_GlobalInvocationID.xy);
  ivec2 size = textureSize(image0, 0);
  if (p.x >= size.x || p.y >= size.y) return;

  uint coords[4] = uint[](0u, uint(p.y), uint(p.x), 0u);
  int8_t flowCodes[4];
  tensorReadARM(flow, coords, flowCodes);
  vec2 flow0 = (vec2(flowCodes[0], flowCodes[1]) - vec2(FLOW_ZERO_POINT)) * FLOW_SCALE;
  vec2 flow1 = (vec2(flowCodes[2], flowCodes[3]) - vec2(FLOW_ZERO_POINT)) * FLOW_SCALE;
  ivec4 rgb0 = quantizeSample(texture(image0, uv(p, vec2(size), flow0)) * 127.0);
  ivec4 rgb1 = quantizeSample(texture(image1, uv(p, vec2(size), flow1)) * 127.0);
  ivec4 feat0 = quantizeSample(texture(image2, uv(p, vec2(size), flow0)) * 127.0);
  ivec4 feat1 = quantizeSample(texture(image3, uv(p, vec2(size), flow1)) * 127.0);

  int8_t values[32];
  for (uint c=0u; c<3u; ++c) {
    values[c] = int8_t(rgb0[c]);
    values[3u+c] = int8_t(rgb1[c]);
  }
  for (uint c=0u; c<4u; ++c) {
    values[6u+c] = int8_t(feat0[c]);
    values[10u+c] = int8_t(feat1[c]);
    values[24u+c] = flowCodes[c];
    values[28u+c] = int8_t(0);
  }
  values[14] = int8_t(@TIMESTEP@);
  int8_t maskCode[1];
  tensorReadARM(mask, coords, maskCode);
  values[15] = maskCode[0];
  int8_t featureCodes[8];
  tensorReadARM(features, coords, featureCodes);
  for (uint c=0u; c<8u; ++c) values[16u+c] = featureCodes[c];

  // Aligned word stores give each invocation exclusive ownership of its bytes.
  uint base = uint(p.y * size.x + p.x) * 8u;
  for (uint word=0u; word<8u; ++word) {
    uint packed = 0u;
    for (uint byte=0u; byte<4u; ++byte)
      packed |= (uint(int(values[word*4u+byte])) & 255u) << (byte*8u);
    output0[base+word] = packed;
  }
}
