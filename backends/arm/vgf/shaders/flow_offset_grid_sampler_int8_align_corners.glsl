// Copyright 2026 Arm Limited and/or its affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#version 450
#extension GL_ARM_tensors : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

layout(set = 0, binding = 0) uniform sampler2D inputImage;
layout(set = 0, binding = 1) uniform tensorARM<int8_t, 4> flow;
layout(set = 0, binding = 2, rgba8_snorm) uniform writeonly image2D outImage;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

const float FLOW_SCALE = @FLOW_SCALE@;
const int FLOW_ZERO_POINT = @FLOW_ZERO_POINT@;
const float INPUT_SCALE = @INPUT_SCALE@;
const int INPUT_ZERO_POINT = @INPUT_ZERO_POINT@;
const float OUTPUT_SCALE = @OUTPUT_SCALE@;
const int OUTPUT_ZERO_POINT = @OUTPUT_ZERO_POINT@;
const uint FLOW_X_CHANNEL = @FLOW_X_CHANNEL@;
const uint FLOW_Y_CHANNEL = @FLOW_Y_CHANNEL@;

float readFlow(ivec2 p, uint channel) {
  uint coords[4] = uint[](0u, channel, uint(p.y), uint(p.x));
  int8_t value[1];
  tensorReadARM(flow, coords, value);
  return (float(value[0]) - float(FLOW_ZERO_POINT)) * FLOW_SCALE;
}

vec2 alignCornersUvFromFlowOffset(ivec2 p) {
  vec2 inputSize = vec2(textureSize(inputImage, 0));
  vec2 flowOffset = vec2(
      readFlow(p, FLOW_X_CHANNEL), readFlow(p, FLOW_Y_CHANNEL));
  vec2 baseGrid =
      (vec2(p) * vec2(2.0) / (inputSize - vec2(1.0))) - vec2(1.0);
  vec2 texel =
      (baseGrid + flowOffset + vec2(1.0)) * vec2(0.5) *
      (inputSize - vec2(1.0));
  return (texel + vec2(0.5)) / inputSize;
}

void main() {
  ivec2 outSize = imageSize(outImage);
  ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
  if (gid.x >= outSize.x || gid.y >= outSize.y) {
    return;
  }

  vec4 inputRaw =
      texture(inputImage, alignCornersUvFromFlowOffset(gid)) * 127.0;
  vec4 inputDequantized =
      (inputRaw - vec4(float(INPUT_ZERO_POINT))) * INPUT_SCALE;
  vec4 outputRaw =
      roundEven(inputDequantized / OUTPUT_SCALE) +
      vec4(float(OUTPUT_ZERO_POINT));
  outputRaw = clamp(outputRaw, vec4(-127.0), vec4(127.0));
  imageStore(outImage, gid, outputRaw / 127.0);
}
