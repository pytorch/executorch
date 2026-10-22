// Copyright 2026 Arm Limited and/or its affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#version 450
#extension GL_ARM_tensors : require

layout(set = 0, binding = 0) uniform sampler2D inputImage;
layout(set = 0, binding = 1) uniform tensorARM<float, 4> flow;
layout(set = 0, binding = 2, rgba8_snorm) uniform writeonly image2D outImage;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// This shader approximates downsampling: each output pixel samples the central
// 2x2 texels in the corresponding full-resolution scale x scale window.
// It is not an area/average-pool downsample over the whole window.
const int kScale = 4;
const int kOffset0 = 1;
const int kOffset1 = 2;

vec2 readFlowXY(ivec2 p) {
  uint xCoords[4] = uint[](0u, 0u, uint(p.y), uint(p.x));
  uint yCoords[4] = uint[](0u, 1u, uint(p.y), uint(p.x));
  float xVal[1];
  float yVal[1];
  tensorReadARM(flow, xCoords, xVal);
  tensorReadARM(flow, yCoords, yVal);
  return vec2(xVal[0], yVal[0]);
}

vec2 alignCornersUv(vec2 gridXY) {
  vec2 inputSize = vec2(textureSize(inputImage, 0));
  vec2 texel = (gridXY + vec2(1.0)) * vec2(0.5) * (inputSize - vec2(1.0));
  return (texel + vec2(0.5)) / inputSize;
}

vec2 baseGridXY(ivec2 p, ivec2 fullSize) {
  return vec2(
      (2.0 * float(p.x)) / float(fullSize.x - 1) - 1.0,
      (2.0 * float(p.y)) / float(fullSize.y - 1) - 1.0);
}

vec4 sampleWarped(ivec2 p, ivec2 fullSize) {
  vec2 flowXY = readFlowXY(p);
  vec2 gridXY = baseGridXY(p, fullSize) + flowXY;
  return texture(inputImage, alignCornersUv(gridXY));
}

void main() {
  ivec2 outSize = imageSize(outImage);
  ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
  if (gid.x >= outSize.x || gid.y >= outSize.y) {
    return;
  }

  ivec2 fullSize = textureSize(inputImage, 0);
  ivec2 base = gid * kScale;
  ivec2 p00 = base + ivec2(kOffset0, kOffset0);
  ivec2 p01 = base + ivec2(kOffset1, kOffset0);
  ivec2 p10 = base + ivec2(kOffset0, kOffset1);
  ivec2 p11 = base + ivec2(kOffset1, kOffset1);

  vec4 value = 0.25 * (
      sampleWarped(p00, fullSize) +
      sampleWarped(p01, fullSize) +
      sampleWarped(p10, fullSize) +
      sampleWarped(p11, fullSize));
  imageStore(outImage, gid, value);
}
