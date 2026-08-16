// Copyright 2026 Arm Limited and/or its affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#version 450
#extension GL_ARM_tensors : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

layout(set = 0, binding = 0) uniform sampler2D inputImage;
layout(set = 0, binding = 1) uniform tensorARM<int8_t, 4> grid;
layout(set = 0, binding = 2, rgba8_snorm) uniform writeonly image2D outImage;
layout(set = 0, binding = 3) readonly buffer GridScaleBuffer {
  float gridScale[];
};
layout(set = 0, binding = 4) readonly buffer GridZeroPointBuffer {
  int gridZeroPoint[];
};

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

vec2 readGridXY(ivec2 p) {
  uint xCoords[4] = uint[](0u, uint(p.y), uint(p.x), 0u);
  uint yCoords[4] = uint[](0u, uint(p.y), uint(p.x), 1u);
  int8_t xVal[1];
  int8_t yVal[1];
  tensorReadARM(grid, xCoords, xVal);
  tensorReadARM(grid, yCoords, yVal);
  return vec2(
      (float(xVal[0]) - float(gridZeroPoint[0])) * gridScale[0],
      (float(yVal[0]) - float(gridZeroPoint[0])) * gridScale[0]);
}

void main() {
  ivec2 outSize = imageSize(outImage);
  ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
  if (gid.x >= outSize.x || gid.y >= outSize.y) {
    return;
  }

  vec2 gridXY = readGridXY(gid);
  vec2 uv = (gridXY + vec2(1.0)) * 0.5;
  imageStore(outImage, gid, texture(inputImage, uv));
}
