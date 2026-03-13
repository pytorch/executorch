// Copyright 2026 Arm Limited and/or its affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#version 450
#extension GL_ARM_tensors : require
layout(set=0, binding=0) uniform sampler2D inputImage;
layout(set=0, binding=1) uniform tensorARM<float, 4> grid;
layout(set=0, binding=2, rgba32f) uniform writeonly image2D outImage;
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
vec2 readGridXY(ivec2 p) {
  uint xCoords[4] = uint[](0u, uint(p.y), uint(p.x), 0u);
  uint yCoords[4] = uint[](0u, uint(p.y), uint(p.x), 1u);
  float xVal[1];
  float yVal[1];
  tensorReadARM(grid, xCoords, xVal);
  tensorReadARM(grid, yCoords, yVal);
  return vec2(xVal[0], yVal[0]);
}
void main() {
  ivec2 outSize = imageSize(outImage);
  ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
  if (gid.x >= outSize.x || gid.y >= outSize.y) { return; }
  vec2 gridXY = readGridXY(gid);
  vec2 uv = (gridXY + vec2(1.0)) * 0.5;
  imageStore(outImage, gid, texture(inputImage, uv));
}
