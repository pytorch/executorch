// Copyright 2026 Arm Limited and/or its affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#version 450
#extension GL_ARM_tensors : require

layout(set=0, binding=0) readonly buffer InputBuffer { float input_data[]; };
layout(set=0, binding=1) uniform tensorARM<float, 4> grid;
layout(set=0, binding=2) buffer OutputBuffer { float out_data[]; };
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

float readInput(uint c, int y, int x) {
  const int width = 8;
  const int height = 8;
  if (x < 0 || x >= width || y < 0 || y >= height) {
    return 0.0;
  }
  uint idx = (c * uint(height) + uint(y)) * uint(width) + uint(x);
  return input_data[idx];
}

void main() {
  const int in_width = 8;
  const int in_height = 8;
  const int out_width = 9;
  const int out_height = 4;

  ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
  if (gid.x >= out_width || gid.y >= out_height) {
    return;
  }

  vec2 gridXY = readGridXY(gid);
  float ix = ((gridXY.x + 1.0) * float(in_width) - 1.0) * 0.5;
  float iy = ((gridXY.y + 1.0) * float(in_height) - 1.0) * 0.5;

  int x0 = int(floor(ix));
  int y0 = int(floor(iy));
  int x1 = x0 + 1;
  int y1 = y0 + 1;

  float wx1 = ix - float(x0);
  float wy1 = iy - float(y0);
  float wx0 = 1.0 - wx1;
  float wy0 = 1.0 - wy1;

  for (uint c = 0u; c < 4u; ++c) {
    float v00 = readInput(c, y0, x0);
    float v01 = readInput(c, y0, x1);
    float v10 = readInput(c, y1, x0);
    float v11 = readInput(c, y1, x1);
    float sample_val =
        v00 * wx0 * wy0 +
        v01 * wx1 * wy0 +
        v10 * wx0 * wy1 +
        v11 * wx1 * wy1;
    uint out_idx =
        (c * uint(out_height) + uint(gid.y)) * uint(out_width) + uint(gid.x);
    out_data[out_idx] = sample_val;
  }
}
