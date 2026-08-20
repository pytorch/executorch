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

void main() {
  const uint width = 9u;
  const uint height = 4u;
  ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
  if (gid.x >= int(width) || gid.y >= int(height)) {
    return;
  }

  uint xCoords[4] = uint[](0u, uint(gid.y), uint(gid.x), 0u);
  uint yCoords[4] = uint[](0u, uint(gid.y), uint(gid.x), 1u);
  float xVal[1];
  float yVal[1];
  tensorReadARM(grid, xCoords, xVal);
  tensorReadARM(grid, yCoords, yVal);

  uint plane_size = width * height;
  uint base = uint(gid.y) * width + uint(gid.x);
  out_data[base] = xVal[0];
  out_data[plane_size + base] = yVal[0];
}
