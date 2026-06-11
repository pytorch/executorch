// Copyright 2026 Arm Limited and/or its affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#version 450
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) buffer In { float x[]; };
layout(set = 0, binding = 1) buffer OutputBuffer { float out_data[]; };
void main() {
  uint idx = gl_GlobalInvocationID.x;
  if (idx >= out_data.length()) {
    return;
  }
  out_data[idx] = x[idx];
}
