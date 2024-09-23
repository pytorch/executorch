/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

layout(std430) buffer;

${layout_declare_buffer(0, "w", "out_buff", DTYPE)}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int NITER = 1;

void main() {

  $for k in range(int(NREG)):
     float reg_data${k} = float(NITER) + ${k};

  int i = 0;
  for (; i < NITER; ++i) {
    reg_data0 *= reg_data${int(NREG)-1};
    $for k in range(1, int(NREG)):
      reg_data${k} *= reg_data${k-1};
  }
  i = i >> 31;

  $for k in range(int(NREG)):
    out_buff[${k} * i] = reg_data${k};
}
