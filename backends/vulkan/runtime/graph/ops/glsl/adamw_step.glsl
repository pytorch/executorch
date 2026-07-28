/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define T ${buffer_scalar_type(DTYPE)}

${define_active_storage_type(STORAGE)}

${define_required_extensions(STORAGE, DTYPE)}

layout(std430) buffer;

${layout_declare_tensor(B, "rw", "t_param", DTYPE, STORAGE)}
${layout_declare_tensor(B, "rw", "t_m", DTYPE, STORAGE)}
${layout_declare_tensor(B, "rw", "t_v", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_grad", DTYPE, STORAGE)}

layout(push_constant) uniform restrict Block {
  int numel;
  float lr;
  float beta1;
  float beta2;
  float eps;
  float weight_decay;
  float bias_correction1;
  float bias_correction2;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const int i = int(gl_GlobalInvocationID.x);
  if (i >= numel) {
    return;
  }
  T g = t_grad[i];
  T p = t_param[i];
  p = p - lr * weight_decay * p;
  T m = beta1 * t_m[i] + (1.0 - beta1) * g;
  T v = beta2 * t_v[i] + (1.0 - beta2) * g * g;
  t_m[i] = m;
  t_v[i] = v;
  T mhat = m / bias_correction1;
  T vhat = v / bias_correction2;
  t_param[i] = p - lr * mhat / (sqrt(vhat) + eps);
}
