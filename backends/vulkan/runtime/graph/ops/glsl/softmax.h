/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// The following two are helper functions to implement `softmax`
// `early_exit` is the global workgroup position-based condition for unnecessary
// invocations to exit.
ivec4 get_early_exit(ivec4 sizes, int in_dim, int compute_dim) {
  ivec4 early_exit = {
      sizes.x, // w
      sizes.y, // h
      divup4(sizes.z) * sizes.w, // divup4(c) * n
      0 // zero pad
  };
  if (in_dim == 4 && compute_dim == 1) {
    return early_exit;
  } else if (in_dim == 4 && compute_dim == 0) {
    early_exit[2] = divup4(sizes.z);
    return early_exit;
  } else {
    early_exit[in_dim - compute_dim - 1] = 1;
    return early_exit;
  }
}

// `input_dim_stride` is the stride to include elements along the softmax
// dimension calculation.
ivec4 get_input_dim_stride(int in_dim, int compute_dim, int in_channel) {
  ivec4 input_dim_stride = ivec4(0);
  if (in_dim == 4 && compute_dim == 1) {
    return input_dim_stride;
  } else if (in_dim == 4 && compute_dim == 0) {
    input_dim_stride[2] = divup4(in_channel);
    return input_dim_stride;
  } else {
    input_dim_stride[in_dim - compute_dim - 1] = 1;
    return input_dim_stride;
  }
}
