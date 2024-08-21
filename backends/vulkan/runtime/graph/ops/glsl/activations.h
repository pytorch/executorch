/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

float hardswish(float x) {
  if (x <= -3) {
    return 0;
  } else if (x >= 3) {
    return x;
  } else {
    return x * (x + 3) / 6;
  }
}

vec4 hardswish(vec4 tex) {
  return vec4(
      hardswish(tex.x), hardswish(tex.y), hardswish(tex.z), hardswish(tex.z));
}

float hardshrink(float x, float lambda, float neg_lambda) {
  return x * (float(x > lambda) + float(x < neg_lambda));
}

vec4 hardshrink(vec4 tex, float lambda, float neg_lambda) {
  return tex *
      (vec4(greaterThan(tex, vec4(lambda))) +
       vec4(lessThan(tex, vec4(neg_lambda))));
}
