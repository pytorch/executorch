/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

float gelu_erf(float x) {
  // Abramowitz and Stegun 7.1.26: maximum absolute erf error is 1.5e-7.
  const float a = abs(x) * 0.7071067811865475;
  const float t = 1.0 / (1.0 + 0.3275911 * a);
  const float polynomial =
      (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t -
        0.284496736) *
           t +
       0.254829592) *
      t;
  const float erf = sign(x) * (1.0 - polynomial * exp(-a * a));
  return 0.5 * x * (1.0 + erf);
}

vec4 gelu_erf(vec4 tex) {
  return vec4(
      gelu_erf(tex.x), gelu_erf(tex.y), gelu_erf(tex.z), gelu_erf(tex.w));
}

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
      hardswish(tex.x), hardswish(tex.y), hardswish(tex.z), hardswish(tex.w));
}

float hardshrink(float x, float lambda, float neg_lambda) {
  return x * (float(x > lambda) + float(x < neg_lambda));
}

vec4 hardshrink(vec4 tex, float lambda, float neg_lambda) {
  return tex *
      (vec4(greaterThan(tex, vec4(lambda))) +
       vec4(lessThan(tex, vec4(neg_lambda))));
}

float hardsigmoid(float x) {
  return mix(float(x >= 0.0), x / 6 + 0.5, float(abs(x) <= 3.0));
}

vec4 hardsigmoid(vec4 tex) {
  return vec4(
      hardsigmoid(tex.x),
      hardsigmoid(tex.y),
      hardsigmoid(tex.z),
      hardsigmoid(tex.w));
}

float leaky_relu(float x, float negative_slope) {
  return x * (float(x > 0.0) + negative_slope * float(x <= 0.0));
}

vec4 leaky_relu(vec4 tex, float negative_slope) {
  return vec4(
      leaky_relu(tex.x, negative_slope),
      leaky_relu(tex.y, negative_slope),
      leaky_relu(tex.z, negative_slope),
      leaky_relu(tex.w, negative_slope));
}
