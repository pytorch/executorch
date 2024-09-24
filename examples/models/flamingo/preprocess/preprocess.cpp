/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "executorch/examples/models/flamingo/preprocess/preprocess.h"

#include <algorithm>
#include <cassert>

std::vector<int> _get_factors(int n) {
  std::vector<int> factors;
  for (int i = 1; i <= n; i++) {
    if (n % i == 0) {
      factors.push_back(i);
    }
  }
  return factors;
}

std::vector<std::vector<int>> find_supported_resolutions(
    int max_num_tiles,
    int tile_size) {
  std::vector<std::vector<int>> supported_resolutions;
  for (int _tile_size = max_num_tiles; _tile_size > 0; _tile_size--) {
    auto factors = _get_factors(_tile_size);
    for (int i = 0; i < factors.size(); i++) {
      int height = factors[i];
      int width = _tile_size / factors[i];
      supported_resolutions.push_back({height * tile_size, width * tile_size});
    }
  }
  return supported_resolutions;
}

std::vector<int> get_canvas_best_fit(
    std::vector<int> image_size,
    std::vector<std::vector<int>> possible_resolutions,
    bool resize_to_max_canvas) {
  assert(image_size.size() == 2);
  int image_h = image_size[0];
  int image_w = image_size[1];

  float best_scale = -0.1;
  std::vector<int> best_resolution;
  int best_area = 0;

  for (int i = 0; i < possible_resolutions.size(); i++) {
    assert(possible_resolutions[i].size() == 2);
    float scale_h = possible_resolutions[i][0] / (float)image_h;
    float scale_w = possible_resolutions[i][1] / (float)image_w;

    // Get limiting side scaling -> no distortion
    float scale = scale_h < scale_w ? scale_h : scale_w;

    bool is_candidate = false;

    if (scale >= 1.0) {
      // Upscaling options.
      if (resize_to_max_canvas) {
        is_candidate = scale >= best_scale;
      } else {
        is_candidate = ((scale <= best_scale) || (best_resolution.size() == 0));
      }
    } else {
      // If no upscaling options, find the minimum downscaling (max scale for
      // scales < 1)
      is_candidate = ((scale >= best_scale) || (best_resolution.size() == 0));
    }

    // Select the best resolution.
    if (is_candidate) {
      // @lint-ignore CLANGTIDY facebook-hte-ParameterUncheckedArrayBounds
      int area = possible_resolutions[i][0] * possible_resolutions[i][1];
      if (scale == best_scale) {
        // If there are multiple resolutions, get the one with minimum area to
        // reduce padding.
        if (scale >= 1.0 && area < best_area) {
          best_resolution = possible_resolutions[i];
          best_area = area;
        }
      } else {
        best_resolution = possible_resolutions[i];
        best_scale = scale;
        best_area = area;
      }
    }
  }
  return best_resolution;
}

std::vector<int> get_inscribed_size(
    std::vector<int> image_size,
    std::vector<int> target_size,
    int max_size) {
  assert(image_size.size() == 2);
  assert(target_size.size() == 2);

  int target_height = target_size[0];
  int target_width = target_size[1];

  if (max_size > 0) {
    target_height = std::min(std::max(image_size[0], max_size), target_size[0]);
    target_width = std::min(std::max(image_size[1], max_size), target_size[1]);
  }

  int resize_height = std::min(
      (int)(image_size[0] * (target_width / (float)image_size[1])),
      target_height);
  int resize_width = std::min(
      (int)(image_size[1] * (target_height / (float)image_size[0])),
      target_width);

  return {resize_height, resize_width};
}
