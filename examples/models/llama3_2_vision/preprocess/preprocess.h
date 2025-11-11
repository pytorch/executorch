/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

// C++ implementation of the python functions in torchtune:
// https://github.com/pytorch/torchtune/tree/main/torchtune/modules/transforms/vision_utils

// Calculate all factors of a given number.
std::vector<int> _get_factors(int n);

// Computes all combinations of resolutions, multiple of tile_size,
// that contain up to max_num_tiles. Useful for when dividing an image into
// tiles. For example, if we want at most 2 tiles per image, then we can support
// the following resolutions: (1x1, 1x2, 2x1) * tile_size Returns a vector of
// tuples of (height, width).
std::vector<std::vector<int>> find_supported_resolutions(
    int max_num_tiles,
    int tile_size);

// Determines the best canvas possible from a list of possible resolutions to
// resize an image to, without distortion.
std::vector<int> get_canvas_best_fit(
    std::vector<int> image_size,
    std::vector<std::vector<int>> possible_resolutions,
    bool resize_to_max_canvas);

// Calculates the size of an image, if it was resized to be inscribed within the
// target_size. It is upscaled or downscaled such that one size is equal to the
// target_size, and the second size is less than or equal to the target_size.
std::vector<int> get_inscribed_size(
    std::vector<int> image_size,
    std::vector<int> canvas_size,
    int max_size);
