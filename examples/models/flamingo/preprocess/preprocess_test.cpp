/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/models/flamingo/preprocess/preprocess.h>
#include <gtest/gtest.h>

using namespace ::testing;

// Mirror the torchtune python testing:
// https://github.com/pytorch/torchtune/tree/main/tests/torchtune/modules/transforms

void test_find_supported_resolutions(
    int max_num_tiles,
    int tile_size,
    std::vector<std::vector<int>> expected_resolutions) {
  std::vector<std::vector<int>> resolutions =
      find_supported_resolutions(max_num_tiles, tile_size);

  EXPECT_EQ(resolutions.size(), expected_resolutions.size());

  for (int i = 0; i < resolutions.size(); i++) {
    EXPECT_EQ(resolutions[i].size(), expected_resolutions[i].size());
    EXPECT_EQ(resolutions[i][0], expected_resolutions[i][0]); // height
    EXPECT_EQ(resolutions[i][1], expected_resolutions[i][1]); // width
  }
}

TEST(PreprocessTest, TestFindSupportedResolution) {
  test_find_supported_resolutions(1, 224, {{224, 224}});
  test_find_supported_resolutions(2, 100, {{100, 200}, {200, 100}, {100, 100}});
  test_find_supported_resolutions(
      3, 50, {{50, 150}, {150, 50}, {50, 100}, {100, 50}, {50, 50}});
  test_find_supported_resolutions(
      4,
      300,
      {
          {300, 1200},
          {600, 600},
          {1200, 300},
          {300, 900},
          {900, 300},
          {300, 600},
          {600, 300},
          {300, 300},
      });
}

void test_get_canvas_best_fit(
    std::vector<int> image_size,
    std::vector<std::vector<int>> possible_resolutions,
    bool resize_to_max_canvas,
    std::vector<int> expected_best_resolution) {
  std::vector<int> best_resolution = get_canvas_best_fit(
      image_size, possible_resolutions, resize_to_max_canvas);
  EXPECT_EQ(best_resolution[0], expected_best_resolution[0]); // height
  EXPECT_EQ(best_resolution[1], expected_best_resolution[1]); // width
}

TEST(PreprocessTest, TestGetCanvasBestFit_200x300_F) {
  std::vector<std::vector<int>> possible_resolutions = {
      {224, 896},
      {448, 448},
      {224, 224},
      {896, 224},
      {224, 672},
      {672, 224},
      {224, 448},
      {448, 224},
  };
  test_get_canvas_best_fit(
      {200, 300},
      possible_resolutions,
      false, // resize_to_max_canvas
      {224, 448});

  test_get_canvas_best_fit(
      {200, 500},
      possible_resolutions,
      true, // resize_to_max_canvas
      {224, 672});
  test_get_canvas_best_fit(
      {200, 200},
      possible_resolutions,
      false, // resize_to_max_canvas
      {224, 224});
  test_get_canvas_best_fit(
      {200, 100},
      possible_resolutions,
      true, // resize_to_max_canvas
      {448, 224});
}

void test_get_inscribed_size(
    std::vector<int> image_size,
    std::vector<int> target_size,
    int max_size,
    std::vector<int> expected_target_size) {
  std::vector<int> result =
      get_inscribed_size(image_size, target_size, max_size);
  EXPECT_EQ(result[0], expected_target_size[0]); // height
  EXPECT_EQ(result[1], expected_target_size[1]); // width
}
TEST(PreprocessTest, GetInscribedSize) {
  test_get_inscribed_size({200, 100}, {1000, 1200}, 600, {600, 300});
  test_get_inscribed_size({2000, 200}, {1000, 1200}, 2000, {1000, 100});
  test_get_inscribed_size({400, 200}, {1000, 1200}, -1, {1000, 500});
  test_get_inscribed_size({1000, 500}, {400, 300}, -1, {400, 200});
}
