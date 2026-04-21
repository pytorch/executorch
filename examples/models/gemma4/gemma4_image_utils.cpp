/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gemma4_image_utils.h"

#include <algorithm>
#include <cstring>

namespace gemma4 {

void patchify(
    const float* image_chw,
    int C,
    int H,
    int W,
    std::vector<float>& pixel_values_out,
    std::vector<int64_t>& pixel_pos_ids_out) {

  // Image must be exactly kImageW × kImageH = 960 × 672 pixels (caller resizes).
  // We extract a kGridW × kGridH = 60 × 42 grid of kPatchSize×kPatchSize patches.
  pixel_values_out.assign(kMaxPatches * kPatchDim, 0.0f);
  pixel_pos_ids_out.assign(kMaxPatches * 2, 0LL);

  const int patches_w = std::min(W / kPatchSize, kGridW);
  const int patches_h = std::min(H / kPatchSize, kGridH);

  for (int py = 0; py < patches_h; ++py) {
    for (int px = 0; px < patches_w; ++px) {
      int patch_idx = py * kGridW + px;
      float* pv = pixel_values_out.data() + patch_idx * kPatchDim;
      int dim_offset = 0;

      // Extract patch in CHW order and normalize to [-1, 1] via 2*(v-0.5)
      for (int c = 0; c < C; ++c) {
        for (int ph = 0; ph < kPatchSize; ++ph) {
          for (int pw2 = 0; pw2 < kPatchSize; ++pw2) {
            int h = py * kPatchSize + ph;
            int w = px * kPatchSize + pw2;
            float raw = image_chw[c * H * W + h * W + w];  // [0,1] after rescale
            pv[dim_offset++] = 2.0f * (raw - 0.5f);        // → [-1,1]
          }
        }
      }

      // Position IDs: (x=column index, y=row index)
      pixel_pos_ids_out[patch_idx * 2 + 0] = static_cast<int64_t>(px);
      pixel_pos_ids_out[patch_idx * 2 + 1] = static_cast<int64_t>(py);
    }
  }
}

} // namespace gemma4
