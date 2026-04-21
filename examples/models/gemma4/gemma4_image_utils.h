/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Gemma4 image patchification: converts a CHW float image to the
// pre-patchified tensor format expected by the exported vision_encoder.
//
// The HF Gemma4ImageProcessor produces:
//   pixel_values       (1, N_max=2520, patch_dim=768)  float32
//   pixel_position_ids (1, N_max=2520, 2)              int64   (-1,-1 = pad)
//
// This C++ implementation replicates that offline step so the runner can
// feed vision_encoder without a Python preprocessing step.

#pragma once

#include <cstdint>
#include <vector>

namespace gemma4 {

// Gemma4 vision constants matching HF Gemma4ImageProcessor config.
// The HF processor resizes images to kGridW*kPatchSize × kGridH*kPatchSize = 960×672,
// producing kMaxPatches = kGridW * kGridH = 2520 patches in a 60×42 grid.
// After 2×2×... spatial pooling with pooling_kernel_size=3:
//   n_soft_tokens = (kGridW/3) * (kGridH/3) = 20 * 14 = 280 visual soft tokens.
constexpr int kPatchSize  = 16;                     // pixels per patch edge
constexpr int kPatchDim   = 3 * 16 * 16;            // = 768
constexpr int kGridW      = 60;                     // patches per row
constexpr int kGridH      = 42;                     // patches per column
constexpr int kMaxPatches = kGridW * kGridH;        // = 2520
constexpr int kImageW     = kGridW * kPatchSize;    // = 960 px
constexpr int kImageH     = kGridH * kPatchSize;    // = 672 px

/**
 * Extract patches from a CHW float image and compute position IDs.
 *
 * The image is first normalised to [-1, 1] via 2*(v - 0.5), matching
 * Gemma4VisionPatchEmbedder.forward().
 *
 * Valid patches are laid out in row-major (y-major) order. Padding slots
 * at positions [n_valid, kMaxPatches) have pixel_values=0 and position_ids=(-1,-1).
 *
 * @param image_chw   Float image data in CHW layout (C=3, H, W), values in [0,1]
 * @param C           Number of channels (must be 3)
 * @param H           Image height (must be multiple of kPatchSize)
 * @param W           Image width  (must be multiple of kPatchSize)
 * @param pixel_values_out  Output buffer of size kMaxPatches * kPatchDim floats
 * @param pixel_pos_ids_out Output buffer of size kMaxPatches * 2 int64s
 */
void patchify(
    const float* image_chw,
    int C,
    int H,
    int W,
    std::vector<float>& pixel_values_out,
    std::vector<int64_t>& pixel_pos_ids_out);

} // namespace gemma4
