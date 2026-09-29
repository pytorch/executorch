/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Internal header for Core Image GPU-accelerated helpers.
// Provides C-linkage functions so image_processor_apple.cpp can call them
// without becoming Objective-C++.

#pragma once

#ifdef __APPLE__

#include <CoreVideo/CVPixelBuffer.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// C-ABI tokens for the pixel formats the GPU raw-bytes path accepts. These are
// mapped to the real CIFormat values (kCIFormat*, which are runtime globals) in
// image_processor_apple_gpu.mm. The values here are private tokens and need not
// match kCIFormat*.
typedef enum {
  CI_PIXEL_FORMAT_BGRA8 = 14,
  CI_PIXEL_FORMAT_RGBA8 = 24,
} CIPixelFormatValue;

// Process interleaved pixel data through Core Image GPU pipeline:
// orient → ROI crop → resize → render to BGRA output at target size.
// Returns 0 on success, non-zero on failure.
int ci_process_to_bgra(
    const uint8_t* pixel_in,
    int32_t width,
    int32_t height,
    int32_t stride,
    CIPixelFormatValue pixel_format,
    int32_t orientation, // Orientation enum value (1-8)
    float roi_x,
    float roi_y,
    float roi_width,
    float roi_height,
    int32_t target_width,
    int32_t target_height,
    uint8_t* bgra_out,
    int32_t out_stride);

// Process NV12 YUV input through Core Image GPU pipeline:
// YUV→RGB + orient → ROI crop → resize → render to BGRA output.
// Returns 0 on success, non-zero on failure.
// Chroma must already be in NV12 (Cb,Cr) order; callers with NV21 input swap
// the chroma beforehand, since CoreVideo has no native NV21 pixel format.
// yuv_range: 0 = video range, 1 = full range
int ci_process_yuv_to_bgra(
    const uint8_t* y_plane,
    int32_t y_stride,
    const uint8_t* uv_plane,
    int32_t uv_stride,
    int32_t width,
    int32_t height,
    int32_t yuv_range,
    int32_t orientation,
    float roi_x,
    float roi_y,
    float roi_width,
    float roi_height,
    int32_t target_width,
    int32_t target_height,
    uint8_t* bgra_out,
    int32_t out_stride);

// Process a CVPixelBuffer directly through the Core Image GPU pipeline,
// rendering to 8-bit BGRA. Zero-copy for camera buffers. Renders 4 B/px
// instead of RGBAf's 16 B/px to cut GPU→CPU readback bandwidth ~4x; the
// uint8→float conversion is done by the normalize step. Returns 0 on success.
int ci_process_pixelbuffer_to_bgra(
    CVPixelBufferRef pixelBuffer,
    int32_t orientation,
    float roi_x,
    float roi_y,
    float roi_width,
    float roi_height,
    int32_t target_width,
    int32_t target_height,
    uint8_t* bgra_out,
    int32_t out_stride);

#ifdef __cplusplus
}
#endif

#endif // __APPLE__
