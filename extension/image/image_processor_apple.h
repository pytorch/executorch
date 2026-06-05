/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Apple-specific ImageProcessor entry point. Available only on Apple
// platforms; used by the Objective-C / Swift bindings to process a
// CVPixelBuffer directly. The normalization/conversion machinery this
// builds on is file-local to image_processor_apple.cpp and is intentionally
// not exposed here.

#pragma once

#ifdef __APPLE__

#include <CoreVideo/CoreVideo.h>

#include <executorch/extension/image/image_processor.h>

namespace executorch {
namespace extension {
namespace image {

/// Process a CVPixelBuffer directly into a normalized float tensor.
///
/// Apple-only entry point that avoids the GPU→CPU→GPU round trip that the
/// generic `process(raw_bytes)` path incurs for IOSurface-backed pixel
/// buffers. When the input qualifies for the GPU path (source pixels >=
/// config.gpu_min_input_pixels), wraps the CVPixelBuffer's IOSurface as a
/// CIImage (zero-copy), runs resize on GPU, reads back to CPU once at the
/// post-resize target dims, and applies vDSP-based normalization. On GPU
/// failure or for CPU-bound inputs, falls
/// back to a CPU pipeline that locks the pixel buffer's base address and
/// dispatches to `process()` / `process_yuv()` based on the pixel format.
///
/// Supported pixel formats: BGRA (32BGRA), RGBA (32RGBA), 8-bit NV12
/// (420YpCbCr8BiPlanar*), and 10-bit P010 (420YpCbCr10BiPlanar*; narrowed
/// to 8-bit NV12 internally before processing). Other formats return
/// Error::InvalidArgument.
///
/// All scratch buffers used by both paths live on the processor's pImpl
/// (`gpu_resized` for the GPU readback, `cpu_proxy` for the GPU→CPU
/// fallback's separate force-CPU processor). Repeated calls on the
/// same processor reuse the same allocations.
///
/// @param orientation Orientation of the pixel-buffer contents. Currently
/// only `Orientation::UP` is supported: the buffer is treated as already
/// upright. The parameter reserves the slot for future orientation correction
/// and is forwarded to the underlying pipeline. Orientation cannot be derived
/// from a CVPixelBuffer, so the caller must supply an upright buffer (e.g. by
/// configuring the capture connection) until non-UP orientations are
/// supported.
runtime::Result<TensorPtr> process_pixelbuffer(
    const ImageProcessor& processor,
    CVPixelBufferRef pixelBuffer,
    Orientation orientation = Orientation::UP);

/// Reuse-friendly variant of process_pixelbuffer that writes into a
/// caller-owned tensor instead of allocating one per call. `out` must be a
/// contiguous Float tensor shaped [1, 3, target_height, target_width]; the
/// result is written into its storage and the same tensor can be reused across
/// frames. The returned result aliases `out`, so the caller must finish
/// consuming the previous result before the next call.
///
/// Supported pixel formats and orientation handling match process_pixelbuffer.
runtime::Error process_pixelbuffer_into(
    const ImageProcessor& processor,
    CVPixelBufferRef pixelBuffer,
    Orientation orientation,
    executorch::aten::Tensor& out);

} // namespace image
} // namespace extension
} // namespace executorch

#endif // __APPLE__
