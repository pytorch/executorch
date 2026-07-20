/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Core Image GPU-accelerated helpers for ImageProcessor.
// Provides C-linkage functions callable from pure C++ code.

#import <CoreImage/CoreImage.h>
#import <CoreVideo/CoreVideo.h>

#include "image_processor_apple_gpu.h"

// Shared CIContext for GPU rendering. Created once per process via dispatch_once.
// CIContext is thread-safe for rendering operations; multiple threads can call
// render:toBitmap: concurrently without synchronization.
static CIContext* sharedCIContext() {
  static CIContext* ctx = nil;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    ctx = [CIContext contextWithOptions:@{
      kCIContextWorkingColorSpace : [NSNull null],
      kCIContextWorkingFormat : @(kCIFormatBGRA8),
      kCIContextCacheIntermediates : @NO,
      kCIContextUseSoftwareRenderer : @NO,
    }];
  });
  return ctx;
}

static CIImage* applyOrientation(CIImage* image, int32_t orientation) {
  if (orientation <= 1 || orientation > 8) {
    return image;
  }
  return [image imageByApplyingOrientation:orientation];
}

static CIImage* applyROI(
    CIImage* image,
    float roi_x,
    float roi_y,
    float roi_width,
    float roi_height) {
  if (roi_x == 0.0f && roi_y == 0.0f && roi_width == 1.0f &&
      roi_height == 1.0f) {
    return image;
  }
  CGRect extent = image.extent;
  // Core Image's coordinate origin is bottom-left (y increases upward), but
  // roi_y is specified top-down (matching the CPU pipeline and the raw pixel
  // buffer). Flip it so the crop selects the same region on both paths:
  // a top-down [roi_y, roi_y + roi_height] maps to a bottom-up origin of
  // (1 - roi_y - roi_height).
  CGRect crop = CGRectMake(
      extent.origin.x + roi_x * extent.size.width,
      extent.origin.y + (1.0f - roi_y - roi_height) * extent.size.height,
      roi_width * extent.size.width,
      roi_height * extent.size.height);
  CIImage* cropped = [image imageByCroppingToRect:crop];
  // Rebase the cropped region to the coordinate-space origin. applyResize
  // scales about (0,0) and the render helpers use bounds {0,0,tw,th}, so a
  // non-zero ROI origin must be removed here — otherwise the content ends up
  // offset by crop.origin * scale and render samples the wrong (largely empty)
  // region. The full-image case returns early above, so this extra transform
  // only runs for actual sub-image ROIs.
  return [cropped
      imageByApplyingTransform:CGAffineTransformMakeTranslation(
                                   -crop.origin.x, -crop.origin.y)];
}

static CIImage* applyResize(
    CIImage* image,
    int32_t target_width,
    int32_t target_height) {
  CGRect extent = image.extent;
  CGFloat sx = (CGFloat)target_width / extent.size.width;
  CGFloat sy = (CGFloat)target_height / extent.size.height;
  return [image imageByApplyingTransform:CGAffineTransformMakeScale(sx, sy)];
}

static int renderToBGRA(
    CIImage* image,
    int32_t target_width,
    int32_t target_height,
    uint8_t* bgra_out,
    int32_t out_stride) {
  CIContext* ctx = sharedCIContext();
  // render:toBitmap: returns void and cannot report a rasterization failure,
  // so validate the inputs here. A failed CIFilter earlier in the pipeline
  // yields a nil or empty-extent image; rejecting it lets the caller fall back
  // to the CPU path.
  if (!ctx || !image || CGRectIsEmpty(image.extent)) {
    return -1;
  }
  CGRect bounds = CGRectMake(0, 0, target_width, target_height);
  [ctx render:image
      toBitmap:bgra_out
      rowBytes:out_stride
        bounds:bounds
        format:kCIFormatBGRA8
    colorSpace:nil];
  return 0;
}

int ci_process_to_bgra(
    const uint8_t* pixel_in,
    int32_t width,
    int32_t height,
    int32_t stride,
    CIPixelFormatValue pixel_format,
    int32_t orientation,
    float roi_x,
    float roi_y,
    float roi_width,
    float roi_height,
    int32_t target_width,
    int32_t target_height,
    uint8_t* bgra_out,
    int32_t out_stride) {
  if (!pixel_in || !bgra_out || width <= 0 || height <= 0 ||
      target_width <= 0 || target_height <= 0) {
    return -1;
  }
  @autoreleasepool {
    NSData* data = [NSData dataWithBytesNoCopy:(void*)pixel_in
                                        length:(NSUInteger)((size_t)stride * (size_t)height)
                                  freeWhenDone:NO];
    // Map the C-ABI format value to the real CIFormat. kCIFormat* are runtime
    // globals (not compile-time constants), so passing the raw enum value as a
    // CIFormat is unsafe. A mismatch yields a misinterpreted (black) image.
    CIFormat ci_format;
    switch (pixel_format) {
      case CI_PIXEL_FORMAT_BGRA8:
        ci_format = kCIFormatBGRA8;
        break;
      case CI_PIXEL_FORMAT_RGBA8:
        ci_format = kCIFormatRGBA8;
        break;
      default:
        return -1; // Unknown format; caller falls back to the CPU path.
    }
    CIImage* image = [CIImage
        imageWithBitmapData:data
                bytesPerRow:stride
                       size:CGSizeMake(width, height)
                     format:ci_format
                 colorSpace:nil];
    if (!image) {
      return -1;
    }
    image = applyOrientation(image, orientation);
    image = applyROI(image, roi_x, roi_y, roi_width, roi_height);
    image = applyResize(image, target_width, target_height);
    return renderToBGRA(image, target_width, target_height, bgra_out, out_stride);
  }
}

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
    int32_t out_stride) {
  if (!y_plane || !uv_plane || !bgra_out || width <= 0 || height <= 0 ||
      target_width <= 0 || target_height <= 0) {
    return -1;
  }
  @autoreleasepool {
    // Create a CVPixelBuffer wrapping the Y and UV planes. Chroma is expected in
    // NV12 (Cb,Cr) order; callers with NV21 input swap the chroma beforehand,
    // since CoreVideo has no native NV21 pixel format.
    //
    // Memory safety: CVPixelBufferCreateWithPlanarBytes wraps the input planes
    // without copying. The planes must remain valid until rendering completes.
    // This is guaranteed here because render completes synchronously within
    // this @autoreleasepool before the function returns.
    const OSType cv_format = (yuv_range != 0)
        ? kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        : kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange;
    CVPixelBufferRef pixelBuffer = NULL;

    const int32_t chroma_w = (width + 1) / 2;
    const int32_t chroma_h = (height + 1) / 2;

    void* planeBaseAddresses[2] = {
        (void*)y_plane, (void*)uv_plane};
    size_t planeWidths[2] = {
        (size_t)width, (size_t)chroma_w};
    size_t planeHeights[2] = {
        (size_t)height, (size_t)chroma_h};
    size_t planeBytesPerRow[2] = {
        (size_t)y_stride, (size_t)uv_stride};

    CVReturn status = CVPixelBufferCreateWithPlanarBytes(
        kCFAllocatorDefault,
        width,
        height,
        cv_format,
        NULL, // dataPtr
        0,    // dataSize
        2,    // numberOfPlanes
        planeBaseAddresses,
        planeWidths,
        planeHeights,
        planeBytesPerRow,
        NULL, // releaseCallback
        NULL, // releaseRefCon
        NULL, // pixelBufferAttributes
        &pixelBuffer);

    if (status != kCVReturnSuccess || !pixelBuffer) {
      return -1;
    }

    // imageWithCVPixelBuffer: retains the pixel buffer, so releasing our
    // reference here is safe: the CIImage keeps the buffer (and the caller-owned
    // planes it wraps without copying) alive through the synchronous render.
    CIImage* image = [CIImage imageWithCVPixelBuffer:pixelBuffer];
    CVPixelBufferRelease(pixelBuffer);

    if (!image) {
      return -1;
    }

    image = applyOrientation(image, orientation);
    image = applyROI(image, roi_x, roi_y, roi_width, roi_height);
    image = applyResize(image, target_width, target_height);
    return renderToBGRA(image, target_width, target_height, bgra_out, out_stride);
  }
}

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
    int32_t out_stride) {
  if (!pixelBuffer || !bgra_out || target_width <= 0 || target_height <= 0) {
    return -1;
  }
  @autoreleasepool {
    // Zero-copy: CIImage wraps the CVPixelBuffer's IOSurface directly. Renders
    // to 8-bit BGRA (4 B/px) rather than RGBAf (16 B/px) to cut readback
    // bandwidth ~4x; the uint8->float conversion happens during normalization.
    CIImage* image = [CIImage imageWithCVPixelBuffer:pixelBuffer];
    if (!image) {
      return -1;
    }
    image = applyOrientation(image, orientation);
    image = applyROI(image, roi_x, roi_y, roi_width, roi_height);
    image = applyResize(image, target_width, target_height);
    return renderToBGRA(image, target_width, target_height, bgra_out, out_stride);
  }
}
