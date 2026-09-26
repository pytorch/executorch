/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>

#import "ExecuTorchTensor.h"

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(uint8_t, ExecuTorchImageResizeMode) {
  ExecuTorchImageResizeModeStretch,
  ExecuTorchImageResizeModeLetterbox,
} NS_SWIFT_NAME(ImageResizeMode);

typedef NS_ENUM(uint8_t, ExecuTorchImageLetterboxAnchor) {
  ExecuTorchImageLetterboxAnchorCenter,
  ExecuTorchImageLetterboxAnchorTopLeft,
} NS_SWIFT_NAME(ImageLetterboxAnchor);

/// Per-side letterbox padding in pixels: `x` is the left/right pad and `y` the
/// top/bottom pad of the resized content.
typedef struct ExecuTorchImageLetterboxPadding {
  NSInteger x;
  NSInteger y;
} ExecuTorchImageLetterboxPadding NS_SWIFT_NAME(ImageLetterboxPadding);

/// EXIF orientation of the source image. The pipeline rotates the content
/// upright before resizing. Only these rotation codes are supported.
typedef NS_ENUM(uint8_t, ExecuTorchImageOrientation) {
  ExecuTorchImageOrientationUp = 1,    // no rotation
  ExecuTorchImageOrientationDown = 3,  // 180 degrees
  ExecuTorchImageOrientationRight = 6, // 90 degrees clockwise
  ExecuTorchImageOrientationLeft = 8,  // 90 degrees counter-clockwise
} NS_SWIFT_NAME(ImageOrientation);
NS_SWIFT_NAME(ImageNormalization)
__attribute__((objc_subclassing_restricted))
@interface ExecuTorchImageNormalization : NSObject

+ (instancetype)zeroToOne;
+ (instancetype)imagenet;

/// Create a normalization with a custom scale factor and per-channel RGB mean
/// and standard deviation. `mean` and `standardDeviation` must each contain
/// exactly 3 elements (R, G, B). Normalization is applied per channel as
/// `(pixel * scaleFactor - mean[c]) / standardDeviation[c]`, so every
/// `standardDeviation` entry must be nonzero.
- (instancetype)initWithScaleFactor:(float)scaleFactor
                               mean:(NSArray<NSNumber *> *)mean
                  standardDeviation:(NSArray<NSNumber *> *)standardDeviation
    NS_REFINED_FOR_SWIFT;

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

@end

NS_SWIFT_NAME(ImageProcessorConfig)
__attribute__((objc_subclassing_restricted))
@interface ExecuTorchImageProcessorConfig : NSObject

@property(nonatomic, readonly) NSInteger targetWidth;
@property(nonatomic, readonly) NSInteger targetHeight;
@property(nonatomic, readonly) ExecuTorchImageResizeMode resizeMode;
@property(nonatomic, readonly) ExecuTorchImageLetterboxAnchor letterboxAnchor;
@property(nonatomic, readonly) float padValue;
@property(nonatomic, readonly) ExecuTorchImageNormalization *normalization;
// Minimum source pixel count (width * height) at which the GPU path may be
// used; smaller inputs run on the CPU. 0 forces GPU, NSIntegerMax forces CPU.
@property(nonatomic, readonly) NSInteger gpuMinInputPixels;

// Default value for gpuMinInputPixels (mirrors the C++ config default).
@property(class, nonatomic, readonly) NSInteger defaultGpuMinInputPixels;

- (instancetype)initWithTargetWidth:(NSInteger)targetWidth
                       targetHeight:(NSInteger)targetHeight
                          resizeMode:(ExecuTorchImageResizeMode)resizeMode
                     letterboxAnchor:(ExecuTorchImageLetterboxAnchor)letterboxAnchor
                            padValue:(float)padValue
                       normalization:(ExecuTorchImageNormalization *)normalization
                   gpuMinInputPixels:(NSInteger)gpuMinInputPixels NS_REFINED_FOR_SWIFT;

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

@end

/// Thread-safety: ExecuTorchImageProcessor is NOT thread-safe per instance.
/// Internal scratch buffers are mutated during processing. Use one instance
/// per concurrent caller. Different instances are safe to use concurrently.
NS_SWIFT_NAME(ImageProcessor)
__attribute__((objc_subclassing_restricted))
@interface ExecuTorchImageProcessor : NSObject

@property(nonatomic, readonly) ExecuTorchImageProcessorConfig *config;

- (instancetype)initWithConfig:(ExecuTorchImageProcessorConfig *)config;

/// Process a CVPixelBuffer into a normalized float tensor, treating the buffer
/// as already upright (orientation `up`). Use
/// processPixelBuffer:orientation:error: to specify a source orientation.
- (nullable ExecuTorchTensor *)processPixelBuffer:(_Nullable CVPixelBufferRef)pixelBuffer
                                            error:(NSError **)error;

/// Reuse-friendly variant of processPixelBuffer:error: that writes into a
/// caller-provided tensor; treats the buffer as already upright (orientation
/// `up`). See processPixelBuffer:orientation:intoTensor:error:.
- (BOOL)processPixelBuffer:(_Nullable CVPixelBufferRef)pixelBuffer
                intoTensor:(ExecuTorchTensor *)tensor
                     error:(NSError **)error;

/// Process a CVPixelBuffer into a normalized float tensor.
///
/// Auto-detects pixel format from the buffer's metadata. Supported
/// formats: BGRA, RGBA, 8-bit NV12, and 10-bit P010 (P010 is narrowed to NV12
/// internally). Other formats return an error.
///
/// `orientation` is the EXIF orientation of the buffer's contents; the pipeline
/// rotates it upright before resizing. It cannot be derived from a
/// CVPixelBuffer, so the caller supplies it (e.g. from capture metadata).
///
/// @param pixelBuffer The input pixel buffer.
/// @param orientation The source orientation.
/// @param error On failure, set to an NSError describing what went wrong.
/// @return An ExecuTorchTensor with shape [1, 3, H, W] (CHW), or nil on failure.
- (nullable ExecuTorchTensor *)processPixelBuffer:(_Nullable CVPixelBufferRef)pixelBuffer
                                      orientation:(ExecuTorchImageOrientation)orientation
                                            error:(NSError **)error;

/// Process a CVPixelBuffer into a caller-provided tensor, reusing its storage.
///
/// Avoids the per-call output allocation of processPixelBuffer:orientation:error:,
/// which matters for sustained video. `tensor` must be a Float tensor shaped
/// [1, 3, targetHeight, targetWidth]; its storage is overwritten and can be
/// reused across frames. The result aliases `tensor`, so the caller must
/// finish using the previous result before the next call.
///
/// @param pixelBuffer The input pixel buffer.
/// @param orientation The source orientation (see processPixelBuffer:orientation:error:).
/// @param tensor The output tensor to fill.
/// @param error On failure, set to an NSError describing what went wrong.
/// @return YES on success, NO on failure.
- (BOOL)processPixelBuffer:(_Nullable CVPixelBufferRef)pixelBuffer
               orientation:(ExecuTorchImageOrientation)orientation
                intoTensor:(ExecuTorchTensor *)tensor
                     error:(NSError **)error;

/// Letterbox padding (per side, in pixels) the processor applies for a source
/// of the given size: `x` is the left/right pad and `y` the top/bottom pad of
/// the resized content. Returns {0, 0} for the stretch resize mode or the
/// top-left anchor. Lets callers map the padded output back to the source
/// region without replicating the resize geometry.
///
/// Treats the source as already upright (orientation `up`). Use
/// computeLetterboxPaddingForInputWidth:height:orientation: for a rotated
/// source.
///
/// @param inputWidth The source pixel width.
/// @param inputHeight The source pixel height.
/// @return The {x, y} padding in pixels.
- (ExecuTorchImageLetterboxPadding)computeLetterboxPaddingForInputWidth:(NSInteger)inputWidth
                                                                height:(NSInteger)inputHeight
    NS_REFINED_FOR_SWIFT;

/// Letterbox padding (per side, in pixels) the processor applies for a source
/// of the given size and orientation. The source dimensions are oriented
/// (width/height swapped for the 90-degree rotations) before the padding is
/// computed, so the result matches the geometry that
/// processPixelBuffer:orientation:error: produces. Returns {0, 0} for the
/// stretch resize mode or the top-left anchor.
///
/// @param inputWidth The source pixel width.
/// @param inputHeight The source pixel height.
/// @param orientation The source orientation (see processPixelBuffer:orientation:error:).
/// @return The {x, y} padding in pixels.
- (ExecuTorchImageLetterboxPadding)computeLetterboxPaddingForInputWidth:(NSInteger)inputWidth
                                                                height:(NSInteger)inputHeight
                                     orientation:(ExecuTorchImageOrientation)orientation
    NS_REFINED_FOR_SWIFT;

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
