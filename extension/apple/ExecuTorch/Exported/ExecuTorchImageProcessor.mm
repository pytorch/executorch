/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ExecuTorchImageProcessor.h"

#import "ExecuTorchError.h"

#import <executorch/extension/image/image_processor.h>
#import <executorch/extension/image/image_processor_apple.h>
#import <executorch/extension/tensor/tensor_ptr.h>

#include <optional>

using executorch::extension::TensorPtr;
using executorch::extension::image::ImageProcessor;
using executorch::extension::image::ImageProcessorConfig;
using executorch::extension::image::LetterboxAnchor;
using executorch::extension::image::Normalization;
using executorch::extension::image::Orientation;
using executorch::extension::image::process_pixelbuffer;
using executorch::extension::image::process_pixelbuffer_into;
using executorch::extension::image::ResizeMode;

// Verify enum value parity between ObjC and C++ at compile time
static_assert((int)ExecuTorchImageResizeModeStretch == (int)ResizeMode::STRETCH, "ExecuTorchImageResizeModeStretch must match ResizeMode::STRETCH");
static_assert((int)ExecuTorchImageResizeModeLetterbox == (int)ResizeMode::LETTERBOX, "ExecuTorchImageResizeModeLetterbox must match ResizeMode::LETTERBOX");
static_assert((int)ExecuTorchImageLetterboxAnchorCenter == (int)LetterboxAnchor::CENTER, "ExecuTorchImageLetterboxAnchorCenter must match LetterboxAnchor::CENTER");
static_assert((int)ExecuTorchImageLetterboxAnchorTopLeft == (int)LetterboxAnchor::TOP_LEFT, "ExecuTorchImageLetterboxAnchorTopLeft must match LetterboxAnchor::TOP_LEFT");
static_assert((int)ExecuTorchImageOrientationUp == (int)Orientation::UP, "ExecuTorchImageOrientationUp must match Orientation::UP");
static_assert((int)ExecuTorchImageOrientationDown == (int)Orientation::DOWN, "ExecuTorchImageOrientationDown must match Orientation::DOWN");
static_assert((int)ExecuTorchImageOrientationRight == (int)Orientation::RIGHT, "ExecuTorchImageOrientationRight must match Orientation::RIGHT");
static_assert((int)ExecuTorchImageOrientationLeft == (int)Orientation::LEFT, "ExecuTorchImageOrientationLeft must match Orientation::LEFT");

// MARK: - Private interfaces

@interface ExecuTorchImageNormalization ()
- (const Normalization &)nativeNormalization;
@end

@interface ExecuTorchImageProcessorConfig ()
- (ImageProcessorConfig)nativeConfig;
@end

static ExecuTorchTensor *tensorFromResult(
    executorch::runtime::Result<TensorPtr> &result,
    NSError **error) {
  if (!result.ok()) {
    if (error) {
      *error = ExecuTorchErrorWithCode((ExecuTorchErrorCode)result.error());
    }
    return nil;
  }
  auto tensorPtr = std::move(result.get());
  // initWithNativeInstance moves out of tensorPtr, leaving it in a moved-from state.
  return [[ExecuTorchTensor alloc] initWithNativeInstance:&tensorPtr];
}

// MARK: - ExecuTorchImageNormalization

@implementation ExecuTorchImageNormalization {
  Normalization _norm;
}

- (instancetype)initWithNormalization:(Normalization)norm {
  if (self = [super init]) {
    _norm = norm;
  }
  return self;
}

+ (instancetype)zeroToOne {
  static ExecuTorchImageNormalization *instance = nil;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    instance = [[self alloc] initWithNormalization:Normalization::zeroToOne()];
  });
  return instance;
}

+ (instancetype)imagenet {
  static ExecuTorchImageNormalization *instance = nil;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    instance = [[self alloc] initWithNormalization:Normalization::imagenet()];
  });
  return instance;
}

- (instancetype)initWithScaleFactor:(float)scaleFactor
                               mean:(NSArray<NSNumber *> *)mean
                  standardDeviation:(NSArray<NSNumber *> *)standardDeviation {
  NSParameterAssert(mean.count == (NSUInteger)ImageProcessorConfig::kOutputChannels);
  NSParameterAssert(standardDeviation.count == (NSUInteger)ImageProcessorConfig::kOutputChannels);
  Normalization norm;
  norm.scale_factor = scaleFactor;
  for (NSUInteger i = 0; i < (NSUInteger)ImageProcessorConfig::kOutputChannels; ++i) {
    norm.mean[i] = mean[i].floatValue;
    norm.std_dev[i] = standardDeviation[i].floatValue;
  }
  // Reserved 4th (alpha) slot: identity so it stays divide-safe if a future
  // path ever reads it (see Normalization in image_processor_config.h).
  norm.mean[ImageProcessorConfig::kOutputChannels] = 0.0f;
  norm.std_dev[ImageProcessorConfig::kOutputChannels] = 1.0f;
  return [self initWithNormalization:norm];
}

- (const Normalization &)nativeNormalization {
  return _norm;
}

@end

// MARK: - ExecuTorchImageProcessorConfig

@implementation ExecuTorchImageProcessorConfig

- (instancetype)initWithTargetWidth:(NSInteger)targetWidth
                       targetHeight:(NSInteger)targetHeight
                          resizeMode:(ExecuTorchImageResizeMode)resizeMode
                     letterboxAnchor:(ExecuTorchImageLetterboxAnchor)letterboxAnchor
                            padValue:(float)padValue
                       normalization:(ExecuTorchImageNormalization *)normalization
                   gpuMinInputPixels:(NSInteger)gpuMinInputPixels {
  if (self = [super init]) {
    _targetWidth = targetWidth;
    _targetHeight = targetHeight;
    _resizeMode = resizeMode;
    _letterboxAnchor = letterboxAnchor;
    _padValue = padValue;
    _normalization = normalization;
    _gpuMinInputPixels = gpuMinInputPixels;
  }
  return self;
}

- (ImageProcessorConfig)nativeConfig {
  ImageProcessorConfig config;
  config.target_width = static_cast<int32_t>(_targetWidth);
  config.target_height = static_cast<int32_t>(_targetHeight);
  config.resize_mode = static_cast<ResizeMode>(_resizeMode);
  config.letterbox_anchor = static_cast<LetterboxAnchor>(_letterboxAnchor);
  config.pad_value = _padValue;
  config.normalization = [_normalization nativeNormalization];
  config.gpu_min_input_pixels = static_cast<int64_t>(_gpuMinInputPixels);
  return config;
}

+ (NSInteger)defaultGpuMinInputPixels {
  return static_cast<NSInteger>(
      ImageProcessorConfig::kDefaultGpuMinInputPixels);
}

@end

// MARK: - ExecuTorchImageProcessor

@implementation ExecuTorchImageProcessor {
  std::optional<ImageProcessor> _processor;
}

- (instancetype)initWithConfig:(ExecuTorchImageProcessorConfig *)config {
  NSParameterAssert(config);
  if (self = [super init]) {
    // Copy the config to avoid external mutations affecting processor.config
    _config = [[ExecuTorchImageProcessorConfig alloc]
        initWithTargetWidth:config.targetWidth
               targetHeight:config.targetHeight
                  resizeMode:config.resizeMode
             letterboxAnchor:config.letterboxAnchor
                    padValue:config.padValue
               normalization:config.normalization
           gpuMinInputPixels:config.gpuMinInputPixels];
    _processor.emplace([_config nativeConfig]);
  }
  return self;
}

- (nullable ExecuTorchTensor *)processPixelBuffer:(_Nullable CVPixelBufferRef)pixelBuffer
                                            error:(NSError **)error {
  return [self processPixelBuffer:pixelBuffer
                      orientation:ExecuTorchImageOrientationUp
                            error:error];
}

- (BOOL)processPixelBuffer:(_Nullable CVPixelBufferRef)pixelBuffer
                intoTensor:(ExecuTorchTensor *)tensor
                     error:(NSError **)error {
  return [self processPixelBuffer:pixelBuffer
                      orientation:ExecuTorchImageOrientationUp
                       intoTensor:tensor
                            error:error];
}

- (nullable ExecuTorchTensor *)processPixelBuffer:(_Nullable CVPixelBufferRef)pixelBuffer
                                      orientation:(ExecuTorchImageOrientation)orientation
                                            error:(NSError **)error {
  if (!pixelBuffer) {
    if (error) {
      *error = ExecuTorchErrorWithCode(ExecuTorchErrorCodeInvalidArgument);
    }
    return nil;
  }
  auto result = process_pixelbuffer(
      *_processor, pixelBuffer, static_cast<Orientation>(orientation));
  return tensorFromResult(result, error);
}

- (BOOL)processPixelBuffer:(_Nullable CVPixelBufferRef)pixelBuffer
               orientation:(ExecuTorchImageOrientation)orientation
                intoTensor:(ExecuTorchTensor *)tensor
                     error:(NSError **)error {
  if (!pixelBuffer || !tensor) {
    if (error) {
      *error = ExecuTorchErrorWithCode(ExecuTorchErrorCodeInvalidArgument);
    }
    return NO;
  }
  auto* tensorPtr = reinterpret_cast<TensorPtr*>(tensor.nativeInstance);
  auto err = process_pixelbuffer_into(
      *_processor, pixelBuffer, static_cast<Orientation>(orientation),
      **tensorPtr);
  if (err != executorch::runtime::Error::Ok) {
    if (error) {
      *error = ExecuTorchErrorWithCode((ExecuTorchErrorCode)err);
    }
    return NO;
  }
  return YES;
}

- (ExecuTorchImageLetterboxPadding)computeLetterboxPaddingForInputWidth:(NSInteger)inputWidth
                                                                height:(NSInteger)inputHeight {
  return [self computeLetterboxPaddingForInputWidth:inputWidth
                                             height:inputHeight
                                        orientation:ExecuTorchImageOrientationUp];
}

- (ExecuTorchImageLetterboxPadding)computeLetterboxPaddingForInputWidth:(NSInteger)inputWidth
                                                                height:(NSInteger)inputHeight
                                     orientation:(ExecuTorchImageOrientation)orientation {
  const auto padding = _processor->compute_letterbox_padding(
      static_cast<int32_t>(inputWidth), static_cast<int32_t>(inputHeight),
      static_cast<Orientation>(orientation));
  return {padding.first, padding.second};
}

@end
