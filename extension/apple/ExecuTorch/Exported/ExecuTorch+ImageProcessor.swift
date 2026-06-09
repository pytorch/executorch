/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import CoreVideo

public extension ImageNormalization {
  /// Create a normalization with a custom scale factor and per-channel RGB mean
  /// and standard deviation. `mean` and `standardDeviation` must each contain
  /// exactly 3 elements (R, G, B); every `standardDeviation` entry must be
  /// nonzero. Applied per channel as
  /// `(pixel * scaleFactor - mean[c]) / standardDeviation[c]`.
  convenience init(scaleFactor: Float, mean: [Float], standardDeviation: [Float]) {
    precondition(mean.count == 3, "mean must have exactly 3 elements (R, G, B)")
    precondition(
      standardDeviation.count == 3,
      "standardDeviation must have exactly 3 elements (R, G, B)")
    self.init(
      __scaleFactor: scaleFactor,
      mean: mean.map { NSNumber(value: $0) },
      standardDeviation: standardDeviation.map { NSNumber(value: $0) })
  }
}

public extension ImageProcessorConfig {
  /// Source pixel count (width * height) sentinels for `gpuMinInputPixels`.
  static let alwaysGPU = 0
  static let alwaysCPU = Int.max

  /// Create an image processor config, specifying only the values that differ
  /// from the defaults.
  ///
  /// `gpuMinInputPixels` is the minimum source pixel count at which the GPU
  /// path may be used; smaller inputs run on the CPU. Use `.alwaysGPU` (0) or
  /// `.alwaysCPU` to force a path.
  convenience init(
    targetWidth: Int,
    targetHeight: Int,
    resizeMode: ImageResizeMode = .stretch,
    letterboxAnchor: ImageLetterboxAnchor = .center,
    padValue: Float = 0,
    normalization: ImageNormalization = .zeroToOne(),
    gpuMinInputPixels: Int = ImageProcessorConfig.defaultGpuMinInputPixels
  ) {
    self.init(
      __targetWidth: targetWidth,
      targetHeight: targetHeight,
      resizeMode: resizeMode,
      letterboxAnchor: letterboxAnchor,
      padValue: padValue,
      normalization: normalization,
      gpuMinInputPixels: gpuMinInputPixels)
  }
}

public extension ImageProcessor {
  /// Process a CVPixelBuffer into a normalized float tensor.
  ///
  /// Auto-detects pixel format from the buffer. Supported formats: BGRA,
  /// RGBA, 8-bit NV12, and 10-bit P010. Output is a `Tensor<Float>` with
  /// shape `[1, 3, target_height, target_width]`.
  ///
  /// The buffer is treated as already upright: orientation correction is not
  /// applied and cannot be derived from a CVPixelBuffer, so the caller is
  /// responsible for supplying an upright buffer.
  func process(_ pixelBuffer: CVPixelBuffer) throws -> Tensor<Float> {
    let anyTensor = try processPixelBuffer(pixelBuffer)
    return Tensor<Float>(anyTensor)
  }

  /// Process a CVPixelBuffer into a caller-provided tensor, reusing its storage.
  ///
  /// Avoids the per-call allocation of `process(_:)`, which matters for
  /// sustained video. `tensor` must be a `Tensor<Float>` with shape
  /// `[1, 3, target_height, target_width]`; its storage is overwritten and can
  /// be reused across frames. The contents are valid until the next call that
  /// writes into the same tensor.
  ///
  /// The buffer is treated as already upright (see `process(_:)`).
  func process(_ pixelBuffer: CVPixelBuffer, into tensor: Tensor<Float>) throws {
    try processPixelBuffer(pixelBuffer, into: tensor.anyTensor)
  }

  /// Letterbox padding (per side, in pixels) applied for a source of the given
  /// size: `x` is the left/right pad and `y` the top/bottom pad of the resized
  /// content. Returns `(0, 0)` for the stretch resize mode or the top-left
  /// anchor. Lets callers map the padded output back to the source region.
  func computeLetterboxPadding(inputWidth: Int, inputHeight: Int) -> (x: Int, y: Int) {
    let padding = __computeLetterboxPadding(forInputWidth: inputWidth, height: inputHeight)
    return (padding.x, padding.y)
  }
}
