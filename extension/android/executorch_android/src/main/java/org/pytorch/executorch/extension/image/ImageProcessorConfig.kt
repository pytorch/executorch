/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch.extension.image

import org.pytorch.executorch.annotations.Experimental

/** How the source image is fitted to the target dimensions. */
@Experimental
enum class ResizeMode {
  /** Scale to the target dimensions directly, ignoring aspect ratio. */
  STRETCH,
  /** Scale to fit inside the target dimensions, then pad with `padValue`. */
  LETTERBOX,
}

/** Where letterboxed content sits within the padded canvas. */
@Experimental
enum class LetterboxAnchor {
  CENTER,
  TOP_LEFT,
}

/**
 * EXIF orientation of the source image. The pipeline rotates the content upright before resizing.
 * Only the four rotation codes are supported (no mirrored variants).
 *
 * @property exifCode The EXIF orientation code this entry represents.
 */
@Experimental
enum class ImageOrientation(val exifCode: Int) {
  UP(1),
  DOWN(3),
  RIGHT(6),
  LEFT(8),
}

/** Chroma layout of semi-planar YUV input. */
@Experimental
enum class YuvFormat {
  NV12,
  NV21,
}

/**
 * Quantization range of YUV samples. VIDEO is studio/limited range (Y in [16, 235], chroma in
 * [16, 240]); FULL spans [0, 255]. Decoding with the wrong range shifts contrast and color.
 */
@Experimental
enum class YuvRange {
  VIDEO,
  FULL,
}

/**
 * Per-channel RGB normalization, applied as `(pixel * scaleFactor - mean[c]) /
 * standardDeviation[c]`.
 *
 * @property scaleFactor Scale applied to the raw 0-255 sample before mean subtraction.
 * @property mean Per-channel mean, exactly 3 entries (R, G, B).
 * @property standardDeviation Per-channel standard deviation, exactly 3 nonzero entries (R, G, B).
 */
@Experimental
class Normalization(
    val scaleFactor: Float,
    mean: FloatArray,
    standardDeviation: FloatArray,
) {
  val mean: FloatArray = mean.copyOf()
  val standardDeviation: FloatArray = standardDeviation.copyOf()

  init {
    require(mean.size == CHANNELS) { "mean must have $CHANNELS entries, got ${mean.size}" }
    require(standardDeviation.size == CHANNELS) {
      "standardDeviation must have $CHANNELS entries, got ${standardDeviation.size}"
    }
    require(standardDeviation.all { it != 0.0f }) { "standardDeviation entries must be nonzero" }
  }

  companion object {
    private const val CHANNELS = 3

    /** Maps 0-255 samples to [0, 1] with no mean subtraction. */
    @JvmStatic
    fun zeroToOne(): Normalization =
        Normalization(1.0f / 255.0f, floatArrayOf(0.0f, 0.0f, 0.0f), floatArrayOf(1.0f, 1.0f, 1.0f))

    /** The standard ImageNet mean and standard deviation over [0, 1] samples. */
    @JvmStatic
    fun imagenet(): Normalization =
        Normalization(
            1.0f / 255.0f,
            floatArrayOf(0.485f, 0.456f, 0.406f),
            floatArrayOf(0.229f, 0.224f, 0.225f),
        )
  }
}

/**
 * Configuration for [ImageProcessor].
 *
 * Warning: These APIs are experimental and subject to change without notice
 *
 * @property targetWidth Width of the produced tensor, in pixels.
 * @property targetHeight Height of the produced tensor, in pixels.
 * @property resizeMode How the source is fitted to the target dimensions.
 * @property letterboxAnchor Where letterboxed content sits; ignored for [ResizeMode.STRETCH].
 * @property padValue Value written to letterbox padding, in normalized output units.
 * @property normalization Per-channel normalization applied to the output.
 */
@Experimental
data class ImageProcessorConfig(
    val targetWidth: Int = 224,
    val targetHeight: Int = 224,
    val resizeMode: ResizeMode = ResizeMode.STRETCH,
    val letterboxAnchor: LetterboxAnchor = LetterboxAnchor.CENTER,
    val padValue: Float = 0.0f,
    val normalization: Normalization = Normalization.zeroToOne(),
) {
  init {
    require(targetWidth > 0) { "targetWidth must be positive" }
    require(targetHeight > 0) { "targetHeight must be positive" }
  }

  /** Builder class for ImageProcessorConfig for Java interoperability. */
  class Builder {
    private var targetWidth: Int = 224
    private var targetHeight: Int = 224
    private var resizeMode: ResizeMode = ResizeMode.STRETCH
    private var letterboxAnchor: LetterboxAnchor = LetterboxAnchor.CENTER
    private var padValue: Float = 0.0f
    private var normalization: Normalization = Normalization.zeroToOne()

    fun setTargetSize(width: Int, height: Int) = apply {
      require(width > 0 && height > 0) { "Target dimensions must be positive" }
      this.targetWidth = width
      this.targetHeight = height
    }

    fun setResizeMode(resizeMode: ResizeMode) = apply { this.resizeMode = resizeMode }

    fun setLetterboxAnchor(letterboxAnchor: LetterboxAnchor) = apply {
      this.letterboxAnchor = letterboxAnchor
    }

    fun setPadValue(padValue: Float) = apply { this.padValue = padValue }

    fun setNormalization(normalization: Normalization) = apply {
      this.normalization = normalization
    }

    fun build() =
        ImageProcessorConfig(
            targetWidth = targetWidth,
            targetHeight = targetHeight,
            resizeMode = resizeMode,
            letterboxAnchor = letterboxAnchor,
            padValue = padValue,
            normalization = normalization,
        )
  }
}

/**
 * Per-side letterbox padding in pixels.
 *
 * @property x The left/right pad of the resized content.
 * @property y The top/bottom pad of the resized content.
 */
@Experimental data class LetterboxPadding(val x: Int, val y: Int)
