/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
package org.pytorch.executorch

import android.graphics.Bitmap
import android.util.Log
import java.nio.FloatBuffer

/**
 * Contains utility functions for [Tensor] creation from [android.graphics.Bitmap] or
 * [android.media.Image] source.
 */
object TensorImageUtils {
  @JvmField var TORCHVISION_NORM_MEAN_RGB: FloatArray = floatArrayOf(0.485f, 0.456f, 0.406f)

  @JvmField var TORCHVISION_NORM_STD_RGB: FloatArray = floatArrayOf(0.229f, 0.224f, 0.225f)

  /**
   * Creates new [Tensor] from full [android.graphics.Bitmap], normalized with specified in
   * parameters mean and std.
   *
   * @param normMeanRGB means for RGB channels normalization, length must equal 3, RGB order
   * @param normStdRGB standard deviation for RGB channels normalization, length must equal 3, RGB
   *   order
   */
  @JvmStatic
  fun bitmapToFloat32Tensor(
      bitmap: Bitmap,
      normMeanRGB: FloatArray,
      normStdRGB: FloatArray,
  ): Tensor {
    checkNormMeanArg(normMeanRGB)
    checkNormStdArg(normStdRGB)

    return bitmapToFloat32Tensor(
        bitmap,
        0,
        0,
        bitmap.width,
        bitmap.height,
        normMeanRGB,
        normStdRGB,
    )
  }

  /**
   * Writes tensor content from specified [android.graphics.Bitmap], normalized with specified in
   * parameters mean and std to specified [java.nio.FloatBuffer] with specified offset.
   *
   * @param bitmap [android.graphics.Bitmap] as a source for Tensor data
   * @param x - x coordinate of top left corner of bitmap's area
   * @param y - y coordinate of top left corner of bitmap's area
   * @param width - width of bitmap's area
   * @param height - height of bitmap's area
   * @param normMeanRGB means for RGB channels normalization, length must equal 3, RGB order
   * @param normStdRGB standard deviation for RGB channels normalization, length must equal 3, RGB
   *   order
   */
  fun bitmapToFloatBuffer(
      bitmap: Bitmap,
      x: Int,
      y: Int,
      width: Int,
      height: Int,
      normMeanRGB: FloatArray,
      normStdRGB: FloatArray,
      outBuffer: FloatBuffer,
      outBufferOffset: Int,
  ) {
    checkOutBufferCapacity(outBuffer, outBufferOffset, width, height)
    checkNormMeanArg(normMeanRGB)
    checkNormStdArg(normStdRGB)
    val pixelsCount = height * width
    val pixels = IntArray(pixelsCount)
    bitmap.getPixels(pixels, 0, width, x, y, width, height)
    val offsetB = 2 * pixelsCount
    for (i in 0..99) {
      val c = pixels[i]
      Log.i("Image", ": " + i + " " + ((c shr 16) and 0xff))
    }
    for (i in 0 until pixelsCount) {
      val c = pixels[i]
      val r = ((c shr 16) and 0xff) / 255.0f
      val g = ((c shr 8) and 0xff) / 255.0f
      val b = ((c) and 0xff) / 255.0f
      outBuffer.put(outBufferOffset + i, (r - normMeanRGB[0]) / normStdRGB[0])
      outBuffer.put(outBufferOffset + pixelsCount + i, (g - normMeanRGB[1]) / normStdRGB[1])
      outBuffer.put(outBufferOffset + offsetB + i, (b - normMeanRGB[2]) / normStdRGB[2])
    }
  }

  /**
   * Creates new [Tensor] from specified area of [android.graphics.Bitmap], normalized with
   * specified in parameters mean and std.
   *
   * @param bitmap [android.graphics.Bitmap] as a source for Tensor data
   * @param x - x coordinate of top left corner of bitmap's area
   * @param y - y coordinate of top left corner of bitmap's area
   * @param width - width of bitmap's area
   * @param height - height of bitmap's area
   * @param normMeanRGB means for RGB channels normalization, length must equal 3, RGB order
   * @param normStdRGB standard deviation for RGB channels normalization, length must equal 3, RGB
   *   order
   */
  fun bitmapToFloat32Tensor(
      bitmap: Bitmap,
      x: Int,
      y: Int,
      width: Int,
      height: Int,
      normMeanRGB: FloatArray,
      normStdRGB: FloatArray,
  ): Tensor {
    checkNormMeanArg(normMeanRGB)
    checkNormStdArg(normStdRGB)

    val floatBuffer = Tensor.allocateFloatBuffer(3 * width * height)
    bitmapToFloatBuffer(bitmap, x, y, width, height, normMeanRGB, normStdRGB, floatBuffer, 0)
    return Tensor.fromBlob(floatBuffer, longArrayOf(1, 3, height.toLong(), width.toLong()))
  }

  private fun checkOutBufferCapacity(
      outBuffer: FloatBuffer,
      outBufferOffset: Int,
      tensorWidth: Int,
      tensorHeight: Int,
  ) {
    check(outBufferOffset + 3 * tensorWidth * tensorHeight <= outBuffer.capacity()) {
      "Buffer underflow"
    }
  }

  private fun checkTensorSize(tensorWidth: Int, tensorHeight: Int) {
    require(!(tensorHeight <= 0 || tensorWidth <= 0)) {
      "tensorHeight and tensorWidth must be positive"
    }
  }

  private fun checkRotateCWDegrees(rotateCWDegrees: Int) {
    require(
        !(rotateCWDegrees != 0 &&
            rotateCWDegrees != 90 &&
            rotateCWDegrees != 180 &&
            rotateCWDegrees != 270)
    ) {
      "rotateCWDegrees must be one of 0, 90, 180, 270"
    }
  }

  private fun checkNormStdArg(normStdRGB: FloatArray) {
    require(normStdRGB.size == 3) { "normStdRGB length must be 3" }
  }

  private fun checkNormMeanArg(normMeanRGB: FloatArray) {
    require(normMeanRGB.size == 3) { "normMeanRGB length must be 3" }
  }
}
