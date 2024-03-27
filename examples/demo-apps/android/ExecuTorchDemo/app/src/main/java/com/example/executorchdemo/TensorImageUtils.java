/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorchdemo;

import android.graphics.Bitmap;
import android.util.Log;
import java.nio.FloatBuffer;
import org.pytorch.executorch.Tensor;

/**
 * Contains utility functions for {@link Tensor} creation from {@link android.graphics.Bitmap} or
 * {@link android.media.Image} source.
 */
public final class TensorImageUtils {

  public static float[] TORCHVISION_NORM_MEAN_RGB = new float[] {0.485f, 0.456f, 0.406f};
  public static float[] TORCHVISION_NORM_STD_RGB = new float[] {0.229f, 0.224f, 0.225f};

  /**
   * Creates new {@link Tensor} from full {@link android.graphics.Bitmap}, normalized with specified
   * in parameters mean and std.
   *
   * @param normMeanRGB means for RGB channels normalization, length must equal 3, RGB order
   * @param normStdRGB standard deviation for RGB channels normalization, length must equal 3, RGB
   *     order
   */
  public static Tensor bitmapToFloat32Tensor(
      final Bitmap bitmap, final float[] normMeanRGB, final float normStdRGB[]) {
    checkNormMeanArg(normMeanRGB);
    checkNormStdArg(normStdRGB);

    return bitmapToFloat32Tensor(
        bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), normMeanRGB, normStdRGB);
  }

  /**
   * Writes tensor content from specified {@link android.graphics.Bitmap}, normalized with specified
   * in parameters mean and std to specified {@link java.nio.FloatBuffer} with specified offset.
   *
   * @param bitmap {@link android.graphics.Bitmap} as a source for Tensor data
   * @param x - x coordinate of top left corner of bitmap's area
   * @param y - y coordinate of top left corner of bitmap's area
   * @param width - width of bitmap's area
   * @param height - height of bitmap's area
   * @param normMeanRGB means for RGB channels normalization, length must equal 3, RGB order
   * @param normStdRGB standard deviation for RGB channels normalization, length must equal 3, RGB
   *     order
   */
  public static void bitmapToFloatBuffer(
      final Bitmap bitmap,
      final int x,
      final int y,
      final int width,
      final int height,
      final float[] normMeanRGB,
      final float[] normStdRGB,
      final FloatBuffer outBuffer,
      final int outBufferOffset) {
    checkOutBufferCapacity(outBuffer, outBufferOffset, width, height);
    checkNormMeanArg(normMeanRGB);
    checkNormStdArg(normStdRGB);
    final int pixelsCount = height * width;
    final int[] pixels = new int[pixelsCount];
    bitmap.getPixels(pixels, 0, width, x, y, width, height);
    final int offset_g = pixelsCount;
    final int offset_b = 2 * pixelsCount;
    for (int i = 0; i < 100; i++) {
      final int c = pixels[i];
      Log.i("Image", ": " + i + " " + ((c >> 16) & 0xff));
    }
    for (int i = 0; i < pixelsCount; i++) {
      final int c = pixels[i];
      float r = ((c >> 16) & 0xff) / 255.0f;
      float g = ((c >> 8) & 0xff) / 255.0f;
      float b = ((c) & 0xff) / 255.0f;
      outBuffer.put(outBufferOffset + i, (r - normMeanRGB[0]) / normStdRGB[0]);
      outBuffer.put(outBufferOffset + offset_g + i, (g - normMeanRGB[1]) / normStdRGB[1]);
      outBuffer.put(outBufferOffset + offset_b + i, (b - normMeanRGB[2]) / normStdRGB[2]);
    }
  }

  /**
   * Creates new {@link Tensor} from specified area of {@link android.graphics.Bitmap}, normalized
   * with specified in parameters mean and std.
   *
   * @param bitmap {@link android.graphics.Bitmap} as a source for Tensor data
   * @param x - x coordinate of top left corner of bitmap's area
   * @param y - y coordinate of top left corner of bitmap's area
   * @param width - width of bitmap's area
   * @param height - height of bitmap's area
   * @param normMeanRGB means for RGB channels normalization, length must equal 3, RGB order
   * @param normStdRGB standard deviation for RGB channels normalization, length must equal 3, RGB
   *     order
   */
  public static Tensor bitmapToFloat32Tensor(
      final Bitmap bitmap,
      int x,
      int y,
      int width,
      int height,
      float[] normMeanRGB,
      float[] normStdRGB) {
    checkNormMeanArg(normMeanRGB);
    checkNormStdArg(normStdRGB);

    final FloatBuffer floatBuffer = Tensor.allocateFloatBuffer(3 * width * height);
    bitmapToFloatBuffer(bitmap, x, y, width, height, normMeanRGB, normStdRGB, floatBuffer, 0);
    return Tensor.fromBlob(floatBuffer, new long[] {1, 3, height, width});
  }

  private static void checkOutBufferCapacity(
      FloatBuffer outBuffer, int outBufferOffset, int tensorWidth, int tensorHeight) {
    if (outBufferOffset + 3 * tensorWidth * tensorHeight > outBuffer.capacity()) {
      throw new IllegalStateException("Buffer underflow");
    }
  }

  private static void checkTensorSize(int tensorWidth, int tensorHeight) {
    if (tensorHeight <= 0 || tensorWidth <= 0) {
      throw new IllegalArgumentException("tensorHeight and tensorWidth must be positive");
    }
  }

  private static void checkRotateCWDegrees(int rotateCWDegrees) {
    if (rotateCWDegrees != 0
        && rotateCWDegrees != 90
        && rotateCWDegrees != 180
        && rotateCWDegrees != 270) {
      throw new IllegalArgumentException("rotateCWDegrees must be one of 0, 90, 180, 270");
    }
  }

  private static void checkNormStdArg(float[] normStdRGB) {
    if (normStdRGB.length != 3) {
      throw new IllegalArgumentException("normStdRGB length must be 3");
    }
  }

  private static void checkNormMeanArg(float[] normMeanRGB) {
    if (normMeanRGB.length != 3) {
      throw new IllegalArgumentException("normMeanRGB length must be 3");
    }
  }
}
