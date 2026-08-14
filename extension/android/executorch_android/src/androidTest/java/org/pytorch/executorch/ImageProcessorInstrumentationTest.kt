/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
package org.pytorch.executorch

import android.graphics.Bitmap
import android.graphics.Color
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import org.junit.runner.RunWith
import org.pytorch.executorch.extension.image.ImageOrientation
import org.pytorch.executorch.extension.image.ImageProcessor
import org.pytorch.executorch.extension.image.ImageProcessorConfig
import org.pytorch.executorch.extension.image.LetterboxAnchor
import org.pytorch.executorch.extension.image.Normalization
import org.pytorch.executorch.extension.image.ResizeMode

/**
 * Instrumentation tests for [ImageProcessor].
 *
 * The processor is pure image math, so these run without a .pte fixture: each test builds a bitmap
 * in memory and asserts on the produced tensor.
 */
@RunWith(AndroidJUnit4::class)
class ImageProcessorInstrumentationTest {

  // ─── Config validation ──────────────────────────────────────────────────────

  @Test
  fun testNonPositiveTargetSizeThrows() {
    try {
      ImageProcessorConfig(targetWidth = 0, targetHeight = 224)
      fail("Should throw for a non-positive target width")
    } catch (_: IllegalArgumentException) {}
  }

  @Test
  fun testZeroStandardDeviationThrows() {
    try {
      Normalization(1.0f, floatArrayOf(0f, 0f, 0f), floatArrayOf(1f, 0f, 1f))
      fail("Should throw for a zero standard deviation")
    } catch (_: IllegalArgumentException) {}
  }

  @Test
  fun testWrongChannelCountThrows() {
    try {
      Normalization(1.0f, floatArrayOf(0f, 0f), floatArrayOf(1f, 1f, 1f))
      fail("Should throw for a mean with the wrong channel count")
    } catch (_: IllegalArgumentException) {}
  }

  @Test
  fun testBuilderMatchesConstructor() {
    val built =
        ImageProcessorConfig.Builder()
            .setTargetSize(320, 240)
            .setResizeMode(ResizeMode.LETTERBOX)
            .setPadValue(0.5f)
            .build()
    assertEquals(320, built.targetWidth)
    assertEquals(240, built.targetHeight)
    assertEquals(ResizeMode.LETTERBOX, built.resizeMode)
    assertEquals(0.5f, built.padValue, 0.0f)
  }

  // ─── Lifecycle ──────────────────────────────────────────────────────────────

  @Test
  fun testCloseIsIdempotent() {
    val processor = ImageProcessor(ImageProcessorConfig())
    assertTrue(processor.isValid)
    processor.close()
    assertFalse(processor.isValid)
    processor.close()
    assertFalse(processor.isValid)
  }

  @Test
  fun testUseAfterCloseThrows() {
    val processor = ImageProcessor(ImageProcessorConfig())
    processor.close()
    try {
      processor.process(solidBitmap(4, 4, Color.RED))
      fail("Should throw after close")
    } catch (_: IllegalStateException) {}
  }

  // ─── Channel order and normalization ────────────────────────────────────────

  @Test
  fun testSolidRedKeepsChannelOrder() {
    ImageProcessor(ImageProcessorConfig(targetWidth = 2, targetHeight = 2)).use { processor ->
      val tensor = processor.process(solidBitmap(4, 4, Color.RED))
      assertArrayEquals(longArrayOf(1, 3, 2, 2), tensor.shape())
      val data = tensor.dataAsFloatArray
      // CHW: the whole R plane is 1.0, G and B planes are 0.0.
      for (i in 0 until 4) {
        assertEquals("R[$i]", 1.0f, data[i], TOLERANCE)
        assertEquals("G[$i]", 0.0f, data[4 + i], TOLERANCE)
        assertEquals("B[$i]", 0.0f, data[8 + i], TOLERANCE)
      }
    }
  }

  @Test
  fun testImagenetNormalizationIsApplied() {
    val config =
        ImageProcessorConfig(
            targetWidth = 1,
            targetHeight = 1,
            normalization = Normalization.imagenet(),
        )
    ImageProcessor(config).use { processor ->
      val data = processor.process(solidBitmap(4, 4, Color.WHITE)).dataAsFloatArray
      assertEquals((1.0f - 0.485f) / 0.229f, data[0], TOLERANCE)
      assertEquals((1.0f - 0.456f) / 0.224f, data[1], TOLERANCE)
      assertEquals((1.0f - 0.406f) / 0.225f, data[2], TOLERANCE)
    }
  }

  // ─── Geometry ───────────────────────────────────────────────────────────────

  @Test
  fun testStretchOutputShapeIgnoresAspectRatio() {
    ImageProcessor(ImageProcessorConfig(targetWidth = 224, targetHeight = 224)).use { processor ->
      assertArrayEquals(longArrayOf(1, 3, 224, 224), processor.computeOutputShape(640, 480))
      assertEquals(0, processor.computeLetterboxPadding(640, 480).x)
      assertEquals(0, processor.computeLetterboxPadding(640, 480).y)
    }
  }

  @Test
  fun testLetterboxPadsTheShorterAxis() {
    val config =
        ImageProcessorConfig(
            targetWidth = 100,
            targetHeight = 100,
            resizeMode = ResizeMode.LETTERBOX,
            letterboxAnchor = LetterboxAnchor.CENTER,
        )
    ImageProcessor(config).use { processor ->
      // A 200x100 source scales to 100x50, leaving 25px above and below.
      val padding = processor.computeLetterboxPadding(200, 100)
      assertEquals(0, padding.x)
      assertEquals(25, padding.y)
    }
  }

  @Test
  fun testTopLeftAnchorHasNoPadding() {
    val config =
        ImageProcessorConfig(
            targetWidth = 100,
            targetHeight = 100,
            resizeMode = ResizeMode.LETTERBOX,
            letterboxAnchor = LetterboxAnchor.TOP_LEFT,
        )
    ImageProcessor(config).use { processor ->
      val padding = processor.computeLetterboxPadding(200, 100)
      assertEquals(0, padding.x)
      assertEquals(0, padding.y)
    }
  }

  @Test
  fun testLetterboxFillsPaddingWithPadValue() {
    val config =
        ImageProcessorConfig(
            targetWidth = 4,
            targetHeight = 4,
            resizeMode = ResizeMode.LETTERBOX,
            padValue = -1.0f,
        )
    ImageProcessor(config).use { processor ->
      // An 8x4 source scales to 4x2, so rows 0 and 3 are padding.
      val data = processor.process(solidBitmap(8, 4, Color.WHITE)).dataAsFloatArray
      for (channel in 0 until 3) {
        val plane = channel * 16
        for (col in 0 until 4) {
          assertEquals("top pad", -1.0f, data[plane + col], TOLERANCE)
          assertEquals("bottom pad", -1.0f, data[plane + 12 + col], TOLERANCE)
        }
      }
    }
  }

  // ─── Orientation ────────────────────────────────────────────────────────────

  @Test
  fun testOrientationMovesLetterboxPaddingAxis() {
    ImageProcessor(ImageProcessorConfig()).use { processor ->
      // The output is always the target size, whatever the source orientation.
      assertArrayEquals(
          longArrayOf(1, 3, 224, 224),
          processor.computeOutputShape(640, 480, ImageOrientation.RIGHT),
      )
    }
    val letterbox =
        ImageProcessorConfig(
            targetWidth = 100,
            targetHeight = 100,
            resizeMode = ResizeMode.LETTERBOX,
        )
    ImageProcessor(letterbox).use { processor ->
      // Upright, a 200x100 source pads vertically; rotated 90 degrees it is
      // 100x200, so the padding moves to the horizontal axis.
      assertEquals(25, processor.computeLetterboxPadding(200, 100, ImageOrientation.UP).y)
      assertEquals(25, processor.computeLetterboxPadding(200, 100, ImageOrientation.RIGHT).x)
    }
  }

  @Test
  fun testRightOrientationRotatesClockwise() {
    val config = ImageProcessorConfig(targetWidth = 1, targetHeight = 2)
    ImageProcessor(config).use { processor ->
      // Source is red on the left, blue on the right. Rotating 90 degrees
      // clockwise puts red on top.
      val bitmap = Bitmap.createBitmap(2, 1, Bitmap.Config.ARGB_8888)
      bitmap.setPixel(0, 0, Color.RED)
      bitmap.setPixel(1, 0, Color.BLUE)
      val data = processor.process(bitmap, ImageOrientation.RIGHT).dataAsFloatArray
      // CHW over a 1x2 output: R plane is data[0..1], B plane is data[4..5].
      assertTrue("red should land on top, got R=${data[0]}", data[0] > 0.5f)
      assertTrue("red should not be at the bottom, got R=${data[1]}", data[1] < 0.5f)
      assertTrue("blue should land at the bottom, got B=${data[5]}", data[5] > 0.5f)
    }
  }

  @Test
  fun testUnsupportedOrientationCodeIsRejected() {
    // Mirrored EXIF codes are not supported; the enum only exposes rotations, so
    // this guards the native validation against a future enum addition.
    assertEquals(4, ImageOrientation.values().size)
    assertArrayEquals(
        intArrayOf(1, 3, 6, 8),
        ImageOrientation.values().map { it.exifCode }.toIntArray(),
    )
  }

  // ─── Reuse ──────────────────────────────────────────────────────────────────

  @Test
  fun testProcessIntoMatchesProcess() {
    ImageProcessor(ImageProcessorConfig(targetWidth = 8, targetHeight = 8)).use { processor ->
      val bitmap = gradientBitmap(16, 16)
      val allocated = processor.process(bitmap)
      val reused = Tensor.fromBlob(Tensor.allocateFloatBuffer(3 * 8 * 8), longArrayOf(1, 3, 8, 8))
      processor.processInto(bitmap, reused)
      assertArrayEquals(allocated.dataAsFloatArray, reused.dataAsFloatArray, TOLERANCE)
    }
  }

  @Test
  fun testProcessIntoRejectsWrongShape() {
    ImageProcessor(ImageProcessorConfig(targetWidth = 8, targetHeight = 8)).use { processor ->
      val wrong = Tensor.fromBlob(Tensor.allocateFloatBuffer(3 * 4 * 4), longArrayOf(1, 3, 4, 4))
      try {
        processor.processInto(solidBitmap(16, 16, Color.RED), wrong)
        fail("Should throw for a tensor with the wrong shape")
      } catch (_: IllegalArgumentException) {}
    }
  }

  @Test
  fun testProcessIntoRejectsWrongDtype() {
    ImageProcessor(ImageProcessorConfig(targetWidth = 8, targetHeight = 8)).use { processor ->
      val wrong = Tensor.fromBlob(IntArray(3 * 8 * 8), longArrayOf(1, 3, 8, 8))
      try {
        processor.processInto(solidBitmap(16, 16, Color.RED), wrong)
        fail("Should throw for a tensor with the wrong dtype")
      } catch (_: IllegalArgumentException) {}
    }
  }

  // ─── Helpers ────────────────────────────────────────────────────────────────

  private fun solidBitmap(width: Int, height: Int, color: Int): Bitmap {
    val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    bitmap.eraseColor(color)
    return bitmap
  }

  private fun gradientBitmap(width: Int, height: Int): Bitmap {
    val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    for (y in 0 until height) {
      for (x in 0 until width) {
        bitmap.setPixel(x, y, Color.rgb(x * 255 / width, y * 255 / height, 128))
      }
    }
    return bitmap
  }

  private companion object {
    const val TOLERANCE = 1e-3f
  }
}
