/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorchllamademo;

import android.content.ContentResolver;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.net.Uri;
import androidx.annotation.Nullable;
import java.io.FileNotFoundException;
import java.io.InputStream;

public class ETImage {
  private int width;
  private int height;
  private final byte[] bytes;
  private final Uri uri;
  private final ContentResolver contentResolver;

  ETImage(ContentResolver contentResolver, Uri uri) {
    this.contentResolver = contentResolver;
    this.uri = uri;
    bytes = getBytesFromImageURI(uri);
  }

  public int getWidth() {
    return width;
  }

  public int getHeight() {
    return height;
  }

  public Uri getUri() {
    return uri;
  }

  public byte[] getBytes() {
    return bytes;
  }

  public int[] getInts() {
    // We need to convert the byte array to an int array because
    // the runner expects an int array as input.
    int[] intArray = new int[bytes.length];
    for (int i = 0; i < bytes.length; i++) {
      intArray[i] = (bytes[i++] & 0xFF);
    }
    return intArray;
  }

  private byte[] getBytesFromImageURI(Uri uri) {
    try {
      int RESIZED_IMAGE_WIDTH = 336;
      Bitmap bitmap = resizeImage(uri, RESIZED_IMAGE_WIDTH);

      if (bitmap == null) {
        ETLogging.getInstance().log("Unable to get bytes from Image URI. Bitmap is null");
        return new byte[0];
      }

      width = bitmap.getWidth();
      height = bitmap.getHeight();

      byte[] rgbValues = new byte[width * height * 3];

      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          // Get the color of the current pixel
          int color = bitmap.getPixel(x, y);

          // Extract the RGB values from the color
          int red = Color.red(color);
          int green = Color.green(color);
          int blue = Color.blue(color);

          // Store the RGB values in the byte array
          rgbValues[y * width + x] = (byte) red;
          rgbValues[(y * width + x) + height * width] = (byte) green;
          rgbValues[(y * width + x) + 2 * height * width] = (byte) blue;
        }
      }
      return rgbValues;
    } catch (FileNotFoundException e) {
      throw new RuntimeException(e);
    }
  }

  @Nullable
  private Bitmap resizeImage(Uri uri, int maxLength) throws FileNotFoundException {
    InputStream inputStream = contentResolver.openInputStream(uri);
    if (inputStream == null) {
      ETLogging.getInstance().log("Unable to resize image, input streams is null");
      return null;
    }
    Bitmap bitmap = BitmapFactory.decodeStream(inputStream);
    if (bitmap == null) {
      ETLogging.getInstance().log("Unable to resize image, bitmap during decode stream is null");
      return null;
    }

    float aspectRatio;
    int finalWidth, finalHeight;

    if (bitmap.getWidth() > bitmap.getHeight()) {
      // width > height --> width = maxLength, height scale with aspect ratio
      aspectRatio = bitmap.getWidth() / (float) bitmap.getHeight();
      finalWidth = maxLength;
      finalHeight = Math.round(maxLength / aspectRatio);
    } else {
      // height >= width --> height = maxLength, width scale with aspect ratio
      aspectRatio = bitmap.getHeight() / (float) bitmap.getWidth();
      finalHeight = maxLength;
      finalWidth = Math.round(maxLength / aspectRatio);
    }

    return Bitmap.createScaledBitmap(bitmap, finalWidth, finalHeight, false);
  }
}
