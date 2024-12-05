/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link Tensor}. */
@RunWith(JUnit4.class)
public class TensorTest {

  @Test
  public void testFloatTensor() {
    float data[] = {Float.MIN_VALUE, 0f, 0.1f, Float.MAX_VALUE};
    long shape[] = {2, 2};
    Tensor tensor = Tensor.fromBlob(data, shape);
    assertEquals(tensor.dtype(), DType.FLOAT);
    assertEquals(shape[0], tensor.shape()[0]);
    assertEquals(shape[1], tensor.shape()[1]);
    assertEquals(4, tensor.numel());
    assertEquals(data[0], tensor.getDataAsFloatArray()[0], 1e-5);
    assertEquals(data[1], tensor.getDataAsFloatArray()[1], 1e-5);
    assertEquals(data[2], tensor.getDataAsFloatArray()[2], 1e-5);
    assertEquals(data[3], tensor.getDataAsFloatArray()[3], 1e-5);

    FloatBuffer floatBuffer = Tensor.allocateFloatBuffer(4);
    floatBuffer.put(data);
    tensor = Tensor.fromBlob(floatBuffer, shape);
    assertEquals(tensor.dtype(), DType.FLOAT);
    assertEquals(shape[0], tensor.shape()[0]);
    assertEquals(shape[1], tensor.shape()[1]);
    assertEquals(4, tensor.numel());
    assertEquals(data[0], tensor.getDataAsFloatArray()[0], 1e-5);
    assertEquals(data[1], tensor.getDataAsFloatArray()[1], 1e-5);
    assertEquals(data[2], tensor.getDataAsFloatArray()[2], 1e-5);
    assertEquals(data[3], tensor.getDataAsFloatArray()[3], 1e-5);
  }

  @Test
  public void testIntTensor() {
    int data[] = {Integer.MIN_VALUE, 0, 1, Integer.MAX_VALUE};
    long shape[] = {1, 4, 1};
    Tensor tensor = Tensor.fromBlob(data, shape);
    assertEquals(tensor.dtype(), DType.INT32);
    assertEquals(shape[0], tensor.shape()[0]);
    assertEquals(shape[1], tensor.shape()[1]);
    assertEquals(shape[2], tensor.shape()[2]);
    assertEquals(4, tensor.numel());
    assertEquals(data[0], tensor.getDataAsIntArray()[0]);
    assertEquals(data[1], tensor.getDataAsIntArray()[1]);
    assertEquals(data[2], tensor.getDataAsIntArray()[2]);
    assertEquals(data[3], tensor.getDataAsIntArray()[3]);

    IntBuffer intBuffer = Tensor.allocateIntBuffer(4);
    intBuffer.put(data);
    tensor = Tensor.fromBlob(intBuffer, shape);
    assertEquals(tensor.dtype(), DType.INT32);
    assertEquals(shape[0], tensor.shape()[0]);
    assertEquals(shape[1], tensor.shape()[1]);
    assertEquals(shape[2], tensor.shape()[2]);
    assertEquals(4, tensor.numel());
    assertEquals(data[0], tensor.getDataAsIntArray()[0]);
    assertEquals(data[1], tensor.getDataAsIntArray()[1]);
    assertEquals(data[2], tensor.getDataAsIntArray()[2]);
    assertEquals(data[3], tensor.getDataAsIntArray()[3]);
  }

  @Test
  public void testDoubleTensor() {
    double data[] = {Double.MIN_VALUE, 0.0d, 0.1d, Double.MAX_VALUE};
    long shape[] = {1, 4};
    Tensor tensor = Tensor.fromBlob(data, shape);
    assertEquals(tensor.dtype(), DType.DOUBLE);
    assertEquals(shape[0], tensor.shape()[0]);
    assertEquals(shape[1], tensor.shape()[1]);
    assertEquals(4, tensor.numel());
    assertEquals(data[0], tensor.getDataAsDoubleArray()[0], 1e-5);
    assertEquals(data[1], tensor.getDataAsDoubleArray()[1], 1e-5);
    assertEquals(data[2], tensor.getDataAsDoubleArray()[2], 1e-5);
    assertEquals(data[3], tensor.getDataAsDoubleArray()[3], 1e-5);

    DoubleBuffer doubleBuffer = Tensor.allocateDoubleBuffer(4);
    doubleBuffer.put(data);
    tensor = Tensor.fromBlob(doubleBuffer, shape);
    assertEquals(tensor.dtype(), DType.DOUBLE);
    assertEquals(shape[0], tensor.shape()[0]);
    assertEquals(shape[1], tensor.shape()[1]);
    assertEquals(4, tensor.numel());
    assertEquals(data[0], tensor.getDataAsDoubleArray()[0], 1e-5);
    assertEquals(data[1], tensor.getDataAsDoubleArray()[1], 1e-5);
    assertEquals(data[2], tensor.getDataAsDoubleArray()[2], 1e-5);
    assertEquals(data[3], tensor.getDataAsDoubleArray()[3], 1e-5);
  }

  @Test
  public void testLongTensor() {
    long data[] = {Long.MIN_VALUE, 0L, 1L, Long.MAX_VALUE};
    long shape[] = {4, 1};
    Tensor tensor = Tensor.fromBlob(data, shape);
    assertEquals(tensor.dtype(), DType.INT64);
    assertEquals(shape[0], tensor.shape()[0]);
    assertEquals(shape[1], tensor.shape()[1]);
    assertEquals(4, tensor.numel());
    assertEquals(data[0], tensor.getDataAsLongArray()[0]);
    assertEquals(data[1], tensor.getDataAsLongArray()[1]);
    assertEquals(data[2], tensor.getDataAsLongArray()[2]);
    assertEquals(data[3], tensor.getDataAsLongArray()[3]);

    LongBuffer longBuffer = Tensor.allocateLongBuffer(4);
    longBuffer.put(data);
    tensor = Tensor.fromBlob(longBuffer, shape);
    assertEquals(tensor.dtype(), DType.INT64);
    assertEquals(shape[0], tensor.shape()[0]);
    assertEquals(shape[1], tensor.shape()[1]);
    assertEquals(4, tensor.numel());
    assertEquals(data[0], tensor.getDataAsLongArray()[0]);
    assertEquals(data[1], tensor.getDataAsLongArray()[1]);
    assertEquals(data[2], tensor.getDataAsLongArray()[2]);
    assertEquals(data[3], tensor.getDataAsLongArray()[3]);
  }

  @Test
  public void testSignedByteTensor() {
    byte data[] = {Byte.MIN_VALUE, (byte) 0, (byte) 1, Byte.MAX_VALUE};
    long shape[] = {1, 1, 4};
    Tensor tensor = Tensor.fromBlob(data, shape);
    assertEquals(tensor.dtype(), DType.INT8);
    assertEquals(shape[0], tensor.shape()[0]);
    assertEquals(shape[1], tensor.shape()[1]);
    assertEquals(shape[2], tensor.shape()[2]);
    assertEquals(4, tensor.numel());
    assertEquals(data[0], tensor.getDataAsByteArray()[0]);
    assertEquals(data[1], tensor.getDataAsByteArray()[1]);
    assertEquals(data[2], tensor.getDataAsByteArray()[2]);
    assertEquals(data[3], tensor.getDataAsByteArray()[3]);

    ByteBuffer byteBuffer = Tensor.allocateByteBuffer(4);
    byteBuffer.put(data);
    tensor = Tensor.fromBlob(byteBuffer, shape);
    assertEquals(tensor.dtype(), DType.INT8);
    assertEquals(shape[0], tensor.shape()[0]);
    assertEquals(shape[1], tensor.shape()[1]);
    assertEquals(shape[2], tensor.shape()[2]);
    assertEquals(4, tensor.numel());
    assertEquals(data[0], tensor.getDataAsByteArray()[0]);
    assertEquals(data[1], tensor.getDataAsByteArray()[1]);
    assertEquals(data[2], tensor.getDataAsByteArray()[2]);
    assertEquals(data[3], tensor.getDataAsByteArray()[3]);
  }

  @Test
  public void testUnsignedByteTensor() {
    byte data[] = {(byte) 0, (byte) 1, (byte) 2, (byte) 255};
    long shape[] = {4, 1, 1};
    Tensor tensor = Tensor.fromBlobUnsigned(data, shape);
    assertEquals(tensor.dtype(), DType.UINT8);
    assertEquals(shape[0], tensor.shape()[0]);
    assertEquals(shape[1], tensor.shape()[1]);
    assertEquals(shape[2], tensor.shape()[2]);
    assertEquals(4, tensor.numel());
    assertEquals(data[0], tensor.getDataAsUnsignedByteArray()[0]);
    assertEquals(data[1], tensor.getDataAsUnsignedByteArray()[1]);
    assertEquals(data[2], tensor.getDataAsUnsignedByteArray()[2]);
    assertEquals(data[3], tensor.getDataAsUnsignedByteArray()[3]);

    ByteBuffer byteBuffer = Tensor.allocateByteBuffer(4);
    byteBuffer.put(data);
    tensor = Tensor.fromBlobUnsigned(byteBuffer, shape);
    assertEquals(tensor.dtype(), DType.UINT8);
    assertEquals(shape[0], tensor.shape()[0]);
    assertEquals(shape[1], tensor.shape()[1]);
    assertEquals(shape[2], tensor.shape()[2]);
    assertEquals(4, tensor.numel());
    assertEquals(data[0], tensor.getDataAsUnsignedByteArray()[0]);
    assertEquals(data[1], tensor.getDataAsUnsignedByteArray()[1]);
    assertEquals(data[2], tensor.getDataAsUnsignedByteArray()[2]);
    assertEquals(data[3], tensor.getDataAsUnsignedByteArray()[3]);
  }

  @Test
  public void testIllegalDataTypeException() {
    float data[] = {Float.MIN_VALUE, 0f, 0.1f, Float.MAX_VALUE};
    long shape[] = {2, 2};
    Tensor tensor = Tensor.fromBlob(data, shape);
    assertEquals(tensor.dtype(), DType.FLOAT);

    try {
      tensor.getDataAsByteArray();
      fail("Should have thrown an exception");
    } catch (IllegalStateException e) {
      // expected
    }
    try {
      tensor.getDataAsUnsignedByteArray();
      fail("Should have thrown an exception");
    } catch (IllegalStateException e) {
      // expected
    }
    try {
      tensor.getDataAsIntArray();
      fail("Should have thrown an exception");
    } catch (IllegalStateException e) {
      // expected
    }
    try {
      tensor.getDataAsDoubleArray();
      fail("Should have thrown an exception");
    } catch (IllegalStateException e) {
      // expected
    }
    try {
      tensor.getDataAsLongArray();
      fail("Should have thrown an exception");
    } catch (IllegalStateException e) {
      // expected
    }
  }

  @Test
  public void testIllegalArguments() {
    float data[] = {Float.MIN_VALUE, 0f, 0.1f, Float.MAX_VALUE};
    long shapeWithNegativeValues[] = {-1, 2};
    long mismatchShape[] = {1, 2};

    try {
      Tensor tensor = Tensor.fromBlob((float[]) null, mismatchShape);
      fail("Should have thrown an exception");
    } catch (IllegalArgumentException e) {
      // expected
    }
    try {
      Tensor tensor = Tensor.fromBlob(data, null);
      fail("Should have thrown an exception");
    } catch (IllegalArgumentException e) {
      // expected
    }
    try {
      Tensor tensor = Tensor.fromBlob(data, shapeWithNegativeValues);
      fail("Should have thrown an exception");
    } catch (IllegalArgumentException e) {
      // expected
    }
    try {
      Tensor tensor = Tensor.fromBlob(data, mismatchShape);
      fail("Should have thrown an exception");
    } catch (IllegalArgumentException e) {
      // expected
    }
  }

  @Test
  public void testLongTensorSerde() {
    long data[] = {1, 2, 3, 4};
    long shape[] = {2, 2};
    Tensor tensor = Tensor.fromBlob(data, shape);
    byte[] bytes = tensor.toByteArray();

    Tensor deser = Tensor.fromByteArray(bytes);
    long[] deserShape = deser.shape();
    long[] deserData = deser.getDataAsLongArray();

    for (int i = 0; i < data.length; i++) {
      assertEquals(data[i], deserData[i]);
    }

    for (int i = 0; i < shape.length; i++) {
      assertEquals(shape[i], deserShape[i]);
    }
  }

  @Test
  public void testFloatTensorSerde() {
    float data[] = {Float.MIN_VALUE, 0f, 0.1f, Float.MAX_VALUE};
    long shape[] = {2, 2};
    Tensor tensor = Tensor.fromBlob(data, shape);
    byte[] bytes = tensor.toByteArray();

    Tensor deser = Tensor.fromByteArray(bytes);
    long[] deserShape = deser.shape();
    float[] deserData = deser.getDataAsFloatArray();

    for (int i = 0; i < data.length; i++) {
      assertEquals(data[i], deserData[i], 1e-5);
    }

    for (int i = 0; i < shape.length; i++) {
      assertEquals(shape[i], deserShape[i]);
    }
  }
}
