/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
package org.pytorch.executorch

import org.assertj.core.api.Assertions.assertThatThrownBy
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

/** Unit tests for [Tensor]. */
@RunWith(JUnit4::class)
class TensorTest {
  @Test
  fun testFloatTensor() {
    val data = floatArrayOf(Float.MIN_VALUE, 0f, 0.1f, Float.MAX_VALUE)
    val shape = longArrayOf(2, 2)
    var tensor = Tensor.fromBlob(data, shape)
    assertEquals(tensor.dtype(), DType.FLOAT)
    assertEquals(shape[0], tensor.shape()[0])
    assertEquals(shape[1], tensor.shape()[1])
    assertEquals(4, tensor.numel())
    assertEquals(data[0].toDouble(), tensor.dataAsFloatArray[0].toDouble(), 1e-5)
    assertEquals(data[1].toDouble(), tensor.dataAsFloatArray[1].toDouble(), 1e-5)
    assertEquals(data[2].toDouble(), tensor.dataAsFloatArray[2].toDouble(), 1e-5)
    assertEquals(data[3].toDouble(), tensor.dataAsFloatArray[3].toDouble(), 1e-5)

    val floatBuffer = Tensor.allocateFloatBuffer(4)
    floatBuffer.put(data)
    tensor = Tensor.fromBlob(floatBuffer, shape)
    assertEquals(tensor.dtype(), DType.FLOAT)
    assertEquals(shape[0], tensor.shape()[0])
    assertEquals(shape[1], tensor.shape()[1])
    assertEquals(4, tensor.numel())
    assertEquals(data[0].toDouble(), tensor.dataAsFloatArray[0].toDouble(), 1e-5)
    assertEquals(data[1].toDouble(), tensor.dataAsFloatArray[1].toDouble(), 1e-5)
    assertEquals(data[2].toDouble(), tensor.dataAsFloatArray[2].toDouble(), 1e-5)
    assertEquals(data[3].toDouble(), tensor.dataAsFloatArray[3].toDouble(), 1e-5)
  }

  @Test
  fun testIntTensor() {
    val data = intArrayOf(Int.MIN_VALUE, 0, 1, Int.MAX_VALUE)
    val shape = longArrayOf(1, 4, 1)
    var tensor = Tensor.fromBlob(data, shape)
    assertEquals(tensor.dtype(), DType.INT32)
    assertEquals(shape[0], tensor.shape()[0])
    assertEquals(shape[1], tensor.shape()[1])
    assertEquals(shape[2], tensor.shape()[2])
    assertEquals(4, tensor.numel())
    assertEquals(data[0].toLong(), tensor.dataAsIntArray[0].toLong())
    assertEquals(data[1].toLong(), tensor.dataAsIntArray[1].toLong())
    assertEquals(data[2].toLong(), tensor.dataAsIntArray[2].toLong())
    assertEquals(data[3].toLong(), tensor.dataAsIntArray[3].toLong())

    val intBuffer = Tensor.allocateIntBuffer(4)
    intBuffer.put(data)
    tensor = Tensor.fromBlob(intBuffer, shape)
    assertEquals(tensor.dtype(), DType.INT32)
    assertEquals(shape[0], tensor.shape()[0])
    assertEquals(shape[1], tensor.shape()[1])
    assertEquals(shape[2], tensor.shape()[2])
    assertEquals(4, tensor.numel())
    assertEquals(data[0].toLong(), tensor.dataAsIntArray[0].toLong())
    assertEquals(data[1].toLong(), tensor.dataAsIntArray[1].toLong())
    assertEquals(data[2].toLong(), tensor.dataAsIntArray[2].toLong())
    assertEquals(data[3].toLong(), tensor.dataAsIntArray[3].toLong())
  }

  @Test
  fun testDoubleTensor() {
    val data = doubleArrayOf(Double.MIN_VALUE, 0.0, 0.1, Double.MAX_VALUE)
    val shape = longArrayOf(1, 4)
    var tensor = Tensor.fromBlob(data, shape)
    assertEquals(tensor.dtype(), DType.DOUBLE)
    assertEquals(shape[0], tensor.shape()[0])
    assertEquals(shape[1], tensor.shape()[1])
    assertEquals(4, tensor.numel())
    assertEquals(data[0], tensor.dataAsDoubleArray[0], 1e-5)
    assertEquals(data[1], tensor.dataAsDoubleArray[1], 1e-5)
    assertEquals(data[2], tensor.dataAsDoubleArray[2], 1e-5)
    assertEquals(data[3], tensor.dataAsDoubleArray[3], 1e-5)

    val doubleBuffer = Tensor.allocateDoubleBuffer(4)
    doubleBuffer.put(data)
    tensor = Tensor.fromBlob(doubleBuffer, shape)
    assertEquals(tensor.dtype(), DType.DOUBLE)
    assertEquals(shape[0], tensor.shape()[0])
    assertEquals(shape[1], tensor.shape()[1])
    assertEquals(4, tensor.numel())
    assertEquals(data[0], tensor.dataAsDoubleArray[0], 1e-5)
    assertEquals(data[1], tensor.dataAsDoubleArray[1], 1e-5)
    assertEquals(data[2], tensor.dataAsDoubleArray[2], 1e-5)
    assertEquals(data[3], tensor.dataAsDoubleArray[3], 1e-5)
  }

  @Test
  fun testLongTensor() {
    val data = longArrayOf(Long.MIN_VALUE, 0L, 1L, Long.MAX_VALUE)
    val shape = longArrayOf(4, 1)
    var tensor = Tensor.fromBlob(data, shape)
    assertEquals(tensor.dtype(), DType.INT64)
    assertEquals(shape[0], tensor.shape()[0])
    assertEquals(shape[1], tensor.shape()[1])
    assertEquals(4, tensor.numel())
    assertEquals(data[0], tensor.dataAsLongArray[0])
    assertEquals(data[1], tensor.dataAsLongArray[1])
    assertEquals(data[2], tensor.dataAsLongArray[2])
    assertEquals(data[3], tensor.dataAsLongArray[3])

    val longBuffer = Tensor.allocateLongBuffer(4)
    longBuffer.put(data)
    tensor = Tensor.fromBlob(longBuffer, shape)
    assertEquals(tensor.dtype(), DType.INT64)
    assertEquals(shape[0], tensor.shape()[0])
    assertEquals(shape[1], tensor.shape()[1])
    assertEquals(4, tensor.numel())
    assertEquals(data[0], tensor.dataAsLongArray[0])
    assertEquals(data[1], tensor.dataAsLongArray[1])
    assertEquals(data[2], tensor.dataAsLongArray[2])
    assertEquals(data[3], tensor.dataAsLongArray[3])
  }

  @Test
  fun testSignedByteTensor() {
    val data = byteArrayOf(Byte.MIN_VALUE, 0.toByte(), 1.toByte(), Byte.MAX_VALUE)
    val shape = longArrayOf(1, 1, 4)
    var tensor = Tensor.fromBlob(data, shape)
    assertEquals(tensor.dtype(), DType.INT8)
    assertEquals(shape[0], tensor.shape()[0])
    assertEquals(shape[1], tensor.shape()[1])
    assertEquals(shape[2], tensor.shape()[2])
    assertEquals(4, tensor.numel())
    assertEquals(data[0].toLong(), tensor.dataAsByteArray[0].toLong())
    assertEquals(data[1].toLong(), tensor.dataAsByteArray[1].toLong())
    assertEquals(data[2].toLong(), tensor.dataAsByteArray[2].toLong())
    assertEquals(data[3].toLong(), tensor.dataAsByteArray[3].toLong())

    val byteBuffer = Tensor.allocateByteBuffer(4)
    byteBuffer.put(data)
    tensor = Tensor.fromBlob(byteBuffer, shape)
    assertEquals(tensor.dtype(), DType.INT8)
    assertEquals(shape[0], tensor.shape()[0])
    assertEquals(shape[1], tensor.shape()[1])
    assertEquals(shape[2], tensor.shape()[2])
    assertEquals(4, tensor.numel())
    assertEquals(data[0].toLong(), tensor.dataAsByteArray[0].toLong())
    assertEquals(data[1].toLong(), tensor.dataAsByteArray[1].toLong())
    assertEquals(data[2].toLong(), tensor.dataAsByteArray[2].toLong())
    assertEquals(data[3].toLong(), tensor.dataAsByteArray[3].toLong())
  }

  @Test
  fun testUnsignedByteTensor() {
    val data = byteArrayOf(0.toByte(), 1.toByte(), 2.toByte(), 255.toByte())
    val shape = longArrayOf(4, 1, 1)
    var tensor = Tensor.fromBlobUnsigned(data, shape)
    assertEquals(tensor.dtype(), DType.UINT8)
    assertEquals(shape[0], tensor.shape()[0])
    assertEquals(shape[1], tensor.shape()[1])
    assertEquals(shape[2], tensor.shape()[2])
    assertEquals(4, tensor.numel())
    assertEquals(data[0].toLong(), tensor.dataAsUnsignedByteArray[0].toLong())
    assertEquals(data[1].toLong(), tensor.dataAsUnsignedByteArray[1].toLong())
    assertEquals(data[2].toLong(), tensor.dataAsUnsignedByteArray[2].toLong())
    assertEquals(data[3].toLong(), tensor.dataAsUnsignedByteArray[3].toLong())

    val byteBuffer = Tensor.allocateByteBuffer(4)
    byteBuffer.put(data)
    tensor = Tensor.fromBlobUnsigned(byteBuffer, shape)
    assertEquals(tensor.dtype(), DType.UINT8)
    assertEquals(shape[0], tensor.shape()[0])
    assertEquals(shape[1], tensor.shape()[1])
    assertEquals(shape[2], tensor.shape()[2])
    assertEquals(4, tensor.numel())
    assertEquals(data[0].toLong(), tensor.dataAsUnsignedByteArray[0].toLong())
    assertEquals(data[1].toLong(), tensor.dataAsUnsignedByteArray[1].toLong())
    assertEquals(data[2].toLong(), tensor.dataAsUnsignedByteArray[2].toLong())
    assertEquals(data[3].toLong(), tensor.dataAsUnsignedByteArray[3].toLong())
  }

  @Test
  fun testIllegalDataTypeException() {
    val data = floatArrayOf(Float.MIN_VALUE, 0f, 0.1f, Float.MAX_VALUE)
    val shape = longArrayOf(2, 2)
    val tensor = Tensor.fromBlob(data, shape)
    assertEquals(tensor.dtype(), DType.FLOAT)

    assertThatThrownBy { tensor.dataAsByteArray }
        .isInstanceOf(IllegalStateException::class.java)
        .hasMessage("Tensor of type Tensor_float32 cannot return data as byte array.")

    assertThatThrownBy { tensor.dataAsUnsignedByteArray }
        .isInstanceOf(IllegalStateException::class.java)
        .hasMessage("Tensor of type Tensor_float32 cannot return data as unsigned byte array.")

    assertThatThrownBy { tensor.dataAsIntArray }
        .isInstanceOf(IllegalStateException::class.java)
        .hasMessage("Tensor of type Tensor_float32 cannot return data as int array.")

    assertThatThrownBy { tensor.dataAsDoubleArray }
        .isInstanceOf(IllegalStateException::class.java)
        .hasMessage("Tensor of type Tensor_float32 cannot return data as double array.")

    assertThatThrownBy { tensor.dataAsLongArray }
        .isInstanceOf(IllegalStateException::class.java)
        .hasMessage("Tensor of type Tensor_float32 cannot return data as long array.")
  }

  @Test
  fun testIllegalArguments() {
    val data = floatArrayOf(Float.MIN_VALUE, 0f, 0.1f, Float.MAX_VALUE)
    val shapeWithNegativeValues = longArrayOf(-1, 2)
    val mismatchShape = longArrayOf(1, 2)

    assertThatThrownBy { Tensor.fromBlob(null as FloatArray?, mismatchShape) }
        .isInstanceOf(IllegalArgumentException::class.java)
        .hasMessage("Data array must be not null")

    assertThatThrownBy { Tensor.fromBlob(data, null) }
        .isInstanceOf(IllegalArgumentException::class.java)
        .hasMessage("Shape must be not null")

    assertThatThrownBy { Tensor.fromBlob(data, shapeWithNegativeValues) }
        .isInstanceOf(IllegalArgumentException::class.java)
        .hasMessage("Shape elements must be non negative")

    assertThatThrownBy { Tensor.fromBlob(data, mismatchShape) }
        .isInstanceOf(IllegalArgumentException::class.java)
        .hasMessage("Inconsistent data capacity:4 and shape number elements:2 shape:[1, 2]")
  }

  @Test
  fun testLongTensorSerde() {
    val data = longArrayOf(1, 2, 3, 4)
    val shape = longArrayOf(2, 2)
    val tensor = Tensor.fromBlob(data, shape)
    val bytes = tensor.toByteArray()

    val deser = Tensor.fromByteArray(bytes)
    val deserShape = deser.shape()
    val deserData = deser.dataAsLongArray

    for (i in data.indices) {
      assertEquals(data[i], deserData[i])
    }

    for (i in shape.indices) {
      assertEquals(shape[i], deserShape[i])
    }
  }

  @Test
  fun testFloatTensorSerde() {
    val data = floatArrayOf(Float.MIN_VALUE, 0f, 0.1f, Float.MAX_VALUE)
    val shape = longArrayOf(2, 2)
    val tensor = Tensor.fromBlob(data, shape)
    val bytes = tensor.toByteArray()

    val deser = Tensor.fromByteArray(bytes)
    val deserShape = deser.shape()
    val deserData = deser.dataAsFloatArray

    for (i in data.indices) {
      assertEquals(data[i].toDouble(), deserData[i].toDouble(), 1e-5)
    }

    for (i in shape.indices) {
      assertEquals(shape[i], deserShape[i])
    }
  }
}
