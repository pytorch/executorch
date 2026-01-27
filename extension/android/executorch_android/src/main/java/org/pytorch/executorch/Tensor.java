/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch;

import java.util.logging.Logger;
import java.util.logging.Level;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;
import java.util.Arrays;
import java.util.Locale;
import org.pytorch.executorch.annotations.Experimental;

/**
 * Representation of an ExecuTorch Tensor. Behavior is similar to PyTorch's tensor objects.
 *
 * <p>Most tensors will be constructed as {@code Tensor.fromBlob(data, shape)}, where {@code data}
 * can be an array or a direct {@link Buffer} (of the proper subclass). Helper methods are provided
 * to allocate buffers properly.
 *
 * <p>To access Tensor data, see {@link #dtype()}, {@link #shape()}, and various {@code getDataAs*}
 * methods.
 *
 * <p>When constructing {@code Tensor} objects with {@code data} as an array, it is not specified
 * whether this data is copied or retained as a reference so it is recommended not to modify it
 * after constructing. {@code data} passed as a {@link Buffer} is not copied, so it can be modified
 * between {@link Module} calls to avoid reallocation. Data retrieved from {@code Tensor} objects
 * may be copied or may be a reference to the {@code Tensor}'s internal data buffer. {@code shape}
 * is always copied.
 *
 * <p>Warning: These APIs are experimental and subject to change without notice
 */
@Experimental
public abstract class Tensor {
  private static final Logger LOGGER = Logger.getLogger(Tensor.class.getName());
  private static final String ERROR_MSG_DATA_BUFFER_NOT_NULL = "Data buffer must be not null";
  private static final String ERROR_MSG_DATA_ARRAY_NOT_NULL = "Data array must be not null";
  private static final String ERROR_MSG_SHAPE_NOT_NULL = "Shape must be not null";
  private static final String ERROR_MSG_SHAPE_NON_NEGATIVE = "Shape elements must be non negative";
  private static final String ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER =
      "Data buffer must have native byte order (java.nio.ByteOrder#nativeOrder)";
  private static final String ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT =
      "Data buffer must be direct (java.nio.ByteBuffer#allocateDirect)";

  final long[] shape;

  private static final int BYTE_SIZE_BYTES = 1;
  private static final int INT_SIZE_BYTES = 4;
  private static final int LONG_SIZE_BYTES = 8;
  private static final int HALF_SIZE_BYTES = 2;
  private static final int FLOAT_SIZE_BYTES = 4;
  private static final int DOUBLE_SIZE_BYTES = 8;

  /**
   * Allocates a new direct {@link ByteBuffer} with native byte order with specified capacity that
   * can be used in {@link Tensor#fromBlob(ByteBuffer, long[])}, {@link
   * Tensor#fromBlobUnsigned(ByteBuffer, long[])}.
   *
   * @param numElements capacity (number of elements) of result buffer.
   */
  public static ByteBuffer allocateByteBuffer(int numElements) {
    return ByteBuffer.allocateDirect(numElements).order(ByteOrder.nativeOrder());
  }

  /**
   * Allocates a new direct {@link IntBuffer} with native byte order with specified capacity that
   * can be used in {@link Tensor#fromBlob(IntBuffer, long[])}.
   *
   * @param numElements capacity (number of elements) of result buffer.
   */
  public static IntBuffer allocateIntBuffer(int numElements) {
    return ByteBuffer.allocateDirect(numElements * INT_SIZE_BYTES)
        .order(ByteOrder.nativeOrder())
        .asIntBuffer();
  }

  /**
   * Allocates a new direct {@link FloatBuffer} with native byte order with specified capacity that
   * can be used in {@link Tensor#fromBlob(FloatBuffer, long[])}.
   *
   * @param numElements capacity (number of elements) of result buffer.
   */
  public static FloatBuffer allocateFloatBuffer(int numElements) {
    return ByteBuffer.allocateDirect(numElements * FLOAT_SIZE_BYTES)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer();
  }

  /**
   * Allocates a new direct {@link LongBuffer} with native byte order with specified capacity that
   * can be used in {@link Tensor#fromBlob(LongBuffer, long[])}.
   *
   * @param numElements capacity (number of elements) of result buffer.
   */
  public static LongBuffer allocateLongBuffer(int numElements) {
    return ByteBuffer.allocateDirect(numElements * LONG_SIZE_BYTES)
        .order(ByteOrder.nativeOrder())
        .asLongBuffer();
  }

  /**
   * Allocates a new direct {@link ShortBuffer} with native byte order and specified capacity that
   * can be used in {@link Tensor#fromBlob(ShortBuffer, long[])}.
   *
   * @param numElements capacity (number of elements) of result buffer.
   */
  public static ShortBuffer allocateHalfBuffer(int numElements) {
    return ByteBuffer.allocateDirect(numElements * HALF_SIZE_BYTES)
        .order(ByteOrder.nativeOrder())
        .asShortBuffer();
  }

  /**
   * Allocates a new direct {@link DoubleBuffer} with native byte order with specified capacity that
   * can be used in {@link Tensor#fromBlob(DoubleBuffer, long[])}.
   *
   * @param numElements capacity (number of elements) of result buffer.
   */
  public static DoubleBuffer allocateDoubleBuffer(int numElements) {
    return ByteBuffer.allocateDirect(numElements * DOUBLE_SIZE_BYTES)
        .order(ByteOrder.nativeOrder())
        .asDoubleBuffer();
  }

  /**
   * Creates a new Tensor instance with dtype torch.uint8 with specified shape and data as array of
   * bytes.
   *
   * @param data Tensor elements
   * @param shape Tensor shape
   */
  public static Tensor fromBlobUnsigned(byte[] data, long[] shape) {
    checkArgument(data != null, ERROR_MSG_DATA_ARRAY_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.length, shape);
    final ByteBuffer byteBuffer = allocateByteBuffer((int) numel(shape));
    byteBuffer.put(data);
    return new Tensor_uint8(byteBuffer, shape);
  }

  /**
   * Creates a new Tensor instance with dtype torch.int8 with specified shape and data as array of
   * bytes.
   *
   * @param data Tensor elements
   * @param shape Tensor shape
   */
  public static Tensor fromBlob(byte[] data, long[] shape) {
    checkArgument(data != null, ERROR_MSG_DATA_ARRAY_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.length, shape);
    final ByteBuffer byteBuffer = allocateByteBuffer((int) numel(shape));
    byteBuffer.put(data);
    return new Tensor_int8(byteBuffer, shape);
  }

  /**
   * Creates a new Tensor instance with dtype torch.int32 with specified shape and data as array of
   * ints.
   *
   * @param data Tensor elements
   * @param shape Tensor shape
   */
  public static Tensor fromBlob(int[] data, long[] shape) {
    checkArgument(data != null, ERROR_MSG_DATA_ARRAY_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.length, shape);
    final IntBuffer intBuffer = allocateIntBuffer((int) numel(shape));
    intBuffer.put(data);
    return new Tensor_int32(intBuffer, shape);
  }

  /**
   * Creates a new Tensor instance with dtype torch.float32 with specified shape and data as array
   * of floats.
   *
   * @param data Tensor elements
   * @param shape Tensor shape
   */
  public static Tensor fromBlob(float[] data, long[] shape) {
    checkArgument(data != null, ERROR_MSG_DATA_ARRAY_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.length, shape);
    final FloatBuffer floatBuffer = allocateFloatBuffer((int) numel(shape));
    floatBuffer.put(data);
    return new Tensor_float32(floatBuffer, shape);
  }

  /**
   * Creates a new Tensor instance with dtype torch.float16 with specified shape and data as array
   * of IEEE-754 half-precision values encoded in {@code short}s.
   *
   * @param data Tensor elements encoded as 16-bit floats.
   * @param shape Tensor shape
   */
  public static Tensor fromBlob(short[] data, long[] shape) {
    checkArgument(data != null, ERROR_MSG_DATA_ARRAY_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.length, shape);
    final ShortBuffer shortBuffer = allocateHalfBuffer((int) numel(shape));
    shortBuffer.put(data);
    return new Tensor_float16(shortBuffer, shape);
  }

  /**
   * Creates a new Tensor instance with dtype torch.int64 with specified shape and data as array of
   * longs.
   *
   * @param data Tensor elements
   * @param shape Tensor shape
   */
  public static Tensor fromBlob(long[] data, long[] shape) {
    checkArgument(data != null, ERROR_MSG_DATA_ARRAY_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.length, shape);
    final LongBuffer longBuffer = allocateLongBuffer((int) numel(shape));
    longBuffer.put(data);
    return new Tensor_int64(longBuffer, shape);
  }

  /**
   * Creates a new Tensor instance with dtype torch.float64 with specified shape and data as array
   * of doubles.
   *
   * @param shape Tensor shape
   * @param data Tensor elements
   */
  public static Tensor fromBlob(double[] data, long[] shape) {
    checkArgument(data != null, ERROR_MSG_DATA_ARRAY_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.length, shape);
    final DoubleBuffer doubleBuffer = allocateDoubleBuffer((int) numel(shape));
    doubleBuffer.put(data);
    return new Tensor_float64(doubleBuffer, shape);
  }

  /**
   * Creates a new Tensor instance with dtype torch.uint8 with specified shape and data.
   *
   * @param data Direct buffer with native byte order that contains {@code Tensor.numel(shape)}
   *     elements. The buffer is used directly without copying, and changes to its content will
   *     change the tensor.
   * @param shape Tensor shape
   */
  public static Tensor fromBlobUnsigned(ByteBuffer data, long[] shape) {
    checkArgument(data != null, ERROR_MSG_DATA_BUFFER_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.capacity(), shape);
    checkArgument(data.isDirect(), ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT);
    checkArgument(
        (data.order() == ByteOrder.nativeOrder()),
        ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER);
    return new Tensor_uint8(data, shape);
  }

  /**
   * Creates a new Tensor instance with dtype torch.int8 with specified shape and data.
   *
   * @param data Direct buffer with native byte order that contains {@code Tensor.numel(shape)}
   *     elements. The buffer is used directly without copying, and changes to its content will
   *     change the tensor.
   * @param shape Tensor shape
   */
  public static Tensor fromBlob(ByteBuffer data, long[] shape) {
    checkArgument(data != null, ERROR_MSG_DATA_BUFFER_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.capacity(), shape);
    checkArgument(data.isDirect(), ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT);
    checkArgument(
        (data.order() == ByteOrder.nativeOrder()),
        ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER);
    return new Tensor_int8(data, shape);
  }

  /**
   * Creates a new Tensor instance with dtype torch.int32 with specified shape and data.
   *
   * @param data Direct buffer with native byte order that contains {@code Tensor.numel(shape)}
   *     elements. The buffer is used directly without copying, and changes to its content will
   *     change the tensor.
   * @param shape Tensor shape
   */
  public static Tensor fromBlob(IntBuffer data, long[] shape) {
    checkArgument(data != null, ERROR_MSG_DATA_BUFFER_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.capacity(), shape);
    checkArgument(data.isDirect(), ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT);
    checkArgument(
        (data.order() == ByteOrder.nativeOrder()),
        ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER);
    return new Tensor_int32(data, shape);
  }

  /**
   * Creates a new Tensor instance with dtype torch.float32 with specified shape and data.
   *
   * @param data Direct buffer with native byte order that contains {@code Tensor.numel(shape)}
   *     elements. The buffer is used directly without copying, and changes to its content will
   *     change the tensor.
   * @param shape Tensor shape
   */
  public static Tensor fromBlob(FloatBuffer data, long[] shape) {
    checkArgument(data != null, ERROR_MSG_DATA_BUFFER_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.capacity(), shape);
    checkArgument(data.isDirect(), ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT);
    checkArgument(
        (data.order() == ByteOrder.nativeOrder()),
        ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER);
    return new Tensor_float32(data, shape);
  }

  /**
   * Creates a new Tensor instance with dtype torch.float16 with specified shape and data.
   *
   * @param data Direct buffer with native byte order that contains {@code Tensor.numel(shape)}
   *     elements encoded as IEEE-754 half-precision floats. The buffer is used directly without
   *     copying.
   * @param shape Tensor shape
   */
  public static Tensor fromBlob(ShortBuffer data, long[] shape) {
    checkArgument(data != null, ERROR_MSG_DATA_BUFFER_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.capacity(), shape);
    checkArgument(data.isDirect(), ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT);
    checkArgument(
        (data.order() == ByteOrder.nativeOrder()),
        ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER);
    return new Tensor_float16(data, shape);
  }

  /**
   * Creates a new Tensor instance with dtype torch.int64 with specified shape and data.
   *
   * @param data Direct buffer with native byte order that contains {@code Tensor.numel(shape)}
   *     elements. The buffer is used directly without copying, and changes to its content will
   *     change the tensor.
   * @param shape Tensor shape
   */
  public static Tensor fromBlob(LongBuffer data, long[] shape) {
    checkArgument(data != null, ERROR_MSG_DATA_BUFFER_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.capacity(), shape);
    checkArgument(data.isDirect(), ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT);
    checkArgument(
        (data.order() == ByteOrder.nativeOrder()),
        ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER);
    return new Tensor_int64(data, shape);
  }

  /**
   * Creates a new Tensor instance with dtype torch.float64 with specified shape and data.
   *
   * @param data Direct buffer with native byte order that contains {@code Tensor.numel(shape)}
   *     elements. The buffer is used directly without copying, and changes to its content will
   *     change the tensor.
   * @param shape Tensor shape
   */
  public static Tensor fromBlob(DoubleBuffer data, long[] shape) {
    checkArgument(data != null, ERROR_MSG_DATA_BUFFER_NOT_NULL);
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    checkShapeAndDataCapacityConsistency(data.capacity(), shape);
    checkArgument(data.isDirect(), ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT);
    checkArgument(
        (data.order() == ByteOrder.nativeOrder()),
        ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER);
    return new Tensor_float64(data, shape);
  }

  /**
   * Creates a new Tensor instance with given data-type and all elements initialized to one.
   *
   * @param shape Tensor shape
   * @param dtype Tensor data-type
   */
  public static Tensor ones(long[] shape, DType dtype) {
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    int numElements = (int) numel(shape);
    switch (dtype) {
      case UINT8:
        byte[] uInt8Data = new byte[numElements];
        Arrays.fill(uInt8Data, (byte) 1);
        return Tensor.fromBlobUnsigned(uInt8Data, shape);
      case INT8:
        byte[] int8Data = new byte[numElements];
        Arrays.fill(int8Data, (byte) 1);
        return Tensor.fromBlob(int8Data, shape);
      case INT32:
        int[] int32Data = new int[numElements];
        Arrays.fill(int32Data, 1);
        return Tensor.fromBlob(int32Data, shape);
      case FLOAT:
        float[] float32Data = new float[numElements];
        Arrays.fill(float32Data, 1.0f);
        return Tensor.fromBlob(float32Data, shape);
      case INT64:
        long[] int64Data = new long[numElements];
        Arrays.fill(int64Data, 1L);
        return Tensor.fromBlob(int64Data, shape);
      case DOUBLE:
        double[] float64Data = new double[numElements];
        Arrays.fill(float64Data, 1.0);
        return Tensor.fromBlob(float64Data, shape);
      default:
        throw new IllegalArgumentException(
            String.format("Tensor.ones() cannot be used with DType %s", dtype));
    }
  }

  /**
   * Creates a new Tensor instance with given data-type and all elements initialized to zero.
   *
   * @param shape Tensor shape
   * @param dtype Tensor data-type
   */
  public static Tensor zeros(long[] shape, DType dtype) {
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    checkShape(shape);
    int numElements = (int) numel(shape);
    switch (dtype) {
      case UINT8:
        byte[] uInt8Data = new byte[numElements];
        return Tensor.fromBlobUnsigned(uInt8Data, shape);
      case INT8:
        byte[] int8Data = new byte[numElements];
        return Tensor.fromBlob(int8Data, shape);
      case INT32:
        int[] int32Data = new int[numElements];
        return Tensor.fromBlob(int32Data, shape);
      case FLOAT:
        float[] float32Data = new float[numElements];
        return Tensor.fromBlob(float32Data, shape);
      case INT64:
        long[] int64Data = new long[numElements];
        return Tensor.fromBlob(int64Data, shape);
      case DOUBLE:
        double[] float64Data = new double[numElements];
        return Tensor.fromBlob(float64Data, shape);
      default:
        throw new IllegalArgumentException(
            String.format("Tensor.zeros() cannot be used with DType %s", dtype));
    }
  }

  private long mNativeHandle;

  private Tensor(long[] shape) {
    checkShape(shape);
    this.shape = Arrays.copyOf(shape, shape.length);
  }

  /** Returns the number of elements in this tensor. */
  public long numel() {
    return numel(this.shape);
  }

  /** Calculates the number of elements in a tensor with the specified shape. */
  public static long numel(long[] shape) {
    checkShape(shape);
    int result = 1;
    for (long s : shape) {
      result *= s;
    }
    return result;
  }

  /** Returns the shape of this tensor. (The array is a fresh copy.) */
  public long[] shape() {
    return Arrays.copyOf(shape, shape.length);
  }

  /**
   * @return data type of this tensor.
   */
  public abstract DType dtype();

  // Called from native

  int dtypeJniCode() {
    return dtype().jniCode;
  }

  /**
   * @return a Java byte array that contains the tensor data. This may be a copy or reference.
   * @throws IllegalStateException if it is called for a non-int8 tensor.
   */
  public byte[] getDataAsByteArray() {
    throw new IllegalStateException(
        "Tensor of type " + getClass().getSimpleName() + " cannot return data as byte array.");
  }

  /**
   * @return a Java short array that contains the tensor data interpreted as IEEE-754 half-precision
   *     bit patterns. This may be a copy or reference.
   * @throws IllegalStateException if it is called for a non-float16 tensor.
   */
  public short[] getDataAsShortArray() {
    throw new IllegalStateException(
        "Tensor of type " + getClass().getSimpleName() + " cannot return data as short array.");
  }

  /**
   * @return a Java byte array that contains the tensor data. This may be a copy or reference.
   * @throws IllegalStateException if it is called for a non-uint8 tensor.
   */
  public byte[] getDataAsUnsignedByteArray() {
    throw new IllegalStateException(
        "Tensor of type "
            + getClass().getSimpleName()
            + " cannot return data as unsigned byte array.");
  }

  /**
   * @return a Java int array that contains the tensor data. This may be a copy or reference.
   * @throws IllegalStateException if it is called for a non-int32 tensor.
   */
  public int[] getDataAsIntArray() {
    throw new IllegalStateException(
        "Tensor of type " + getClass().getSimpleName() + " cannot return data as int array.");
  }

  /**
   * @return a Java float array that contains the tensor data. This may be a copy or reference.
   * @throws IllegalStateException if it is called for a non-float32 tensor.
   */
  public float[] getDataAsFloatArray() {
    throw new IllegalStateException(
        "Tensor of type " + getClass().getSimpleName() + " cannot return data as float array.");
  }

  /**
   * @return a Java long array that contains the tensor data. This may be a copy or reference.
   * @throws IllegalStateException if it is called for a non-int64 tensor.
   */
  public long[] getDataAsLongArray() {
    throw new IllegalStateException(
        "Tensor of type " + getClass().getSimpleName() + " cannot return data as long array.");
  }

  /**
   * @return a Java double array that contains the tensor data. This may be a copy or reference.
   * @throws IllegalStateException if it is called for a non-float64 tensor.
   */
  public double[] getDataAsDoubleArray() {
    throw new IllegalStateException(
        "Tensor of type " + getClass().getSimpleName() + " cannot return data as double array.");
  }


  Buffer getRawDataBuffer() {
    throw new IllegalStateException(
        "Tensor of type " + getClass().getSimpleName() + " cannot " + "return raw data buffer.");
  }

  static class Tensor_uint8 extends Tensor {
    private final ByteBuffer data;

    private Tensor_uint8(ByteBuffer data, long[] shape) {
      super(shape);
      this.data = data;
    }

    @Override
    public DType dtype() {
      return DType.UINT8;
    }

    @Override
    Buffer getRawDataBuffer() {
      return data;
    }

    @Override
    public byte[] getDataAsUnsignedByteArray() {
      data.rewind();
      byte[] arr = new byte[data.remaining()];
      data.get(arr);
      return arr;
    }

    @Override
    public String toString() {
      return String.format("Tensor(%s, dtype=torch.uint8)", Arrays.toString(shape));
    }
  }

  static class Tensor_int8 extends Tensor {
    private final ByteBuffer data;

    private Tensor_int8(ByteBuffer data, long[] shape) {
      super(shape);
      this.data = data;
    }

    @Override
    public DType dtype() {
      return DType.INT8;
    }

    @Override
    Buffer getRawDataBuffer() {
      return data;
    }

    @Override
    public byte[] getDataAsByteArray() {
      data.rewind();
      byte[] arr = new byte[data.remaining()];
      data.get(arr);
      return arr;
    }

    @Override
    public String toString() {
      return String.format("Tensor(%s, dtype=torch.int8)", Arrays.toString(shape));
    }
  }

  static class Tensor_int32 extends Tensor {
    private final IntBuffer data;

    private Tensor_int32(IntBuffer data, long[] shape) {
      super(shape);
      this.data = data;
    }

    @Override
    public DType dtype() {
      return DType.INT32;
    }

    @Override
    Buffer getRawDataBuffer() {
      return data;
    }

    @Override
    public int[] getDataAsIntArray() {
      data.rewind();
      int[] arr = new int[data.remaining()];
      data.get(arr);
      return arr;
    }

    @Override
    public String toString() {
      return String.format("Tensor(%s, dtype=torch.int32)", Arrays.toString(shape));
    }
  }

  static class Tensor_float32 extends Tensor {
    private final FloatBuffer data;

    Tensor_float32(FloatBuffer data, long[] shape) {
      super(shape);
      this.data = data;
    }

    @Override
    public float[] getDataAsFloatArray() {
      data.rewind();
      float[] arr = new float[data.remaining()];
      data.get(arr);
      return arr;
    }

    @Override
    public DType dtype() {
      return DType.FLOAT;
    }

    @Override
    Buffer getRawDataBuffer() {
      return data;
    }

    @Override
    public String toString() {
      return String.format("Tensor(%s, dtype=torch.float32)", Arrays.toString(shape));
    }
  }

  static class Tensor_float16 extends Tensor {
    private final ShortBuffer data;

    private Tensor_float16(ShortBuffer data, long[] shape) {
      super(shape);
      this.data = data;
    }

    @Override
    public DType dtype() {
      return DType.HALF;
    }

    @Override
    Buffer getRawDataBuffer() {
      return data;
    }

    @Override
    public short[] getDataAsShortArray() {
      data.rewind();
      short[] arr = new short[data.remaining()];
      data.get(arr);
      return arr;
    }

    @Override
    public float[] getDataAsFloatArray() {
      data.rewind();
      int remaining = data.remaining();
      float[] arr = new float[remaining];
      for (int i = 0; i < remaining; i++) {
        arr[i] = halfBitsToFloat(data.get());
      }
      return arr;
    }

    @Override
    public String toString() {
      return String.format("Tensor(%s, dtype=torch.float16)", Arrays.toString(shape));
    }

    private static float halfBitsToFloat(short halfBits) {
      int h = halfBits & 0xFFFF;
      int sign = (h >>> 15) & 0x1;
      int exp = (h >>> 10) & 0x1F;
      int mant = h & 0x3FF;

      if (exp == 0) {
        if (mant == 0) {
          return sign == 0 ? 0.0f : -0.0f;
        }
        float result = mant * 5.9604645e-8f; // 2^-24
        return sign == 0 ? result : -result;
      } else if (exp == 0x1F) {
        if (mant == 0) {
          return sign == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
        }
        int bits = (sign << 31) | 0x7f800000 | (mant << 13);
        return Float.intBitsToFloat(bits);
      } else {
        int exp32 = exp + 112; // 127 (float bias) - 15 (half bias)
        int bits = (sign << 31) | (exp32 << 23) | (mant << 13);
        return Float.intBitsToFloat(bits);
      }
    }
  }

  static class Tensor_int64 extends Tensor {
    private final LongBuffer data;

    private Tensor_int64(LongBuffer data, long[] shape) {
      super(shape);
      this.data = data;
    }

    @Override
    public DType dtype() {
      return DType.INT64;
    }

    @Override
    Buffer getRawDataBuffer() {
      return data;
    }

    @Override
    public long[] getDataAsLongArray() {
      data.rewind();
      long[] arr = new long[data.remaining()];
      data.get(arr);
      return arr;
    }

    @Override
    public String toString() {
      return String.format("Tensor(%s, dtype=torch.int64)", Arrays.toString(shape));
    }
  }

  static class Tensor_float64 extends Tensor {
    private final DoubleBuffer data;

    private Tensor_float64(DoubleBuffer data, long[] shape) {
      super(shape);
      this.data = data;
    }

    @Override
    public DType dtype() {
      return DType.DOUBLE;
    }

    @Override
    Buffer getRawDataBuffer() {
      return data;
    }

    @Override
    public double[] getDataAsDoubleArray() {
      data.rewind();
      double[] arr = new double[data.remaining()];
      data.get(arr);
      return arr;
    }

    @Override
    public String toString() {
      return String.format("Tensor(%s, dtype=torch.float64)", Arrays.toString(shape));
    }
  }

  static class Tensor_unsupported extends Tensor {
    private final ByteBuffer data;
    private final DType mDtype;

    private Tensor_unsupported(ByteBuffer data, long[] shape, DType dtype) {
      super(shape);
      this.data = data;
      this.mDtype = dtype;
      LOGGER.log(Level.SEVERE, toString() + " in Java. Please consider re-export the model with proper return type");
    }

    @Override
    public DType dtype() {
      return mDtype;
    }

    @Override
    public String toString() {
      return String.format("Unsupported tensor(%s, dtype=%d)", Arrays.toString(shape), this.mDtype);
    }
  }

  // region checks
  private static void checkArgument(boolean expression, String errorMessage, Object... args) {
    if (!expression) {
      throw new IllegalArgumentException(String.format(Locale.US, errorMessage, args));
    }
  }

  private static void checkShape(long[] shape) {
    checkArgument(shape != null, ERROR_MSG_SHAPE_NOT_NULL);
    for (int i = 0; i < shape.length; i++) {
      checkArgument(shape[i] >= 0, ERROR_MSG_SHAPE_NON_NEGATIVE);
    }
  }

  private static void checkShapeAndDataCapacityConsistency(int dataCapacity, long[] shape) {
    final long numel = numel(shape);
    checkArgument(
        numel == dataCapacity,
        "Inconsistent data capacity:%d and shape number elements:%d shape:%s",
        dataCapacity,
        numel,
        Arrays.toString(shape));
  }

  // endregion checks

  // Called from native

  private static Tensor nativeNewTensor(
      ByteBuffer data, long[] shape, int dtype, long nativeHandle) {
    Tensor tensor = null;

    if (DType.FLOAT.jniCode == dtype) {
      tensor = new Tensor_float32(data.asFloatBuffer(), shape);
    } else if (DType.HALF.jniCode == dtype) {
      tensor = new Tensor_float16(data.asShortBuffer(), shape);
    } else if (DType.INT32.jniCode == dtype) {
      tensor = new Tensor_int32(data.asIntBuffer(), shape);
    } else if (DType.INT64.jniCode == dtype) {
      tensor = new Tensor_int64(data.asLongBuffer(), shape);
    } else if (DType.DOUBLE.jniCode == dtype) {
      tensor = new Tensor_float64(data.asDoubleBuffer(), shape);
    } else if (DType.UINT8.jniCode == dtype) {
      tensor = new Tensor_uint8(data, shape);
    } else if (DType.INT8.jniCode == dtype) {
      tensor = new Tensor_int8(data, shape);
    } else {
      tensor = new Tensor_unsupported(data, shape, DType.fromJniCode(dtype));
    }
    tensor.mNativeHandle = nativeHandle;
    return tensor;
  }

  /**
   * Serializes a {@code Tensor} into a byte array. Note: This method is experimental and subject to
   * change without notice. This does NOT supoprt list type.
   *
   * @return The serialized byte array.
   */
  public byte[] toByteArray() {
    int dtypeSize = 0;
    byte[] tensorAsByteArray = null;
    if (dtype() == DType.UINT8) {
      dtypeSize = BYTE_SIZE_BYTES;
      tensorAsByteArray = new byte[(int) numel()];
      Tensor_uint8 thiz = (Tensor_uint8) this;
      ByteBuffer.wrap(tensorAsByteArray).put(thiz.getDataAsUnsignedByteArray());
    } else if (dtype() == DType.INT8) {
      dtypeSize = BYTE_SIZE_BYTES;
      tensorAsByteArray = new byte[(int) numel()];
      Tensor_int8 thiz = (Tensor_int8) this;
      ByteBuffer.wrap(tensorAsByteArray).put(thiz.getDataAsByteArray());
    } else if (dtype() == DType.HALF) {
      dtypeSize = HALF_SIZE_BYTES;
      tensorAsByteArray = new byte[(int) numel() * dtypeSize];
      Tensor_float16 thiz = (Tensor_float16) this;
      ByteBuffer.wrap(tensorAsByteArray).asShortBuffer().put(thiz.getDataAsShortArray());
    } else if (dtype() == DType.INT16) {
      throw new IllegalArgumentException("DType.INT16 is not supported in Java so far");
    } else if (dtype() == DType.INT32) {
      dtypeSize = INT_SIZE_BYTES;
      tensorAsByteArray = new byte[(int) numel() * dtypeSize];
      Tensor_int32 thiz = (Tensor_int32) this;
      ByteBuffer.wrap(tensorAsByteArray).asIntBuffer().put(thiz.getDataAsIntArray());
    } else if (dtype() == DType.INT64) {
      dtypeSize = LONG_SIZE_BYTES;
      tensorAsByteArray = new byte[(int) numel() * dtypeSize];
      Tensor_int64 thiz = (Tensor_int64) this;
      ByteBuffer.wrap(tensorAsByteArray).asLongBuffer().put(thiz.getDataAsLongArray());
    } else if (dtype() == DType.FLOAT) {
      dtypeSize = FLOAT_SIZE_BYTES;
      tensorAsByteArray = new byte[(int) numel() * dtypeSize];
      Tensor_float32 thiz = (Tensor_float32) this;
      ByteBuffer.wrap(tensorAsByteArray).asFloatBuffer().put(thiz.getDataAsFloatArray());
    } else if (dtype() == DType.DOUBLE) {
      dtypeSize = DOUBLE_SIZE_BYTES;
      tensorAsByteArray = new byte[(int) numel() * dtypeSize];
      Tensor_float64 thiz = (Tensor_float64) this;
      ByteBuffer.wrap(tensorAsByteArray).asDoubleBuffer().put(thiz.getDataAsDoubleArray());
    } else {
      throw new IllegalArgumentException("Unknown Tensor dtype");
    }
    ByteBuffer byteBuffer =
        ByteBuffer.allocate(1 + 1 + 4 * shape.length + dtypeSize * (int) numel());
    byteBuffer.put((byte) dtype().jniCode);
    byteBuffer.put((byte) shape.length);
    for (long s : shape) {
      byteBuffer.putInt((int) s);
    }
    byteBuffer.put(tensorAsByteArray);
    return byteBuffer.array();
  }

  /**
   * Deserializes a {@code Tensor} from a byte[]. Note: This method is experimental and subject to
   * change without notice. This does NOT supoprt list type.
   *
   * @param bytes The byte array to deserialize from.
   * @return The deserialized {@code Tensor}.
   */
  public static Tensor fromByteArray(byte[] bytes) {
    if (bytes == null) {
      throw new IllegalArgumentException("bytes cannot be null");
    }
    ByteBuffer buffer = ByteBuffer.wrap(bytes);
    if (!buffer.hasRemaining()) {
      throw new IllegalArgumentException("invalid buffer");
    }
    byte dtype = buffer.get();
    byte shapeLength = buffer.get();
    long[] shape = new long[(int) shapeLength];
    long numel = 1;
    for (int i = 0; i < shapeLength; i++) {
      int dim = buffer.getInt();
      if (dim < 0) {
        throw new IllegalArgumentException("invalid shape");
      }
      shape[i] = dim;
      numel *= dim;
    }
    if (dtype == DType.UINT8.jniCode) {
      return new Tensor_uint8(buffer, shape);
    } else if (dtype == DType.INT8.jniCode) {
      return new Tensor_int8(buffer, shape);
    } else if (dtype == DType.HALF.jniCode) {
      return new Tensor_float16(buffer.asShortBuffer(), shape);
    } else if (dtype == DType.INT32.jniCode) {
      return new Tensor_int32(buffer.asIntBuffer(), shape);
    } else if (dtype == DType.INT64.jniCode) {
      return new Tensor_int64(buffer.asLongBuffer(), shape);
    } else if (dtype == DType.FLOAT.jniCode) {
      return new Tensor_float32(buffer.asFloatBuffer(), shape);
    } else if (dtype == DType.DOUBLE.jniCode) {
      return new Tensor_float64(buffer.asDoubleBuffer(), shape);
    } else {
      throw new IllegalArgumentException("Unknown Tensor dtype");
    }
  }
}
