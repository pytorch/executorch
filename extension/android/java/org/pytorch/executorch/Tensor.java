// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

package org.pytorch.executorch;

import androidx.annotation.NonNull;

/**
 * Represents a tensor in ExecuTorch, managing its native memory and providing operations
 * to interact with it.
 *
 * This class is designed to be used by Android applications to create, manipulate,
 * and pass tensors to ExecuTorch models. It implements {@link AutoCloseable} to
 * ensure proper release of native resources.
 */
public class Tensor implements AutoCloseable {

    // The pointer to the native (C++) Tensor object.
    // A value of 0 indicates that the native object has been released.
    private long nativeHandle;

    // --- Existing Native Method Declarations ---
    private static native long nativeNew(Object data, long[] shape, int dtype);
    private static native void nativeRelease(long nativeHandle);
    private static native long[] nativeGetShape(long nativeHandle);
    private static native int nativeGetDtype(long nativeHandle);

    // --- NEW NATIVE DECLARATIONS ---
    private static native long nativeOnes(long[] shape, int dtype);
    private static native long nativeZeros(long[] shape, int dtype);

    /**
     * Constructs a Tensor object from a native handle.
     * This constructor is primarily used internally after JNI calls create a native tensor.
     *
     * @param nativeHandle The native pointer to the underlying C++ Tensor object. Must not be 0.
     * @throws IllegalArgumentException if the nativeHandle is 0.
     */
    public Tensor(long nativeHandle) {
        if (nativeHandle == 0) {
            throw new IllegalArgumentException("Native handle cannot be 0.");
        }
        this.nativeHandle = nativeHandle;
    }

    /**
     * Creates a new tensor from a flat array of float data and a shape.
     * The data type of the tensor will be {@code ScalarType.FLOAT}.
     *
     * @param data The flat array containing the tensor's float data.
     * @param shape The desired shape of the tensor. An empty array {@code new long[0]}
     *              represents a scalar (0-D) tensor.
     * @return A new Tensor.
     * @throws IllegalArgumentException if data is null, shape is null, or native allocation fails.
     */
    public static Tensor fromBlob(@NonNull float[] data, @NonNull long[] shape) {
        if (data == null) {
            throw new IllegalArgumentException("Data cannot be null.");
        }
        if (shape == null) {
            throw new IllegalArgumentException("Shape cannot be null.");
        }
        // It's generally good practice for the Java side to validate data.length against product(shape)
        // for early error detection, but we rely on the native side for now.
        long nativeHandle = nativeNew(data, shape, ScalarType.FLOAT.getValue());
        if (nativeHandle == 0) {
            throw new IllegalArgumentException("Failed to create native Tensor from float blob.");
        }
        return new Tensor(nativeHandle);
    }

    // TODO: Add other `fromBlob` overloads for different primitive types (e.g., int[], byte[]).

    /**
     * Returns the shape of the tensor.
     *
     * @return An array of long representing the dimensions of the tensor. An empty array signifies
     * a scalar (0-D) tensor.
     * @throws IllegalStateException if the tensor has been released.
     */
    @NonNull
    public long[] getShape() {
        if (nativeHandle == 0) {
            throw new IllegalStateException("Tensor has been released.");
        }
        return nativeGetShape(nativeHandle);
    }

    /**
     * Returns the data type of the tensor.
     *
     * @return The {@code ScalarType} enum value representing the tensor's data type.
     * @throws IllegalStateException if the tensor has been released.
     */
    @NonNull
    public ScalarType getDtype() {
        if (nativeHandle == 0) {
            throw new IllegalStateException("Tensor has been released.");
        }
        return ScalarType.fromValue(nativeGetDtype(nativeHandle));
    }

    // --- NEW PUBLIC STATIC CONVENIENCE METHODS ---

    /**
     * Creates a new tensor with the specified shape and fills it with ones.
     * The data type of the tensor will be {@code ScalarType.FLOAT} by default.
     *
     * @param shape The desired shape of the tensor. An empty array {@code new long[0]}
     *              represents a scalar (0-D) tensor.
     * @return A new Tensor filled with ones.
     * @throws IllegalArgumentException if the shape is null or native allocation fails.
     */
    public static Tensor ones(@NonNull long[] shape) {
        return ones(shape, ScalarType.FLOAT);
    }

    /**
     * Creates a new tensor with the specified shape and fills it with ones.
     *
     * @param shape The desired shape of the tensor. An empty array {@code new long[0]}
     *              represents a scalar (0-D) tensor.
     * @param dtype The desired data type of the tensor.
     * @return A new Tensor filled with ones.
     * @throws IllegalArgumentException if the shape is null, dtype is null, or native allocation fails.
     */
    public static Tensor ones(@NonNull long[] shape, @NonNull ScalarType dtype) {
        if (shape == null) {
            throw new IllegalArgumentException("Shape cannot be null.");
        }
        if (dtype == null) {
            throw new IllegalArgumentException("Dtype cannot be null.");
        }
        long nativeHandle = nativeOnes(shape, dtype.getValue());
        if (nativeHandle == 0) {
            throw new IllegalArgumentException("Failed to create native Tensor with ones.");
        }
        return new Tensor(nativeHandle);
    }

    /**
     * Creates a new tensor with the specified shape and fills it with zeros.
     * The data type of the tensor will be {@code ScalarType.FLOAT} by default.
     *
     * @param shape The desired shape of the tensor. An empty array {@code new long[0]}
     *              represents a scalar (0-D) tensor.
     * @return A new Tensor filled with zeros.
     * @throws IllegalArgumentException if the shape is null or native allocation fails.
     */
    public static Tensor zeros(@NonNull long[] shape) {
        return zeros(shape, ScalarType.FLOAT);
    }

    /**
     * Creates a new tensor with the specified shape and fills it with zeros.
     *
     * @param shape The desired shape of the tensor. An empty array {@code new long[0]}
     *              represents a scalar (0-D) tensor.
     * @param dtype The desired data type of the tensor.
     * @return A new Tensor filled with zeros.
     * @throws IllegalArgumentException if the shape is null, dtype is null, or native allocation fails.
     */
    public static Tensor zeros(@NonNull long[] shape, @NonNull ScalarType dtype) {
        if (shape == null) {
            throw new IllegalArgumentException("Shape cannot be null.");
        }
        if (dtype == null) {
            throw new IllegalArgumentException("Dtype cannot be null.");
        }
        long nativeHandle = nativeZeros(shape, dtype.getValue());
        if (nativeHandle == 0) {
            throw new IllegalArgumentException("Failed to create native Tensor with zeros.");
        }
        return new Tensor(nativeHandle);
    }

    /**
     * Releases the native resources associated with this tensor.
     * After this method is called, the Tensor object becomes invalid.
     * This method is automatically called when the Tensor is used in a try-with-resources statement.
     */
    @Override
    public void close() {
        if (nativeHandle != 0) {
            nativeRelease(nativeHandle);
            nativeHandle = 0;
        }
    }

    // Static initializer to load the JNI library.
    // In a typical Android application, the JNI library might be loaded
    // once in a higher-level entry point (e.g., Application class or a Module class).
    // This block ensures the library is loaded if not already.
    static {
        try {
            System.loadLibrary("executorch_android_jni");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Failed to load native library 'executorch_android_jni': " + e.getMessage());
            // For a core library class like Tensor, if native functionality is essential,
            // it's appropriate to rethrow the error to indicate a critical setup failure.
            throw e;
        }
    }
}