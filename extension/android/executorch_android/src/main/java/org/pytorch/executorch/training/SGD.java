/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch.training;

import java.util.Map;
import org.pytorch.executorch.Tensor;
import org.pytorch.executorch.annotations.Experimental;

/**
 * Java wrapper for ExecuTorch SGD Optimizer.
 *
 * <p>Warning: These APIs are experimental and subject to change without notice
 */
@Experimental
public class SGD {

  static {
    // Loads libexecutorch.so from jniLibs
    System.loadLibrary("executorch");
  }

  private final long mNativeHandle;

  private static native long nativeInit(
      Map<String, Tensor> namedParameters,
      double learningRate,
      double momentum,
      double dampening,
      double weightDecay,
      boolean nesterov);

  private SGD(
      Map<String, Tensor> namedParameters,
      double learningRate,
      double momentum,
      double dampening,
      double weightDecay,
      boolean nesterov) {
    mNativeHandle =
        nativeInit(namedParameters, learningRate, momentum, dampening, weightDecay, nesterov);
  }

  /**
   * Creates a new SGD optimizer with the specified parameters and options.
   *
   * @param namedParameters Map of parameter names to tensors to be optimized
   * @param learningRate The learning rate for the optimizer
   * @param momentum The momentum value
   * @param dampening The dampening value
   * @param weightDecay The weight decay value
   * @param nesterov Whether to use Nesterov momentum
   * @return new {@link SGD} object
   */
  public static SGD create(
      Map<String, Tensor> namedParameters,
      double learningRate,
      double momentum,
      double dampening,
      double weightDecay,
      boolean nesterov) {
    return new SGD(namedParameters, learningRate, momentum, dampening, weightDecay, nesterov);
  }

  /**
   * Creates a new SGD optimizer with default options.
   *
   * @param namedParameters Map of parameter names to tensors to be optimized
   * @param learningRate The learning rate for the optimizer
   * @return new {@link SGD} object
   */
  public static SGD create(Map<String, Tensor> namedParameters, double learningRate) {
    return create(namedParameters, learningRate, 0.0, 0.0, 0.0, false);
  }

  /**
   * Performs a single optimization step using the provided gradients.
   *
   * @param namedGradients Map of parameter names to gradient tensors
   */
  public void step(Map<String, Tensor> namedGradients) {
    if (mNativeHandle == 0) {
      throw new RuntimeException("Attempt to use a destroyed SGD optimizer");
    }
    nativeStep(mNativeHandle, namedGradients);
  }

  private native void nativeStep(long handle, Map<String, Tensor> namedGradients);

  public void destroy() {
      nativeDestroy(mNativeHandle);
  }
  private native void nativeDestroy(long handle);
}
