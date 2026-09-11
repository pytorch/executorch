/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch.training

import com.facebook.jni.HybridData
import com.facebook.jni.annotations.DoNotStrip
import com.facebook.soloader.nativeloader.NativeLoader
import com.facebook.soloader.nativeloader.SystemDelegate
import org.pytorch.executorch.Tensor
import org.pytorch.executorch.annotations.Experimental

/**
 * Kotlin wrapper for ExecuTorch SGD Optimizer.
 *
 * Warning: These APIs are experimental and subject to change without notice
 */
@Experimental
class SGD
private constructor(
    namedParameters: Map<String, Tensor>,
    learningRate: Double,
    momentum: Double,
    dampening: Double,
    weightDecay: Double,
    nesterov: Boolean,
) {

  private val mHybridData: HybridData =
      initHybrid(namedParameters, learningRate, momentum, dampening, weightDecay, nesterov)

  /**
   * Performs a single optimization step using the provided gradients.
   *
   * @param namedGradients Map of parameter names to gradient tensors
   */
  fun step(namedGradients: Map<String, Tensor>) {
    check(mHybridData.isValid) { "SGD optimizer has been destroyed" }
    stepNative(namedGradients)
  }

  @DoNotStrip private external fun stepNative(namedGradients: Map<String, Tensor>)

  companion object {
    init {
      if (!NativeLoader.isInitialized()) {
        NativeLoader.init(SystemDelegate())
      }
      NativeLoader.loadLibrary("executorch")
    }

    @DoNotStrip
    @JvmStatic
    private external fun initHybrid(
        namedParameters: Map<String, Tensor>,
        learningRate: Double,
        momentum: Double,
        dampening: Double,
        weightDecay: Double,
        nesterov: Boolean,
    ): HybridData

    /**
     * Creates a new SGD optimizer with the specified parameters and options.
     *
     * @param namedParameters Map of parameter names to tensors to be optimized
     * @param learningRate The learning rate for the optimizer
     * @param momentum The momentum value
     * @param dampening The dampening value
     * @param weightDecay The weight decay value
     * @param nesterov Whether to use Nesterov momentum
     * @return new [SGD] object
     */
    @JvmStatic
    fun create(
        namedParameters: Map<String, Tensor>,
        learningRate: Double,
        momentum: Double,
        dampening: Double,
        weightDecay: Double,
        nesterov: Boolean,
    ): SGD = SGD(namedParameters, learningRate, momentum, dampening, weightDecay, nesterov)

    /**
     * Creates a new SGD optimizer with default options.
     *
     * @param namedParameters Map of parameter names to tensors to be optimized
     * @param learningRate The learning rate for the optimizer
     * @return new [SGD] object
     */
    @JvmStatic
    fun create(namedParameters: Map<String, Tensor>, learningRate: Double): SGD =
        create(namedParameters, learningRate, 0.0, 0.0, 0.0, false)
  }
}
