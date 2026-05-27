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
import java.io.Closeable
import java.util.concurrent.locks.ReentrantLock
import org.pytorch.executorch.EValue
import org.pytorch.executorch.ExecuTorchRuntime
import org.pytorch.executorch.Tensor
import org.pytorch.executorch.annotations.Experimental

/**
 * Kotlin wrapper for ExecuTorch TrainingModule.
 *
 * Warning: These APIs are experimental and subject to change without notice
 */
@Experimental
class TrainingModule private constructor(moduleAbsolutePath: String, dataAbsolutePath: String) :
    Closeable {

  private val mHybridData: HybridData = initHybrid(moduleAbsolutePath, dataAbsolutePath)
  private val mLock = ReentrantLock()

  @Volatile private var mDestroyed = false

  private fun checkNotDestroyed() {
    check(!mDestroyed) { "TrainingModule has been destroyed" }
  }

  /**
   * Runs the specified joint-graph method of this module with the specified arguments.
   *
   * @param methodName name of the ExecuTorch method to run.
   * @param inputs arguments that will be passed to ExecuTorch method.
   * @return return value(s) from the method.
   */
  fun executeForwardBackward(methodName: String, vararg inputs: EValue): Array<EValue> {
    mLock.lock()
    try {
      checkNotDestroyed()
      return executeForwardBackwardNative(methodName, *inputs)
    } finally {
      mLock.unlock()
    }
  }

  @DoNotStrip
  private external fun executeForwardBackwardNative(
      methodName: String,
      vararg inputs: EValue,
  ): Array<EValue>

  fun namedParameters(methodName: String): Map<String, Tensor> {
    mLock.lock()
    try {
      checkNotDestroyed()
      return namedParametersNative(methodName)
    } finally {
      mLock.unlock()
    }
  }

  @DoNotStrip private external fun namedParametersNative(methodName: String): Map<String, Tensor>

  fun namedGradients(methodName: String): Map<String, Tensor> {
    mLock.lock()
    try {
      checkNotDestroyed()
      return namedGradientsNative(methodName)
    } finally {
      mLock.unlock()
    }
  }

  @DoNotStrip private external fun namedGradientsNative(methodName: String): Map<String, Tensor>

  override fun close() {
    if (mLock.tryLock()) {
      try {
        if (!mDestroyed) {
          mDestroyed = true
          mHybridData.resetNative()
        }
      } finally {
        mLock.unlock()
      }
    } else {
      throw IllegalStateException("Cannot close module while method is executing")
    }
  }

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
        moduleAbsolutePath: String,
        dataAbsolutePath: String,
    ): HybridData

    /**
     * Loads a serialized ExecuTorch Training Module from the specified path on the disk.
     *
     * @param modelPath path to file that contains the serialized ExecuTorch module.
     * @param dataPath path to file that contains the ExecuTorch module external weights.
     * @return new [TrainingModule] object which owns the model module.
     */
    @JvmStatic
    fun load(modelPath: String, dataPath: String): TrainingModule {
      ExecuTorchRuntime.validateFilePath(modelPath, "model path")
      ExecuTorchRuntime.validateFilePath(dataPath, "data path")
      return TrainingModule(modelPath, dataPath)
    }

    /**
     * Loads a serialized ExecuTorch training module from the specified path on the disk.
     *
     * @param modelPath path to file that contains the serialized ExecuTorch module. This PTE does
     *   not rely on external weights.
     * @return new [TrainingModule] object which owns the model module.
     */
    @JvmStatic
    fun load(modelPath: String): TrainingModule {
      ExecuTorchRuntime.validateFilePath(modelPath, "model path")
      return TrainingModule(modelPath, "")
    }
  }
}
