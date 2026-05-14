/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch.training;

import com.facebook.jni.HybridData;
import com.facebook.jni.annotations.DoNotStrip;
import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;
import java.io.Closeable;
import java.util.Map;
import java.util.concurrent.locks.ReentrantLock;
import org.pytorch.executorch.EValue;
import org.pytorch.executorch.ExecuTorchRuntime;
import org.pytorch.executorch.Tensor;
import org.pytorch.executorch.annotations.Experimental;

/**
 * Java wrapper for ExecuTorch TrainingModule.
 *
 * <p>Warning: These APIs are experimental and subject to change without notice
 */
@Experimental
public class TrainingModule implements Closeable {

  static {
    if (!NativeLoader.isInitialized()) {
      NativeLoader.init(new SystemDelegate());
    }
    // Loads libexecutorch.so from jniLibs
    NativeLoader.loadLibrary("executorch");
  }

  private final HybridData mHybridData;
  private final ReentrantLock mLock = new ReentrantLock();
  private volatile boolean mDestroyed = false;

  @DoNotStrip
  private static native HybridData initHybrid(String moduleAbsolutePath, String dataAbsolutePath);

  private TrainingModule(String moduleAbsolutePath, String dataAbsolutePath) {
    mHybridData = initHybrid(moduleAbsolutePath, dataAbsolutePath);
  }

  private void checkNotDestroyed() {
    if (mDestroyed) throw new IllegalStateException("TrainingModule has been destroyed");
  }

  /**
   * Loads a serialized ExecuTorch Training Module from the specified path on the disk.
   *
   * @param modelPath path to file that contains the serialized ExecuTorch module.
   * @param dataPath path to file that contains the ExecuTorch module external weights.
   * @return new {@link TrainingModule} object which owns the model module.
   */
  public static TrainingModule load(final String modelPath, final String dataPath) {
    ExecuTorchRuntime.validateFilePath(modelPath, "model path");
    ExecuTorchRuntime.validateFilePath(dataPath, "data path");
    return new TrainingModule(modelPath, dataPath);
  }

  /**
   * Loads a serialized ExecuTorch training module from the specified path on the disk.
   *
   * @param modelPath path to file that contains the serialized ExecuTorch module. This PTE does not
   *     rely on external weights.
   * @return new {@link TrainingModule} object which owns the model module.
   */
  public static TrainingModule load(final String modelPath) {
    ExecuTorchRuntime.validateFilePath(modelPath, "model path");
    return new TrainingModule(modelPath, "");
  }

  /**
   * Runs the specified joint-graph method of this module with the specified arguments.
   *
   * @param methodName name of the ExecuTorch method to run.
   * @param inputs arguments that will be passed to ExecuTorch method.
   * @return return value(s) from the method.
   */
  public EValue[] executeForwardBackward(String methodName, EValue... inputs) {
    mLock.lock();
    try {
      checkNotDestroyed();
      return executeForwardBackwardNative(methodName, inputs);
    } finally {
      mLock.unlock();
    }
  }

  @DoNotStrip
  private native EValue[] executeForwardBackwardNative(String methodName, EValue... inputs);

  public Map<String, Tensor> namedParameters(String methodName) {
    mLock.lock();
    try {
      checkNotDestroyed();
      return namedParametersNative(methodName);
    } finally {
      mLock.unlock();
    }
  }

  @DoNotStrip
  private native Map<String, Tensor> namedParametersNative(String methodName);

  public Map<String, Tensor> namedGradients(String methodName) {
    mLock.lock();
    try {
      checkNotDestroyed();
      return namedGradientsNative(methodName);
    } finally {
      mLock.unlock();
    }
  }

  @DoNotStrip
  private native Map<String, Tensor> namedGradientsNative(String methodName);

  @Override
  public void close() {
    if (mLock.tryLock()) {
      try {
        if (!mDestroyed) {
          mDestroyed = true;
          mHybridData.resetNative();
        }
      } finally {
        mLock.unlock();
      }
    } else {
      throw new IllegalStateException("Cannot close module while method is executing");
    }
  }
}
