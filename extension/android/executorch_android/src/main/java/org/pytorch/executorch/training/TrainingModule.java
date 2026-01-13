/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch.training;

import android.util.Log;
import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;
import java.io.File;
import java.util.HashMap;
import java.util.Map;
import org.pytorch.executorch.EValue;
import org.pytorch.executorch.Tensor;
import org.pytorch.executorch.annotations.Experimental;

/**
 * Java wrapper for ExecuTorch TrainingModule.
 *
 * <p>Warning: These APIs are experimental and subject to change without notice
 */
@Experimental
public class TrainingModule {

  static {
    if (!NativeLoader.isInitialized()) {
      NativeLoader.init(new SystemDelegate());
    }
    // Loads libexecutorch.so from jniLibs
    NativeLoader.loadLibrary("executorch");
  }

  private long mNativeHandle;

  private static native long initHybrid(String moduleAbsolutePath, String dataAbsolutePath);

  private static native void nativeDestroy(long nativeHandle);

  private TrainingModule(String moduleAbsolutePath, String dataAbsolutePath) {
    mNativeHandle = initHybrid(moduleAbsolutePath, dataAbsolutePath);
  }

  /**
   * Loads a serialized ExecuTorch Training Module from the specified path on the disk.
   *
   * @param modelPath path to file that contains the serialized ExecuTorch module.
   * @param dataPath path to file that contains the ExecuTorch module external weights.
   * @return new {@link TrainingModule} object which owns the model module.
   */
  public static TrainingModule load(final String modelPath, final String dataPath) {
    File modelFile = new File(modelPath);
    if (!modelFile.canRead() || !modelFile.isFile()) {
      throw new RuntimeException("Cannot load model path!! " + modelPath);
    }
    File dataFile = new File(dataPath);
    if (!dataFile.canRead() || !dataFile.isFile()) {
      throw new RuntimeException("Cannot load data path!! " + dataPath);
    }
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
    File modelFile = new File(modelPath);
    if (!modelFile.canRead() || !modelFile.isFile()) {
      throw new RuntimeException("Cannot load model path!! " + modelPath);
    }
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
    if (mNativeHandle == 0) {
      Log.e("ExecuTorch", "Attempt to use a destroyed module");
      return new EValue[0];
    }
    return executeForwardBackwardNative(mNativeHandle, methodName, inputs);
  }

  private static native EValue[] executeForwardBackwardNative(
      long nativeHandle, String methodName, EValue... inputs);

  public Map<String, Tensor> namedParameters(String methodName) {
    if (mNativeHandle == 0) {
      Log.e("ExecuTorch", "Attempt to use a destroyed module");
      return new HashMap<String, Tensor>();
    }
    return namedParametersNative(mNativeHandle, methodName);
  }

  private static native Map<String, Tensor> namedParametersNative(
      long nativeHandle, String methodName);

  public Map<String, Tensor> namedGradients(String methodName) {
    if (mNativeHandle == 0) {
      Log.e("ExecuTorch", "Attempt to use a destroyed module");
      return new HashMap<String, Tensor>();
    }
    return namedGradientsNative(mNativeHandle, methodName);
  }

  private static native Map<String, Tensor> namedGradientsNative(
      long nativeHandle, String methodName);

  @Override
  protected void finalize() throws Throwable {
    try {
      if (mNativeHandle != 0) {
        nativeDestroy(mNativeHandle);
        mNativeHandle = 0;
      }
    } finally {
      super.finalize();
    }
  }
}
