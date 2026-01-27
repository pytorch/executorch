/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch.training;

import java.util.logging.Logger;
import java.util.logging.Level;
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
  private static final Logger LOGGER = Logger.getLogger(TrainingModule.class.getName());

  static {
    // Loads libexecutorch.so from jniLibs
    System.loadLibrary("executorch");
  }

  private final long mNativeHandle;

  private static native long nativeInit(String moduleAbsolutePath, String dataAbsolutePath);

  private TrainingModule(String moduleAbsolutePath, String dataAbsolutePath) {
    mNativeHandle = nativeInit(moduleAbsolutePath, dataAbsolutePath);
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
      LOGGER.log(Level.SEVERE, "Attempt to use a destroyed module");
      return new EValue[0];
    }
    return nativeExecuteForwardBackward(mNativeHandle, methodName, inputs);
  }

  private native EValue[] nativeExecuteForwardBackward(long handle, String methodName, EValue... inputs);

  public Map<String, Tensor> namedParameters(String methodName) {
    if (mNativeHandle == 0) {
      LOGGER.log(Level.SEVERE, "Attempt to use a destroyed module");
      return new HashMap<String, Tensor>();
    }
    return nativeNamedParameters(mNativeHandle, methodName);
  }

  private native Map<String, Tensor> nativeNamedParameters(long handle, String methodName);

  public Map<String, Tensor> namedGradients(String methodName) {
    if (mNativeHandle == 0) {
      LOGGER.log(Level.SEVERE, "Attempt to use a destroyed module");
      return new HashMap<String, Tensor>();
    }
    return nativeNamedGradients(mNativeHandle, methodName);
  }

  private native Map<String, Tensor> nativeNamedGradients(long handle, String methodName);

  public void destroy() {
      nativeDestroy(mNativeHandle);
  }
  private native void nativeDestroy(long handle);
}
