/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch;

import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;
import java.util.Map;

/** Java wrapper for ExecuTorch Module. */
public class Module {

  /** Reference to the INativePeer object of this module. */
  private INativePeer mNativePeer;

  /**
   * Loads a serialized ExecuTorch module from the specified path on the disk.
   *
   * @param modelPath path to file that contains the serialized ExecuTorch module.
   * @param extraFiles map with extra files names as keys, content of them will be loaded to values.
   * @return new {@link org.pytorch.executorch.Module} object which owns torch::jit::Module.
   */
  public static Module load(final String modelPath, final Map<String, String> extraFiles) {
    if (!NativeLoader.isInitialized()) {
      NativeLoader.init(new SystemDelegate());
    }
    return new Module(new NativePeer(modelPath, extraFiles));
  }

  /**
   * Loads a serialized ExecuTorch module from the specified path on the disk to run on CPU.
   *
   * @param modelPath path to file that contains the serialized ExecuTorch module.
   * @return new {@link org.pytorch.executorch.Module} object which owns torch::jit::Module.
   */
  public static Module load(final String modelPath) {
    return load(modelPath, null);
  }

  Module(INativePeer nativePeer) {
    this.mNativePeer = nativePeer;
  }

  /**
   * Runs the 'forward' method of this module with the specified arguments.
   *
   * @param inputs arguments for the ExecuTorch module's 'forward' method.
   * @return return value from the 'forward' method.
   */
  public EValue[] forward(EValue... inputs) {
    return mNativePeer.forward(inputs);
  }

  /**
   * Runs the specified method of this module with the specified arguments.
   *
   * @param methodName name of the ExecuTorch method to run.
   * @param inputs arguments that will be passed to ExecuTorch method.
   * @return return value from the method.
   */
  public EValue[] execute(String methodName, EValue... inputs) {
    return mNativePeer.execute(methodName, inputs);
  }

  /**
   * Load a method on this module. This might help with the first time inference performance,
   * because otherwise the method is loaded lazily when it's execute. Note: this function is
   * synchronous, and will block until the method is loaded. Therefore, it is recommended to call
   * this on a background thread. However, users need to make sure that they don't execute before
   * this function returns.
   *
   * @return the Error code if there was an error loading the method
   */
  public int loadMethod(String methodName) {
    return mNativePeer.loadMethod(methodName);
  }

  /**
   * Explicitly destroys the native torch::jit::Module. Calling this method is not required, as the
   * native object will be destroyed when this object is garbage-collected. However, the timing of
   * garbage collection is not guaranteed, so proactively calling {@code destroy} can free memory
   * more quickly. See {@link com.facebook.jni.HybridData#resetNative}.
   */
  public void destroy() {
    mNativePeer.resetNative();
  }
}
