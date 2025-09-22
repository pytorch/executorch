/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch;

import android.util.Log;
import com.facebook.jni.HybridData;
import com.facebook.jni.annotations.DoNotStrip;
import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;
import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import org.pytorch.executorch.annotations.Experimental;

/**
 * Java wrapper for ExecuTorch Module.
 *
 * <p>Warning: These APIs are experimental and subject to change without notice
 */
@Experimental
public class Module {

  static {
    if (!NativeLoader.isInitialized()) {
      NativeLoader.init(new SystemDelegate());
    }
    // Loads libexecutorch.so from jniLibs
    NativeLoader.loadLibrary("executorch");
  }

  /** Load mode for the module. Load the whole file as a buffer. */
  public static final int LOAD_MODE_FILE = 0;

  /** Load mode for the module. Use mmap to load pages into memory. */
  public static final int LOAD_MODE_MMAP = 1;

  /** Load mode for the module. Use memory locking and handle errors. */
  public static final int LOAD_MODE_MMAP_USE_MLOCK = 2;

  /** Load mode for the module. Use memory locking and ignore errors. */
  public static final int LOAD_MODE_MMAP_USE_MLOCK_IGNORE_ERRORS = 3;

  private final HybridData mHybridData;

  private final Map<String, MethodMetadata> mMethodMetadata;

  @DoNotStrip
  private static native HybridData initHybrid(
      String moduleAbsolutePath, int loadMode, int initHybrid);

  private Module(String moduleAbsolutePath, int loadMode, int numThreads) {
    ExecuTorchRuntime runtime = ExecuTorchRuntime.getRuntime();

    mHybridData = initHybrid(moduleAbsolutePath, loadMode, numThreads);

    mMethodMetadata = populateMethodMeta();
  }

  Map<String, MethodMetadata> populateMethodMeta() {
    String[] methods = getMethods();
    Map<String, MethodMetadata> metadata = new HashMap<String, MethodMetadata>();
    for (int i = 0; i < methods.length; i++) {
      String name = methods[i];
      metadata.put(name, new MethodMetadata().setName(name));
    }

    return metadata;
  }

  /** Lock protecting the non-thread safe methods in mHybridData. */
  private Lock mLock = new ReentrantLock();

  /**
   * Loads a serialized ExecuTorch module from the specified path on the disk.
   *
   * @param modelPath path to file that contains the serialized ExecuTorch module.
   * @param loadMode load mode for the module. See constants in {@link Module}.
   * @return new {@link org.pytorch.executorch.Module} object which owns the model module.
   */
  public static Module load(final String modelPath, int loadMode) {
    return load(modelPath, loadMode, 0);
  }

  /**
   * Loads a serialized ExecuTorch module from the specified path on the disk.
   *
   * @param modelPath path to file that contains the serialized ExecuTorch module.
   * @param loadMode load mode for the module. See constants in {@link Module}.
   * @param numThreads the number of threads to use for inference. A value of 0 defaults to a
   *     hardware-specific default.
   * @return new {@link org.pytorch.executorch.Module} object which owns the model module.
   */
  public static Module load(final String modelPath, int loadMode, int numThreads) {
    File modelFile = new File(modelPath);
    if (!modelFile.canRead() || !modelFile.isFile()) {
      throw new RuntimeException("Cannot load model path " + modelPath);
    }
    return new Module(modelPath, loadMode, numThreads);
  }

  /**
   * Loads a serialized ExecuTorch module from the specified path on the disk to run on CPU.
   *
   * @param modelPath path to file that contains the serialized ExecuTorch module.
   * @return new {@link org.pytorch.executorch.Module} object which owns the model module.
   */
  public static Module load(final String modelPath) {
    return load(modelPath, LOAD_MODE_FILE);
  }

  /**
   * Runs the 'forward' method of this module with the specified arguments.
   *
   * @param inputs arguments for the ExecuTorch module's 'forward' method. Note: if method 'forward'
   *     requires inputs but no inputs are given, the function will not error out, but run 'forward'
   *     with sample inputs.
   * @return return value from the 'forward' method.
   */
  public EValue[] forward(EValue... inputs) {
    return execute("forward", inputs);
  }

  /**
   * Runs the specified method of this module with the specified arguments.
   *
   * @param methodName name of the ExecuTorch method to run.
   * @param inputs arguments that will be passed to ExecuTorch method.
   * @return return value from the method.
   */
  public EValue[] execute(String methodName, EValue... inputs) {
    try {
      mLock.lock();
      if (!mHybridData.isValid()) {
        Log.e("ExecuTorch", "Attempt to use a destroyed module");
        return new EValue[0];
      }
      return executeNative(methodName, inputs);
    } finally {
      mLock.unlock();
    }
  }

  @DoNotStrip
  private native EValue[] executeNative(String methodName, EValue... inputs);

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
    try {
      mLock.lock();
      if (!mHybridData.isValid()) {
        Log.e("ExecuTorch", "Attempt to use a destroyed module");
        return 0x2; // InvalidState
      }
      return loadMethodNative(methodName);
    } finally {
      mLock.unlock();
    }
  }

  @DoNotStrip
  private native int loadMethodNative(String methodName);

  /**
   * Returns the names of the backends in a certain method.
   *
   * @param methodName method name to query
   * @return an array of backend name
   */
  @DoNotStrip
  private native String[] getUsedBackends(String methodName);

  /**
   * Returns the names of methods.
   *
   * @return name of methods in this Module
   */
  @DoNotStrip
  public native String[] getMethods();

  /**
   * Get the corresponding @MethodMetadata for a method
   *
   * @param name method name
   * @return @MethodMetadata for this method
   */
  public MethodMetadata getMethodMetadata(String name) {
    if (!mMethodMetadata.containsKey(name)) {
      throw new RuntimeException("method " + name + "does not exist for this module");
    }

    MethodMetadata methodMetadata = mMethodMetadata.get(name);
    if (methodMetadata != null) {
      methodMetadata.setBackends(getUsedBackends(name));
    }
    return methodMetadata;
  }

  /** Retrieve the in-memory log buffer, containing the most recent ExecuTorch log entries. */
  public String[] readLogBuffer() {
    return readLogBufferNative();
  }

  @DoNotStrip
  private native String[] readLogBufferNative();

  /**
   * Dump the ExecuTorch ETRecord file to /data/local/tmp/result.etdump.
   *
   * <p>Currently for internal (minibench) use only.
   *
   * @return true if the etdump was successfully written, false otherwise.
   */
  @Experimental
  @DoNotStrip
  public native boolean etdump();

  /**
   * Explicitly destroys the native Module object. Calling this method is not required, as the
   * native object will be destroyed when this object is garbage-collected. However, the timing of
   * garbage collection is not guaranteed, so proactively calling {@code destroy} can free memory
   * more quickly. See {@link com.facebook.jni.HybridData#resetNative}.
   */
  public void destroy() {
    if (mLock.tryLock()) {
      try {
        mHybridData.resetNative();
      } finally {
        mLock.unlock();
      }
    } else {
      Log.w(
          "ExecuTorch",
          "Destroy was called while the module was in use. Resources will not be immediately"
              + " released.");
    }
  }
}
