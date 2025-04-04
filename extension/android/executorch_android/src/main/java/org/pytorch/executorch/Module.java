/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch;

import android.util.Log;
import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;
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

  /** Load mode for the module. Load the whole file as a buffer. */
  public static final int LOAD_MODE_FILE = 0;

  /** Load mode for the module. Use mmap to load pages into memory. */
  public static final int LOAD_MODE_MMAP = 1;

  /** Load mode for the module. Use memory locking and handle errors. */
  public static final int LOAD_MODE_MMAP_USE_MLOCK = 2;

  /** Load mode for the module. Use memory locking and ignore errors. */
  public static final int LOAD_MODE_MMAP_USE_MLOCK_IGNORE_ERRORS = 3;

  /** Reference to the NativePeer object of this module. */
  private NativePeer mNativePeer;

  /** Lock protecting the non-thread safe methods in NativePeer. */
  private Lock mLock = new ReentrantLock();

  /**
   * Loads a serialized ExecuTorch module from the specified path on the disk.
   *
   * @param modelPath path to file that contains the serialized ExecuTorch module.
   * @param loadMode load mode for the module. See constants in {@link Module}.
   * @return new {@link org.pytorch.executorch.Module} object which owns the model module.
   */
  public static Module load(final String modelPath, int loadMode) {
    if (!NativeLoader.isInitialized()) {
      NativeLoader.init(new SystemDelegate());
    }
    return new Module(new NativePeer(modelPath, loadMode));
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

  Module(NativePeer nativePeer) {
    this.mNativePeer = nativePeer;
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
    try {
      mLock.lock();
      if (mNativePeer == null) {
        Log.e("ExecuTorch", "Attempt to use a destroyed module");
        return new EValue[0];
      }
      return mNativePeer.forward(inputs);
    } finally {
      mLock.unlock();
    }
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
      if (mNativePeer == null) {
        Log.e("ExecuTorch", "Attempt to use a destroyed module");
        return new EValue[0];
      }
      return mNativePeer.execute(methodName, inputs);
    } finally {
      mLock.unlock();
    }
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
    try {
      mLock.lock();
      if (mNativePeer == null) {
        Log.e("ExecuTorch", "Attempt to use a destroyed module");
        return 0x2; // InvalidState
      }
      return mNativePeer.loadMethod(methodName);
    } finally {
      mLock.unlock();
    }
  }

  /** Retrieve the in-memory log buffer, containing the most recent ExecuTorch log entries. */
  public String[] readLogBuffer() {
    return mNativePeer.readLogBuffer();
  }

  /**
   * Explicitly destroys the native torch::jit::Module. Calling this method is not required, as the
   * native object will be destroyed when this object is garbage-collected. However, the timing of
   * garbage collection is not guaranteed, so proactively calling {@code destroy} can free memory
   * more quickly. See {@link com.facebook.jni.HybridData#resetNative}.
   */
  public void destroy() {
    if (mLock.tryLock()) {
      try {
        mNativePeer.resetNative();
      } finally {
        mNativePeer = null;
        mLock.unlock();
      }
    } else {
      mNativePeer = null;
      Log.w(
          "ExecuTorch",
          "Destroy was called while the module was in use. Resources will not be immediately"
              + " released.");
    }
  }
}
