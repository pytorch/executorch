/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch;

import android.util.Log;

/**
 * ETDump provides runtime control for ExecuTorch profiling.
 *
 * <p>Enable profiling before loading models to capture execution traces. Use
 * Module.writeETDumpToPath() to write profiling data to custom locations.
 *
 * <p>Example usage:
 *
 * <pre>{@code
 * // Enable profiling
 * ETDump.enableProfiling();
 *
 * // Load and run model
 * Module module = Module.load("model.pte");
 * module.forward(inputs);
 *
 * // Write profiling data to custom path (no root access needed!)
 * module.writeETDumpToPath(getCacheDir() + "/profile.etdump");
 *
 * // Disable profiling
 * ETDump.disableProfiling();
 * }</pre>
 */
public class ETDump {
  private static final String TAG = "ExecuTorch-ETDump";

  static {
    try {
      System.loadLibrary("executorch");
      nativeInit();
    } catch (UnsatisfiedLinkError e) {
      Log.e(TAG, "Failed to load executorch library", e);
      throw e;
    }
  }

  /** Initialize the ETDump subsystem. Called automatically when the class is loaded. */
  private static native void nativeInit();

  /**
   * Enable profiling for subsequently loaded models.
   *
   * <p>Modules loaded after calling this method will capture profiling data. This has zero runtime
   * overhead until a model is actually loaded.
   *
   * @return true if profiling was successfully enabled
   */
  public static boolean enableProfiling() {
    boolean result = nativeEnableProfiling();
    if (result) {
      Log.i(TAG, "Profiling enabled");
    } else {
      Log.e(TAG, "Failed to enable profiling");
    }
    return result;
  }

  /**
   * Disable profiling.
   *
   * <p>Modules loaded after calling this method will not capture profiling data.
   */
  public static void disableProfiling() {
    nativeDisableProfiling();
    Log.i(TAG, "Profiling disabled");
  }

  /**
   * Check if profiling is currently enabled.
   *
   * @return true if profiling is enabled
   */
  public static boolean isProfilingEnabled() {
    return nativeIsProfilingEnabled();
  }

  // Native methods
  private static native boolean nativeEnableProfiling();

  private static native void nativeDisableProfiling();

  private static native boolean nativeIsProfilingEnabled();
}
