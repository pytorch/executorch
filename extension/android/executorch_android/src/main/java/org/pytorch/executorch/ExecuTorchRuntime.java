/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch;

import android.app.ActivityManager;
import android.content.Context;
import com.facebook.jni.annotations.DoNotStrip;
import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;
import java.io.File;

/** Class for entire ExecuTorch Runtime related functions. */
public class ExecuTorchRuntime {

  static {
    if (!NativeLoader.isInitialized()) {
      NativeLoader.init(new SystemDelegate());
    }
    // Loads libexecutorch.so from jniLibs
    NativeLoader.loadLibrary("executorch");
  }

  private static final ExecuTorchRuntime sInstance = new ExecuTorchRuntime();

  private ExecuTorchRuntime() {}

  /** Get the runtime instance. */
  public static ExecuTorchRuntime getRuntime() {
    return sInstance;
  }

  /**
   * Validates that the given path points to a readable file.
   *
   * @throws RuntimeException if the file does not exist or is not readable.
   */
  public static void validateFilePath(String path, String description) {
    File file = new File(path);
    if (!file.canRead() || !file.isFile()) {
      throw new RuntimeException("Cannot load " + description + " " + path);
    }
  }

  /**
   * Heuristic check: compares reported available memory against model file size. A true result does
   * not guarantee that loading will succeed (runtime overhead, fragmentation, other allocations may
   * still cause OOM). A false result may indicate the model is too large, but can be a false negative
   * (for example with mmap-based loads, shared page cache behavior, or compressed model files).
   *
   * @param context Android context for accessing system services (must not be null)
   * @param modelPath Path to the model file (must not be null or empty)
   * @return true if reported available memory exceeds the file size, false otherwise
   * @throws IllegalArgumentException if context is null or modelPath is null/empty
   * @throws RuntimeException if the file does not exist or is not readable
   * @throws IllegalStateException if the ActivityManager system service is unavailable
   */
  public static boolean checkMemoryFit(Context context, String modelPath) {
    if (context == null) {
      throw new IllegalArgumentException("context must not be null");
    }
    if (modelPath == null || modelPath.isEmpty()) {
      throw new IllegalArgumentException("modelPath must not be null or empty");
    }
    validateFilePath(modelPath, "model file");
    long fileSize = new File(modelPath).length();
    ActivityManager am = (ActivityManager) context.getSystemService(Context.ACTIVITY_SERVICE);
    if (am == null) {
      throw new IllegalStateException("ActivityManager system service is not available");
    }
    ActivityManager.MemoryInfo memInfo = new ActivityManager.MemoryInfo();
    am.getMemoryInfo(memInfo);
    return memInfo.availMem > fileSize;
  }

  /** Get all registered ops. */
  @DoNotStrip
  public static native String[] getRegisteredOps();

  /** Get all registered backends. */
  @DoNotStrip
  public static native String[] getRegisteredBackends();
}
