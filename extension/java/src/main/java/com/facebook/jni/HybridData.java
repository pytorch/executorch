/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.jni;

import com.facebook.jni.annotations.DoNotStrip;

/**
 * HybridData holds a C++ pointer created by fbjni.
 *
 * <p>This class is a shim for non-Android platforms. On Android, fbjni provides this class
 * directly. For desktop platforms, this provides the same interface so that Java code can work
 * unchanged across platforms.
 *
 * <p>The actual implementation is provided by fbjni's native code.
 */
@DoNotStrip
public class HybridData {

  @DoNotStrip private long mNativePointer;

  static {
    // Ensure native library is loaded before any HybridData operations
    org.pytorch.executorch.NativeLibraryLoader.loadLibrary("executorch_jni");
  }

  /** Check if the native object is still valid (not destroyed). */
  public boolean isValid() {
    return mNativePointer != 0;
  }

  /**
   * Explicitly release the C++ object. After calling this, {@link #isValid()} will return false.
   *
   * <p>This is safe to call multiple times.
   */
  public synchronized native void resetNative();

  /**
   * Release native resources when garbage collected. Users should prefer calling {@link
   * #resetNative()} explicitly when the lifecycle is known.
   */
  @Override
  protected void finalize() throws Throwable {
    resetNative();
    super.finalize();
  }
}
