/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch;

import com.facebook.jni.HybridData;
import com.facebook.jni.annotations.DoNotStrip;
import com.facebook.soloader.nativeloader.NativeLoader;
import java.util.Map;

class NativePeer implements INativePeer {
  static {
    // Loads libexecutorch.so from jniLibs
    NativeLoader.loadLibrary("executorch");
  }

  private final HybridData mHybridData;

  @DoNotStrip
  private static native HybridData initHybrid(
      String moduleAbsolutePath, Map<String, String> extraFiles);

  NativePeer(String moduleAbsolutePath, Map<String, String> extraFiles) {
    mHybridData = initHybrid(moduleAbsolutePath, extraFiles);
  }

  public void resetNative() {
    mHybridData.resetNative();
  }

  @DoNotStrip
  public native EValue[] forward(EValue... inputs);

  @DoNotStrip
  public native EValue[] execute(String methodName, EValue... inputs);

  @DoNotStrip
  public native int loadMethod(String methodName);
}
