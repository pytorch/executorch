/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorchdemo.executor;

import com.facebook.jni.HybridData;
import com.facebook.jni.annotations.DoNotStrip;
import com.facebook.soloader.nativeloader.NativeLoader;
import java.util.Map;

class NativePeer implements INativePeer {
  static {
    // Loads libexecutorchdemo.so from jniLibs
    NativeLoader.loadLibrary("executorchdemo");
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
  public native EValue forward(EValue... inputs);
}
