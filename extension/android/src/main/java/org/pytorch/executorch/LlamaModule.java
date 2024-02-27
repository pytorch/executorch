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
import com.facebook.soloader.nativeloader.SystemDelegate;

public class LlamaModule {
  static {
    if (!NativeLoader.isInitialized()) {
      NativeLoader.init(new SystemDelegate());
    }
    NativeLoader.loadLibrary("executorch_llama_jni");
  }

  private final HybridData mHybridData;

  @DoNotStrip
  private static native HybridData initHybrid(
      String modulePath, String tokenizerPath, float temperature);

  public LlamaModule(String modulePath, String tokenizerPath, float temperature) {
    mHybridData = initHybrid(modulePath, tokenizerPath, temperature);
  }

  public void resetNative() {
    mHybridData.resetNative();
  }

  @DoNotStrip
  public native int generate(String prompt, LlamaCallback llamaCallback);

  @DoNotStrip
  public native void stop();
}
