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
  private static final int DEFAULT_SEQ_LEN = 128;

  @DoNotStrip
  private static native HybridData initHybrid(
      String modulePath, String tokenizerPath, float temperature);

  /** Constructs a LLAMA Module for a model with given path, tokenizer, and temperature. */
  public LlamaModule(String modulePath, String tokenizerPath, float temperature) {
    mHybridData = initHybrid(modulePath, tokenizerPath, temperature);
  }

  public void resetNative() {
    mHybridData.resetNative();
  }

  /**
   * Start generating tokens from the module.
   *
   * @param prompt Input prompt
   * @param llamaCallback callback object to receive results.
   */
  public int generate(String prompt, LlamaCallback llamaCallback) {
    return generate(prompt, DEFAULT_SEQ_LEN, llamaCallback);
  }

  /**
   * Start generating tokens from the module.
   *
   * @param prompt Input prompt
   * @param seqLen sequence length
   * @param llamaCallback callback object to receive results.
   */
  @DoNotStrip
  public native int generate(String prompt, int seqLen, LlamaCallback llamaCallback);

  /** Stop current generate() before it finishes. */
  @DoNotStrip
  public native void stop();

  /** Force loading the module. Otherwise the model is loaded during first generate(). */
  @DoNotStrip
  public native int load();
}
