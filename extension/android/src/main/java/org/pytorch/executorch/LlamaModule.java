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

  public static final int MODEL_TYPE_TEXT = 1;
  public static final int MODEL_TYPE_TEXT_VISION = 2;

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
      int modelType, String modulePath, String tokenizerPath, float temperature);

  /** Constructs a LLAMA Module for a model with given path, tokenizer, and temperature. */
    public LlamaModule(String modulePath, String tokenizerPath, float temperature) {
    mHybridData = initHybrid(MODEL_TYPE_TEXT, modulePath, tokenizerPath, temperature);
  }

  /** Constructs a LLM Module for a model with given path, tokenizer, and temperature. */
  public LlamaModule(int modelType, String modulePath, String tokenizerPath, float temperature) {
    mHybridData = initHybrid(modelType, modulePath, tokenizerPath, temperature);
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
  public int generate(String prompt, int seqLen, LlamaCallback llamaCallback) {
    return generate(null, 0, 0, 0, prompt, seqLen, llamaCallback);
  }

  /** Start generating tokens from the module.
   * @param image Input image as a byte array
   * @param width Input image width
   * @param height Input image height
   * @param channels Input image number of channels
   * @param prompt Input prompt
   * @param seqLen sequence length
   * @param llamaCallback callback object to receive results.
   */
  @DoNotStrip
  public native int generate(int[] image, int width, int height, int channels, String prompt, int seqLen, LlamaCallback llamaCallback);

  /** Stop current generate() before it finishes. */
  @DoNotStrip
  public native void stop();

  /** Force loading the module. Otherwise the model is loaded during first generate(). */
  @DoNotStrip
  public native int load();
}
