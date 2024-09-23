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
import org.pytorch.executorch.annotations.Experimental;

/**
 * LlamaModule is a wrapper around the Executorch Llama model. It provides a simple interface to
 * generate text from the model.
 *
 * <p>Warning: These APIs are experimental and subject to change without notice
 */
@Experimental
public class LlamaModule {

  public static final int MODEL_TYPE_TEXT = 1;
  public static final int MODEL_TYPE_TEXT_VISION = 2;

  static {
    if (!NativeLoader.isInitialized()) {
      NativeLoader.init(new SystemDelegate());
    }
    NativeLoader.loadLibrary("executorch");
  }

  private final HybridData mHybridData;
  private static final int DEFAULT_SEQ_LEN = 128;
  private static final boolean DEFAULT_ECHO = true;

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
    return generate(prompt, DEFAULT_SEQ_LEN, llamaCallback, DEFAULT_ECHO);
  }

  /**
   * Start generating tokens from the module.
   *
   * @param prompt Input prompt
   * @param seqLen sequence length
   * @param llamaCallback callback object to receive results.
   */
  public int generate(String prompt, int seqLen, LlamaCallback llamaCallback) {
    return generate(null, 0, 0, 0, prompt, seqLen, llamaCallback, DEFAULT_ECHO);
  }

  /**
   * Start generating tokens from the module.
   *
   * @param prompt Input prompt
   * @param llamaCallback callback object to receive results
   * @param echo indicate whether to echo the input prompt or not (text completion vs chat)
   */
  public int generate(String prompt, LlamaCallback llamaCallback, boolean echo) {
    return generate(null, 0, 0, 0, prompt, DEFAULT_SEQ_LEN, llamaCallback, echo);
  }

  /**
   * Start generating tokens from the module.
   *
   * @param prompt Input prompt
   * @param seqLen sequence length
   * @param llamaCallback callback object to receive results
   * @param echo indicate whether to echo the input prompt or not (text completion vs chat)
   */
  public int generate(String prompt, int seqLen, LlamaCallback llamaCallback, boolean echo) {
    return generate(null, 0, 0, 0, prompt, seqLen, llamaCallback, echo);
  }

  /**
   * Start generating tokens from the module.
   *
   * @param image Input image as a byte array
   * @param width Input image width
   * @param height Input image height
   * @param channels Input image number of channels
   * @param prompt Input prompt
   * @param seqLen sequence length
   * @param llamaCallback callback object to receive results.
   * @param echo indicate whether to echo the input prompt or not (text completion vs chat)
   */
  @DoNotStrip
  public native int generate(
      int[] image,
      int width,
      int height,
      int channels,
      String prompt,
      int seqLen,
      LlamaCallback llamaCallback,
      boolean echo);

  /**
   * Prefill an LLaVA Module with the given images input.
   *
   * @param image Input image as a byte array
   * @param width Input image width
   * @param height Input image height
   * @param channels Input image number of channels
   * @param startPos The starting position in KV cache of the input in the LLM.
   * @return The updated starting position in KV cache of the input in the LLM.
   * @throws RuntimeException if the prefill failed
   */
  public long prefillImages(int[] image, int width, int height, int channels, long startPos) {
    long[] nativeResult = prefillImagesNative(image, width, height, channels, startPos);
    if (nativeResult[0] != 0) {
      throw new RuntimeException("Prefill failed with error code: " + nativeResult[0]);
    }
    return nativeResult[1];
  }

  // returns a tuple of (status, updated startPos)
  private native long[] prefillImagesNative(
      int[] image, int width, int height, int channels, long startPos);

  /**
   * Prefill an LLaVA Module with the given text input.
   *
   * @param prompt The text prompt to LLaVA.
   * @param startPos The starting position in KV cache of the input in the LLM. It's passed as
   *     reference and will be updated inside this function.
   * @param bos The number of BOS (begin of sequence) token.
   * @param eos The number of EOS (end of sequence) token.
   * @return The updated starting position in KV cache of the input in the LLM.
   * @throws RuntimeException if the prefill failed
   */
  public long prefillPrompt(String prompt, long startPos, int bos, int eos) {
    long[] nativeResult = prefillPromptNative(prompt, startPos, bos, eos);
    if (nativeResult[0] != 0) {
      throw new RuntimeException("Prefill failed with error code: " + nativeResult[0]);
    }
    return nativeResult[1];
  }

  // returns a tuple of (status, updated startPos)
  private native long[] prefillPromptNative(String prompt, long startPos, int bos, int eos);

  /**
   * Generate tokens from the given prompt, starting from the given position.
   *
   * @param prompt The text prompt to LLaVA.
   * @param seqLen The total sequence length, including the prompt tokens and new tokens.
   * @param startPos The starting position in KV cache of the input in the LLM.
   * @param callback callback object to receive results.
   * @param echo indicate whether to echo the input prompt or not.
   * @return The error code.
   */
  public native int generateFromPos(
      String prompt, int seqLen, long startPos, LlamaCallback callback, boolean echo);

  /** Stop current generate() before it finishes. */
  @DoNotStrip
  public native void stop();

  /** Force loading the module. Otherwise the model is loaded during first generate(). */
  @DoNotStrip
  public native int load();
}
