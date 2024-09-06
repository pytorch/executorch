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

/**
 * LlamaModule is a wrapper around the Executorch Llama model. It provides a simple interface to
 * generate text from the model.
 *
 * <p>Warning: These APIs are experimental and subject to change without notice
 */
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
   */
  @DoNotStrip
  public native int generate(
      int[] image,
      int width,
      int height,
      int channels,
      String prompt,
      int seqLen,
      LlamaCallback llamaCallback);

  /**
   * Prefill an LLaVA Module with the given text input.
   *
   * @param prompt The text prompt to LLaVA.
   * @param startPos The starting position in KV cache of the input in the LLM. It's passed as
   *     reference and will be updated inside this function.
   * @param bos The number of BOS (begin of sequence) token.
   * @param eos The number of EOS (end of sequence) token.
   * @return a tuple of (error, token, updated startPos)
   */
  public static native long[] prefill_prompt(
      String prompt, long startPos, int bos, int eos, long generatedToken);

  /**
   * Prefill an LLaVA Module with the given images input.
   *
   * @param image Input image as a byte array
   * @param width Input image width
   * @param height Input image height
   * @param channels Input image number of channels
   * @param startPos The starting position in KV cache of the input in the LLM.
   * @return a tuple of (error code, updated startPos)
   */
  public static native long[] prefill_images(
      int[] image, int width, int height, int channels, long startPos);

  /**
   * Generate tokens from the given prompt, starting from the given position.
   *
   * @param prompt The text prompt to LLaVA.
   * @param seqLen The total sequence length, including the prompt tokens and new tokens.
   * @param startPos The starting position in KV cache of the input in the LLM.
   * @param llamaCallback callback object to receive results.
   * @return The error code.
   */
  public static native int generate_from_pos(
      String prompt, int seqLen, long startPos, ExecuTorchLlamaCallback callback);

  /** Stop current generate() before it finishes. */
  @DoNotStrip
  public native void stop();

  /** Force loading the module. Otherwise the model is loaded during first generate(). */
  @DoNotStrip
  public native int load();
}
