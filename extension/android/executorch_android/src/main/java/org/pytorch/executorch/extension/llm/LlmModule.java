/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch.extension.llm;

import com.facebook.jni.HybridData;
import com.facebook.jni.annotations.DoNotStrip;
import java.io.File;
import org.pytorch.executorch.ExecuTorchRuntime;
import org.pytorch.executorch.annotations.Experimental;

/**
 * LlmModule is a wrapper around the Executorch LLM. It provides a simple interface to generate text
 * from the model.
 *
 * <p>Warning: These APIs are experimental and subject to change without notice
 */
@Experimental
public class LlmModule {

  public static final int MODEL_TYPE_TEXT = 1;
  public static final int MODEL_TYPE_TEXT_VISION = 2;

  private final HybridData mHybridData;
  private static final int DEFAULT_SEQ_LEN = 128;
  private static final boolean DEFAULT_ECHO = true;

  @DoNotStrip
  private static native HybridData initHybrid(
      int modelType, String modulePath, String tokenizerPath, float temperature, String dataPath);

  /**
   * Constructs a LLM Module for a model with given type, model path, tokenizer, temperature, and
   * data path.
   */
  public LlmModule(
      int modelType, String modulePath, String tokenizerPath, float temperature, String dataPath) {
    ExecuTorchRuntime runtime = ExecuTorchRuntime.getRuntime();

    File modelFile = new File(modulePath);
    if (!modelFile.canRead() || !modelFile.isFile()) {
      throw new RuntimeException("Cannot load model path " + modulePath);
    }
    File tokenizerFile = new File(tokenizerPath);
    if (!tokenizerFile.canRead() || !tokenizerFile.isFile()) {
      throw new RuntimeException("Cannot load tokenizer path " + tokenizerPath);
    }
    mHybridData = initHybrid(modelType, modulePath, tokenizerPath, temperature, dataPath);
  }

  /** Constructs a LLM Module for a model with given model path, tokenizer, temperature. */
  public LlmModule(String modulePath, String tokenizerPath, float temperature) {
    this(MODEL_TYPE_TEXT, modulePath, tokenizerPath, temperature, null);
  }

  /**
   * Constructs a LLM Module for a model with given model path, tokenizer, temperature and data
   * path.
   */
  public LlmModule(String modulePath, String tokenizerPath, float temperature, String dataPath) {
    this(MODEL_TYPE_TEXT, modulePath, tokenizerPath, temperature, dataPath);
  }

  /** Constructs a LLM Module for a model with given path, tokenizer, and temperature. */
  public LlmModule(int modelType, String modulePath, String tokenizerPath, float temperature) {
    this(modelType, modulePath, tokenizerPath, temperature, null);
  }

  /** Constructs a LLM Module for a model with the given LlmModuleConfig */
  public LlmModule(LlmModuleConfig config) {
    this(
        config.getModelType(),
        config.getModulePath(),
        config.getTokenizerPath(),
        config.getTemperature(),
        config.getDataPath());
  }

  public void resetNative() {
    mHybridData.resetNative();
  }

  /**
   * Start generating tokens from the module.
   *
   * @param prompt Input prompt
   * @param llmCallback callback object to receive results.
   */
  public int generate(String prompt, LlmCallback llmCallback) {
    return generate(prompt, DEFAULT_SEQ_LEN, llmCallback, DEFAULT_ECHO);
  }

  /**
   * Start generating tokens from the module.
   *
   * @param prompt Input prompt
   * @param seqLen sequence length
   * @param llmCallback callback object to receive results.
   */
  public int generate(String prompt, int seqLen, LlmCallback llmCallback) {
    return generate(null, 0, 0, 0, prompt, seqLen, llmCallback, DEFAULT_ECHO);
  }

  /**
   * Start generating tokens from the module.
   *
   * @param prompt Input prompt
   * @param llmCallback callback object to receive results
   * @param echo indicate whether to echo the input prompt or not (text completion vs chat)
   */
  public int generate(String prompt, LlmCallback llmCallback, boolean echo) {
    return generate(null, 0, 0, 0, prompt, DEFAULT_SEQ_LEN, llmCallback, echo);
  }

  /**
   * Start generating tokens from the module.
   *
   * @param prompt Input prompt
   * @param seqLen sequence length
   * @param llmCallback callback object to receive results
   * @param echo indicate whether to echo the input prompt or not (text completion vs chat)
   */
  public native int generate(String prompt, int seqLen, LlmCallback llmCallback, boolean echo);

  /**
   * Start generating tokens from the module.
   *
   * @param prompt Input prompt
   * @param config the config for generation
   * @param llmCallback callback object to receive results
   */
  public int generate(String prompt, LlmGenerationConfig config, LlmCallback llmCallback) {
    int seqLen = config.getSeqLen();
    boolean echo = config.isEcho();
    return generate(null, 0, 0, 0, prompt, seqLen, llmCallback, echo);
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
   * @param llmCallback callback object to receive results.
   * @param echo indicate whether to echo the input prompt or not (text completion vs chat)
   */
  public int generate(
      int[] image,
      int width,
      int height,
      int channels,
      String prompt,
      int seqLen,
      LlmCallback llmCallback,
      boolean echo) {
    prefillPrompt(prompt);
    prefillImages(image, width, height, channels);
    return generate("", llmCallback, echo);
  }

  /**
   * Prefill an multimodal Module with the given images input.
   *
   * @param image Input image as a byte array
   * @param width Input image width
   * @param height Input image height
   * @param channels Input image number of channels
   * @return 0, as the updated starting position in KV cache of the input in the LLM is no longer
   *     exposed to user.
   * @throws RuntimeException if the prefill failed
   */
  @Deprecated
  public long prefillImages(int[] image, int width, int height, int channels) {
    int nativeResult = appendImagesInput(image, width, height, channels);
    if (nativeResult != 0) {
      throw new RuntimeException("Prefill failed with error code: " + nativeResult);
    }
    return 0;
  }

  private native int appendImagesInput(int[] image, int width, int height, int channels);

  /**
   * Prefill an multimodal Module with the given text input.
   *
   * @param prompt The text prompt to multimodal model.
   * @return 0, as the updated starting position in KV cache of the input in the LLM is no longer
   *     exposed to user.
   * @throws RuntimeException if the prefill failed
   */
  @Deprecated
  public long prefillPrompt(String prompt) {
    int nativeResult = appendTextInput(prompt);
    if (nativeResult != 0) {
      throw new RuntimeException("Prefill failed with error code: " + nativeResult);
    }
    return 0;
  }

  // returns status
  private native int appendTextInput(String prompt);

  /**
   * Prefill a multimodal Module with the given text input.
   *
   * @param prompt The text prompt to multimodal model.
   * @return 0, as the updated starting position in KV cache of the input in the LLM is no longer
   *     exposed to user.
   * @throws RuntimeException if the prefill failed
   */
  public int prefillAudio(String filePath) {
    java.io.File file = new java.io.File(filePath);
    try (java.io.FileInputStream fis = new java.io.FileInputStream(file)) {
      byte[] fileBytes = new byte[(int) file.length()];
      int bytesRead = fis.read(fileBytes);
      if (bytesRead != fileBytes.length) {
          throw new RuntimeException("Could not completely read file " + file.getName());
      }
      int nFloats = fileBytes.length / 4;
      int batchSize = nFloats / (128 * 3000);
      return appendAudioInput(fileBytes, batchSize, 128, 3000);
    } catch (java.io.IOException e) {
      throw new RuntimeException("Failed to read file: " + e);
    }
  }

  // For Audio (option B), not RawAudio
  // Use batch_size = ceil(n_floats / (n_bins * n_frames)), n_bins = 128, n_frames = 3000
  // returns status
  private native int appendAudioInput(byte[] audio, int batch_size, int n_bins, int n_frames);

  /**
   * Reset the context of the LLM. This will clear the KV cache and reset the state of the LLM.
   *
   * <p>The startPos will be reset to 0.
   */
  public native void resetContext();

  /** Stop current generate() before it finishes. */
  @DoNotStrip
  public native void stop();

  /** Force loading the module. Otherwise the model is loaded during first generate(). */
  @DoNotStrip
  public native int load();
}
