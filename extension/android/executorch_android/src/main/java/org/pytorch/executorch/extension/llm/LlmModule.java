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
import java.nio.ByteBuffer;
import java.util.List;
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
  public static final int MODEL_TYPE_MULTIMODAL = 2;

  private final HybridData mHybridData;
  private static final int DEFAULT_SEQ_LEN = 128;
  private static final boolean DEFAULT_ECHO = true;
  private static final float DEFAULT_TEMPERATURE = -1.0f;
  private static final int DEFAULT_BOS = 0;
  private static final int DEFAULT_EOS = 0;

  @DoNotStrip
  private static native HybridData initHybrid(
      int modelType,
      String modulePath,
      String tokenizerPath,
      float temperature,
      List<String> dataFiles,
      int numBos,
      int numEos);

  /**
   * Constructs a LLM Module for a model with given type, model path, tokenizer, temperature, and
   * dataFiles.
   */
  public LlmModule(
      int modelType,
      String modulePath,
      String tokenizerPath,
      float temperature,
      List<String> dataFiles,
      int numBos,
      int numEos) {
    ExecuTorchRuntime.getRuntime();
    ExecuTorchRuntime.validateFilePath(modulePath, "model path");
    ExecuTorchRuntime.validateFilePath(tokenizerPath, "tokenizer path");

    mHybridData =
        initHybrid(modelType, modulePath, tokenizerPath, temperature, dataFiles, numBos, numEos);
  }

  /**
   * Constructs a LLM Module for a model with given type, model path, tokenizer, temperature, and
   * dataFiles.
   */
  public LlmModule(
      int modelType,
      String modulePath,
      String tokenizerPath,
      float temperature,
      List<String> dataFiles) {
    this(modelType, modulePath, tokenizerPath, temperature, dataFiles, DEFAULT_BOS, DEFAULT_EOS);
  }

  /**
   * Constructs a LLM Module for a model with given type, model path, tokenizer, temperature, and
   * data path.
   */
  public LlmModule(
      int modelType,
      String modulePath,
      String tokenizerPath,
      float temperature,
      String dataPath,
      int numBos,
      int numEos) {
    this(
        modelType,
        modulePath,
        tokenizerPath,
        temperature,
        dataPath != null ? List.of(dataPath) : List.of(),
        numBos,
        numEos);
  }

  /**
   * Constructs a LLM Module for a model with given type, model path, tokenizer, temperature, and
   * data path.
   */
  public LlmModule(
      int modelType, String modulePath, String tokenizerPath, float temperature, String dataPath) {
    this(modelType, modulePath, tokenizerPath, temperature, dataPath, DEFAULT_BOS, DEFAULT_EOS);
  }

  /** Constructs a LLM Module for a model with given model path, tokenizer, temperature. */
  public LlmModule(String modulePath, String tokenizerPath, float temperature) {
    this(
        MODEL_TYPE_TEXT,
        modulePath,
        tokenizerPath,
        temperature,
        List.of(),
        DEFAULT_BOS,
        DEFAULT_EOS);
  }

  /**
   * Constructs a LLM Module for a model with given model path, tokenizer, temperature and data
   * path.
   */
  public LlmModule(String modulePath, String tokenizerPath, float temperature, String dataPath) {
    this(
        MODEL_TYPE_TEXT,
        modulePath,
        tokenizerPath,
        temperature,
        List.of(dataPath),
        DEFAULT_BOS,
        DEFAULT_EOS);
  }

  /** Constructs a LLM Module for a model with given path, tokenizer, and temperature. */
  public LlmModule(int modelType, String modulePath, String tokenizerPath, float temperature) {
    this(modelType, modulePath, tokenizerPath, temperature, List.of(), DEFAULT_BOS, DEFAULT_EOS);
  }

  /** Constructs a LLM Module for a model with the given LlmModuleConfig */
  public LlmModule(LlmModuleConfig config) {
    this(
        config.getModelType(),
        config.getModulePath(),
        config.getTokenizerPath(),
        config.getTemperature(),
        config.getDataPath(),
        config.getNumBos(),
        config.getNumEos());
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
    return generate(
        prompt,
        DEFAULT_SEQ_LEN,
        llmCallback,
        DEFAULT_ECHO,
        DEFAULT_TEMPERATURE,
        DEFAULT_BOS,
        DEFAULT_EOS);
  }

  /**
   * Start generating tokens from the module.
   *
   * @param prompt Input prompt
   * @param seqLen sequence length
   * @param llmCallback callback object to receive results.
   */
  public int generate(String prompt, int seqLen, LlmCallback llmCallback) {
    return generate(
        null,
        0,
        0,
        0,
        prompt,
        seqLen,
        llmCallback,
        DEFAULT_ECHO,
        DEFAULT_TEMPERATURE,
        DEFAULT_BOS,
        DEFAULT_EOS);
  }

  /**
   * Start generating tokens from the module.
   *
   * @param prompt Input prompt
   * @param llmCallback callback object to receive results
   * @param echo indicate whether to echo the input prompt or not (text completion vs chat)
   */
  public int generate(String prompt, LlmCallback llmCallback, boolean echo) {
    return generate(
        null,
        0,
        0,
        0,
        prompt,
        DEFAULT_SEQ_LEN,
        llmCallback,
        echo,
        DEFAULT_TEMPERATURE,
        DEFAULT_BOS,
        DEFAULT_EOS);
  }

  /**
   * Start generating tokens from the module.
   *
   * @param prompt Input prompt
   * @param seqLen sequence length
   * @param llmCallback callback object to receive results
   * @param echo indicate whether to echo the input prompt or not (text completion vs chat)
   */
  public int generate(String prompt, int seqLen, LlmCallback llmCallback, boolean echo) {
    return generate(
        prompt, seqLen, llmCallback, echo, DEFAULT_TEMPERATURE, DEFAULT_BOS, DEFAULT_EOS);
  }

  /**
   * Start generating tokens from the module.
   *
   * @param prompt Input prompt
   * @param seqLen sequence length
   * @param llmCallback callback object to receive results
   * @param echo indicate whether to echo the input prompt or not (text completion vs chat)
   * @param temperature temperature for sampling (use negative value to use module default)
   * @param numBos number of BOS tokens to prepend
   * @param numEos number of EOS tokens to append
   */
  public native int generate(
      String prompt,
      int seqLen,
      LlmCallback llmCallback,
      boolean echo,
      float temperature,
      int numBos,
      int numEos);

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
    float temperature = config.getTemperature();
    int numBos = config.getNumBos();
    int numEos = config.getNumEos();
    return generate(null, 0, 0, 0, prompt, seqLen, llmCallback, echo, temperature, numBos, numEos);
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
    return generate(
        image,
        width,
        height,
        channels,
        prompt,
        seqLen,
        llmCallback,
        echo,
        DEFAULT_TEMPERATURE,
        DEFAULT_BOS,
        DEFAULT_EOS);
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
   * @param temperature temperature for sampling (use negative value to use module default)
   */
  public int generate(
      int[] image,
      int width,
      int height,
      int channels,
      String prompt,
      int seqLen,
      LlmCallback llmCallback,
      boolean echo,
      float temperature) {
    return generate(
        image,
        width,
        height,
        channels,
        prompt,
        seqLen,
        llmCallback,
        echo,
        temperature,
        DEFAULT_BOS,
        DEFAULT_EOS);
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
   * @param temperature temperature for sampling (use negative value to use module default)
   * @param numBos number of BOS tokens to prepend
   * @param numEos number of EOS tokens to append
   */
  public int generate(
      int[] image,
      int width,
      int height,
      int channels,
      String prompt,
      int seqLen,
      LlmCallback llmCallback,
      boolean echo,
      float temperature,
      int numBos,
      int numEos) {
    if (image != null) {
      prefillImages(image, width, height, channels);
    }
    return generate(prompt, seqLen, llmCallback, echo, temperature, numBos, numEos);
  }

  /**
   * Prefill a multimodal Module with the given images input.
   *
   * @param image Input image as a byte array
   * @param width Input image width
   * @param height Input image height
   * @param channels Input image number of channels
   * @return 0, as the updated starting position in KV cache of the input in the LLM is no longer
   *     exposed to user.
   * @throws RuntimeException if the prefill failed
   */
  @Experimental
  @Deprecated
  public long prefillImages(int[] image, int width, int height, int channels) {
    int nativeResult = appendImagesInput(image, width, height, channels);
    if (nativeResult != 0) {
      throw new RuntimeException("Prefill failed with error code: " + nativeResult);
    }
    return 0;
  }

  /**
   * Prefill a multimodal Module with the given image input via a direct ByteBuffer. The buffer data
   * is accessed directly without JNI array copies, unlike {@link #prefillImages(int[], int, int,
   * int)}. The ByteBuffer must contain raw uint8 pixel data in CHW format with at least channels *
   * height * width bytes remaining. Only the first channels * height * width bytes from the
   * buffer's current position are consumed.
   *
   * @param image Input image as a direct ByteBuffer containing uint8 pixel data
   * @param width Input image width
   * @param height Input image height
   * @param channels Input image number of channels
   * @throws IllegalArgumentException if the ByteBuffer is not direct or has insufficient remaining
   *     bytes
   * @throws RuntimeException if the prefill failed
   */
  @Experimental
  public void prefillImages(ByteBuffer image, int width, int height, int channels) {
    if (!image.isDirect()) {
      throw new IllegalArgumentException("Input ByteBuffer must be direct.");
    }
    long expectedBytes = (long) width * height * channels;
    if (width <= 0 || height <= 0 || channels <= 0 || image.remaining() < expectedBytes) {
      throw new IllegalArgumentException(
          "ByteBuffer remaining ("
              + image.remaining()
              + ") must be at least width*height*channels ("
              + expectedBytes
              + ").");
    }
    // slice() so that getDirectBufferAddress on the native side returns a pointer
    // starting at the current position, not the base address.
    int nativeResult = appendImagesInputBuffer(image.slice(), width, height, channels);
    if (nativeResult != 0) {
      throw new RuntimeException("Prefill failed with error code: " + nativeResult);
    }
  }

  /**
   * Prefill a multimodal Module with the given normalized image input via a direct ByteBuffer. The
   * buffer data is accessed directly without JNI array copies, unlike {@link
   * #prefillImages(float[], int, int, int)}. The ByteBuffer must contain normalized float pixel
   * data in CHW format with at least channels * height * width * 4 bytes remaining. Only the first
   * channels * height * width floats from the buffer's current position are consumed. The buffer
   * must use the platform's native byte order (set via {@code
   * buffer.order(ByteOrder.nativeOrder())}).
   *
   * @param image Input normalized image as a direct ByteBuffer containing float pixel data in
   *     native byte order
   * @param width Input image width
   * @param height Input image height
   * @param channels Input image number of channels
   * @throws IllegalArgumentException if the ByteBuffer is not direct, has insufficient remaining
   *     bytes, is not float-aligned, or does not use native byte order
   * @throws RuntimeException if the prefill failed
   */
  @Experimental
  public void prefillNormalizedImages(ByteBuffer image, int width, int height, int channels) {
    if (!image.isDirect()) {
      throw new IllegalArgumentException("Input ByteBuffer must be direct.");
    }
    if (image.order() != java.nio.ByteOrder.nativeOrder()) {
      throw new IllegalArgumentException(
          "Input ByteBuffer must use native byte order (ByteOrder.nativeOrder()).");
    }
    if (image.position() % Float.BYTES != 0) {
      throw new IllegalArgumentException(
          "Input ByteBuffer position (" + image.position() + ") must be 4-byte aligned.");
    }
    long expectedBytes = (long) width * height * channels * Float.BYTES;
    if (width <= 0 || height <= 0 || channels <= 0 || image.remaining() < expectedBytes) {
      throw new IllegalArgumentException(
          "ByteBuffer remaining ("
              + image.remaining()
              + ") must be at least width*height*channels*4 ("
              + expectedBytes
              + ").");
    }
    if (image.remaining() % Float.BYTES != 0) {
      throw new IllegalArgumentException(
          "ByteBuffer remaining (" + image.remaining() + ") must be a multiple of 4 (float size).");
    }
    // slice() so that getDirectBufferAddress on the native side returns a pointer
    // starting at the current position, not the base address.
    int nativeResult = appendNormalizedImagesInputBuffer(image.slice(), width, height, channels);
    if (nativeResult != 0) {
      throw new RuntimeException("Prefill failed with error code: " + nativeResult);
    }
  }

  private native int appendImagesInput(int[] image, int width, int height, int channels);

  private native int appendImagesInputBuffer(ByteBuffer image, int width, int height, int channels);

  private native int appendNormalizedImagesInputBuffer(
      ByteBuffer image, int width, int height, int channels);

  /**
   * Prefill a multimodal Module with the given images input.
   *
   * @param image Input normalized image as a float array
   * @param width Input image width
   * @param height Input image height
   * @param channels Input image number of channels
   * @return 0, as the updated starting position in KV cache of the input in the LLM is no longer
   *     exposed to user.
   * @throws RuntimeException if the prefill failed
   */
  @Experimental
  @Deprecated
  public long prefillImages(float[] image, int width, int height, int channels) {
    int nativeResult = appendNormalizedImagesInput(image, width, height, channels);
    if (nativeResult != 0) {
      throw new RuntimeException("Prefill failed with error code: " + nativeResult);
    }
    return 0;
  }

  private native int appendNormalizedImagesInput(
      float[] image, int width, int height, int channels);

  /**
   * Prefill a multimodal Module with the given audio input.
   *
   * @param audio Input preprocessed audio as a byte array
   * @param batch_size Input batch size
   * @param n_bins Input number of bins
   * @param n_frames Input number of frames
   * @return 0, as the updated starting position in KV cache of the input in the LLM is no longer
   *     exposed to user.
   * @throws RuntimeException if the prefill failed
   */
  @Experimental
  public long prefillAudio(byte[] audio, int batch_size, int n_bins, int n_frames) {
    int nativeResult = appendAudioInput(audio, batch_size, n_bins, n_frames);
    if (nativeResult != 0) {
      throw new RuntimeException("Prefill failed with error code: " + nativeResult);
    }
    return 0;
  }

  private native int appendAudioInput(byte[] audio, int batch_size, int n_bins, int n_frames);

  /**
   * Prefill a multimodal Module with the given audio input.
   *
   * @param audio Input preprocessed audio as a float array
   * @param batch_size Input batch size
   * @param n_bins Input number of bins
   * @param n_frames Input number of frames
   * @return 0, as the updated starting position in KV cache of the input in the LLM is no longer
   *     exposed to user.
   * @throws RuntimeException if the prefill failed
   */
  @Experimental
  public long prefillAudio(float[] audio, int batch_size, int n_bins, int n_frames) {
    int nativeResult = appendAudioInputFloat(audio, batch_size, n_bins, n_frames);
    if (nativeResult != 0) {
      throw new RuntimeException("Prefill failed with error code: " + nativeResult);
    }
    return 0;
  }

  private native int appendAudioInputFloat(float[] audio, int batch_size, int n_bins, int n_frames);

  /**
   * Prefill a multimodal Module with the given raw audio input.
   *
   * @param audio Input raw audio as a byte array
   * @param batch_size Input batch size
   * @param n_channels Input number of channels
   * @param n_samples Input number of samples
   * @return 0, as the updated starting position in KV cache of the input in the LLM is no longer
   *     exposed to user.
   * @throws RuntimeException if the prefill failed
   */
  @Experimental
  public long prefillRawAudio(byte[] audio, int batch_size, int n_channels, int n_samples) {
    int nativeResult = appendRawAudioInput(audio, batch_size, n_channels, n_samples);
    if (nativeResult != 0) {
      throw new RuntimeException("Prefill failed with error code: " + nativeResult);
    }
    return 0;
  }

  private native int appendRawAudioInput(
      byte[] audio, int batch_size, int n_channels, int n_samples);

  /**
   * Prefill a multimodal Module with the given text input.
   *
   * @param prompt The text prompt to prefill.
   * @return 0, as the updated starting position in KV cache of the input in the LLM is no longer
   *     exposed to user.
   * @throws RuntimeException if the prefill failed
   */
  @Experimental
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
