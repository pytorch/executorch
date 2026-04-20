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
import java.io.Closeable;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import org.pytorch.executorch.ExecuTorchRuntime;
import org.pytorch.executorch.ExecutorchRuntimeException;
import org.pytorch.executorch.annotations.Experimental;

/**
 * LlmModule is a wrapper around the Executorch LLM. It provides a simple interface to generate text
 * from the model.
 *
 * <p>Warning: These APIs are experimental and subject to change without notice
 */
@Experimental
public class LlmModule implements Closeable {

  public static final int MODEL_TYPE_TEXT = 1;
  public static final int MODEL_TYPE_TEXT_VISION = 2;
  public static final int MODEL_TYPE_MULTIMODAL = 2;

  private final HybridData mHybridData;
  private final ReentrantReadWriteLock mLock = new ReentrantReadWriteLock(true);
  private volatile boolean mClosed = false;
  private static final int DEFAULT_SEQ_LEN = 128;
  private static final boolean DEFAULT_ECHO = true;
  private static final float DEFAULT_TEMPERATURE = -1.0f;
  private static final int DEFAULT_BOS = 0;
  private static final int DEFAULT_EOS = 0;
  private static final int DEFAULT_LOAD_MODE = LlmModuleConfig.LOAD_MODE_MMAP;

  @DoNotStrip
  private static native HybridData initHybrid(
      int modelType,
      String modulePath,
      String tokenizerPath,
      float temperature,
      List<String> dataFiles,
      int numBos,
      int numEos,
      int loadMode);

  private LlmModule(
      int modelType,
      String modulePath,
      String tokenizerPath,
      float temperature,
      List<String> dataFiles,
      int numBos,
      int numEos,
      int loadMode) {
    ExecuTorchRuntime.getRuntime();
    ExecuTorchRuntime.validateFilePath(modulePath, "model path");
    ExecuTorchRuntime.validateFilePath(tokenizerPath, "tokenizer path");

    mHybridData =
        initHybrid(
            modelType, modulePath, tokenizerPath, temperature, dataFiles, numBos, numEos, loadMode);
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
      List<String> dataFiles,
      int numBos,
      int numEos) {
    this(
        modelType,
        modulePath,
        tokenizerPath,
        temperature,
        dataFiles,
        numBos,
        numEos,
        DEFAULT_LOAD_MODE);
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
    this(
        modelType,
        modulePath,
        tokenizerPath,
        temperature,
        dataFiles,
        DEFAULT_BOS,
        DEFAULT_EOS,
        DEFAULT_LOAD_MODE);
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
        config.getDataPath() != null ? List.of(config.getDataPath()) : List.of(),
        config.getNumBos(),
        config.getNumEos(),
        config.getLoadMode());
  }

  private void checkNotClosed() {
    if (mClosed) throw new IllegalStateException("LlmModule has been closed");
  }

  @Override
  public void close() {
    stopNative();
    mLock.writeLock().lock();
    try {
      if (mClosed) return;
      mClosed = true;
      mHybridData.resetNative();
    } finally {
      mLock.writeLock().unlock();
    }
  }

  /** @deprecated Use {@link #close()} instead. */
  @Deprecated
  public void resetNative() {
    close();
  }

  /**
   * Start generating tokens from the module.
   *
   * @param prompt Input prompt
   * @param llmCallback callback object to receive results.
   */
  public void generate(String prompt, LlmCallback llmCallback) {
    generate(
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
  public void generate(String prompt, int seqLen, LlmCallback llmCallback) {
    generate(
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
  public void generate(String prompt, LlmCallback llmCallback, boolean echo) {
    generate(
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
  public void generate(String prompt, int seqLen, LlmCallback llmCallback, boolean echo) {
    generate(prompt, seqLen, llmCallback, echo, DEFAULT_TEMPERATURE, DEFAULT_BOS, DEFAULT_EOS);
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
  public void generate(
      String prompt,
      int seqLen,
      LlmCallback llmCallback,
      boolean echo,
      float temperature,
      int numBos,
      int numEos) {
    mLock.readLock().lock();
    try {
      checkNotClosed();
      int err = generateNative(prompt, seqLen, llmCallback, echo, temperature, numBos, numEos);
      if (err != 0) {
        throw ExecutorchRuntimeException.makeExecutorchException(err, "Failed to generate");
      }
    } finally {
      mLock.readLock().unlock();
    }
  }

  @DoNotStrip
  private native int generateNative(
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
  public void generate(String prompt, LlmGenerationConfig config, LlmCallback llmCallback) {
    int seqLen = config.getSeqLen();
    boolean echo = config.isEcho();
    float temperature = config.getTemperature();
    int numBos = config.getNumBos();
    int numEos = config.getNumEos();
    generate(null, 0, 0, 0, prompt, seqLen, llmCallback, echo, temperature, numBos, numEos);
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
  public void generate(
      int[] image,
      int width,
      int height,
      int channels,
      String prompt,
      int seqLen,
      LlmCallback llmCallback,
      boolean echo) {
    generate(
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
  public void generate(
      int[] image,
      int width,
      int height,
      int channels,
      String prompt,
      int seqLen,
      LlmCallback llmCallback,
      boolean echo,
      float temperature) {
    generate(
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
  public void generate(
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
    mLock.readLock().lock();
    try {
      checkNotClosed();
      if (image != null) {
        int nativeResult = prefillImagesInput(image, width, height, channels);
        if (nativeResult != 0) {
          throw ExecutorchRuntimeException.makeExecutorchException(nativeResult, "Prefill failed");
        }
      }
      int err = generateNative(prompt, seqLen, llmCallback, echo, temperature, numBos, numEos);
      if (err != 0) {
        throw ExecutorchRuntimeException.makeExecutorchException(err, "Failed to generate");
      }
    } finally {
      mLock.readLock().unlock();
    }
  }

  /**
   * Prefill the KV cache with the given image input.
   *
   * @param image Input image as a byte array
   * @param width Input image width
   * @param height Input image height
   * @param channels Input image number of channels
   * @throws ExecutorchRuntimeException if the prefill failed
   */
  @Experimental
  public void prefillImages(int[] image, int width, int height, int channels) {
    mLock.readLock().lock();
    try {
      checkNotClosed();
      int nativeResult = prefillImagesInput(image, width, height, channels);
      if (nativeResult != 0) {
        throw ExecutorchRuntimeException.makeExecutorchException(nativeResult, "Prefill failed");
      }
    } finally {
      mLock.readLock().unlock();
    }
  }

  /**
   * Prefill a multimodal Module with the given image input via a direct ByteBuffer. The buffer data
   * is accessed directly without JNI array copies, unlike {@link #prefillImages(int[], int, int,
   * int)}. The ByteBuffer must contain raw uint8 pixel data in CHW format with at least channels *
   * height * width bytes remaining. Only the first channels * height * width bytes from the
   * buffer's current position are read; the position of the original ByteBuffer is not modified.
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
    mLock.readLock().lock();
    try {
      checkNotClosed();
    if (!image.isDirect()) {
      throw new IllegalArgumentException("Input ByteBuffer must be direct.");
    }
    long expectedBytes;
    try {
      long pixels = Math.multiplyExact((long) width, (long) height);
      expectedBytes = Math.multiplyExact(pixels, (long) channels);
    } catch (ArithmeticException ex) {
      throw new IllegalArgumentException(
          "width*height*channels is too large and overflows the allowed range.", ex);
    }
    if (width <= 0
        || height <= 0
        || channels <= 0
        || expectedBytes > Integer.MAX_VALUE
        || image.remaining() < expectedBytes) {
      throw new IllegalArgumentException(
          "ByteBuffer remaining ("
              + image.remaining()
              + ") must be at least width*height*channels ("
              + expectedBytes
              + ").");
    }
    // slice() so that getDirectBufferAddress on the native side returns a pointer
    // starting at the current position, not the base address.
    int nativeResult = prefillImagesInputBuffer(image.slice(), width, height, channels);
    if (nativeResult != 0) {
      throw ExecutorchRuntimeException.makeExecutorchException(nativeResult, "Prefill failed");
    }
    } finally {
      mLock.readLock().unlock();
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
  public void prefillNormalizedImage(ByteBuffer image, int width, int height, int channels) {
    mLock.readLock().lock();
    try {
      checkNotClosed();
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
    final long expectedBytes;
    try {
      int wh = Math.multiplyExact(width, height);
      long whc = Math.multiplyExact((long) wh, (long) channels);
      long totalBytes = Math.multiplyExact(whc, (long) Float.BYTES);
      if (totalBytes > Integer.MAX_VALUE) {
        throw new IllegalArgumentException(
            "ByteBuffer size (width*height*channels*4) exceeds Integer.MAX_VALUE bytes: "
                + totalBytes);
      }
      expectedBytes = totalBytes;
    } catch (ArithmeticException e) {
      throw new IllegalArgumentException(
          "Overflow while computing width*height*channels*4 for ByteBuffer size.", e);
    }
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
    int nativeResult = prefillNormalizedImagesInputBuffer(image.slice(), width, height, channels);
    if (nativeResult != 0) {
      throw ExecutorchRuntimeException.makeExecutorchException(nativeResult, "Prefill failed");
    }
    } finally {
      mLock.readLock().unlock();
    }
  }

  private native int prefillImagesInput(int[] image, int width, int height, int channels);

  private native int prefillImagesInputBuffer(
      ByteBuffer image, int width, int height, int channels);

  private native int prefillNormalizedImagesInputBuffer(
      ByteBuffer image, int width, int height, int channels);

  /**
   * Prefill the KV cache with the given normalized image input.
   *
   * @param image Input normalized image as a float array
   * @param width Input image width
   * @param height Input image height
   * @param channels Input image number of channels
   * @throws ExecutorchRuntimeException if the prefill failed
   */
  @Experimental
  public void prefillImages(float[] image, int width, int height, int channels) {
    mLock.readLock().lock();
    try {
      checkNotClosed();
      int nativeResult = prefillNormalizedImagesInput(image, width, height, channels);
      if (nativeResult != 0) {
        throw ExecutorchRuntimeException.makeExecutorchException(nativeResult, "Prefill failed");
      }
    } finally {
      mLock.readLock().unlock();
    }
  }

  private native int prefillNormalizedImagesInput(
      float[] image, int width, int height, int channels);

  /**
   * Prefill the KV cache with the given preprocessed audio input.
   *
   * @param audio Input preprocessed audio as a byte array
   * @param batch_size Input batch size
   * @param n_bins Input number of bins
   * @param n_frames Input number of frames
   * @throws ExecutorchRuntimeException if the prefill failed
   */
  @Experimental
  public void prefillAudio(byte[] audio, int batch_size, int n_bins, int n_frames) {
    mLock.readLock().lock();
    try {
      checkNotClosed();
      int nativeResult = prefillAudioInput(audio, batch_size, n_bins, n_frames);
      if (nativeResult != 0) {
        throw ExecutorchRuntimeException.makeExecutorchException(nativeResult, "Prefill failed");
      }
    } finally {
      mLock.readLock().unlock();
    }
  }

  private native int prefillAudioInput(byte[] audio, int batch_size, int n_bins, int n_frames);

  /**
   * Prefill the KV cache with the given preprocessed audio input.
   *
   * @param audio Input preprocessed audio as a float array
   * @param batch_size Input batch size
   * @param n_bins Input number of bins
   * @param n_frames Input number of frames
   * @throws ExecutorchRuntimeException if the prefill failed
   */
  @Experimental
  public void prefillAudio(float[] audio, int batch_size, int n_bins, int n_frames) {
    mLock.readLock().lock();
    try {
      checkNotClosed();
      int nativeResult = prefillAudioInputFloat(audio, batch_size, n_bins, n_frames);
      if (nativeResult != 0) {
        throw ExecutorchRuntimeException.makeExecutorchException(nativeResult, "Prefill failed");
      }
    } finally {
      mLock.readLock().unlock();
    }
  }

  private native int prefillAudioInputFloat(
      float[] audio, int batch_size, int n_bins, int n_frames);

  /**
   * Prefill the KV cache with the given raw audio input.
   *
   * @param audio Input raw audio as a byte array
   * @param batch_size Input batch size
   * @param n_channels Input number of channels
   * @param n_samples Input number of samples
   * @throws ExecutorchRuntimeException if the prefill failed
   */
  @Experimental
  public void prefillRawAudio(byte[] audio, int batch_size, int n_channels, int n_samples) {
    mLock.readLock().lock();
    try {
      checkNotClosed();
      int nativeResult = prefillRawAudioInput(audio, batch_size, n_channels, n_samples);
      if (nativeResult != 0) {
        throw ExecutorchRuntimeException.makeExecutorchException(nativeResult, "Prefill failed");
      }
    } finally {
      mLock.readLock().unlock();
    }
  }

  private native int prefillRawAudioInput(
      byte[] audio, int batch_size, int n_channels, int n_samples);

  /**
   * Prefill the KV cache with the given text prompt.
   *
   * @param prompt The text prompt to prefill.
   * @throws ExecutorchRuntimeException if the prefill failed
   */
  @Experimental
  public void prefillPrompt(String prompt) {
    mLock.readLock().lock();
    try {
      checkNotClosed();
      int nativeResult = prefillTextInput(prompt);
      if (nativeResult != 0) {
        throw ExecutorchRuntimeException.makeExecutorchException(nativeResult, "Prefill failed");
      }
    } finally {
      mLock.readLock().unlock();
    }
  }

  // returns status
  private native int prefillTextInput(String prompt);

  /**
   * Reset the context of the LLM. This will clear the KV cache and reset the state of the LLM.
   *
   * <p>The startPos will be reset to 0.
   */
  public void resetContext() {
    mLock.readLock().lock();
    try {
      checkNotClosed();
      resetContextNative();
    } finally {
      mLock.readLock().unlock();
    }
  }

  @DoNotStrip
  private native void resetContextNative();

  /**
   * Stop current generate() before it finishes. Safe to call from any thread. Does not acquire any
   * lock. After close(), this is a no-op.
   */
  public void stop() {
    if (mClosed) return;
    stopNative();
  }

  @DoNotStrip
  private native void stopNative();

  /** Force loading the module. Otherwise the model is loaded during first generate(). */
  @DoNotStrip
  public void load() {
    mLock.readLock().lock();
    try {
      checkNotClosed();
      int err = loadNative();
      if (err != 0) {
        throw ExecutorchRuntimeException.makeExecutorchException(err, "Failed to load model");
      }
    } finally {
      mLock.readLock().unlock();
    }
  }

  @DoNotStrip
  private native int loadNative();
}
