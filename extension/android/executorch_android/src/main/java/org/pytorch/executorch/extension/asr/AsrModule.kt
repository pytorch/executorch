/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch.extension.asr

import java.io.Closeable
import java.io.File
import java.util.concurrent.atomic.AtomicLong
import org.pytorch.executorch.annotations.Experimental

/**
 * AsrModule is a wrapper around the ExecuTorch ASR Runner. It provides a simple interface to
 * transcribe audio from WAV files using speech recognition models like Whisper.
 *
 * The module loads a WAV file, optionally preprocesses it using a preprocessor module (e.g., for
 * mel-spectrogram extraction), and then runs the ASR model to generate transcriptions.
 *
 * Warning: These APIs are experimental and subject to change without notice
 *
 * @param modelPath Path to the ExecuTorch model file (.pte). The model must expose exactly two
 *   callable methods named "encoder" and "text_decoder" (these names are required).
 * @param tokenizerPath Path to the tokenizer directory containing tokenizer.json
 * @param dataPath Optional path to additional data file (e.g., for delegate data)
 * @param preprocessorPath Optional path to preprocessor .pte for converting raw audio to features.
 *   If not provided, raw audio samples will be passed directly to the model.
 */
@Experimental
class AsrModule(
    modelPath: String,
    tokenizerPath: String,
    dataPath: String? = null,
    preprocessorPath: String? = null,
) : Closeable {

  private val nativeHandle = AtomicLong(0L)

  init {
    val modelFile = File(modelPath)
    require(modelFile.canRead() && modelFile.isFile) { "Cannot load model path $modelPath" }
    val tokenizerFile = File(tokenizerPath)
    require(tokenizerFile.exists()) { "Cannot load tokenizer path $tokenizerPath" }
    if (preprocessorPath != null) {
      val preprocessorFile = File(preprocessorPath)
      require(preprocessorFile.canRead() && preprocessorFile.isFile) {
        "Cannot load preprocessor path $preprocessorPath"
      }
    }

    val handle = nativeCreate(modelPath, tokenizerPath, dataPath, preprocessorPath)
    if (handle == 0L) {
      throw RuntimeException("Failed to create native AsrModule")
    }
    nativeHandle.set(handle)
  }

  companion object {
    init {
      System.loadLibrary("executorch")
    }

    @JvmStatic
    private external fun nativeCreate(
        modelPath: String,
        tokenizerPath: String,
        dataPath: String?,
        preprocessorPath: String?,
    ): Long

    @JvmStatic private external fun nativeDestroy(nativeHandle: Long)

    @JvmStatic private external fun nativeLoad(nativeHandle: Long): Int

    @JvmStatic private external fun nativeIsLoaded(nativeHandle: Long): Boolean

    @JvmStatic
    private external fun nativeTranscribe(
        nativeHandle: Long,
        wavPath: String,
        maxNewTokens: Long,
        temperature: Float,
        decoderStartTokenId: Long,
        callback: AsrCallback?,
    ): Int
  }

  /** Check if the native handle is valid. */
  val isValid: Boolean
    get() = nativeHandle.get() != 0L

  /** Check if the module is loaded and ready for inference. */
  val isLoaded: Boolean
    get() {
      val handle = nativeHandle.get()
      return handle != 0L && nativeIsLoaded(handle)
    }

  /** Releases native resources. Call this when done with the module. */
  fun destroy() {
    val handle = nativeHandle.getAndSet(0L)
    if (handle != 0L) {
      nativeDestroy(handle)
    }
  }

  /** Closeable implementation for use with use {} blocks. */
  override fun close() {
    destroy()
  }

  /**
   * Force loading the module. Otherwise the model is loaded during first transcribe() call.
   *
   * @return 0 on success, error code otherwise
   * @throws IllegalStateException if the module has been destroyed
   */
  fun load(): Int {
    val handle = nativeHandle.get()
    check(handle != 0L) { "AsrModule has been destroyed" }
    return nativeLoad(handle)
  }

  /**
   * Transcribe audio from a WAV file.
   *
   * This is a synchronous blocking call that returns the complete transcription.
   *
   * @param wavPath Path to the WAV audio file
   * @param config Configuration for transcription (null uses default configuration)
   * @param callback Optional callback to receive tokens as they are generated (can be null)
   * @return The complete transcribed text
   * @throws IllegalStateException if the module has been destroyed
   * @throws RuntimeException if transcription fails (non-zero result code)
   */
  @JvmOverloads
  fun transcribe(
      wavPath: String,
      config: AsrTranscribeConfig? = null,
      callback: AsrCallback? = null,
  ): String {
    val handle = nativeHandle.get()
    check(handle != 0L) { "AsrModule has been destroyed" }
    val wavFile = File(wavPath)
    require(wavFile.canRead() && wavFile.isFile) { "Cannot read WAV file: $wavPath" }

    val effectiveConfig = config ?: AsrTranscribeConfig()
    val result = StringBuilder()
    val status =
        nativeTranscribe(
            handle,
            wavPath,
            effectiveConfig.maxNewTokens,
            effectiveConfig.temperature,
            effectiveConfig.decoderStartTokenId,
            object : AsrCallback {
              override fun onToken(token: String) {
                result.append(token)
                callback?.onToken(token)
              }
            },
        )

    if (status != 0) {
      throw RuntimeException("Transcription failed with error code: $status")
    }

    return result.toString()
  }
}
