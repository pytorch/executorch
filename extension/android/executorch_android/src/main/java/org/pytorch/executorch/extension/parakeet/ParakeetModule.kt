/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch.extension.parakeet

import java.io.Closeable
import java.io.File
import java.util.concurrent.atomic.AtomicLong
import org.pytorch.executorch.annotations.Experimental

/**
 * ParakeetModule is a wrapper around the ExecuTorch Parakeet TDT Runner. It provides a simple
 * interface to transcribe audio from WAV files using the NVIDIA Parakeet TDT speech recognition
 * model.
 *
 * The module loads a WAV file, runs preprocessing (mel-spectrogram extraction), encoding, and TDT
 * greedy decoding to generate transcriptions with optional timestamps.
 *
 * Warning: These APIs are experimental and subject to change without notice
 *
 * @param modelPath Path to the ExecuTorch Parakeet model file (.pte). The model must expose
 *   callable methods: "preprocessor", "encoder", "decoder_step", "joint", and metadata methods.
 * @param tokenizerPath Path to the SentencePiece tokenizer model file.
 * @param dataPath Optional path to additional data file (e.g., for delegate data like CUDA).
 */
@Experimental
class ParakeetModule(
    modelPath: String,
    tokenizerPath: String,
    dataPath: String? = null,
) : Closeable {

  private val nativeHandle = AtomicLong(0L)

  init {
    val modelFile = File(modelPath)
    require(modelFile.canRead() && modelFile.isFile) { "Cannot load model path $modelPath" }
    val tokenizerFile = File(tokenizerPath)
    require(tokenizerFile.exists()) { "Cannot load tokenizer path $tokenizerPath" }

    val handle = nativeCreate(modelPath, tokenizerPath, dataPath)
    if (handle == 0L) {
      throw RuntimeException("Failed to create native ParakeetModule")
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
    ): Long

    @JvmStatic private external fun nativeDestroy(nativeHandle: Long)

    @JvmStatic
    private external fun nativeTranscribe(
        nativeHandle: Long,
        wavPath: String,
        timestamps: String,
    ): String
  }

  /** Check if the native handle is valid (not yet closed). */
  val isValid: Boolean
    get() = nativeHandle.get() != 0L

  /** Releases native resources. Call this when done with the module. */
  override fun close() {
    val handle = nativeHandle.getAndSet(0L)
    if (handle != 0L) {
      nativeDestroy(handle)
    }
  }

  /**
   * Transcribe audio from a WAV file.
   *
   * This is a blocking call that returns the complete transcription.
   *
   * @param wavPath Path to the WAV audio file (must be 16kHz mono)
   * @param timestamps Timestamp output mode: "none", "token", "word", "segment", or "all". Default
   *   is "segment" which returns sentence-level timestamps.
   * @return The transcribed text, optionally with timestamps depending on the mode
   * @throws IllegalStateException if the module has been destroyed
   * @throws RuntimeException if transcription fails
   */
  @JvmOverloads
  fun transcribe(
      wavPath: String,
      timestamps: String = "segment",
  ): String {
    val handle = nativeHandle.get()
    check(handle != 0L) { "ParakeetModule has been destroyed" }
    val wavFile = File(wavPath)
    require(wavFile.canRead() && wavFile.isFile) { "Cannot read WAV file: $wavPath" }

    return nativeTranscribe(handle, wavPath, timestamps)
  }
}
