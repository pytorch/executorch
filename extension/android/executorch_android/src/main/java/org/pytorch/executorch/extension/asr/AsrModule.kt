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
import org.pytorch.executorch.annotations.Experimental

/**
 * AsrModule is a wrapper around the ExecuTorch ASR Runner. It provides a simple interface to
 * transcribe audio using speech recognition models like Whisper.
 *
 * The module expects preprocessed audio features (e.g., mel-spectrogram) as input. The model
 * should expose two callable methods: "encoder" and "text_decoder".
 *
 * Warning: These APIs are experimental and subject to change without notice
 *
 * @param modelPath Path to the ExecuTorch model file (.pte)
 * @param tokenizerPath Path to the tokenizer file
 * @param dataPath Optional path to additional data file (e.g., for delegate data)
 */
@Experimental
class AsrModule(
    modelPath: String,
    tokenizerPath: String,
    dataPath: String? = null
) : Closeable {

    private var nativeHandle: Long

    init {
        val modelFile = File(modelPath)
        require(modelFile.canRead() && modelFile.isFile) {
            "Cannot load model path $modelPath"
        }
        val tokenizerFile = File(tokenizerPath)
        require(tokenizerFile.canRead() && tokenizerFile.isFile) {
            "Cannot load tokenizer path $tokenizerPath"
        }

        nativeHandle = nativeCreate(modelPath, tokenizerPath, dataPath)
        if (nativeHandle == 0L) {
            throw RuntimeException("Failed to create native AsrModule")
        }
    }

    companion object {
        init {
            System.loadLibrary("executorch_jni")
        }

        @JvmStatic
        private external fun nativeCreate(
            modelPath: String,
            tokenizerPath: String,
            dataPath: String?
        ): Long

        @JvmStatic
        private external fun nativeDestroy(nativeHandle: Long)

        @JvmStatic
        private external fun nativeLoad(nativeHandle: Long): Int

        @JvmStatic
        private external fun nativeIsLoaded(nativeHandle: Long): Boolean

        @JvmStatic
        private external fun nativeTranscribe(
            nativeHandle: Long,
            features: FloatArray,
            batchSize: Int,
            timeSteps: Int,
            featureDim: Int,
            maxNewTokens: Long,
            temperature: Float,
            decoderStartTokenId: Long,
            callback: AsrCallback?
        ): Int
    }

    /**
     * Check if the native handle is valid.
     */
    val isValid: Boolean
        get() = nativeHandle != 0L

    /**
     * Check if the module is loaded and ready for inference.
     */
    val isLoaded: Boolean
        get() = nativeHandle != 0L && nativeIsLoaded(nativeHandle)

    /**
     * Releases native resources. Call this when done with the module.
     */
    fun destroy() {
        if (nativeHandle != 0L) {
            nativeDestroy(nativeHandle)
            nativeHandle = 0L
        }
    }

    /**
     * Closeable implementation for use with use {} blocks.
     */
    override fun close() {
        destroy()
    }

    @Throws(Throwable::class)
    protected fun finalize() {
        destroy()
    }

    /**
     * Force loading the module. Otherwise the model is loaded during first transcribe() call.
     *
     * @return 0 on success, error code otherwise
     * @throws IllegalStateException if the module has been destroyed
     */
    fun load(): Int {
        checkNotDestroyed()
        return nativeLoad(nativeHandle)
    }

    /**
     * Transcribe preprocessed audio features with default configuration.
     *
     * @param features Preprocessed audio features as a float array
     * @param batchSize Batch size (typically 1)
     * @param timeSteps Number of time steps in the features
     * @param featureDim Feature dimension (e.g., 80 for mel-spectrogram)
     * @param callback Callback to receive tokens, can be null
     * @return 0 on success, error code otherwise
     * @throws IllegalStateException if the module has been destroyed
     */
    fun transcribe(
        features: FloatArray,
        batchSize: Int,
        timeSteps: Int,
        featureDim: Int,
        callback: AsrCallback? = null
    ): Int = transcribe(features, batchSize, timeSteps, featureDim, AsrTranscribeConfig(), callback)

    /**
     * Transcribe preprocessed audio features with custom configuration.
     *
     * @param features Preprocessed audio features as a float array
     * @param batchSize Batch size (typically 1)
     * @param timeSteps Number of time steps in the features
     * @param featureDim Feature dimension (e.g., 80 for mel-spectrogram)
     * @param config Configuration for transcription
     * @param callback Callback to receive tokens, can be null
     * @return 0 on success, error code otherwise
     * @throws IllegalStateException if the module has been destroyed
     */
    fun transcribe(
        features: FloatArray,
        batchSize: Int,
        timeSteps: Int,
        featureDim: Int,
        config: AsrTranscribeConfig,
        callback: AsrCallback? = null
    ): Int {
        checkNotDestroyed()
        return nativeTranscribe(
            nativeHandle,
            features,
            batchSize,
            timeSteps,
            featureDim,
            config.maxNewTokens,
            config.temperature,
            config.decoderStartTokenId,
            callback
        )
    }

    /**
     * Transcribe preprocessed audio features and return the full transcription.
     *
     * This is a blocking call that collects all tokens and returns the complete transcription.
     *
     * @param features Preprocessed audio features as a float array
     * @param batchSize Batch size (typically 1)
     * @param timeSteps Number of time steps in the features
     * @param featureDim Feature dimension (e.g., 80 for mel-spectrogram)
     * @param config Configuration for transcription
     * @return The transcribed text
     * @throws RuntimeException if transcription fails
     */
    @JvmOverloads
    fun transcribeBlocking(
        features: FloatArray,
        batchSize: Int,
        timeSteps: Int,
        featureDim: Int,
        config: AsrTranscribeConfig = AsrTranscribeConfig()
    ): String {
        val result = StringBuilder()
        val status = transcribe(
            features,
            batchSize,
            timeSteps,
            featureDim,
            config,
            object : AsrCallback {
                override fun onToken(token: String) {
                    result.append(token)
                }

                override fun onComplete(transcription: String) {
                    // Tokens already collected
                }
            }
        )

        if (status != 0) {
            throw RuntimeException("Transcription failed with error code: $status")
        }

        return result.toString()
    }

    private fun checkNotDestroyed() {
        check(nativeHandle != 0L) { "AsrModule has been destroyed" }
    }
}
