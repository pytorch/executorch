/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch

import com.facebook.jni.annotations.DoNotStrip

/**
 * Base exception for all ExecuTorch runtime errors. Each instance carries an integer error code
 * corresponding to the native `runtime/core/error.h` values, accessible via [getErrorCode].
 */
open class ExecutorchRuntimeException
@DoNotStrip
constructor(
    val errorCode: Int,
    details: String?,
) : RuntimeException(ErrorHelper.formatMessage(errorCode, details)) {

  constructor(
      errorCode: Int,
      details: String?,
      cause: Throwable?,
  ) : this(errorCode, details) {
    if (cause != null) initCause(cause)
  }

  /** Returns detailed log output captured from the native runtime, if available. */
  fun getDetailedError(): String = ErrorHelper.getDetailedErrorLogs()

  @DoNotStrip
  class ExecutorchInvalidArgumentException @DoNotStrip constructor(details: String?) :
      ExecutorchRuntimeException(INVALID_ARGUMENT, details)

  private object ErrorHelper {
    private val ERROR_CODE_MESSAGES: Map<Int, String> =
        mapOf(
            // System errors
            OK to "Operation successful",
            INTERNAL to "Internal error",
            INVALID_STATE to "Invalid state",
            END_OF_METHOD to "End of method reached",
            ALREADY_LOADED to "Already loaded",
            // Logical errors
            NOT_SUPPORTED to "Operation not supported",
            NOT_IMPLEMENTED to "Operation not implemented",
            INVALID_ARGUMENT to "Invalid argument",
            INVALID_TYPE to "Invalid type",
            OPERATOR_MISSING to "Operator missing",
            REGISTRATION_EXCEEDING_MAX_KERNELS to "Exceeded max kernels",
            REGISTRATION_ALREADY_REGISTERED to "Kernel already registered",
            // Resource errors
            NOT_FOUND to "Resource not found",
            MEMORY_ALLOCATION_FAILED to "Memory allocation failed",
            ACCESS_FAILED to "Access failed",
            INVALID_PROGRAM to "Invalid program",
            INVALID_EXTERNAL_DATA to "Invalid external data",
            OUT_OF_RESOURCES to "Out of resources",
            // Delegate errors
            DELEGATE_INVALID_COMPATIBILITY to "Delegate invalid compatibility",
            DELEGATE_MEMORY_ALLOCATION_FAILED to "Delegate memory allocation failed",
            DELEGATE_INVALID_HANDLE to "Delegate invalid handle",
        )

    fun formatMessage(errorCode: Int, details: String?): String {
      val baseMessage =
          ERROR_CODE_MESSAGES[errorCode] ?: "Unknown error code 0x${Integer.toHexString(errorCode)}"
      val safeDetails = details ?: "No details provided"
      return "[ExecuTorch Error 0x${Integer.toHexString(errorCode)}] $baseMessage: $safeDetails"
    }

    fun getDetailedErrorLogs(): String {
      val sb = StringBuilder()
      try {
        val logEntries = Module.readLogBufferStatic() // JNI call
        if (logEntries != null && logEntries.isNotEmpty()) {
          sb.append("\nDetailed logs:\n")
          for (entry in logEntries) {
            sb.append(entry).append("\n")
          }
        }
      } catch (e: Exception) {
        return ""
      }
      return sb.toString()
    }
  }

  companion object {
    // Error code constants - keep in sync with runtime/core/error.h

    // System errors
    const val OK = 0x00
    const val INTERNAL = 0x01
    const val INVALID_STATE = 0x02
    const val END_OF_METHOD = 0x03
    const val ALREADY_LOADED = 0x04

    // Logical errors
    const val NOT_SUPPORTED = 0x10
    const val NOT_IMPLEMENTED = 0x11
    const val INVALID_ARGUMENT = 0x12
    const val INVALID_TYPE = 0x13
    const val OPERATOR_MISSING = 0x14
    const val REGISTRATION_EXCEEDING_MAX_KERNELS = 0x15
    const val REGISTRATION_ALREADY_REGISTERED = 0x16

    // Resource errors
    const val NOT_FOUND = 0x20
    const val MEMORY_ALLOCATION_FAILED = 0x21
    const val ACCESS_FAILED = 0x22
    const val INVALID_PROGRAM = 0x23
    const val INVALID_EXTERNAL_DATA = 0x24
    const val OUT_OF_RESOURCES = 0x25

    // Delegate errors
    const val DELEGATE_INVALID_COMPATIBILITY = 0x30
    const val DELEGATE_MEMORY_ALLOCATION_FAILED = 0x31
    const val DELEGATE_INVALID_HANDLE = 0x32

    @DoNotStrip
    @JvmStatic
    fun makeExecutorchException(errorCode: Int, details: String?): RuntimeException {
      val nativeTail =
          try {
            ErrorHelper.getDetailedErrorLogs()
                .removePrefix("\nDetailed logs:\n")
                .replace(Regex("\\s+"), " ")
                .trim()
          } catch (t: Throwable) {
            ""
          }
      val enrichedDetails =
          if (nativeTail.isNotBlank()) {
            "${details ?: "No details provided"} | nativeLog=${nativeTail.takeLast(NATIVE_LOG_TAIL_MAX_CHARS)}"
          } else {
            details
          }
      return when (errorCode) {
        INVALID_ARGUMENT -> ExecutorchInvalidArgumentException(enrichedDetails)
        else -> ExecutorchRuntimeException(errorCode, enrichedDetails)
      }
    }

    private const val NATIVE_LOG_TAIL_MAX_CHARS = 2048
  }
}
