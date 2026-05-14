/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch;

import com.facebook.jni.annotations.DoNotStrip;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Base exception for all ExecuTorch runtime errors. Each instance carries an integer error code
 * corresponding to the native {@code runtime/core/error.h} values, accessible via {@link
 * #getErrorCode()}.
 */
public class ExecutorchRuntimeException extends RuntimeException {
  // Error code constants - keep in sync with runtime/core/error.h

  // System errors

  /** Operation completed successfully. */
  public static final int OK = 0x00;

  /** An unexpected internal error occurred in the runtime. */
  public static final int INTERNAL = 0x01;

  /** The runtime or method is in an invalid state for the requested operation. */
  public static final int INVALID_STATE = 0x02;

  /** The method has finished execution and has no more work to do. */
  public static final int END_OF_METHOD = 0x03;

  /** A required resource has already been loaded. */
  public static final int ALREADY_LOADED = 0x04;

  // Logical errors

  /** The requested operation is not supported by this build or backend. */
  public static final int NOT_SUPPORTED = 0x10;

  /** The requested operation has not been implemented. */
  public static final int NOT_IMPLEMENTED = 0x11;

  /** One or more arguments passed to the operation are invalid. */
  public static final int INVALID_ARGUMENT = 0x12;

  /** A value or tensor has an unexpected type. */
  public static final int INVALID_TYPE = 0x13;

  /** A required operator kernel is not registered. */
  public static final int OPERATOR_MISSING = 0x14;

  /** The maximum number of registered kernels has been exceeded. */
  public static final int REGISTRATION_EXCEEDING_MAX_KERNELS = 0x15;

  /** A kernel with the same name is already registered. */
  public static final int REGISTRATION_ALREADY_REGISTERED = 0x16;

  // Resource errors

  /** A required resource (file, tensor, program) was not found. */
  public static final int NOT_FOUND = 0x20;

  /** A memory allocation failed. */
  public static final int MEMORY_ALLOCATION_FAILED = 0x21;

  /** Access to a resource was denied or failed. */
  public static final int ACCESS_FAILED = 0x22;

  /** The loaded program is malformed or incompatible. */
  public static final int INVALID_PROGRAM = 0x23;

  /** External data referenced by the program is invalid or missing. */
  public static final int INVALID_EXTERNAL_DATA = 0x24;

  /** The system has run out of a required resource. */
  public static final int OUT_OF_RESOURCES = 0x25;

  // Delegate errors

  /** A delegate reported an incompatible model or configuration. */
  public static final int DELEGATE_INVALID_COMPATIBILITY = 0x30;

  /** A delegate failed to allocate required memory. */
  public static final int DELEGATE_MEMORY_ALLOCATION_FAILED = 0x31;

  /** A delegate received an invalid or stale handle. */
  public static final int DELEGATE_INVALID_HANDLE = 0x32;

  private static final Map<Integer, String> ERROR_CODE_MESSAGES;

  static {
    Map<Integer, String> map = new HashMap<>();

    // System errors
    map.put(OK, "Operation successful");
    map.put(INTERNAL, "Internal error");
    map.put(INVALID_STATE, "Invalid state");
    map.put(END_OF_METHOD, "End of method reached");
    map.put(ALREADY_LOADED, "Already loaded");
    // Logical errors
    map.put(NOT_SUPPORTED, "Operation not supported");
    map.put(NOT_IMPLEMENTED, "Operation not implemented");
    map.put(INVALID_ARGUMENT, "Invalid argument");
    map.put(INVALID_TYPE, "Invalid type");
    map.put(OPERATOR_MISSING, "Operator missing");
    map.put(REGISTRATION_EXCEEDING_MAX_KERNELS, "Exceeded max kernels");
    map.put(REGISTRATION_ALREADY_REGISTERED, "Kernel already registered");
    // Resource errors
    map.put(NOT_FOUND, "Resource not found");
    map.put(MEMORY_ALLOCATION_FAILED, "Memory allocation failed");
    map.put(ACCESS_FAILED, "Access failed");
    map.put(INVALID_PROGRAM, "Invalid program");
    map.put(INVALID_EXTERNAL_DATA, "Invalid external data");
    map.put(OUT_OF_RESOURCES, "Out of resources");
    // Delegate errors
    map.put(DELEGATE_INVALID_COMPATIBILITY, "Delegate invalid compatibility");
    map.put(DELEGATE_MEMORY_ALLOCATION_FAILED, "Delegate memory allocation failed");
    map.put(DELEGATE_INVALID_HANDLE, "Delegate invalid handle");
    ERROR_CODE_MESSAGES = Collections.unmodifiableMap(map);
  }

  static class ErrorHelper {
    static String formatMessage(int errorCode, String details) {
      String baseMessage = ERROR_CODE_MESSAGES.get(errorCode);
      if (baseMessage == null) {
        baseMessage = "Unknown error code 0x" + Integer.toHexString(errorCode);
      }

      String safeDetails = details != null ? details : "No details provided";
      return String.format(
          "[ExecuTorch Error 0x%s] %s: %s",
          Integer.toHexString(errorCode), baseMessage, safeDetails);
    }

    static String getDetailedErrorLogs() {
      StringBuilder sb = new StringBuilder();
      try {
        String[] logEntries = Module.readLogBufferStatic(); // JNI call
        if (logEntries != null && logEntries.length > 0) {
          sb.append("\nDetailed logs:\n");
          for (String entry : logEntries) {
            sb.append(entry).append("\n");
          }
        }
      } catch (Exception e) {
        sb.append("Failed to retrieve detailed logs: ").append(e.getMessage());
      }
      return sb.toString();
    }
  }

  private final int errorCode;

  @DoNotStrip
  public ExecutorchRuntimeException(int errorCode, String details) {
    super(ErrorHelper.formatMessage(errorCode, details));
    this.errorCode = errorCode;
  }

  public ExecutorchRuntimeException(int errorCode, String details, Throwable cause) {
    super(ErrorHelper.formatMessage(errorCode, details), cause);
    this.errorCode = errorCode;
  }

  /** Returns the numeric error code from {@code runtime/core/error.h}. */
  public int getErrorCode() {
    return errorCode;
  }

  /** Returns detailed log output captured from the native runtime, if available. */
  public String getDetailedError() {
    return ErrorHelper.getDetailedErrorLogs();
  }

  @DoNotStrip
  public static class ExecutorchInvalidArgumentException extends ExecutorchRuntimeException {
    @DoNotStrip
    public ExecutorchInvalidArgumentException(String details) {
      super(INVALID_ARGUMENT, details);
    }
  }

  @DoNotStrip
  public static RuntimeException makeExecutorchException(int errorCode, String details) {
    switch (errorCode) {
      case INVALID_ARGUMENT:
        return new ExecutorchInvalidArgumentException(details);
      default:
        return new ExecutorchRuntimeException(errorCode, details);
    }
  }
}
