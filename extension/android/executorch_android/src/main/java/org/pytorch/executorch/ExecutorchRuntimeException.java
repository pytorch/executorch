/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

public class ExecutorchRuntimeException extends RuntimeException {
  // Error code constants - keep in sync with runtime/core/error.h
  // System errors
  public static final int OK = 0x00;
  public static final int INTERNAL = 0x01;
  public static final int INVALID_STATE = 0x02;
  public static final int END_OF_METHOD = 0x03;

  // Logical errors
  public static final int NOT_SUPPORTED = 0x10;
  public static final int NOT_IMPLEMENTED = 0x11;
  public static final int INVALID_ARGUMENT = 0x12;
  public static final int INVALID_TYPE = 0x13;
  public static final int OPERATOR_MISSING = 0x14;
  public static final int REGISTRATION_EXCEEDING_MAX_KERNELS = 0x15;
  public static final int REGISTRATION_ALREADY_REGISTERED = 0x16;

  // Resource errors
  public static final int NOT_FOUND = 0x20;
  public static final int MEMORY_ALLOCATION_FAILED = 0x21;
  public static final int ACCESS_FAILED = 0x22;
  public static final int INVALID_PROGRAM = 0x23;
  public static final int INVALID_EXTERNAL_DATA = 0x24;
  public static final int OUT_OF_RESOURCES = 0x25;

  // Delegate errors
  public static final int DELEGATE_INVALID_COMPATIBILITY = 0x30;
  public static final int DELEGATE_MEMORY_ALLOCATION_FAILED = 0x31;
  public static final int DELEGATE_INVALID_HANDLE = 0x32;

  private static final Map<Integer, String> ERROR_CODE_MESSAGES;

  static {
    Map<Integer, String> map = new HashMap<>();

    // System errors
    map.put(OK, "Operation successful");
    map.put(INTERNAL, "Internal error");
    map.put(INVALID_STATE, "Invalid state");
    map.put(END_OF_METHOD, "End of method reached");
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
    private static final boolean ENABLE_READ_LOG_BUFFER = false;
    // Reusable StringBuilder instance
    private static final StringBuilder sb = new StringBuilder();

    static String formatMessage(int errorCode, String details) {
      synchronized (sb) {
        sb.setLength(0); // Clear the StringBuilder before use

        String baseMessage = ERROR_CODE_MESSAGES.get(errorCode);
        if (baseMessage == null) {
          baseMessage = "Unknown error code 0x" + Integer.toHexString(errorCode);
        }

        sb.append("[Executorch Error 0x")
            .append(Integer.toHexString(errorCode))
            .append("] ")
            .append(baseMessage)
            .append(": ")
            .append(details);
        if (ENABLE_READ_LOG_BUFFER) {
          try {
            sb.append("\nDetailed Logs:\n");
            String[] logEntries = readLogBuffer(); // JNI call
            formatLogEntries(sb, logEntries);
          } catch (Exception e) {
            sb.append("Failed to retrieve detailed logs: ").append(e.getMessage());
          }
        }

        return sb.toString();
      }
    }

    // Native JNI method declaration
    private static native String[] readLogBuffer();

    // Append log entries to the provided StringBuilder
    private static void formatLogEntries(StringBuilder sb, String[] logEntries) {
      if (logEntries == null || logEntries.length == 0) {
        sb.append("No detailed logs available.");
        return;
      }
      for (String entry : logEntries) {
        sb.append(entry).append("\n");
      }
    }
  }

  private final int errorCode;

  public ExecutorchRuntimeException(int errorCode, String details) {
    super(ErrorHelper.formatMessage(errorCode, details));
    this.errorCode = errorCode;
  }

  public int getErrorCode() {
    return errorCode;
  }

  // Idiomatic Java exception for invalid arguments.
  public static class ExecutorchInvalidArgumentException extends IllegalArgumentException {
    private final int errorCode = INVALID_ARGUMENT;

    public ExecutorchInvalidArgumentException(String details) {
      super(ErrorHelper.formatMessage(INVALID_ARGUMENT, details));
    }

    public int getErrorCode() {
      return errorCode;
    }
  }

  // Factory method to create an exception of the appropriate subclass.
  public static RuntimeException makeExecutorchException(int errorCode, String details) {
    switch (errorCode) {
      case INVALID_ARGUMENT:
        return new ExecutorchInvalidArgumentException(details);
      default:
        return new ExecutorchRuntimeException(errorCode, details);
    }
  }
}
