/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/// Logging level of the delegate and QNN backend.
typedef enum { // NOLINT(modernize-use-using)
  kLogOff = 0,
  kLogLevelError,
  kLogLevelWarn,
  kLogLevelInfo,
  kLogLevelVerbose,
  kLogLevelDebug,
} QnnExecuTorchLogLevel;

/// The QNN backend used to delegate the model's nodes. Each backend has
/// its own set of supported ops and tensor types.
typedef enum { // NOLINT(modernize-use-using)
  kUndefinedBackend = 0,
  /// Backend for Adreno<sup>TM</sup> GPU hardware accelerator.
  kGpuBackend,
  /// Backend for Hexagon HTP hardware accelerator.
  kHtpBackend,
  /// Backend for Hexagon DSP hardware accelerator.
  kDspBackend,
} QnnExecuTorchBackendType;

/// Defines performance modes available for HTP backend.
typedef enum { // NOLINT(modernize-use-using)
  kHtpDefault = 0,
  kHtpSustainedHighPerformance,
  kHtpBurst,
  kHtpHighPerformance,
  kHtpPowerSaver,
  kHtpLowPowerSaver,
  kHtpHighPowerSaver,
  kHtpLowBalanced,
  kHtpBalanced,
} QnnExecuTorchHtpPerformanceMode;

/// Defines pd sessions available for HTP backend.
typedef enum { // NOLINT(modernize-use-using)
  kHtpUnsignedPd = 0,
  kHtpSignedPd,
} QnnExecuTorchHtpPdSession;

/// Defines the optimization levels of the graph tensors that are not input nor
/// output tensors. This enum controls the trade-off between performance and
/// accuracy.
typedef enum { // NOLINT(modernize-use-using)
  kHtpQuantized = 0,
  kHtpFp16,
} QnnExecuTorchHtpPrecision;

/// Please add new entries only at the end, and assign the next value. Do not
/// reuse values of retired chipsets.
typedef enum {
  UNKNOWN_SM = 0,
  SM8450 = 36, // v69
  SM8475 = 42, // v69
  SM8550 = 43, // v73
  SM8650 = 57, // v75
} QcomChipset;

/// Specifies the backend options for the HTP backend.
typedef struct { // NOLINT
  /// Specify SoC to generate HTP Offline Cache for.
  QcomChipset soc_model;
  /// The default performance mode sets no configurations on the HTP.
  QnnExecuTorchHtpPerformanceMode performance_mode;
  /// The default precision mode supports quantized networks. Other precision
  /// modes may only be supported on certain SoCs.
  QnnExecuTorchHtpPrecision precision;
  /// Signed or unsigned HTP PD session. The default PD session is unsigned.
  QnnExecuTorchHtpPdSession pd_session;
  /// With using conv hmx with short depths, we might have better performance,
  /// but convolution that have short depth and/or weights that are not
  /// symmetric could exhibit inaccurate results.
  bool use_conv_hmx;
  /// With using fold relu, we might have better performance, this optimization
  /// is correct when quantization ranges for convolution are equal or subset of
  /// the Relu operation.
  bool use_fold_relu;
} QnnExecuTorchHtpBackendOptions;

// clang-format off
#define QNN_EXECUTORCH_HTP_OPTION_INIT           \
  {                                              \
    SM8550,             /*soc_model*/            \
    kHtpDefault,        /*performance_mode*/     \
    kHtpQuantized,      /*precision*/            \
    kHtpUnsignedPd,     /*pd_session*/           \
    true,               /*use_conv_hmx*/           \
    true,               /*use_fold_relu*/          \
  }
// clang-format on

typedef struct {
  /// qnn_context_binary_blob
  void* buffer;
  /// number of bytes of buffer
  uint64_t nbytes;
} QnnExecuTorchContextBinary;

// clang-format off
#define QNN_EXECUTORCH_CONTEXT_BINARY    \
  {                                      \
    nullptr,        /*buffer*/           \
    0,              /*nbytes*/           \
  }
// clang-format on

typedef struct { // NOLINT
  /// The backend QNN library to open and execute the graph with. This is a
  /// required argument and will error out if kUndefinedBackend is supplied.
  QnnExecuTorchBackendType backend_type;

  /// Optional parameter to override the QNN backend library.
  const char* library_path;

  /// Optional parameter specifying the directory of QNN Skel library. Only
  /// useful for backends which have a Skel library.
  const char* skel_library_dir;

  /// Optional parameter to create qnn graph if not give QNN context blob
  const char* graph_name;

  /// Optional backend specific options for the HTP backend.
  QnnExecuTorchHtpBackendOptions htp_options;

  /// Logging level of the delegate and the backend. Default is off.
  QnnExecuTorchLogLevel log_level;

  /// QNN context blob is the same as SizedBuffer given by user
  QnnExecuTorchContextBinary qnn_context_blob;

} QnnExecuTorchOptions;

// clang-format off
#define QNN_EXECUTORCH_OPTION_INIT                                      \
  {                                                                     \
    kUndefinedBackend,                   /*backend_type*/               \
    "",                                  /*library_path*/               \
    "",                                  /*skel_library_dir*/           \
    "",                                  /*graph_name*/                 \
    QNN_EXECUTORCH_HTP_OPTION_INIT,      /*htp_options*/                \
    kLogOff,                             /*log_level*/                  \
    QNN_EXECUTORCH_CONTEXT_BINARY,       /*qnn_context_blob*/           \
  }
// clang-format on

/// Create the QNN Delegate options structure and populate with default values.
QnnExecuTorchOptions QnnExecuTorchOptionsDefault();

#ifdef __cplusplus
}
#endif // __cplusplus
