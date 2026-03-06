/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>

namespace executorch {
namespace backends {
namespace tensorrt {

/**
 * TensorRT backend for executing models on NVIDIA GPUs.
 *
 * This backend deserializes TensorRT engines from blobs created by the
 * Python preprocess() function and executes them using the TensorRT runtime.
 */
class TensorRTBackend final : public runtime::BackendInterface {
 public:
  TensorRTBackend() = default;
  ~TensorRTBackend() override = default;

  /**
   * Returns true if TensorRT is available on this device.
   *
   * Checks for:
   * - TensorRT runtime library availability
   * - CUDA device availability
   */
  bool is_available() const override;

  /**
   * Initialize the TensorRT backend with a serialized engine blob.
   *
   * Parses the blob header, extracts I/O binding metadata, and deserializes
   * the TensorRT engine. Creates an execution context for inference.
   *
   * @param context Backend initialization context.
   * @param processed Blob containing the serialized TensorRT engine.
   * @param compile_specs Compilation specifications (unused at runtime).
   * @return DelegateHandle pointer on success, error otherwise.
   */
  runtime::Result<runtime::DelegateHandle*> init(
      runtime::BackendInitContext& context,
      runtime::FreeableBuffer* processed,
      runtime::ArrayRef<runtime::CompileSpec> compile_specs) const override;

  /**
   * Execute inference using the TensorRT engine.
   *
   * Binds input tensors from args to TensorRT input bindings, runs inference,
   * and copies results to output tensors.
   *
   * @param context Backend execution context.
   * @param handle DelegateHandle returned by init().
   * @param args Input and output EValues.
   * @return Error::Ok on success.
   */
  runtime::Error execute(
      runtime::BackendExecutionContext& context,
      runtime::DelegateHandle* handle,
      runtime::Span<runtime::EValue*> args) const override;

  /**
   * Destroy the delegate handle and release TensorRT resources.
   *
   * @param handle DelegateHandle to destroy.
   */
  void destroy(runtime::DelegateHandle* handle) const override;
};

} // namespace tensorrt
} // namespace backends
} // namespace executorch
