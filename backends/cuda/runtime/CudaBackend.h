// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>

namespace executorch::backends::cuda {

class ET_EXPERIMENTAL CudaBackend final
    : public ::executorch::runtime::BackendInterface {
 private:
  /**
   * Load AOTI function pointers from the shared library into the handle.
   */
  ::executorch::runtime::Error load_function_pointers_into_handle(
      void* so_handle,
      struct AOTIDelegateHandle* handle) const;

 public:
  /**
   * Check if the CUDA backend is available.
   */
  bool is_available() const override;

  /**
   * Initialize the backend with the given context and compile specs.
   * Called once per loaded binary blob.
   */
  ::executorch::runtime::Result<::executorch::runtime::DelegateHandle*> init(
      ::executorch::runtime::BackendInitContext& context,
      ::executorch::runtime::FreeableBuffer* processed,
      ::executorch::runtime::ArrayRef<::executorch::runtime::CompileSpec>
          compile_specs) const override;

  /**
   * Execute the backend with the given context and arguments.
   * Called once per execution.
   */
  ::executorch::runtime::Error execute(
      ::executorch::runtime::BackendExecutionContext& context,
      ::executorch::runtime::DelegateHandle* handle,
      ::executorch::runtime::Span<::executorch::runtime::EValue*> args)
      const override;

  /**
   * Destroy the backend handle and clean up resources.
   */
  void destroy(::executorch::runtime::DelegateHandle* handle) const override;
};

} // namespace executorch::backends::cuda
