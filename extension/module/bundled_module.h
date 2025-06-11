/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/module/module.h>

#ifdef USE_ATEN_LIB
#define ET_BUNDLED_MODULE_NAMESPACE bundled_module::aten
#else // !USE_ATEN_LIB
#define ET_BUNDLED_MODULE_NAMESPACE bundled_module
#endif // USE_ATEN_LIB

namespace executorch {
namespace extension {

using ET_MODULE_NAMESPACE::Module;

namespace ET_BUNDLED_MODULE_NAMESPACE {

/**
 * A facade class for loading bundled programs and executing methods within
 * them.
 */
class BundledModule : public Module {
 public:
  /**
   * Constructs an instance with the bundled program buffer pointer.
   *
   * This constructor reads the program from bundled program buffer to load the
   * module with data loader. The bundled program pointer is preserved so that
   * the portion outside of program is accessible.
   *
   * @param[in] bundled_program_ptr A DataLoader used for loading program data.
   * @param[in] memory_allocator A MemoryAllocator used for memory management.
   * @param[in] temp_allocator A MemoryAllocator to use when allocating
   * temporary data during kernel or delegate execution.
   * @param[in] event_tracer A EventTracer used for tracking and logging events.
   * @param[in] data_map_loader A DataLoader used for loading external weights.
   */
  explicit BundledModule(
      const void* bundled_program_ptr,
      std::unique_ptr<runtime::MemoryAllocator> memory_allocator = nullptr,
      std::unique_ptr<runtime::MemoryAllocator> temp_allocator = nullptr,
      std::unique_ptr<runtime::EventTracer> event_tracer = nullptr,
      std::unique_ptr<runtime::DataLoader> data_map_loader = nullptr);

  // Disallow copying
  BundledModule(const BundledModule&) = delete;
  BundledModule& operator=(const BundledModule&) = delete;
  // Disallow copying
  BundledModule(BundledModule&&) = delete;
  BundledModule& operator=(BundledModule&&) = delete;
  // Default destructor
  ~BundledModule() {
    if (is_loaded_from_file_) {
      delete[] static_cast<const uint8_t*>(bundled_program_ptr_);
    }
  }

  /**
   * Constructs an instance by loading a bundled program from a file with
   * specified memory locking behavior.
   *
   * @param[in] file_path The path to the ExecuTorch bundled program file to
   * load.
   * @param[in] memory_allocator A MemoryAllocator used for memory management.
   * @param[in] temp_allocator A MemoryAllocator to use when allocating
   * temporary data during kernel or delegate execution.
   * @param[in] event_tracer A EventTracer used for tracking and logging events.
   * @param[in] data_map_loader A DataLoader used for loading external weights.
   */
  ET_NODISCARD static runtime::Result<std::unique_ptr<BundledModule>> from_file(
      const std::string& file_path,
      std::unique_ptr<runtime::MemoryAllocator> memory_allocator = nullptr,
      std::unique_ptr<runtime::MemoryAllocator> temp_allocator = nullptr,
      std::unique_ptr<runtime::EventTracer> event_tracer = nullptr,
      std::unique_ptr<runtime::DataLoader> data_map_loader = nullptr);

  using Module::execute;

  /**
   * Execute a specific method with the input value at the given `testset_idx`
   * from the bundle to the method. Loads the program and method before
   * executing if needed.
   *
   * This function is a wrapper of `load_bundled_input` in `bundled_program`.
   *
   * @param[in] method_name The name of the method to execute.
   * @param[in] testset_idx The index of the input value to be passed to the
   * method.
   *
   * @returns Return Error::Ok on a successful load, or the error happens during
   * execution.
   */
  ET_NODISCARD
  runtime::Result<std::vector<runtime::EValue>> execute(
      const std::string& method_name,
      const size_t testset_idx);

  /**
   * Verify the output of a specific method with the expected output from the
   * program bundle at the given `testset_idx`.
   *
   * This function is a wrapper of `verify_method_outputs` in `bundled_program`.
   *
   * @param[in] method_name The name of the method to extract outputs from.
   * @param[in] testset_idx  The index of expected output needs to be compared.
   * @param[in] rtol Relative tolerance used for data comparsion.
   * @param[in] atol Absolute tolerance used for data comparsion.
   *
   * @returns Return Error::Ok if two outputs match, or the error happens during
   * execution.
   */
  ET_NODISCARD
  runtime::Error verify_method_outputs(
      const std::string& method_name,
      const size_t testset_idx,
      double rtol = 1e-5,
      double atol = 1e-8);

 private:
  const void* bundled_program_ptr_;
  bool is_loaded_from_file_ = false;
};

} // namespace ET_BUNDLED_MODULE_NAMESPACE
} // namespace extension
} // namespace executorch
