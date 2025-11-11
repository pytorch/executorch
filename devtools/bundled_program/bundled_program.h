/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * Copyright 2025 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/method.h>
#ifdef USE_ATEN_LIB
#define BUNDLED_PROGRAM_NAMESPACE bundled_program::aten
#else // !USE_ATEN_LIB
#define BUNDLED_PROGRAM_NAMESPACE bundled_program
#endif // USE_ATEN_LIB

namespace executorch {
namespace BUNDLED_PROGRAM_NAMESPACE {

/**
 * An opaque pointer to a serialized bundled program.
 */
using SerializedBundledProgram = const void;
using ::executorch::ET_RUNTIME_NAMESPACE::Method;
/**
 * Load testset_idx-th bundled input of method_idx-th Method test in
 * bundled_program_ptr to given Method.
 *
 * @param[in] method The Method to verify.
 * @param[in] bundled_program_ptr The bundled program contains expected output.
 * @param[in] testset_idx  The index of input needs to be set into given Method.
 *
 * @returns Return Error::Ok if load successfully, or the error happens during
 * execution.
 */
ET_NODISCARD ::executorch::runtime::Error load_bundled_input(
    Method& method,
    SerializedBundledProgram* bundled_program_ptr,
    size_t testset_idx);

struct ErrorStats {
  ::executorch::runtime::Error status;
  double mean_abs_error;
  double max_abs_error;
  double mean_relative_error;
  double max_relative_error;
};

/**
 * Compute error stats for method.outputs() vs. the bundled "expected_outputs"
 * for testset_idx.
 *
 * @param[in] method The Method to extract outputs from.
 * @param[in] bundled_program_ptr The bundled program contains expected output.
 * @param[in] testset_idx  The index of expected output needs to be compared.
 *
 * @returns Return ErrorStats with status set to Error::Ok if stats are filled
 * in.
 */

ET_NODISCARD ErrorStats compute_method_output_error_stats(
    Method& method,
    SerializedBundledProgram* bundled_program_ptr,
    size_t testset_idx);

/**
 * Compare the Method's output with testset_idx-th bundled expected
 * output in method_idx-th Method test.
 *
 * @param[in] method The Method to extract outputs from.
 * @param[in] bundled_program_ptr The bundled program contains expected output.
 * @param[in] testset_idx  The index of expected output needs to be compared.
 * @param[in] rtol Relative tolerance used for data comparsion.
 * @param[in] atol Absolute tolerance used for data comparsion.
 *
 * @returns Return Error::Ok if two outputs match, or the error happens during
 * execution.
 */
ET_NODISCARD ::executorch::runtime::Error verify_method_outputs(
    Method& method,
    SerializedBundledProgram* bundled_program_ptr,
    size_t testset_idx,
    double rtol = 1e-5,
    double atol = 1e-8);

/**
 * Finds the serialized ExecuTorch program data in the provided bundled program
 * file data.
 *
 * The returned buffer is appropriate for constructing a
 * torch::executor::Program.
 *
 * @param[in] file_data The contents of an ExecuTorch program or bundled program
 *                      file.
 * @param[in] file_data_len The length of file_data, in bytes.
 * @param[out] out_program_data The serialized Program data, if found.
 * @param[out] out_program_data_len The length of out_program_data, in bytes.
 *
 * @returns Error::Ok if the given file is bundled program, a program was found
 * in it, and out_program_data/out_program_data_len point to the data. Other
 * values on failure.
 */
ET_NODISCARD ::executorch::runtime::Error get_program_data(
    void* file_data,
    size_t file_data_len,
    const void** out_program_data,
    size_t* out_program_data_len);

/**
 * Checks whether the given file is a bundled program.
 *
 * @param[in] file_data The contents of the given file.
 * @param[in] file_data_len The length of file_data, in bytes.
 *
 * @returns true if the given file is a bundled program, false otherwise
 */
bool is_bundled_program(void* file_data, size_t file_data_len);

/// DEPRECATED: Use the version with the file_data_len parameter.
ET_DEPRECATED inline bool is_bundled_program(void* file_data) {
  // 128 is enough data to contain the identifier in the flatbuffer header.
  return is_bundled_program(file_data, 128);
}

} // namespace BUNDLED_PROGRAM_NAMESPACE
} // namespace executorch

namespace torch {
namespace executor {
namespace bundled_program {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using serialized_bundled_program =
    ::executorch::BUNDLED_PROGRAM_NAMESPACE::SerializedBundledProgram;

ET_NODISCARD inline ::executorch::runtime::Error LoadBundledInput(
    Method& method,
    serialized_bundled_program* bundled_program_ptr,
    size_t testset_idx) {
  return ::executorch::BUNDLED_PROGRAM_NAMESPACE::load_bundled_input(
      method, bundled_program_ptr, testset_idx);
}

ET_NODISCARD inline ::executorch::runtime::Error
VerifyResultWithBundledExpectedOutput(
    Method& method,
    serialized_bundled_program* bundled_program_ptr,
    size_t testset_idx,
    double rtol = 1e-5,
    double atol = 1e-8) {
  return ::executorch::BUNDLED_PROGRAM_NAMESPACE::verify_method_outputs(
      method, bundled_program_ptr, testset_idx, rtol, atol);
}

ET_NODISCARD inline ::executorch::runtime::Error GetProgramData(
    void* file_data,
    size_t file_data_len,
    const void** out_program_data,
    size_t* out_program_data_len) {
  return ::executorch::BUNDLED_PROGRAM_NAMESPACE::get_program_data(
      file_data, file_data_len, out_program_data, out_program_data_len);
}

inline bool IsBundledProgram(void* file_data) {
  // 128 is enough data to contain the identifier in the flatbuffer header.
  return ::executorch::BUNDLED_PROGRAM_NAMESPACE::is_bundled_program(
      file_data, 128);
}
} // namespace bundled_program
} // namespace executor
} // namespace torch
