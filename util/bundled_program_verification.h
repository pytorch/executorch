/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/method.h>

namespace torch {
namespace executor {
namespace util {

/**
 * An opaque pointer to a serialized bundled program.
 */
using serialized_bundled_program = const void;

/**
 * Load testset_idx-th bundled input of method_idx-th Method test in
 * bundled_program_ptr to given Method.
 *
 * @param[in] method The Method to verify.
 * @param[in] bundled_program_ptr The bundled program contains expected output.
 * @param[in] method_name  The name of the Method being verified.
 * @param[in] testset_idx  The index of input needs to be set into given Method.
 *
 * @returns Return Error::Ok if load successfully, or the error happens during
 * execution.
 */
__ET_NODISCARD Error LoadBundledInput(
    Method& method,
    serialized_bundled_program* bundled_program_ptr,
    MemoryAllocator* memory_allocator,
    const char* method_name,
    size_t testset_idx);

/**
 * Compare the Method's output with testset_idx-th bundled expected
 * output in method_idx-th Method test.
 *
 * @param[in] method The Method to extract outputs from.
 * @param[in] bundled_program_ptr The bundled program contains expected output.
 * @param[in] method_name  The name of the Method being verified.
 * @param[in] testset_idx  The index of expected output needs to be compared.
 * @param[in] rtol Relative tolerance used for data comparsion.
 * @param[in] atol Absolute tolerance used for data comparsion.
 *
 * @returns Return Error::Ok if two outputs match, or the error happens during
 * execution.
 */
__ET_NODISCARD Error VerifyResultWithBundledExpectedOutput(
    Method& method,
    serialized_bundled_program* bundled_program_ptr,
    MemoryAllocator* memory_allocator,
    const char* method_name,
    size_t testset_idx,
    double rtol = 1e-5,
    double atol = 1e-8);

/**
 * Finds the serialized ExecuTorch program data in the provided file data.
 *
 * The returned buffer is appropriate for constructing a
 * torch::executor::Program.
 *
 * Calling this is only necessary if the file could be a bundled program. If the
 * file will only contain an unwrapped ExecuTorch program, callers can construct
 * torch::executor::Program with file_data directly.
 *
 * @param[in] file_data The contents of an ExecuTorch program or bundled program
 *                      file.
 * @param[in] file_data_len The length of file_data, in bytes.
 * @param[out] out_program_data The serialized Program data, if found.
 * @param[out] out_program_data_len The length of out_program_data, in bytes.
 *
 * @returns Error::Ok if the program was found, and
 *     out_program_data/out_program_data_len point to the data. Other values
 *     on failure.
 */
__ET_NODISCARD Error GetProgramData(
    void* file_data,
    size_t file_data_len,
    const void** out_program_data,
    size_t* out_program_data_len);

} // namespace util
} // namespace executor
} // namespace torch
