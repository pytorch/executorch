#pragma once

#include <executorch/runtime/executor/executor.h>
#include <executorch/runtime/executor/memory_manager.h>

namespace torch {
namespace executor {
namespace util {

/**
 * An opaque pointer to a serialized bundled program.
 */
using serialized_bundled_program = const void;

/**
 * Load testset_idx-th bundled input of plan_idx-th execution plan test in
 * bundled_program_ptr to given execution plan.
 *
 * @param[in] plan The execution plan going to be verified.
 * @param[in] bundled_program_ptr The bundled program contains expected output.
 * @param[in] plan_idx  The index of execution plan being verified.
 * @param[in] testset_idx  The index of input needs to be set into given
 * execution plan.
 *
 * @returns Return Error::Ok if load successfully, or the error happens during
 * execution.
 */
__ET_NODISCARD Error LoadBundledInput(
    ExecutionPlan& plan,
    serialized_bundled_program* bundled_program_ptr,
    MemoryAllocator* memory_allocator,
    size_t plan_idx,
    size_t testset_idx);

/**
 * Compare the execution plan's output with testset_idx-th bundled expected
 * output in plan_idx-th execution plan test.
 *
 * @param[in] plan The execution plan contains output.
 * @param[in] bundled_program_ptr The bundled program contains expected output.
 * @param[in] plan_idx  The index of execution plan being verified.
 * @param[in] testset_idx  The index of expected output needs to be compared.
 * @param[in] rtol Relative tolerance used for data comparsion.
 * @param[in] atol Absolute tolerance used for data comparsion.
 *
 * @returns Return Error::Ok if two outputs match, or the error happens during
 * execution.
 */
__ET_NODISCARD Error VerifyResultWithBundledExpectedOutput(
    ExecutionPlan& plan,
    serialized_bundled_program* bundled_program_ptr,
    MemoryAllocator* memory_allocator,
    size_t plan_idx,
    size_t testset_idx,
    double rtol = 1e-5,
    double atol = 1e-8);

/**
 * Finds the serialized Executorch program data in the provided file data.
 *
 * The returned buffer is appropriate for constructing a
 * torch::executor::Program.
 *
 * Calling this is only necessary if the file could be a bundled program. If the
 * file will only contain an unwrapped Executorch program, callers can construct
 * torch::executor::Program with file_data directly.
 *
 * @param[in] file_data The contents of an Executorch program or bundled program
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
