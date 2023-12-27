/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cinttypes>
#include <cstdint>

#include <executorch/runtime/core/data_loader.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/event_tracer.h>
#include <executorch/runtime/core/freeable_buffer.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/method_meta.h>
#include <executorch/runtime/platform/compiler.h>

// Forward declare flatbuffer types. This is a public header and must not
// include the generated flatbuffer header.
namespace executorch_flatbuffer {
struct Program;
} // namespace executorch_flatbuffer

namespace torch {
namespace executor {

namespace testing {
// Provides test access to private Program methods.
class ProgramTestFriend;
} // namespace testing

/**
 * A deserialized ExecuTorch program binary.
 */
class Program final {
 public:
  /**
   * Types of validation that the Program can do before parsing the data.
   */
  enum class Verification : uint8_t {
    /**
     * Do minimal verification of the data, ensuring that the header appears
     * correct.
     *
     * Has minimal runtime overhead.
     */
    Minimal,
    /**
     * Do full verification of the data, ensuring that internal pointers are
     * self-consistent and that the data has not been truncated or obviously
     * corrupted. May not catch all types of corruption, but should guard
     * against illegal memory operations during parsing.
     *
     * Will have higher runtime overhead, scaling with the complexity of the
     * proram data.
     */
    InternalConsistency,
  };

  /**
   * Loads a Program from the provided loader. The Program will hold a pointer
   * to the loader, which must outlive the returned Program instance.
   *
   * @param[in] loader The source to load program data from. The Program will
   *     hold a pointer to this loader, which must outlive the returned Program
   *     instance.
   * @param[in] verification The type of verification to do before returning
   *     success.
   */
  __ET_NODISCARD static Result<Program> load(
      DataLoader* loader,
      Verification verification = Verification::Minimal);

  /// DEPRECATED: Use the lowercase `load()` instead.
  __ET_DEPRECATED __ET_NODISCARD static Result<Program> Load(
      DataLoader* loader,
      Verification verification = Verification::Minimal) {
    return load(loader, verification);
  }

  // Movable, to be compatible with Result.
  Program(Program&&) noexcept = default;
  ~Program() = default;

  /**
   * Get the constant buffer inside Program with index buffer_idx.
   * @param[in] buffer_idx the index of the buffer in the constant_buffer.
   * @param[in] nbytes the number of bytes to read from the buffer.
   * @return The buffer with corresponding index.
   */
  Result<const void*> get_constant_buffer_data(size_t buffer_idx, size_t nbytes)
      const;

  /**
   * Returns the number of methods in the program.
   */
  size_t num_methods() const;

  /**
   * Returns the name of the method at particular index.
   *
   * @param[in] method_index The index of the method name to retrieve. Must be
   * less than the value returned by `num_methods()`.
   *
   * @returns The name of the requested method. The pointer is owned by the
   * Program, and has the same lifetime as the Program.
   */
  Result<const char*> get_method_name(size_t method_index) const;

  /**
   * Loads the named method and prepares it for execution.
   *
   * @param[in] method_name The name of the method to load.
   * @param[in] memory_manager The allocators to use during initialization and
   *     execution of the loaded method.
   * @param[in] event_tracer The event tracer to use for this method run.
   *
   * @returns The loaded method on success, or an error on failure.
   */
  Result<Method> load_method(
      const char* method_name,
      MemoryManager* memory_manager,
      EventTracer* event_tracer = nullptr) const;

  /**
   * Gathers metadata for the named method.
   *
   * @param[in] method_name The name of the method to get metadata for.
   */
  Result<MethodMeta> method_meta(const char* method_name) const;

  /**
   * DEPRECATED: Use MethodMeta instead.
   *
   * Get the size of the buffer with index buffer_index. Note that this function
   * does not return the correct value for index 0 which denotes constant
   * memory. Only index >= 1 should be used to retrieve the size of
   * non-constant pools.
   * @param[in] buffer_index the index of the buffer in the non_const_buffer
   * list
   * @param[in] method_name The name of the method to retrieve buffer
   * information from.
   * @return The size of the non_constant buffer corresponding to buffer_index,
   * or Error if it cannot be retrieved.
   */
  __ET_DEPRECATED Result<int64_t> get_non_const_buffer_size(
      size_t buffer_index,
      const char* method_name = "forward") const;

  /**
   * DEPRECATED: Use MethodMeta instead.
   *
   * Get the number of non_constant buffers.
   * @param[in] method_name The name of the method to get the buffer amounts
   * for.
   * @return The number of non_constant buffers, or Error if it cannot be
   * retrieved.
   */
  __ET_DEPRECATED Result<size_t> num_non_const_buffers(
      const char* method_name = "forward") const;

  /**
   * DEPRECATED: Get the pytree encoding string for the output. Deprecated as
   * this functionality will eventually move out of the core program into a
   * higher level structure, but that does not exist at this time.
   * @param[in] method_name The name of the method to get the encoding for.
   *
   * @return The pytree encoding string for the output
   */
  __ET_DEPRECATED Result<const char*> get_output_flattening_encoding(
      const char* method_name = "forward") const;

  /**
   * Describes the presence of an ExecuTorch program header.
   */
  enum HeaderStatus {
    /**
     * An ExecuTorch program header is present, and its version is compatible
     * with this version of the runtime.
     */
    CompatibleVersion,

    /**
     * An ExecuTorch program header is present, but its version is not
     * compatible with this version of the runtime.
     */
    IncompatibleVersion,

    /**
     * An ExecuTorch program header is not present.
     */
    NotPresent,

    /**
     * The data provided was too short to find the program header.
     */
    ShortData,
  };

  /**
   * The minimum number of bytes necessary for calls to `check_header`.
   */
  static constexpr size_t kMinHeadBytes = 64;

  /**
   * Looks for an ExecuTorch program header in the provided data.
   *
   * @param[in] data The data from the beginning of a file that might contain
   *     an ExecuTorch program.
   * @param[in] size The size of `data` in bytes. Must be >= `kMinHeadBytes`.
   *
   * @returns A value describing the presence of a header in the data.
   */
  static HeaderStatus check_header(const void* data, size_t size);

 private:
  // Let some classes call these private methods.
  friend class BackendDelegate;
  friend class Executor;
  friend class Method;
  friend class testing::ProgramTestFriend;

  const executorch_flatbuffer::Program* get_internal_program() const {
    return internal_program_;
  }

  // Used by Method to look up entries in the delegate data table.
  Error get_backend_delegate_data(
      size_t index,
      const void** out_data,
      size_t* out_size) const;

  /**
   * Loads a segment by index.
   *
   * @param[in] index The sement index to load. This should be an index into
   *     the Program.segments list.
   *
   * @returns The data as a FreeableBuffer, if the index is valid.
   * @retval Error::NotFound The program does not contain any segments or the
   *     index is out of range.
   * @returns Other errors depending on the implementation of
   *     DataLoader: The Program.segment table is inconsistent, or the
   *     data cannot be accessed.
   */
  __ET_NODISCARD Result<FreeableBuffer> LoadSegment(size_t index) const;

 private:
  Program(
      DataLoader* loader,
      size_t segment_base_offset,
      FreeableBuffer&& program_data,
      const executorch_flatbuffer::Program* internal_program,
      FreeableBuffer&& constant_segment_data)
      : program_data_(std::move(program_data)),
        // Don't need the loader if there are no segments.
        loader_(segment_base_offset > 0 ? loader : nullptr),
        internal_program_(internal_program),
        segment_base_offset_(segment_base_offset),
        constant_segment_data_(std::move(constant_segment_data)) {}

  // Not copyable or assignable.
  Program(const Program& rhs) = delete;
  Program& operator=(Program&& rhs) noexcept = delete;
  Program& operator=(const Program& rhs) = delete;

  /// The serialized program data. Tensors will point directly into this buffer.
  FreeableBuffer program_data_;

  /// Used to load segment data. Null if there are no segments.
  DataLoader* loader_;

  /// The flatbuffer representation of the program. Must not be exposed to
  /// users.
  const executorch_flatbuffer::Program* internal_program_;

  /// The offset to the first segment, in bytes. If zero, no segments should
  /// be present in internal_program_.
  size_t segment_base_offset_;

  /// Constant segment data.
  FreeableBuffer constant_segment_data_;
};

} // namespace executor
} // namespace torch
