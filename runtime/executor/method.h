/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/event_tracer.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/executor/method_meta.h>
#include <executorch/runtime/platform/compiler.h>

// Forward declare flatbuffer types. This is a public header and must not
// include the generated flatbuffer header.
namespace executorch_flatbuffer {
struct Chain;
struct ExecutionPlan;
struct EValue;
} // namespace executorch_flatbuffer

namespace executorch {
namespace runtime {

// Forward declare Program to avoid a circular reference.
class Program;

// Forward declare internal types.
class BackendDelegate;
struct Chain;
class KernelRuntimeContext;
using OpFunction = void (*)(KernelRuntimeContext&, EValue**);
/// A list of pointers into the master values table that together compose the
/// argument list for a single instruction
using InstructionArgs = Span<EValue*>;

/**
 * An executable method of an executorch program. Maps to a python method like
 * `forward()` on the original nn.Module.
 */
class Method final {
 public:
  /**
   * Move ctor. Takes ownership of resources previously owned by `rhs`,
   * and leaves `rhs` in an uninitialized state.
   */
  Method(Method&& rhs) noexcept
      : step_state_(rhs.step_state_),
        program_(rhs.program_),
        memory_manager_(rhs.memory_manager_),
        serialization_plan_(rhs.serialization_plan_),
        event_tracer_(rhs.event_tracer_),
        n_value_(rhs.n_value_),
        values_(rhs.values_),
        n_delegate_(rhs.n_delegate_),
        delegates_(rhs.delegates_),
        n_chains_(rhs.n_chains_),
        chains_(rhs.chains_),
        init_state_(rhs.init_state_),
        pre_allocated_input_(rhs.pre_allocated_input_),
        pre_allocated_output_(rhs.pre_allocated_output_) {
    // Required: clear out fields that the dtor looks at, so that we don't free
    // anything twice.
    rhs.n_value_ = 0;
    rhs.values_ = nullptr;
    rhs.n_delegate_ = 0;
    rhs.delegates_ = nullptr;

    // Helpful: Try to ensure that any other interactions with the old object
    // result in failures.
    rhs.init_state_ = InitializationState::Uninitialized;
    rhs.step_state_ = {};
    rhs.program_ = nullptr;
    rhs.memory_manager_ = nullptr;
    rhs.serialization_plan_ = nullptr;
    rhs.event_tracer_ = nullptr;
    rhs.n_chains_ = 0;
    rhs.chains_ = nullptr;
    rhs.pre_allocated_input_ = false;
    rhs.pre_allocated_output_ = false;
  }

  /**
   * Sets the internal input value to be equivalent to the to the provided
   * value.
   *
   * @param[in] input_evalue The evalue to copy into the method input. If the
   *     evalue is a tensor, the data is copied in most cases, so the tensor
   *     passed in here does not always need to outlive this call. But there is
   *     a case where the Method will keep a pointer to the tensor's data.
   *     Based on the memory plan of the method, the inputs may not have
   *     buffer space pre-allocated for them. In this case the executor will
   *     alias the memory of the tensors provided as inputs here rather then
   *     deepcopy the input into the memory planned arena.
   *
   * @param[in] input_idx Zero-based index of the input to set. Must be less
   *     than the value returned by inputs_size().
   *
   * @returns Error::Ok on success, non-Ok on failure.
   */
  __ET_NODISCARD Error set_input(const EValue& input_evalue, size_t input_idx);

  /**
   * Sets the values of all method inputs.
   *
   * See set_input() for a more detailed description of the behavior.
   *
   * @param[in] input_evalues The new values for all of the method inputs. The
   *     type of each element must match the type of corresponding input. If the
   *     value of an element is a tensor, attempts to allow dynamic shape, but
   *     the dtype must always agree.
   *
   * @returns Error::Ok on success, non-Ok on failure.
   */
  __ET_NODISCARD Error
  set_inputs(const exec_aten::ArrayRef<EValue>& input_evalues);

  /**
   * Sets the data buffer of the specified method output to the provided value.
   *
   * NOTE: Based on the memory plan of the method, the output tensors may not
   * have buffer space pre-allocated for them, in this case the executor will
   * point those tensors to the buffer provided here, so the user should take
   * care that the life span of this memory outlasts the executor forward.
   *
   * @param[in] buffer The block of memory to point the specified tensor at.
   *
   * @param[in] size the length of buffer in bytes, must be >= the nbytes of the
   * specified tensor.
   *
   * @param[in] output_idx The index of the output to set the data_ptr for. Must
   *     correspond to a tensor, and that tensor must not have had a buffer
   *     allocated by the memory plan.
   *
   * @returns Error::Ok on success, non-Ok on failure.
   */
  __ET_NODISCARD Error
  set_output_data_ptr(void* buffer, size_t size, size_t output_idx);

  /**
   * Copies the method's outputs into the provided array.
   *
   * WARNING: The output contains shallow copies of internal tensor outputs.
   * Please do not mutate returned Tensor elements.
   *
   * TODO(T139259264): Add checks to detect output mutation, or deep-copy
   * outputs.
   *
   * @param[in] output_evalues The array to copy the outputs into. The first
   *     `outputs_size()` elements will be set to the corresponding output
   *     values. The rest of the array will be set to the EValue value None.
   * @param[in] length The size of the `output_evalues` array in elements. Must
   *     be greater than or equal to `outputs_size()`.
   *
   * @returns Error::Ok on success, non-Ok on failure.
   */
  __ET_NODISCARD Error get_outputs(EValue* output_evalues, size_t length);

  /**
   * Copies the method's inputs into the provided array.
   *
   * WARNING: The input contains shallow copies of internal tensor inputs.
   * Please do not mutate returned Tensor elements.
   *
   * @param[in] input_evalues The array to copy the inputs into. The first
   *     `inputs_size()` elements will be set to the corresponding input
   *     values. The rest of the array will be set to the EValue value None.
   * @param[in] length The size of the `input_evalues` array in elements. Must
   *     be greater than or equal to `inputs_size()`.
   *
   * @returns Error::Ok on success, non-Ok on failure.
   */
  __ET_NODISCARD Error get_inputs(EValue* input_evalues, size_t length);

  /**
   * Execute the method.
   *
   * NOTE: Will fail if the method has been partially executed using the
   * `experimental_step()` api.
   *
   * @returns Error::Ok on success, non-Ok on failure.
   */
  __ET_NODISCARD Error execute();

  /**
   * Advances/executes a single instruction in the method.
   *
   * NOTE: Prototype API; subject to change.
   *
   * @retval Error::Ok step succeeded
   * @retval non-Ok step failed
   * @retval Error::EndOfMethod method finished executing successfully
   */
  __ET_NODISCARD Error experimental_step();

  /**
   * Resets execution state to the start of the Method. For use with the
   * `experimental_step()` API.
   *
   * NOTE: Prototype API; subject to change.
   *
   * @retval Error:Ok on success
   * @retval Error::InvalidState if called before step-based execution reached
   *     the end of the Method. This means it is not possible to recover a
   *     Method that failed mid-execution.
   */
  __ET_NODISCARD Error experimental_reset_execution();

  /**
   * Returns the MethodMeta that corresponds to the calling Method.
   */
  MethodMeta method_meta() const;

  /**
   * Returns the number of inputs the Method expects.
   */
  size_t inputs_size() const;

  /**
   * Returns the number of outputs the Method returns.
   */
  size_t outputs_size() const;

  /**
   * Retrieves the output at the specified index.
   */
  const EValue& get_output(size_t i) const;

  EventTracer* get_event_tracer();

  /// DEPRECATED: Use MethodMeta instead to access metadata, and set_input to
  /// update Method inputs.
  __ET_DEPRECATED const EValue& get_input(size_t i) const;
  /// DEPRECATED: Use MethodMeta instead to access metadata, and set_input to
  /// update Method inputs.
  __ET_DEPRECATED EValue& mutable_input(size_t i);
  /// DEPRECATED: Use MethodMeta instead to access metadata, and get_output to
  /// retrieve Method outputs.
  __ET_DEPRECATED EValue& mutable_output(size_t i);

  ~Method();

 private:
  // Delete other rule-of-five methods.
  Method(const Method&) = delete;
  Method& operator=(const Method&) noexcept = delete;
  Method& operator=(Method&&) = delete;

  // Let Program call load().
  friend class Program;
  // Let Executor call the ctor and init().
  friend class Executor;

  enum class InitializationState : uint8_t {
    Uninitialized,
    Initialized,
    InitializationFailed,
  };

  /// Tracks what step in program execution we are on
  struct StepState {
    size_t chain_idx;
    size_t instr_idx;
  };

  Method(
      const Program* program,
      MemoryManager* memory_manager,
      EventTracer* event_tracer)
      : step_state_(),
        program_(program),
        memory_manager_(memory_manager),
        serialization_plan_(nullptr),
        event_tracer_(event_tracer),
        n_value_(0),
        values_(nullptr),
        n_delegate_(0),
        delegates_(nullptr),
        n_chains_(0),
        chains_(nullptr),
        init_state_(InitializationState::Uninitialized),
        pre_allocated_input_(false),
        pre_allocated_output_(false) {}

  /// Static factory used by Program.
  __ET_NODISCARD static Result<Method> load(
      executorch_flatbuffer::ExecutionPlan* s_plan,
      const Program* program,
      MemoryManager* memory_manager,
      EventTracer* event_tracer);

  /**
   * Initialize the method from its serialized representation.
   *
   * @returns Error::Ok on success, non-Ok on failure.
   */
  __ET_NODISCARD Error init(executorch_flatbuffer::ExecutionPlan* s_plan);

  /// Returns true if the Method was successfully initialized.
  inline bool initialized() const {
    return init_state_ == InitializationState::Initialized;
  }

  const EValue& get_value(size_t i) const;
  EValue& mutable_value(size_t i);
  size_t get_input_index(size_t i) const;
  size_t get_output_index(size_t i) const;

  // Executes a single instruction using the state in step_state_
  __ET_NODISCARD Error execute_instruction();

  StepState step_state_;
  const Program* program_;
  MemoryManager* memory_manager_;
  executorch_flatbuffer::ExecutionPlan* serialization_plan_;
  EventTracer* event_tracer_;

  size_t n_value_;
  EValue* values_;

  size_t n_delegate_;
  BackendDelegate* delegates_;

  size_t n_chains_;
  Chain* chains_;

  InitializationState init_state_;
  bool pre_allocated_input_;
  bool pre_allocated_output_;

  /**
   * Parses the elements of the values_ array. On error, n_value_ will be set to
   * the number of successfully-initialized entries so that ~Method doesn't try
   * to clean up uninitialized entries.
   */
  __ET_NODISCARD Error parse_values();

  __ET_NODISCARD Error resolve_operator(
      int32_t op_index,
      OpFunction* kernels,
      size_t kernel_index,
      InstructionArgs args,
      size_t n_args);

  void log_outputs();
};

} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::Method;
} // namespace executor
} // namespace torch
