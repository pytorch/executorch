#pragma once

#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/compiler.h>

// Forward declare flatbuffer types. This is a public header and must not
// include the generated flatbuffer header.
namespace flatbuffers {
template <typename T>
class Vector;
template <typename T>
struct Offset;
} // namespace flatbuffers
namespace executorch_flatbuffer {
struct Chain;
struct ExecutionPlan;
struct EValue;
} // namespace executorch_flatbuffer

namespace torch {
namespace executor {

// Forward declare internal types.
class BackendDelegate;
struct Chain;
template <typename Fn>
class FunctionRef;
template <typename T>
class Span;
class KernelRuntimeContext;
using OpFunction = FunctionRef<void(KernelRuntimeContext&, EValue**)>;
/// A list of pointers into the master values table that together compose the
/// argument list for a single instruction
using InstructionArgs = Span<EValue*>;

/**
 * An executable method of an executorch program. Maps to a python method like
 * `forward()` on the original nn.Module.
 */
class Method {
 public:
  Method(const Program* program, MemoryManager* memory_manager)
      : step_state_(),
        program_(program),
        memory_manager_(memory_manager),
        serialization_plan_(nullptr),
        n_value_(0),
        values_(nullptr),
        n_delegate_(0),
        delegates_(nullptr),
        n_chains_(0),
        chains_(nullptr),
        init_state_(InitializationState::Uninitialized),
        pre_allocated_input_(false) {}

  /**
   * Initialize the method from its serialized representation.
   *
   * @returns Error::Ok on success, non-Ok on failure.
   */
  __ET_NODISCARD Error init(executorch_flatbuffer::ExecutionPlan* s_plan);

  /**
   * Sets a specific method input to the provided value.
   *
   * NOTE: Based on the memory plan of the method, the inputs may not have
   * buffer space pre-allocated for them, in this case the executor will alias
   * the memory of the tensors provided as inputs here, so the user should take
   * care that the life span of this memory outlasts the executor forward.
   *
   * @param[in] input_evalue The value to set the input to. The type of this
   *     must match the type of the corresponding input. If this value is a
   *     tensor, attempts to allow dynamic shape, but the dtype must always
   *     agree.
   * @param[in] input_idx Zero-based index of the input to set. Must be less
   *     than the value returned by inputs_size().
   *
   * @returns Error::Ok on success, non-Ok on failure.
   */
  __ET_NODISCARD Error set_input(const EValue& input_evalue, size_t input_idx);

  /**
   * Sets the values of all method inputs.
   *
   * See NOTE on set_input().
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

  // Moveable only as it owns unique_ptrs to tensors_, tensor_lists_
  Method(const Method&) = delete;
  Method& operator=(const Method&) = delete;
  Method(Method&&) = default;
  Method& operator=(Method&&) = default;

  size_t values_size() const;
  const EValue& get_value(size_t i) const;
  EValue& mutable_value(size_t i);
  size_t inputs_size() const;
  size_t get_input_index(size_t i) const;
  const EValue& get_input(size_t i) const;
  EValue& mutable_input(size_t i);
  size_t outputs_size() const;
  size_t get_output_index(size_t i) const;
  const EValue& get_output(size_t i) const;
  EValue& mutable_output(size_t i);
  ~Method();

 private:
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

  /// Returns true if the Method was successfully initialized.
  inline bool initialized() const {
    return init_state_ == InitializationState::Initialized;
  }

  // Executes a single instruction using the state in step_state_
  __ET_NODISCARD Error execute_instruction();

  StepState step_state_;
  const Program* program_;
  MemoryManager* memory_manager_;
  executorch_flatbuffer::ExecutionPlan* serialization_plan_;

  size_t n_value_;
  EValue* values_;

  size_t n_delegate_;
  BackendDelegate* delegates_;

  size_t n_chains_;
  Chain* chains_;

  InitializationState init_state_;
  bool pre_allocated_input_;

  /**
   * Initialize the EValue table of the program. *num_parsed is the number
   * of elements actually parsed, which may be less than n_value_ on failure.
   */
  __ET_NODISCARD Error parse_values(
      const flatbuffers::Vector<
          flatbuffers::Offset<executorch_flatbuffer::EValue>>* fb_values,
      size_t* num_parsed);

  __ET_NODISCARD Error resolve_operator(
      int32_t op_index,
      OpFunction* kernels,
      size_t kernel_index,
      InstructionArgs args,
      size_t n_args);
};

} // namespace executor
} // namespace torch
