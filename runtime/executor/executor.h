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
namespace executorch {
struct Chain;
struct ExecutionPlan;
struct EValue;
} // namespace executorch

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
// ExecutionPlan in executor (runtime) namespace.
// Differences from executorch::ExecutionPlan in serialization:
// It holds Evalues with APIs that are compatible operator unboxing.
// The data pointers of the Evalues should be mapped to serialization buffer.
// It holds function pointers of kernels, instead of operator names.
// TODO: use static memory planning to create all executor related data
class ExecutionPlan {
 public:
  ExecutionPlan(const Program* program, MemoryManager* memory_manager)
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
   * Initialize the runtime plan using a serialized plan.
   *
   * @retval Error::Ok on successful initialization.
   */
  __ET_NODISCARD Error init(executorch::ExecutionPlan* s_plan);

  /**
   * Sets the specific index input of execution plan with input_evalue
   *
   * Expects that the type of input_evalue align with the idx-th input of the
   * plan. If input_evalue is a tensor, attempts to allow dynamic shape, but
   * dtype is fixed. Also, the idx should be smaller the number of inputs in the
   * execution plan.
   *
   * NOTE: Based on the memory plan of the execution plan, the inputs may not
   * have buffer space pre-allocated for them, in this case the executor will
   * alias the memory of the tensors provided as inputs here, so the user should
   * take care that the life span of this memory outlasts the executor forward.
   *
   * @retval The type of Error occurs during execution. Error::Ok on successful
   * setting completion.
   */
  __ET_NODISCARD Error set_input(const EValue& input_evalue, size_t input_idx);

  /**
   * Sets the inputs for the execution plan from given EValue list.
   *
   * Expects that the type of elements in the list align with the inputs of the
   * plan. If input is tensor, attempts to allow dynamic shape, but dtype is
   * fixed.
   *
   * There're some memory issue worth noticed. Check NOTE of set_input
   * function for details.
   *
   * @retval The type of Error occurs during execution. Error::Ok on successful
   * setting completion.
   */
  __ET_NODISCARD Error
  set_inputs(const exec_aten::ArrayRef<EValue>& input_evalues);

  /**
   * Load plan's outputs to the given array with given length.
   *
   * Expects that the plan has been initialized, and the length of the given
   * array shall be not smaller than the number of outputs.
   *
   * The function will only update first plan.output_size() elements of
   * output_evalues, and set rest of the array as none EValue.
   *
   * NOTE: This function exposes the internel output tensors. Please do not try
   * to mess up the underlying data of output EValue tensors. Future updates
   * will prevent the exposures from happening.
   *
   * TODO (T139259264): Add checks to execution_plan's output to prevent
   * mess-up, and/or return deepcopy version of the output to prevent the
   * exposion.
   *
   * @returns Type of error occurs during execution. Error::Ok on successful
   * setting completion.
   */
  __ET_NODISCARD Error get_outputs(EValue* output_evalues, size_t length);

  /**
   * Execute the runtime plan. Cannot be used if the program is mid execution
   * through the 'experimental_step' api.
   *
   * @retval Error::Ok on successful execution completion.
   */
  __ET_NODISCARD Error execute();

  /**
   * Advances/executes a single instruction in runtime plan. Prototype api. Do
   * not rely on as it is liable to change.
   *
   * @retval Error::Ok on successful step completion
   * @retval Error::EndOfProgram if no more steps to take
   */
  __ET_NODISCARD Error experimental_step();

  /**
   * Resets executor state back to the beginning of the program
   *
   * @retval Error:Ok on successful reset
   * @retval ErrorInvalidState if called before step based execution reached the
   * end of the program. This means it is not possible to recover a program that
   * failed mid execution
   */
  __ET_NODISCARD Error experimental_reset_execution();

  // Moveable only as it owns unique_ptrs to tensors_, tensor_lists_
  ExecutionPlan(const ExecutionPlan&) = delete;
  ExecutionPlan& operator=(const ExecutionPlan&) = delete;
  ExecutionPlan(ExecutionPlan&&) = default;
  ExecutionPlan& operator=(ExecutionPlan&&) = default;

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
  ~ExecutionPlan();

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

  /// Returns true if the ExecutionPlan was successfully initialized.
  inline bool initialized() const {
    return init_state_ == InitializationState::Initialized;
  }

  // Executes a single instruction using the state in step_state_
  __ET_NODISCARD Error execute_instruction();

  StepState step_state_;
  const Program* program_;
  MemoryManager* memory_manager_;
  executorch::ExecutionPlan* serialization_plan_;

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
      const flatbuffers::Vector<flatbuffers::Offset<executorch::EValue>>*
          fb_values,
      size_t* num_parsed);

  __ET_NODISCARD Error resolve_operator(
      int32_t op_index,
      OpFunction* kernels,
      size_t kernel_index,
      InstructionArgs args,
      size_t n_args);
};

class Executor {
 public:
  // Executes a PyTorch executor program.
  Executor(const Program* program, MemoryManager* memory_manager);
  Executor(const Executor&) = delete;
  Executor& operator=(const Executor&) = delete;
  Executor(Executor&&) = default;
  Executor& operator=(Executor&&) = default;

  /**
   * DEPRECATED: Use init_execution_plan(const char*)
   *
   * Initializes the execution plan to use the specified entry point
   * of the model. `execution_plan()` returns this plan.
   *
   * May only be called once for the lifetime of the Executor.
   *
   * @param[in] index The index of the entry point to use for this plan.
   *     Defaults to using the `forward()` method.
   * @retval Error::Ok on successful initialization.
   */
  __ET_DEPRECATED __ET_NODISCARD Error
  init_execution_plan(size_t index = Program::kForwardMethodIndex);

  /**
   * Initializes the execution plan to use the specified entry point of the
   * model. `execution_plan()` returns this plan.
   *
   * May only be called once for the lifetime of the Executor.
   *
   * @param[in] name The name of the entry point to use for this plan.
   * @retval Error::Ok on successful initialization.
   */
  __ET_NODISCARD Error init_execution_plan(const char* method_name);

  /**
   * Returns the plan that was initialized by `init_execution_plan()`.
   */
  ExecutionPlan& execution_plan() {
    return plan_;
  }

  ~Executor() = default;

 private:
  const Program* program_;
  ExecutionPlan plan_;
};

} // namespace executor
} // namespace torch
