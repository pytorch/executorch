/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/executor/method.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/event_tracer_hooks.h>
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/executor/tensor_parser.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/profiler.h>
#include <executorch/schema/program_generated.h>

namespace torch {
namespace executor {

/**
 * Runtime state for a backend delegate.
 */
// This lint wants to wrap the class in an anonymous namespace, but it must be
// visible because it's forward-declared and used in Executor.h.
// @lint-ignore CLANGTIDY facebook-hte-ShadowingClass
class BackendDelegate final {
 public:
  /**
   * Initializes an already-allocated BackendDelegate from its serialized
   * representation.
   *
   * @param[in] delegate The serialized backend delegate to load.
   * @param[in] program The serialized program to load from.
   * @param[in] backend_init_context The context pointer to pass to the
   *     backend's init() method.
   * @param[out] out The BackendDelegate to initialize.
   *
   * @returns Error::Ok if the initialization succeeded, or an error otherwise.
   */
  static Error Init(
      const executorch_flatbuffer::BackendDelegate& delegate,
      const Program* program,
      BackendInitContext& backend_init_context,
      BackendDelegate* out) {
    // Look up the backend.
    const char* backend_id = delegate.id()->c_str();
    PyTorchBackendInterface* backend = get_backend_class(backend_id);
    ET_CHECK_OR_RETURN_ERROR(
        backend != nullptr,
        NotFound,
        "Backend %s is not registered.",
        backend_id);
    ET_CHECK_OR_RETURN_ERROR(
        backend->is_available(),
        NotFound,
        "Backend %s is not available.",
        backend_id);

    // Get the delegate data.
    Result<FreeableBuffer> processed_data = GetProcessedData(delegate, program);
    if (!processed_data.ok()) {
      ET_LOG(Error, "Failed to load data for backend %s", backend_id);
      return processed_data.error();
    }

    // Parse compilation specs from program
    CompileSpec* compile_specs;
    Error err = PopulateCompileSpecs(
        delegate.compile_specs(), backend_init_context, &compile_specs);
    if (err != Error::Ok) {
      ET_LOG(Error, "Failed to get compile specs for backend %s", backend_id);
      return err;
    }
    size_t num_compile_specs = delegate.compile_specs()->size();

    out->backend_ = backend;
    out->handle_ = nullptr;
    // Pass a pointer to this buffer to the backend. It's safe for the backend
    // to point its handle to this object, since it will outlive the backend.
    new (&out->segment_) FreeableBuffer(std::move(processed_data.get()));

    // Initialize the delegate.
    Result<DelegateHandle*> handle = backend->init(
        backend_init_context,
        &out->segment_,
        ArrayRef<CompileSpec>(compile_specs, num_compile_specs));
    if (!handle.ok()) {
      ET_LOG(
          Error,
          "Init failed for backend %s: 0x%" PRIx32,
          backend_id,
          static_cast<uint32_t>(handle.error()));
      out->segment_.Free();
      return handle.error();
    }
    out->handle_ = handle.get();
    return Error::Ok;
  }

  ~BackendDelegate() {
    if (backend_ != nullptr) {
      backend_->destroy(handle_);
    }
  }

  Error Execute(
      BackendExecutionContext& backend_execution_context,
      EValue** args) const {
    EXECUTORCH_SCOPE_PROF("delegate_execute");
    return backend_->execute(backend_execution_context, handle_, args);
  }

 private:
  // Not constructible.
  BackendDelegate() = delete;

  // Disallow copy/move.
  BackendDelegate(const BackendDelegate&) = delete;
  BackendDelegate& operator=(const BackendDelegate&) = delete;
  BackendDelegate(BackendDelegate&&) = delete;
  BackendDelegate& operator=(BackendDelegate&&) = delete;

  static Error PopulateCompileSpecs(
      const flatbuffers::Vector<flatbuffers::Offset<
          executorch_flatbuffer::CompileSpec>>* compile_specs_in_program,
      BackendInitContext& backend_init_context,
      CompileSpec** out_spec) {
    auto number_of_compile_specs = compile_specs_in_program->size();

    CompileSpec* compile_specs_list = ET_ALLOCATE_LIST_OR_RETURN_ERROR(
        backend_init_context.get_runtime_allocator(),
        CompileSpec,
        number_of_compile_specs);

    // Initialize the spec list for each method spec
    for (size_t j = 0; j < number_of_compile_specs; j++) {
      auto compile_spec_in_program = compile_specs_in_program->Get(j);

      compile_specs_list[j].key = compile_spec_in_program->key()->c_str();
      compile_specs_list[j].value = {
          /*buffer*/ static_cast<void*>(
              const_cast<uint8_t*>(compile_spec_in_program->value()->Data())),
          /*nbytes*/ compile_spec_in_program->value()->size(),
      };
    }

    *out_spec = compile_specs_list;
    return Error::Ok;
  }

  static Result<FreeableBuffer> GetProcessedData(
      const executorch_flatbuffer::BackendDelegate& delegate,
      const Program* program) {
    const executorch_flatbuffer::BackendDelegateDataReference* processed =
        delegate.processed();
    switch (processed->location()) {
      case executorch_flatbuffer::DataLocation::INLINE: {
        const void* data;
        size_t size;
        Error err = program->get_backend_delegate_data(
            processed->index(), &data, &size);
        if (err != Error::Ok) {
          return err;
        }
        return FreeableBuffer(
            data,
            size,
            /*free_fn=*/nullptr);
      }
      case executorch_flatbuffer::DataLocation::SEGMENT: {
        return program->LoadSegment(processed->index());
      }
      default:
        ET_LOG(
            Error,
            "Unknown data location %u",
            static_cast<unsigned int>(processed->location()));
        return Error::Internal;
    }
  }

  FreeableBuffer segment_;
  const PyTorchBackendInterface* backend_;
  DelegateHandle* handle_;
};

/**
 * Runtime state for a chain of instructions.
 */
struct Chain {
  /// Pointer to the associated flatbuffer chain.
  const executorch_flatbuffer::Chain* s_chain_;

  /// Each entry is a list of parameters for a kernel or delegate call.
  Span<InstructionArgs> argument_lists_;
  /// Each instruction will have one kernel (not for delegate).
  OpFunction* kernels_;
};

namespace {

Result<InstructionArgs> gen_instruction_arguments(
    MemoryAllocator* method_allocator,
    EValue* values,
    size_t num_args,
    const int32_t* arg_idxs) {
  EValue** arg_list =
      ET_ALLOCATE_LIST_OR_RETURN_ERROR(method_allocator, EValue*, num_args);
  for (size_t i = 0; i < num_args; ++i) {
    arg_list[i] = &values[arg_idxs[i]];
  }
  return InstructionArgs(arg_list, num_args);
}

bool parse_cond_value(const EValue& cond_value) {
  // The cond value attached to the JF instruction at the beginning of an
  // if/else branch is a Tensor which we parse and decide whether to continue
  // to execute the if branch or jump to the else branch.
  // The cond value attached to the JF instruction at the end of the if branch
  // is a Bool Scalar which resolves to false and points us to the instruction
  // to jump to which will take us to a point that is after the else branch.
  if (cond_value.isTensor()) {
    const exec_aten::Tensor& cond_val = cond_value.toTensor();

    // All the tensors and scalar cond values should be of bool type
    // currently. If that's not the case then something is wrong in the model
    // and we should exit.
    ET_CHECK_MSG(
        ScalarType::Bool == cond_val.scalar_type(),
        "Expected dtype of %" PRId8 " got %" PRId8,
        static_cast<int8_t>(ScalarType::Bool),
        static_cast<int8_t>(cond_val.scalar_type()));

    const bool* cond_data = cond_val.const_data_ptr<bool>();
    for (size_t i = 0; i < cond_val.numel(); i++) {
      if (!cond_data[i]) {
        return false;
      }
    }
  } else if (cond_value.isBool()) {
    if (!cond_value.toBool()) {
      return false;
    }
  } else {
    ET_CHECK_MSG(false, "Unsupported EValue was passed in for JF instruction");
  }

  return true;
}

} // namespace

Error Method::parse_values() {
  auto flatbuffer_values = serialization_plan_->values();
  ET_CHECK(flatbuffer_values != nullptr);
  size_t n_value = flatbuffer_values->size();
  values_ = ET_ALLOCATE_LIST_OR_RETURN_ERROR(
      memory_manager_->method_allocator(), EValue, n_value);

  // n_value_ counts the number of successfully-initialized values for ~Method()
  // to clean up, and is incremented at the bottom of the loop. This makes it
  // safe for errors to return without updating any state.
  n_value_ = 0;

  for (size_t i = 0; i < n_value; ++i) {
    auto serialization_value = flatbuffer_values->Get(i);
    switch (serialization_value->val_type()) {
      case executorch_flatbuffer::KernelTypes::Null: {
        // Placement new as the list elements are not initialized, so calling
        // copy assignment is not defined if its non trivial (Imagine the
        // garbage in values_[i] thinks its an at::Tensor).
        new (&values_[i]) EValue();
      } break;
      case executorch_flatbuffer::KernelTypes::Int: {
        new (&values_[i]) EValue(serialization_value->val_as_Int()->int_val());
      } break;
      case executorch_flatbuffer::KernelTypes::Double: {
        new (&values_[i])
            EValue(serialization_value->val_as_Double()->double_val());
      } break;
      case executorch_flatbuffer::KernelTypes::Bool: {
        new (&values_[i])
            EValue(serialization_value->val_as_Bool()->bool_val());
      } break;
      case executorch_flatbuffer::KernelTypes::IntList: {
        const auto items = serialization_value->val_as_IntList()->items();
        // Allocate space for boxed and unboxed list representations using
        // values_ as source of truth
        auto* evalp_list =
            memory_manager_->method_allocator()->allocateList<EValue*>(
                items->size());
        auto* int_list =
            memory_manager_->method_allocator()->allocateList<int64_t>(
                items->size());

        // initialize boxed list
        for (size_t j = 0; j < items->size(); j++) {
          evalp_list[j] = &values_[static_cast<size_t>(items->Get(j))];
        }
        new (&values_[i]) EValue(
            BoxedEvalueList<int64_t>(evalp_list, int_list, items->size()));
      } break;
      case executorch_flatbuffer::KernelTypes::BoolList: {
        const auto items = serialization_value->val_as_BoolList()->items();
        // NOTE: This is technically not portable. A platform could technically
        // define boolean as something longer than a byte. This would be an
        // exceptionally rare case, and this type is currently unused in any
        // operators in ATen that we would need to support. To be properly
        // portable here we need to allocate a new array of bool and copy cast
        // the flatbuffer data into it, but because of how exceptionally rare
        // this case is its low prio TODO: jakeszwe
        new (&values_[i]) EValue(exec_aten::ArrayRef<bool>(
            (const bool*)items->data(), items->size()));
      } break;
      case executorch_flatbuffer::KernelTypes::DoubleList: {
        const auto items = serialization_value->val_as_DoubleList()->items();
        new (&values_[i])
            EValue(exec_aten::ArrayRef<double>(items->data(), items->size()));
      } break;
      case executorch_flatbuffer::KernelTypes::String: {
        const auto fb_str = serialization_value->val_as_String()->string_val();
        new (&values_[i]) EValue(fb_str->c_str(), fb_str->size());
      } break;
      case executorch_flatbuffer::KernelTypes::Tensor: {
        auto t = deserialization::parseTensor(
            program_, memory_manager_, serialization_value->val_as_Tensor());
        if (!t.ok()) {
          ET_LOG(
              Error,
              "Failed parsing tensor at index %zu: 0x%" PRIx32,
              i,
              static_cast<uint32_t>(t.error()));
          return t.error();
        }
        new (&values_[i]) EValue(t.get());
      } break;
      case executorch_flatbuffer::KernelTypes::TensorList: {
        // get list of serialization tensors and allocate storage for executor
        // tensors
        auto tensors = deserialization::parseTensorList(
            serialization_value->val_as_TensorList()->items(),
            values_,
            memory_manager_);
        if (!tensors.ok()) {
          ET_LOG(
              Error,
              "Failed parsing tensor list at index %zu: 0x%" PRIx32,
              i,
              static_cast<uint32_t>(tensors.error()));
          return tensors.error();
        }
        new (&values_[i]) EValue(tensors.get());
      } break;
      case executorch_flatbuffer::KernelTypes::OptionalTensorList: {
        // Same as TensorList but optional<Tensor> instead of Tensor
        auto tensors =
            deserialization::parseListOptionalType<exec_aten::Tensor>(
                serialization_value->val_as_OptionalTensorList()->items(),
                values_,
                memory_manager_);
        if (!tensors.ok()) {
          ET_LOG(
              Error,
              "Failed parsing optional tensor list at index %zu: 0x%" PRIx32,
              i,
              static_cast<uint32_t>(tensors.error()));
          return tensors.error();
        }
        new (&values_[i]) EValue(tensors.get());
      } break;
      default:
        // flatbuffer enums start at 0, but they generate a hidden NONE enum
        // and give it that value. schema.fbs doesnt show this type, so I
        // subtract one to keep the output in 0 based indexing for a
        // disgruntled debugger seeing this error message and checking
        // schema.fbs
        ET_CHECK_MSG(
            false,
            "Enum KernelTypes type: %" PRIu32
            " not supported. Please look in executorch/schema/program.fbs "
            "to see which type this is.",
            static_cast<uint32_t>(serialization_value->val_type()) - 1);
    }

    // ~Method() will try to clean up n_value_ entries in the values_ array.
    // Only increment this once we know the entry is valid, so that we don't try
    // to clean up an uninitialized entry.
    n_value_ = i + 1;
  }
  return Error::Ok;
}

/**
 * Private/helper method for populating operator_name from the Operator.
 * operator_name is a char pointer that is already allocated. The size of
 * of this buffer is of size operator_name_size.
 */
static void populateOperatorName(
    const executorch_flatbuffer::Operator* const& op,
    const size_t operator_name_size,
    char* operator_name) {
  int cx;
  const bool has_overload = (op->overload()->size() > 0);
  // Don't append any overload if the overload string is empty.
  cx = snprintf(
      operator_name,
      operator_name_size,
      "%s%s%s",
      op->name()->c_str(),
      has_overload ? "." : "",
      has_overload ? op->overload()->c_str() : "");

  ET_CHECK_MSG(cx >= 0, "String encoding error occured.");
  ET_CHECK_MSG(
      cx < operator_name_size,
      "Aborting. Operator name %s%s%s with length %d "
      "truncated to %zu due to internal buffer limit.",
      op->name()->c_str(),
      has_overload ? "." : "",
      has_overload ? op->overload()->c_str() : "",
      cx,
      operator_name_size);
}

Error Method::resolve_operator(
    int32_t op_index,
    OpFunction* kernels,
    size_t kernel_index,
    InstructionArgs args,
    size_t n_args) {
  // TODO(T153505381, T153506819) Investigate optimizing this function for both
  // space and time.

  // resolve name
  constexpr size_t kTempBufferSizeForName = 100;
  char operator_name[kTempBufferSizeForName];
  const auto ops = serialization_plan_->operators();
  const auto& op = ops->Get(op_index);

  populateOperatorName(op, kTempBufferSizeForName, operator_name);

  // resolve tensor meta
  auto method_allocator = memory_manager_->method_allocator();
  TensorMeta* meta =
      ET_ALLOCATE_LIST_OR_RETURN_ERROR(method_allocator, TensorMeta, n_args);
  size_t count = 0;
  for (size_t i = 0; i < n_args; i++) {
    EValue* eval = args[i];
    // handle tensor list as well
    if (eval->isTensor()) {
      auto tensor = eval->toTensor();
      meta[count].dtype_ = tensor.scalar_type();
      exec_aten::DimOrderType* dim_order_ptr = ET_ALLOCATE_LIST_OR_RETURN_ERROR(
          method_allocator, exec_aten::DimOrderType, tensor.dim());
      size_t size = tensor.dim();
      Error err = get_dim_order(tensor, dim_order_ptr, size);
      ET_CHECK_OR_RETURN_ERROR(
          err == Error::Ok,
          InvalidArgument,
          "Error setting dim_order %zu: 0x%" PRIx32,
          i,
          static_cast<uint32_t>(err));
      meta[count].dim_order_ =
          ArrayRef<exec_aten::DimOrderType>(dim_order_ptr, size);
      count++;
    }
  }
  // search kernel
  if (hasOpsFn(operator_name, ArrayRef<TensorMeta>(meta, count))) {
    kernels[kernel_index] =
        getOpsFn(operator_name, ArrayRef<TensorMeta>(meta, count));
    return Error::Ok;
  } else {
    ET_LOG(Error, "Missing operator: [%d] %s", op_index, operator_name);
    return Error::OperatorMissing;
  }
}

Result<Method> Method::load(
    executorch_flatbuffer::ExecutionPlan* s_plan,
    const Program* program,
    MemoryManager* memory_manager,
    EventTracer* event_tracer) {
  Method method(program, memory_manager, event_tracer);
  Error err = method.init(s_plan);
  if (err != Error::Ok) {
    return err;
  } else {
    ET_CHECK(method.initialized());
    return method;
  }
}

Error Method::init(executorch_flatbuffer::ExecutionPlan* s_plan) {
  EXECUTORCH_SCOPE_PROF("Method::init");
  internal::EventTracerProfileScope event_tracer_profile_scope =
      internal::EventTracerProfileScope(event_tracer_, "Method::init");
  ET_CHECK_OR_RETURN_ERROR(
      // Don't use !initialized() here because we also want to fail on the
      // InitializationFailed state.
      init_state_ == InitializationState::Uninitialized,
      InvalidState,
      "Method already initialized, or previously failed to initialize.");
  init_state_ =
      InitializationState::InitializationFailed; // Until proven otherwise
  serialization_plan_ = s_plan;
  auto method_allocator = memory_manager_->method_allocator();

  {
    // Parse the elements of the values_ array.
    Error err = parse_values();
    if (err != Error::Ok) {
      return err;
    }
  }

  {
    // Resolve delegates
    const auto delegates = serialization_plan_->delegates();
    ET_CHECK(delegates != nullptr);
    size_t n_delegate = delegates->size();
    delegates_ = ET_ALLOCATE_LIST_OR_RETURN_ERROR(
        method_allocator, BackendDelegate, n_delegate);

    // n_delegate_ counts the number of successfully-initialized delegates for
    // ~Method() to clean up, and is incremented at the bottom of the loop. This
    // makes it safe for errors to return without updating any state.
    n_delegate_ = 0;

    for (size_t i = 0; i < n_delegate; ++i) {
      const auto& delegate = *delegates->Get(i);
      BackendInitContext backend_init_context(method_allocator);
      Error err = BackendDelegate::Init(
          delegate, program_, backend_init_context, &delegates_[i]);
      if (err != Error::Ok) {
        return err;
      }
      // ~Method() will try to clean up n_delegate_ entries in the delegates_
      // array. Only increment this once we know the entry is valid, so that
      // we don't try to clean up an uninitialized entry.
      n_delegate_ = i + 1;
    }
  }

  {
    // Load chains
    const auto chains = serialization_plan_->chains();
    ET_CHECK(chains != nullptr);
    n_chains_ = chains->size();

    chains_ =
        ET_ALLOCATE_LIST_OR_RETURN_ERROR(method_allocator, Chain, n_chains_);
    int32_t num_instructions_missing_op = 0;
    for (size_t i = 0; i < n_chains_; ++i) {
      auto s_chain = chains->Get(i);
      auto num_instructions = s_chain->instructions()->size();
      auto chain_instruction_kernels = ET_ALLOCATE_LIST_OR_RETURN_ERROR(
          method_allocator, OpFunction, num_instructions);
      auto chain_instruction_arg_lists = ET_ALLOCATE_LIST_OR_RETURN_ERROR(
          method_allocator, InstructionArgs, num_instructions);

      // Set up the argument lists ahead of time and store pointers to them to
      // use when the instructions are called
      for (size_t instr_idx = 0; instr_idx < s_chain->instructions()->size();
           ++instr_idx) {
        const auto instruction = s_chain->instructions()->Get(instr_idx);
        switch (instruction->instr_args_type()) {
          case executorch_flatbuffer::InstructionArguments::KernelCall: {
            const auto arg_idxs =
                instruction->instr_args_as_KernelCall()->args();
            auto res = gen_instruction_arguments(
                method_allocator, values_, arg_idxs->size(), arg_idxs->data());
            if (!res.ok()) {
              return res.error();
            }
            chain_instruction_arg_lists[instr_idx] = res.get();
            auto err = resolve_operator(
                instruction->instr_args_as_KernelCall()->op_index(),
                chain_instruction_kernels,
                instr_idx,
                res.get(),
                arg_idxs->size());
            if (err == Error::OperatorMissing) {
              num_instructions_missing_op++;
            } else if (err == Error::MemoryAllocationFailed) {
              return err;
            }
          } break;
          case executorch_flatbuffer::InstructionArguments::DelegateCall: {
            const auto arg_idxs =
                instruction->instr_args_as_DelegateCall()->args();
            auto res = gen_instruction_arguments(
                method_allocator, values_, arg_idxs->size(), arg_idxs->data());
            if (!res.ok()) {
              return res.error();
            }
            chain_instruction_arg_lists[instr_idx] = res.get();
          } break;
          default:
            // wasteful but non kernel/delegate instructions are fairly sparse
            // maybe need to revisit if FreeCall becomes to populous
            chain_instruction_arg_lists[instr_idx] = InstructionArgs();
        }
      }
      chains_[i] = Chain{
          s_chain,
          Span<InstructionArgs>(chain_instruction_arg_lists, num_instructions),
          chain_instruction_kernels,
      };
    }
    ET_CHECK_OR_RETURN_ERROR(
        num_instructions_missing_op == 0,
        OperatorMissing,
        "There are %d instructions don't have corresponding operator registered. "
        "See logs for details",
        num_instructions_missing_op);
  }

  pre_allocated_input_ = false;

  // Get pre_allocation info for input tensors
  for (int i = 0; i < inputs_size(); i++) {
    if (get_input(i).isTensor()) {
      pre_allocated_input_ =
          get_input(i).toTensor().const_data_ptr() != nullptr;
      break;
    }
  }

  pre_allocated_output_ = false;

  // Get pre_allocation info for output tensors
  for (int i = 0; i < outputs_size(); i++) {
    if (get_output(i).isTensor()) {
      pre_allocated_output_ =
          get_output(i).toTensor().const_data_ptr() != nullptr;
      break;
    }
  }

  ET_CHECK_OR_RETURN_ERROR(
      n_chains_ > 0,
      Internal,
      "Expected program to have at least one chain received %zu",
      n_chains_);

  step_state_ = StepState{0, 0};

  init_state_ = InitializationState::Initialized;
  return Error::Ok;
}

__ET_NODISCARD Error
Method::set_input(const EValue& input_evalue, size_t input_idx) {
  ET_CHECK_OR_RETURN_ERROR(
      initialized(),
      InvalidState,
      "Input can not be set until method has been initialized.");

  ET_CHECK_OR_RETURN_ERROR(
      step_state_.instr_idx == 0 && step_state_.chain_idx == 0,
      InvalidState,
      "Inputs can not be set mid execution.");

  ET_CHECK_OR_RETURN_ERROR(
      input_idx < inputs_size(),
      InvalidArgument,
      "Given input index must be less than the number of inputs in method, but got %zu and %zu",
      input_idx,
      inputs_size());

  const auto& e = get_input(input_idx);
  ET_CHECK_OR_RETURN_ERROR(
      e.isTensor() || e.isScalar(),
      InvalidArgument,
      "The %zu-th input in method is expected Tensor or prim, but received %" PRIu32,
      input_idx,
      static_cast<uint32_t>(e.tag));

  ET_CHECK_OR_RETURN_ERROR(
      e.tag == input_evalue.tag,
      InvalidArgument,
      "The %zu-th input of method should have the same type as the input_evalue, but get tag %" PRIu32
      " and tag %" PRIu32,
      input_idx,
      static_cast<uint32_t>(e.tag),
      static_cast<uint32_t>(input_evalue.tag));

  if (e.isTensor()) {
    const auto& t_dst = e.toTensor();
    const auto& t_src = input_evalue.toTensor();
    ET_CHECK_OR_RETURN_ERROR(
        t_dst.scalar_type() == t_src.scalar_type(),
        InvalidArgument,
        "The input tensor's scalartype does not meet requirement: found %" PRId8
        " but expected %" PRId8,
        static_cast<int8_t>(t_src.scalar_type()),
        static_cast<int8_t>(t_dst.scalar_type()));
    // Reset the shape for the Method's input as the size of forwarded input
    // tensor for shape dynamism. Also is a safety check if need memcpy.
    Error err = resize_tensor(t_dst, t_src.sizes());
    ET_CHECK_OR_RETURN_ERROR(
        err == Error::Ok,
        InvalidArgument,
        "Error setting input %zu: 0x%" PRIx32,
        input_idx,
        static_cast<uint32_t>(err));
    Error error;
    if (pre_allocated_input_) {
      error = internal::copy_tensor_data(t_dst, t_src);
    } else {
      error = internal::share_tensor_data(t_dst, t_src);
    }
    ET_CHECK_OR_RETURN_ERROR(
        error == Error::Ok,
        InvalidArgument,
        "Error setting data_ptr %zu: 0x%" PRIx32,
        input_idx,
        static_cast<uint32_t>(error));
    // Prims have to be the same as what was traced
  } else if (e.isInt()) {
    ET_CHECK_OR_RETURN_ERROR(
        e.toInt() == input_evalue.toInt(),
        InvalidArgument,
        "The %zu-th input of method should have the same value as the input_evalue, but got %" PRId64
        " and %" PRId64,
        input_idx,
        e.toInt(),
        input_evalue.toInt());
  } else if (e.isBool()) {
    ET_CHECK_OR_RETURN_ERROR(
        e.toBool() == input_evalue.toBool(),
        InvalidArgument,
        "The %zu-th input of method should have the same value as the input_evalue, but got %" PRId64
        " and %" PRId64,
        input_idx,
        (int64_t)e.toBool(),
        (int64_t)input_evalue.toBool());
  } else if (e.isDouble()) {
    double lhs = input_evalue.toDouble();
    double rhs = e.toDouble();
    double atol = 1e-4;
    double rtol = 1e-5;
    bool is_equal = true;
    if (std::isnan(lhs) && std::isnan(rhs)) {
      // NaN == NaN
    } else if (
        !std::isfinite(lhs) && !std::isfinite(rhs) &&
        ((lhs > 0) == (rhs > 0))) {
      // -Inf == -Inf
      // +Inf == +Inf
    } else {
      auto allowed_error = atol + std::abs(rtol * rhs);
      auto actual_error = std::abs(lhs - rhs);
      if (!std::isfinite(actual_error) || actual_error > allowed_error) {
        is_equal = false;
      }
    }
    ET_CHECK_OR_RETURN_ERROR(
        is_equal,
        InvalidArgument,
        "The %zu-th input of method should have the same value as the input_evalue, but get %f and %f",
        input_idx,
        lhs,
        rhs);
  } else if (e.isString()) {
    ET_CHECK_OR_RETURN_ERROR(
        e.toString() == input_evalue.toString(),
        InvalidArgument,
        "The %zu-th input of method should have the same value as the input_evalue, but get %s and %s",
        input_idx,
        e.toString().data(),
        input_evalue.toString().data());
  } else {
    ET_LOG(Error, "Unsupported input type: %d", (int32_t)e.tag);
    return Error::InvalidArgument;
  }
  return Error::Ok;
}

__ET_NODISCARD Error
Method::set_inputs(const exec_aten::ArrayRef<EValue>& input_evalues) {
  ET_CHECK_OR_RETURN_ERROR(
      initialized(),
      InvalidState,
      "Inputs can not be set until method has been initialized.");

  ET_CHECK_OR_RETURN_ERROR(
      step_state_.instr_idx == 0 && step_state_.chain_idx == 0,
      InvalidState,
      "Inputs can not be set mid execution.");

  size_t input_size = inputs_size();
  ET_CHECK_OR_RETURN_ERROR(
      input_size == input_evalues.size(),
      InvalidArgument,
      "The length of given input array (%zu) must be same as the number of inputs in method (%zu).",
      input_evalues.size(),
      input_size);

  for (size_t i = 0; i < input_size; i++) {
    Error status = set_input(input_evalues[i], i);
    if (status != Error::Ok) {
      return status;
    }
  }
  return Error::Ok;
}

__ET_NODISCARD Error
Method::set_output_data_ptr(void* buffer, size_t size, size_t output_idx) {
  // Check method state
  ET_CHECK_OR_RETURN_ERROR(
      initialized(),
      InvalidState,
      "Outputs can not be retrieved until method has been initialized.");

  ET_CHECK_OR_RETURN_ERROR(
      !pre_allocated_output_,
      InvalidState,
      "Overriding output data pointer allocated by memory plan is not allowed.");

  // Check the args
  ET_CHECK_OR_RETURN_ERROR(
      output_idx <= outputs_size(),
      InvalidArgument,
      "output_idx: %zu num_outputs: %zu",
      output_idx,
      outputs_size());

  auto& output = mutable_output(output_idx);
  ET_CHECK_OR_RETURN_ERROR(
      output.isTensor(),
      InvalidArgument,
      "output type: %zu is not tensor",
      (size_t)output.tag);

  auto& t = output.toTensor();
  ET_CHECK_OR_RETURN_ERROR(
      output.isTensor(),
      InvalidArgument,
      "output type: %zu is not tensor",
      (size_t)output.tag);
  ET_CHECK_OR_RETURN_ERROR(
      t.nbytes() <= size,
      InvalidArgument,
      "buffer size: %zu is smaller then expected tensor size: %zu",
      size,
      t.nbytes());

  // Set data
  return internal::set_tensor_data(t, buffer, size);
}

__ET_NODISCARD Error
Method::get_outputs(EValue* output_evalues, size_t length) {
  ET_CHECK_OR_RETURN_ERROR(
      initialized(),
      InvalidState,
      "Outputs can not be retrieved until method has been initialized.");

  ET_CHECK_OR_RETURN_ERROR(
      length >= outputs_size(),
      InvalidArgument,
      "The given array is not large enough to hold all outputs.");

  for (size_t i = 0; i < outputs_size(); i++) {
    output_evalues[i] = values_[get_output_index(i)];
  }

  for (size_t i = outputs_size(); i < length; i++) {
    output_evalues[i] = EValue();
  }

  return Error::Ok;
}

Error Method::execute_instruction() {
  // TODO(jakeszwe): remove all the ET_CHECKS in this function and properly
  // return the error instead

  auto& chain = chains_[step_state_.chain_idx];
  auto instructions = chain.s_chain_->instructions();

  ET_CHECK_OR_RETURN_ERROR(
      step_state_.instr_idx < instructions->size(),
      Internal,
      "Instr index %zu >= chain[%zu] instr count %zu",
      step_state_.instr_idx,
      step_state_.chain_idx,
      (size_t)instructions->size());

  auto instruction = instructions->Get(step_state_.instr_idx);
  size_t next_instr_idx = step_state_.instr_idx + 1;
  Error err = Error::Ok;
  switch (instruction->instr_args_type()) {
    case executorch_flatbuffer::InstructionArguments::KernelCall: {
      EXECUTORCH_SCOPE_PROF("OPERATOR_CALL");
      internal::EventTracerProfileScope event_tracer_scope =
          internal::EventTracerProfileScope(event_tracer_, "OPERATOR_CALL");
      // TODO(T147221312): Also expose the temp allocator and tensor resizer
      // via the context.
      KernelRuntimeContext context(event_tracer_);
      auto args = chain.argument_lists_[step_state_.instr_idx];
      chain.kernels_[step_state_.instr_idx](context, args.data());
      err = context.failure_state();
      if (err != Error::Ok) {
        auto op_index = instruction->instr_args_as_KernelCall()->op_index();
        auto op = serialization_plan_->operators()->Get(op_index);
        ET_LOG(
            Error,
            "KernelCall failed at instruction %zu:%zu in operator %s.%s: 0x%x",
            step_state_.chain_idx,
            step_state_.instr_idx,
            op->name()->c_str(),
            op->overload()->c_str(),
            (unsigned int)err);
        for (size_t i = 0; i < args.size(); ++i) {
          ET_LOG(
              Error,
              "arg %u with type id %u",
              (unsigned int)i,
              (unsigned int)args[i]->tag);
        }
        // TODO(T153804650): Consider logging the EValues to help with
        // debugging. This is a failure path, and it doesn't matter if it's a
        // little slow. Do the same for DelegateCall errors.
      }
    } break;
    case executorch_flatbuffer::InstructionArguments::DelegateCall: {
      EXECUTORCH_SCOPE_PROF("DELEGATE_CALL");
      internal::EventTracerProfileScope event_tracer_profile_scope =
          internal::EventTracerProfileScope(event_tracer_, "DELEGATE_CALL");
      auto delegate_idx =
          instruction->instr_args_as_DelegateCall()->delegate_index();
      ET_CHECK_OR_RETURN_ERROR(
          delegate_idx < n_delegate_,
          Internal,
          "DELEGATE_CALL index %" PRIu32
          " >= num delegates %zu at instruction %zu",
          delegate_idx,
          n_delegate_,
          step_state_.instr_idx);
      BackendExecutionContext backend_execution_context(event_tracer_);
      err = delegates_[delegate_idx].Execute(
          backend_execution_context,
          chain.argument_lists_[step_state_.instr_idx].data());
      if (err != Error::Ok) {
        ET_LOG(
            Error,
            "CALL_DELEGATE execute failed at instruction %zu: 0x%" PRIx32,
            step_state_.instr_idx,
            static_cast<uint32_t>(err));
      }
    } break;
    case executorch_flatbuffer::InstructionArguments::JumpFalseCall: {
      EXECUTORCH_SCOPE_PROF("JF_CALL");
      internal::EventTracerProfileScope event_tracer_profile_scope =
          internal::EventTracerProfileScope(event_tracer_, "JF_CALL");
      auto jf_call = instruction->instr_args_as_JumpFalseCall();
      bool jf_result = parse_cond_value(values_[jf_call->cond_value_index()]);
      if (!jf_result) {
        next_instr_idx = jf_call->destination_instruction();
        // return Error::Ok;
      }
    } break;
    case executorch_flatbuffer::InstructionArguments::MoveCall: {
      EXECUTORCH_SCOPE_PROF("MOVE_CALL");
      internal::EventTracerProfileScope event_tracer_profile_scope =
          internal::EventTracerProfileScope(event_tracer_, "MOVE_CALL");
      auto move_call = instruction->instr_args_as_MoveCall();
      mutable_value(move_call->move_to()) = get_value(move_call->move_from());
    } break;
    case executorch_flatbuffer::InstructionArguments::FreeCall: {
      EXECUTORCH_SCOPE_PROF("FREE_CALL");
      internal::EventTracerProfileScope event_tracer_profile_scope =
          internal::EventTracerProfileScope(event_tracer_, "FREE_CALL");
      auto free_call = instruction->instr_args_as_FreeCall();
      auto t = values_[free_call->value_index()].toTensor();
      internal::reset_data_ptr(t);
    } break;
    default:
      ET_CHECK_MSG(
          false,
          "Instruction is not supported. %hhu",
          static_cast<uint8_t>(instruction->instr_args_type()));
  }
  if (err == Error::Ok) {
    step_state_.instr_idx = next_instr_idx;
  }
  return err;
}

Error Method::experimental_reset_execution() {
  ET_CHECK_OR_RETURN_ERROR(
      step_state_.chain_idx == n_chains_,
      InvalidState,
      "Cannot reset until EndOfMethod has been reached.");
  step_state_ = StepState{0, 0};
  return Error::Ok;
}

Error Method::experimental_step() {
  EXECUTORCH_PROFILE_INSTRUCTION_SCOPE(
      static_cast<int32_t>(step_state_.chain_idx),
      static_cast<uint32_t>(step_state_.instr_idx));
  internal::EventTracerProfileInstructionScope event_tracer_instr_scope =
      internal::EventTracerProfileInstructionScope(
          event_tracer_,
          static_cast<int32_t>(step_state_.chain_idx),
          static_cast<uint32_t>(step_state_.instr_idx));
  EXECUTORCH_SCOPE_PROF("Method::step");
  internal::EventTracerProfileScope event_tracer_profile_scope =
      internal::EventTracerProfileScope(event_tracer_, "Method::step");
  ET_CHECK_OR_RETURN_ERROR(
      initialized(),
      InvalidState,
      "Cannot execute until method has been initialized.");

  // If chain_step_ is on n_chains_, then we have no instructions run.
  if (step_state_.chain_idx == n_chains_) {
    return Error::EndOfMethod;
  }

  auto num_instructions =
      chains_[step_state_.chain_idx].s_chain_->instructions()->size();

  // Special case chains with no instructions. These appear for example in a
  // model that just returns the input/a constant.
  if (num_instructions == 0) {
    step_state_.chain_idx += 1;
    return Error::Ok;
  }

  auto status = execute_instruction();
  if (status != Error::Ok) {
    return status;
  }

  // end of the current chain, advance to the next chain
  if (step_state_.instr_idx == num_instructions) {
    step_state_.instr_idx = 0;
    step_state_.chain_idx += 1;
  }
  return Error::Ok;
}

Error Method::execute() {
  internal::event_tracer_create_event_block(event_tracer_, "Execute");
  internal::EventTracerProfileScope event_tracer_profile_scope =
      internal::EventTracerProfileScope(event_tracer_, "Method::execute");
  EXECUTORCH_SCOPE_PROF("Method::execute");
  ET_CHECK_OR_RETURN_ERROR(
      initialized(),
      NotSupported,
      "Cannot execute until method has been initialized.");

  // Chains are executed sequentially today, but future async designs may
  // branch and run many in parallel or out of order.
  for (step_state_.chain_idx = 0; step_state_.chain_idx < n_chains_;
       ++step_state_.chain_idx) {
    Chain& chain = chains_[step_state_.chain_idx];
    auto instructions = chain.s_chain_->instructions();
    ET_CHECK_OR_RETURN_ERROR(
        instructions != nullptr,
        Internal,
        "chain %zu has no instructions field",
        step_state_.chain_idx);

    // Loop over instructions
    step_state_.instr_idx = 0;
    while (step_state_.instr_idx < chain.s_chain_->instructions()->size()) {
      EXECUTORCH_PROFILE_INSTRUCTION_SCOPE(
          static_cast<int32_t>(step_state_.chain_idx),
          static_cast<uint32_t>(step_state_.instr_idx));
      internal::EventTracerProfileInstructionScope event_tracer_instr_scope =
          internal::EventTracerProfileInstructionScope(
              event_tracer_,
              static_cast<ChainID>(step_state_.chain_idx),
              static_cast<DebugHandle>(step_state_.instr_idx));
      auto status = execute_instruction();
      if (status != Error::Ok) {
        return status;
      }
    }
  }
  // TODO(jakeszwe, dbort): Decide on calling execute back to back without
  // going through the reset api first.
  return experimental_reset_execution();
}

MethodMeta Method::method_meta() const {
  auto name = serialization_plan_->name()->c_str();
  auto method_meta = program_->method_meta(name);
  ET_CHECK_MSG(
      method_meta.ok(),
      "Internal error: method_meta(%s) returned 0x%" PRIx32,
      name,
      static_cast<uint32_t>(method_meta.error()));
  return method_meta.get();
}

size_t Method::values_size() const {
  return n_value_;
}

const EValue& Method::get_value(size_t i) const {
  return values_[i];
}

EValue& Method::mutable_value(size_t i) {
  return values_[i];
}

size_t Method::inputs_size() const {
  return serialization_plan_->inputs()->size();
}

size_t Method::get_input_index(size_t i) const {
  return static_cast<size_t>(serialization_plan_->inputs()->Get(i));
}

const EValue& Method::get_input(size_t i) const {
  return get_value(get_input_index(i));
}

EValue& Method::mutable_input(size_t i) {
  return mutable_value(get_input_index(i));
}

size_t Method::outputs_size() const {
  return serialization_plan_->outputs()->size();
}

size_t Method::get_output_index(size_t i) const {
  return static_cast<size_t>(serialization_plan_->outputs()->Get(i));
}

const EValue& Method::get_output(size_t i) const {
  return get_value(get_output_index(i));
}

EValue& Method::mutable_output(size_t i) {
  return mutable_value(get_output_index(i));
}

Method::~Method() {
  // Destroy the values. It's necessary in ATen mode, where the refcount of
  // Tensors needs to be decremented properly.
  if (values_ != nullptr) {
    for (int i = 0; i < n_value_; ++i) {
      values_[i].~EValue();
    }
  }
  // Free any resources associated with delegate backends.
  if (delegates_ != nullptr) {
    for (int i = 0; i < n_delegate_; i++) {
      delegates_[i].~BackendDelegate();
    }
  }
  // All other fields are trivially destructible.
}
} // namespace executor
} // namespace torch
