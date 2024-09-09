/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/event_tracer_hooks_delegate.h>

#include <cstdlib> /* strtol */
#include <cstring>

using executorch::runtime::ArrayRef;
using executorch::runtime::Backend;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::BackendInitContext;
using executorch::runtime::BackendInterface;
using executorch::runtime::CompileSpec;
using executorch::runtime::DelegateHandle;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::Result;

struct DemoOp {
  const char* name;
  long int debug_handle;
};

struct DemoOpList {
  DemoOp* ops;
  size_t numops;
};

class BackendWithDelegateMapping final : public BackendInterface {
 public:
  ~BackendWithDelegateMapping() override = default;

  bool is_available() const override {
    return true;
  }

  // The delegate blob schema will be a list of instruction:
  // {op_name:{str},delegate debug identifier:{int}}
  // Instructions will be separated by #, for example:
  // `op_name:demo_linear,delegate debug
  // identifier:0#op_name:mm_decomp_from_addmm,\ delegate debug
  // identifier:1#op_name:mm_decomp_from_addmm,delegate debug identifier:2`
  Error parse_delegate(
      const char* str,
      DemoOpList* op_list,
      MemoryAllocator* runtime_allocator) const {
    char* op_name;
    char* delegate_debug_identifier;
    size_t num_ops = 0;
    char* copy = strdup(str);

    while (true) {
      char* saveptr = nullptr;
      op_name = strtok_r(copy, ",", &saveptr);
      delegate_debug_identifier = strtok_r(nullptr, ",", &saveptr);

      if (op_name == nullptr || delegate_debug_identifier == nullptr) {
        break;
      }

      if (op_name != nullptr && delegate_debug_identifier != nullptr) {
        char* op_name_mem = (char*)ET_ALLOCATE_OR_RETURN_ERROR(
            runtime_allocator, strlen(op_name) + 1);
        memcpy(op_name_mem, op_name, strlen(op_name) + 1);
        op_list->ops[num_ops].name = op_name_mem;
        op_list->ops[num_ops].debug_handle = atoi(delegate_debug_identifier);
      }

      num_ops += 1;
      if (num_ops == op_list->numops) {
        break;
      }
      copy = nullptr;
    }

    free(copy);
    return Error::Ok;
  }

  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec> compile_specs) const override {
    MemoryAllocator* runtime_allocator = context.get_runtime_allocator();
    (void)compile_specs;
    const char* kSignLiteral = "#";
    // The first number is the number of total instruction
    const char* start = static_cast<const char*>(processed->data());
    char* instruction_number_end =
        const_cast<char*>(strstr(start, kSignLiteral));
    long int instruction_number = strtol(start, &instruction_number_end, 10);
    ET_CHECK_OR_RETURN_ERROR(
        instruction_number >= 0,
        InvalidArgument,
        "Instruction count must be non-negative: %ld",
        instruction_number);

    auto op_list =
        ET_ALLOCATE_INSTANCE_OR_RETURN_ERROR(runtime_allocator, DemoOpList);
    op_list->ops = ET_ALLOCATE_LIST_OR_RETURN_ERROR(
        runtime_allocator, DemoOp, instruction_number);
    op_list->numops = static_cast<size_t>(instruction_number);

    Error error =
        parse_delegate(instruction_number_end + 1, op_list, runtime_allocator);
    if (error != Error::Ok) {
      return error;
    }

    return op_list;
  }

  // This function doesn't actually execute the op but just prints out the op
  // name and the corresponding delegate debug identifier.
  Error execute(
      ET_UNUSED BackendExecutionContext& context,
      DelegateHandle* handle,
      EValue** args) const override {
    (void)args;
    // example: [('prim::Constant#1', 14), ('aten::add', 15)]
    auto op_list = static_cast<const DemoOpList*>(handle);

    for (size_t index = 0; index < op_list->numops; index++) {
      ET_LOG(
          Info,
          "Op name = %s Delegate debug index = %ld",
          op_list->ops[index].name,
          op_list->ops[index].debug_handle);
      event_tracer_log_profiling_delegate(
          context.event_tracer(),
          nullptr,
          op_list->ops[index].debug_handle,
          0,
          1);
      /**
       If you used string based delegate debug identifiers then the profiling
       call would be as below.
       event_tracer_log_profiling_delegate(
          context.event_tracer(),
          pointer_to_delegate_debug_string,
          -1,
          0,
          1);
       */
    }

    return Error::Ok;
  }
};

namespace {
auto cls = BackendWithDelegateMapping();
Backend backend{"BackendWithDelegateMappingDemo", &cls};
static auto success_with_compiler = register_backend(backend);
} // namespace
