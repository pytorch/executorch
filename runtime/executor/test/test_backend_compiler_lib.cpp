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
#include <executorch/runtime/platform/profiler.h>
#include <cstdio>
#include <cstdlib> /* strtol */

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
  long int numel;
  const char* dtype;
  long int debug_handle;
};

struct DemoOpList {
  DemoOp* ops;
  size_t numops;
};

class BackendWithCompiler final : public BackendInterface {
  int max_shape = 4;

 public:
  ~BackendWithCompiler() override = default;

  bool is_available() const override {
    return true;
  }

  // The delegate blob schema will be a list of instruction:
  // {op: {str}, numel: {long}, dtype: {type}}<debug_handle>n
  // Instruction will be separated by #, for example:
  // 'op:demo::mul.Tensor, numel:4, dtype:torch.float32<debug_handle>2\
  // #op:demo::add.Tensor, numel:4, dtype:torch.float32<debug_handle>4#'
  void parse_delegate(const char* str, const char* sub, DemoOp* op_list) const {
    const char* kOpLiteral = "op:";
    const char* kNumelLiteral = "numel:";
    const char* kDtypeliteral = "dtype:";
    const char* kDebugHandleLiteral = "<debug_handle>";

    const char* kComma = ",";

    int cnt = 0;
    const char* left = str;
    const char* right;

    // iter 0:
    // op:demo::sin.default, numel:1, dtype:torch.float32<debug_handle>1#
    // |<--left                                                 right-->|
    // iter 1:
    // op:demo::add.Tensor, numel:4, dtype:torch.float32<debug_handle>4#
    // |<--left                                                right-->|
    while ((right = strstr(left, sub))) {
      // Get operator name
      const char* op_start = strstr(left, kOpLiteral) + strlen(kOpLiteral);
      const char* op_end = strstr(op_start, kComma);

      op_list[cnt].name = op_start;

      // Get numel
      const char* numel_start =
          strstr(op_end, kNumelLiteral) + strlen(kNumelLiteral);
      char* numel_end = const_cast<char*>(strstr(numel_start, kComma));
      op_list[cnt].numel = strtol(numel_start, &numel_end, 10);

      // Get dtype
      const char* dtype_start =
          strstr(numel_end, kDtypeliteral) + strlen(kDtypeliteral);
      const char* dtype_end = strstr(dtype_start, kDebugHandleLiteral);
      op_list[cnt].dtype = dtype_start;

      // Get debug handle
      const char* debug_handle_start =
          strstr(dtype_end, kDebugHandleLiteral) + strlen(kDebugHandleLiteral);
      char* debug_end = const_cast<char*>(strstr(debug_handle_start, kComma));
      op_list[cnt].debug_handle = strtol(debug_handle_start, &debug_end, 10);

      // Move left pointer to the start of next instruction
      left = right + 1;
      cnt++;
    }
  }

  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec> compile_specs) const override {
    MemoryAllocator* runtime_allocator = context.get_runtime_allocator();
    int shape = *(int*)(compile_specs.at(0).value.buffer);
    ET_CHECK_OR_RETURN_ERROR(
        shape <= max_shape,
        InvalidArgument,
        "The input number is %d and it's larger than the max number %d "
        "supported by this backend.",
        shape,
        max_shape);

    const char* kSignLiteral = "#";
    // The first number is the number of total instruction
    const char* start = static_cast<const char*>(processed->data());

    const char* kVersion = "version:";
    const long int kRuntimeVersion = 0;
    char* version_start =
        const_cast<char*>(strstr(start, kVersion)) + strlen(kVersion);
    char* version_end;
    char* instruction_set_start =
        const_cast<char*>(strstr(start, kSignLiteral));

    long int version = strtol(version_start, &version_end, 10);
    ET_CHECK_OR_RETURN_ERROR(
        version == kRuntimeVersion,
        DelegateInvalidCompatibility,
        "The version of BackendWithCompiler runtime is %ld, but received an incompatible version %ld instead.",
        kRuntimeVersion,
        version);
    char* instruction_number_end;
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

    parse_delegate(instruction_set_start + 1, kSignLiteral, op_list->ops);

    // Can't call `processed->Free()` because op_list points into it.

    return op_list;
  }

  // Function that actually executes the model in the backend. Here there is
  // nothing to dispatch to, so the backend is implemented locally within
  // execute and it only supports add, subtract, and constant. In a non toy
  // backend you can imagine how this function could be used to actually
  // dispatch the inputs to the relevant backend/device.
  Error execute(
      ET_UNUSED BackendExecutionContext& context,
      DelegateHandle* handle,
      EValue** args) const override {
    EXECUTORCH_SCOPE_PROF("BackendWithCompiler::execute");

    // example: [('prim::Constant#1', 14), ('aten::add', 15)]
    auto op_list = static_cast<const DemoOpList*>(handle);

    const char* kDemoAdd = "demo::aten.add.Tensor";
    const char* kDemoMul = "demo::aten.mm.default";
    const char* kDemoSin = "demo::aten.sin.default";
    const char* kTorchFloat32 = "torch.float32";

    for (size_t index = 0; index < op_list->numops; index++) {
      auto instruction = op_list->ops[index];
      ET_CHECK_OR_RETURN_ERROR(
          strncmp(instruction.dtype, kTorchFloat32, strlen(kTorchFloat32)) == 0,
          NotSupported,
          "BackendWithCompiler only support float and doesn't support %s, "
          "debug handle is: %ld",
          instruction.dtype,
          instruction.debug_handle);
      if (strncmp(instruction.name, kDemoAdd, strlen(kDemoAdd)) == 0) {
        // z = z + b
        const float* b_ptr = args[2]->toTensor().const_data_ptr<float>();
        float* z_ptr = args[3]->toTensor().mutable_data_ptr<float>();
        for (size_t j = 0; j < instruction.numel; j++) {
          z_ptr[j] = b_ptr[j] + z_ptr[j];
        }
      } else if (strncmp(instruction.name, kDemoMul, strlen(kDemoMul)) == 0) {
        ET_CHECK_OR_RETURN_ERROR(
            instruction.numel == 4,
            NotSupported,
            "BackendWithCompiler only support 2 x 2 matrix multiplication, "
            "debug handle is %ld",
            instruction.debug_handle);
        // z = a * x
        const float* a_ptr = args[0]->toTensor().const_data_ptr<float>();
        const float* x_ptr = args[1]->toTensor().const_data_ptr<float>();
        float* z_ptr = args[3]->toTensor().mutable_data_ptr<float>();

        z_ptr[0] = a_ptr[0] * x_ptr[0] + a_ptr[1] * x_ptr[2];
        z_ptr[1] = a_ptr[0] * x_ptr[1] + a_ptr[1] * x_ptr[3];
        z_ptr[2] = a_ptr[2] * x_ptr[0] + a_ptr[3] * x_ptr[2];
        z_ptr[3] = a_ptr[2] * x_ptr[1] + a_ptr[3] * x_ptr[3];
      } else if (strncmp(instruction.name, kDemoSin, strlen(kDemoSin)) == 0) {
        const float* x_ptr = args[0]->toTensor().const_data_ptr<float>();
        float* y_ptr = args[1]->toTensor().mutable_data_ptr<float>();
        // Taylor series: an approximation of sin x around the point x = 0
        // sin(x) = x - x^3 / 3! + x^5 / 5! - x^7 / 7! ...
        // Use the first two items as proof of concept
        for (size_t j = 0; j < instruction.numel; j++) {
          y_ptr[j] = x_ptr[j] - x_ptr[j] * x_ptr[j] * x_ptr[j] / 6.0;
        }
      }
    }
    return Error::Ok;
  }
};

namespace {
auto cls = BackendWithCompiler();
Backend backend{"BackendWithCompilerDemo", &cls};
static auto success_with_compiler = register_backend(backend);
} // namespace
