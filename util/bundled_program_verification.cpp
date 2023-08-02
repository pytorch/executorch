/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/util/bundled_program_verification.h>

#include <cstddef>
#include <cstring>

#ifdef USE_ATEN_LIB
#include <ATen/ATen.h>
#endif // USE_ATEN_LIB

#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/schema/bundled_program_schema_generated.h>
#include <executorch/schema/program_generated.h>

namespace torch {
namespace executor {
namespace util {

namespace {

#ifdef USE_ATEN_LIB

#define kMaxDim 16

// Create an aten tensor with same content using bundled tensor
at::Tensor tensor_like(executorch_flatbuffer::BundledTensor* bundled_tensor) {
  ET_CHECK(bundled_tensor->sizes()->size() <= kMaxDim);
  int64_t ret_t_sizes[kMaxDim];

  for (size_t i = 0; i < bundled_tensor->sizes()->size(); i++) {
    ret_t_sizes[i] = static_cast<int64_t>(bundled_tensor->sizes()->data()[i]);
  }

  at::Tensor ret_tensor = at::zeros(
      {ret_t_sizes, bundled_tensor->sizes()->size()},
      at::dtype(static_cast<ScalarType>(bundled_tensor->scalar_type())));
  memcpy(
      ret_tensor.data_ptr(),
      static_cast<const void*>(bundled_tensor->data()->Data()),
      ret_tensor.nbytes());
  return ret_tensor;
}

#else // !USE_ATEN_LIB
// Create a tensorimpl with same content using bundled tensor
TensorImpl impl_like(
    executorch_flatbuffer::BundledTensor* bundled_tensor,
    MemoryAllocator* runtime_allocator) {
  ScalarType scalar_type =
      static_cast<ScalarType>(bundled_tensor->scalar_type());
  ssize_t dim = bundled_tensor->sizes()->size();
  exec_aten::SizesType* sizes = bundled_tensor->mutable_sizes()->data();
  void* data = bundled_tensor->mutable_data()->data();
  exec_aten::DimOrderType* dim_order =
      bundled_tensor->mutable_dim_order()->data();
  exec_aten::StridesType* strides =
      ET_TRY_ALLOCATE_LIST_OR(runtime_allocator, exec_aten::StridesType, dim, {
        ET_CHECK_MSG(false, "Failed to allocate memory for strides");
      });
  auto status =
      torch::executor::dim_order_to_stride(sizes, dim_order, dim, strides);
  ET_CHECK_MSG(
      status == Error::Ok, "dim_order_to_stride returned invalid status");

  return TensorImpl(scalar_type, dim, sizes, data, dim_order, strides);
}
#endif
} // namespace

// Load testset_idx-th bundled data into the Method
__ET_NODISCARD Error LoadBundledInput(
    Method& method,
    serialized_bundled_program* bundled_program_ptr,
    MemoryAllocator* memory_allocator,
    size_t method_idx,
    size_t testset_idx) {
  ET_CHECK_OR_RETURN_ERROR(
      executorch_flatbuffer::BundledProgramBufferHasIdentifier(
          bundled_program_ptr),
      NotSupported,
      "The input buffer should be a bundled program.");

  auto bundled_inputs =
      executorch_flatbuffer::GetBundledProgram(bundled_program_ptr)
          ->execution_plan_tests()
          ->Get(method_idx)
          ->test_sets()
          ->Get(testset_idx)
          ->inputs();

  for (size_t input_idx = 0; input_idx < method.inputs_size(); input_idx++) {
    auto bundled_input = bundled_inputs->GetMutableObject(input_idx);

    // The EValue variable will contain the info set to input_idx Method input.
    EValue e_input;

    // Status for set_input function in this scope.
    Error status;

    // Set e_input with bundled_input based on different types.
    switch (bundled_input->val_type()) {
      case executorch_flatbuffer::BundledValueUnion::BundledTensor: {
        auto bundled_input_tensor =
            static_cast<executorch_flatbuffer::BundledTensor*>(
                bundled_input->mutable_val());

#ifdef USE_ATEN_LIB
        Tensor t = tensor_like(bundled_input_tensor);
#else // !USE_ATEN_LIB
        TensorImpl impl = impl_like(bundled_input_tensor, memory_allocator);
        Tensor t = Tensor(&impl);
#endif
        // Use t to create EValue as Method's input.
        e_input = EValue(t);
        // Setting input like this a bit problematic because
        // tensor that EValue is storing has pointer to tensorIml.
        // This pointer is from the stack of this function whose lifetime
        // is not beyond this function.
        // TODO(T148052964): use runtime allocator to allocate `TensorImpl impl`
        status = method.set_input(e_input, input_idx);
        break;
      }
      case executorch_flatbuffer::BundledValueUnion::BundledInt: {
        auto bundled_input_int = bundled_input->val_as_BundledInt();
        e_input = EValue(bundled_input_int->int_val());
        status = method.set_input(e_input, input_idx);
        break;
      }
      case executorch_flatbuffer::BundledValueUnion::BundledDouble: {
        auto bundled_input_int = bundled_input->val_as_BundledDouble();
        e_input = EValue(bundled_input_int->double_val());
        status = method.set_input(e_input, input_idx);
        break;
      }
      case executorch_flatbuffer::BundledValueUnion::BundledBool: {
        auto bundled_input_int = bundled_input->val_as_BundledBool();
        e_input = EValue(bundled_input_int->bool_val());
        status = method.set_input(e_input, input_idx);
        break;
      }
      default: {
        ET_CHECK_OR_RETURN_ERROR(
            false,
            NotSupported,
            "Data type %hhd not supported",
            bundled_input->val_type());
        break;
      }
    }

    ET_CHECK_OR_RETURN_ERROR(
        status == Error::Ok,
        NotSupported,
        "set_input failed during load bundled inputs with status %" PRIu32,
        status);
  }

  return Error::Ok;
}

__ET_NODISCARD Error VerifyResultWithBundledExpectedOutput(
    Method& method,
    serialized_bundled_program* bundled_program_ptr,
    MemoryAllocator* memory_allocator,
    size_t method_idx,
    size_t testset_idx,
    double rtol,
    double atol) {
  ET_CHECK_OR_RETURN_ERROR(
      executorch_flatbuffer::BundledProgramBufferHasIdentifier(
          bundled_program_ptr),
      NotSupported,
      "The input buffer should be a bundled program.");

  auto bundled_expected_outputs =
      executorch_flatbuffer::GetBundledProgram(bundled_program_ptr)
          ->execution_plan_tests()
          ->Get(method_idx)
          ->test_sets()
          ->Get(testset_idx)
          ->expected_outputs();

  for (size_t output_idx = 0; output_idx < method.outputs_size();
       output_idx++) {
    auto bundled_expected_output =
        bundled_expected_outputs->GetMutableObject(output_idx);
    auto method_output = method.get_output(output_idx);
    switch (bundled_expected_output->val_type()) {
      case executorch_flatbuffer::BundledValueUnion::BundledTensor: {
        auto bundled_expected_output_tensor =
            static_cast<executorch_flatbuffer::BundledTensor*>(
                bundled_expected_output->mutable_val());
        const auto method_output_tensor = method_output.toTensor();

#ifdef USE_ATEN_LIB
        Tensor t = tensor_like(bundled_expected_output_tensor);
#else // !USE_ATEN_LIB
        TensorImpl impl =
            impl_like(bundled_expected_output_tensor, memory_allocator);
        Tensor t = Tensor(&impl);
#endif
        ET_CHECK_OR_RETURN_ERROR(
            testing::tensors_are_close(t, method_output_tensor, rtol, atol),
            NotFound, // maybe some new error tag?
            "Method's output data mismatched the expected one.");
        break;
      }
      default: {
        ET_CHECK_OR_RETURN_ERROR(
            false,
            NotSupported,
            "Data type %hhd not supported",
            bundled_expected_output->val_type());
        break;
      }
    }
  }

  return Error::Ok;
}

__ET_NODISCARD Error GetProgramData(
    void* file_data,
    size_t file_data_len,
    const void** out_program_data,
    size_t* out_program_data_len) {
  if (executorch_flatbuffer::ProgramBufferHasIdentifier(file_data)) {
    *out_program_data = file_data;
    *out_program_data_len = file_data_len;
  } else if (executorch_flatbuffer::BundledProgramBufferHasIdentifier(
                 file_data)) {
    auto program_bundled = executorch_flatbuffer::GetBundledProgram(file_data);
    *out_program_data = program_bundled->program()->data();
    *out_program_data_len = program_bundled->program()->size();
  } else {
    ET_LOG(
        Error,
        "Unrecognized flatbuffer identifier '%.4s'",
        flatbuffers::GetBufferIdentifier(file_data));
    return Error::NotSupported;
  }
  return Error::Ok;
}

} // namespace util
} // namespace executor
} // namespace torch
