/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/util/bundled_program_verification.h>

#include <cmath>
#include <cstddef>
#include <cstring>

#ifdef USE_ATEN_LIB
#include <ATen/ATen.h>
#endif // USE_ATEN_LIB

#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/schema/bundled_program_schema_generated.h>

namespace torch {
namespace executor {
namespace util {

namespace {

#ifdef USE_ATEN_LIB

#define kMaxDim 16

// Create an aten tensor with same content using bundled tensor
at::Tensor tensor_like(bundled_program_flatbuffer::Tensor* bundled_tensor) {
  ET_CHECK(bundled_tensor->sizes()->size() <= kMaxDim);
  int64_t ret_t_sizes[kMaxDim];

  for (size_t i = 0; i < bundled_tensor->sizes()->size(); i++) {
    ret_t_sizes[i] = static_cast<int64_t>(bundled_tensor->sizes()->data()[i]);
  }

  at::Tensor ret_tensor = at::zeros(
      {ret_t_sizes, bundled_tensor->sizes()->size()},
      at::dtype(static_cast<ScalarType>(bundled_tensor->scalar_type())));
  memcpy(
      ret_tensor.mutable_data_ptr(),
      static_cast<const void*>(bundled_tensor->data()->Data()),
      ret_tensor.nbytes());
  return ret_tensor;
}

#else // !USE_ATEN_LIB
// Create a tensorimpl with same content using bundled tensor
TensorImpl impl_like(
    bundled_program_flatbuffer::Tensor* bundled_tensor,
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

/**
 * Returns true if the two arrays are close according to the description on
 * `tensors_are_close()`.
 *
 * T must be a floating point type. Non-floating point data should be compared
 * directly.
 */
template <
    typename T,
    typename = std::enable_if_t<std::is_floating_point<T>::value>>
bool data_is_close(
    const T* a,
    const T* b,
    size_t numel,
    double rtol,
    double atol) {
  for (size_t i = 0; i < numel; i++) {
    const auto ai = a[i];
    const auto bi = b[i];

    if (std::isnan(ai) && std::isnan(bi)) {
      // NaN == NaN
    } else if (
        !std::isfinite(ai) && !std::isfinite(bi) && ((ai > 0) == (bi > 0))) {
      // -Inf == -Inf
      // +Inf == +Inf
    } else if (rtol == 0 && atol == 0) {
      // Exact comparison; avoid unnecessary math.
      if (ai != bi) {
        return false;
      }
    } else {
      auto allowed_error = atol + std::abs(rtol * bi);
      auto actual_error = std::abs(ai - bi);
      if (!std::isfinite(actual_error) || actual_error > allowed_error) {
        return false;
      }
    }
  }
  return true;
}

bool tensors_are_close(
    const Tensor& a,
    const Tensor& b,
    double rtol,
    double atol) {
  if (a.scalar_type() != b.scalar_type() || a.sizes() != b.sizes()) {
    return false;
  }

  // TODO(T132992348): support comparison between tensors of different strides
  ET_CHECK_MSG(
      a.strides() == b.strides(),
      "The two inputs of `tensors_are_close` function shall have same strides");

  // Since the two tensors have same shape and strides, any two elements that
  // share same index from underlying data perspective will also share same
  // index from tensor perspective, whatever the size and strides really are.
  // e.g. if a[i_1, i_2, ... i_n] = a.const_data_ptr()[m], we can assert
  // b[i_1, i_2, ... i_n] = b.const_data_ptr()[m])
  // So we can just compare the two underlying data sequentially to figure out
  // if the two tensors are same.

  if (a.nbytes() == 0) {
    // Note that this case is important. It's valid for a zero-size tensor to
    // have a null data pointer, but in some environments it's invalid to pass a
    // null pointer to memcmp() even when the size is zero.
    return true;
  } else if (a.scalar_type() == ScalarType::Float) {
    return data_is_close<float>(
        a.const_data_ptr<float>(),
        b.const_data_ptr<float>(),
        a.numel(),
        rtol,
        atol);
  } else if (a.scalar_type() == ScalarType::Double) {
    return data_is_close<double>(
        a.const_data_ptr<double>(),
        b.const_data_ptr<double>(),
        a.numel(),
        rtol,
        atol);
  } else {
    // Non-floating-point types can be compared bitwise.
    return memcmp(a.const_data_ptr(), b.const_data_ptr(), a.nbytes()) == 0;
  }
}

Result<bundled_program_flatbuffer::BundledMethodTestSuite*>
get_method_test_suite(
    const bundled_program_flatbuffer::BundledProgram* bundled_program,
    const char* method_name) {
  auto method_test_suites = bundled_program->method_test_suites();
  for (size_t i = 0; i < method_test_suites->size(); i++) {
    auto m_test = method_test_suites->GetMutableObject(i);
    if (std::strcmp(m_test->method_name()->c_str(), method_name) == 0) {
      return m_test;
    }
  }
  ET_LOG(Error, "No method named '%s' in given bundled program", method_name);
  return Error::InvalidArgument;
}

} // namespace

// Load testset_idx-th bundled data into the Method
__ET_NODISCARD Error LoadBundledInput(
    Method& method,
    serialized_bundled_program* bundled_program_ptr,
    MemoryAllocator* memory_allocator,
    const char* method_name,
    size_t testset_idx) {
  ET_CHECK_OR_RETURN_ERROR(
      bundled_program_flatbuffer::BundledProgramBufferHasIdentifier(
          bundled_program_ptr),
      NotSupported,
      "The input buffer should be a bundled program.");

  auto method_test = get_method_test_suite(
      bundled_program_flatbuffer::GetBundledProgram(bundled_program_ptr),
      method_name);

  if (!method_test.ok()) {
    return method_test.error();
  }

  auto bundled_inputs =
      method_test.get()->test_cases()->Get(testset_idx)->inputs();

  for (size_t input_idx = 0; input_idx < method.inputs_size(); input_idx++) {
    auto bundled_input = bundled_inputs->GetMutableObject(input_idx);

    // The EValue variable will contain the info set to input_idx Method input.
    EValue e_input;

    // Status for set_input function in this scope.
    Error status;

    // Set e_input with bundled_input based on different types.
    switch (bundled_input->val_type()) {
      case bundled_program_flatbuffer::ValueUnion::Tensor: {
        auto bundled_input_tensor =
            static_cast<bundled_program_flatbuffer::Tensor*>(
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
      case bundled_program_flatbuffer::ValueUnion::Int: {
        auto bundled_input_int = bundled_input->val_as_Int();
        e_input = EValue(bundled_input_int->int_val());
        status = method.set_input(e_input, input_idx);
        break;
      }
      case bundled_program_flatbuffer::ValueUnion::Double: {
        auto bundled_input_int = bundled_input->val_as_Double();
        e_input = EValue(bundled_input_int->double_val());
        status = method.set_input(e_input, input_idx);
        break;
      }
      case bundled_program_flatbuffer::ValueUnion::Bool: {
        auto bundled_input_int = bundled_input->val_as_Bool();
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
    const char* method_name,
    size_t testset_idx,
    double rtol,
    double atol) {
  ET_CHECK_OR_RETURN_ERROR(
      bundled_program_flatbuffer::BundledProgramBufferHasIdentifier(
          bundled_program_ptr),
      NotSupported,
      "The input buffer should be a bundled program.");

  auto method_test = get_method_test_suite(
      bundled_program_flatbuffer::GetBundledProgram(bundled_program_ptr),
      method_name);

  if (!method_test.ok()) {
    return method_test.error();
  }

  auto bundled_expected_outputs =
      method_test.get()->test_cases()->Get(testset_idx)->expected_outputs();

  for (size_t output_idx = 0; output_idx < method.outputs_size();
       output_idx++) {
    auto bundled_expected_output =
        bundled_expected_outputs->GetMutableObject(output_idx);
    auto method_output = method.get_output(output_idx);
    switch (bundled_expected_output->val_type()) {
      case bundled_program_flatbuffer::ValueUnion::Tensor: {
        auto bundled_expected_output_tensor =
            static_cast<bundled_program_flatbuffer::Tensor*>(
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
            tensors_are_close(t, method_output_tensor, rtol, atol),
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
  if (IsBundledProgram(file_data)) {
    auto program_bundled =
        bundled_program_flatbuffer::GetBundledProgram(file_data);
    *out_program_data = program_bundled->program()->data();
    *out_program_data_len = program_bundled->program()->size();
  } else {
    ET_LOG(
        Error,
        "Unrecognized bundled program flatbuffer identifier '%.4s'",
        flatbuffers::GetBufferIdentifier(file_data));
    return Error::NotSupported;
  }
  return Error::Ok;
}

bool IsBundledProgram(void* file_data) {
  return bundled_program_flatbuffer::BundledProgramBufferHasIdentifier(
      file_data);
}

} // namespace util
} // namespace executor
} // namespace torch
