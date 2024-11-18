/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/devtools/bundled_program/bundled_program.h>

#include <cmath>
#include <cstddef>
#include <cstring>

#ifdef USE_ATEN_LIB
#include <ATen/ATen.h>
#endif // USE_ATEN_LIB

#include <executorch/devtools/bundled_program/schema/bundled_program_schema_generated.h>
#include <executorch/runtime/core/event_tracer_hooks.h>
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/platform/log.h>

using exec_aten::ArrayRef;
using exec_aten::Half;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;
using ::executorch::runtime::Method;
using ::executorch::runtime::Result;

namespace executorch {
namespace bundled_program {

namespace {

constexpr size_t kMaxDim = 16;

#ifdef USE_ATEN_LIB

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
using torch::executor::TensorImpl;
// Create a tensorimpl with same content using bundled tensor
TensorImpl impl_like(bundled_program_flatbuffer::Tensor* bundled_tensor) {
  ScalarType scalar_type =
      static_cast<ScalarType>(bundled_tensor->scalar_type());
  ssize_t dim = bundled_tensor->sizes()->size();
  exec_aten::SizesType* sizes = bundled_tensor->mutable_sizes()->data();
  void* data = bundled_tensor->mutable_data()->data();
  exec_aten::DimOrderType* dim_order =
      bundled_tensor->mutable_dim_order()->data();

  // The strides of created tensorimpl will only be actually used when
  // comparsion (`tensor_are_close` below). To eliminate the usage of memory
  // allocator, here we set the initial strides as null and reconstruct the
  // stride array as temporary varible when comparsion.
  exec_aten::StridesType* strides = nullptr;
  return TensorImpl(scalar_type, dim, sizes, data, dim_order, strides);
}
#endif

/**
 * Returns true if the two elements are close according to the description on
 * `tensors_are_close()`.
 *
 * T must be a floating point type. Non-floating point data should be compared
 * directly.
 */
template <
    typename T,
    typename = std::enable_if_t<std::is_floating_point<T>::value>>
bool elem_is_close(const T& ai, const T& bi, double rtol, double atol) {
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
  return true;
}

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
    if (!elem_is_close(a[i], b[i], rtol, atol)) {
      return false;
    }
  }
  return true;
}

bool data_is_close_half(
    const Half* a,
    const Half* b,
    size_t numel,
    double rtol,
    double atol) {
  for (size_t i = 0; i < numel; i++) {
    if (!elem_is_close(
            static_cast<double>(a[i]), static_cast<double>(b[i]), rtol, atol)) {
      return false;
    }
  }
  return true;
}

bool tensors_are_close(
    const Tensor& bundled_tensor,
    const Tensor& method_output_tensor,
    double rtol,
    double atol) {
  if (bundled_tensor.scalar_type() != method_output_tensor.scalar_type() ||
      bundled_tensor.sizes() != method_output_tensor.sizes()) {
    return false;
  }

#ifdef USE_ATEN_LIB

  ET_CHECK_MSG(
      bundled_tensor.strides() == method_output_tensor.strides(),
      "The two inputs of `tensors_are_close` function shall have same strides");

#else // !USE_ATEN_LIB

  // Contruct stride array for bundled tensor based on its dim order since
  // strides of bundled_tensor in lean mode is null.
  exec_aten::StridesType strides[kMaxDim] = {0};
  auto status = torch::executor::dim_order_to_stride(
      bundled_tensor.sizes().data(),
      bundled_tensor.dim_order().data(),
      bundled_tensor.dim(),
      strides);
  ET_CHECK_MSG(
      status == Error::Ok, "dim_order_to_stride returned invalid status");

  // TODO(T132992348): support comparison between tensors of different strides
  ET_CHECK_MSG(
      ArrayRef<exec_aten::StridesType>(strides, bundled_tensor.dim()) ==
          method_output_tensor.strides(),
      "The two inputs of `tensors_are_close` function shall have same strides");
#endif

  // Since the two tensors have same shape and strides, any two elements that
  // share same index from underlying data perspective will also share same
  // index from tensor perspective, whatever the size and strides really are.
  // e.g. if a[i_1, i_2, ... i_n] = a.const_data_ptr()[m], we can assert
  // b[i_1, i_2, ... i_n] = b.const_data_ptr()[m])
  // So we can just compare the two underlying data sequentially to figure out
  // if the two tensors are same.

  if (bundled_tensor.nbytes() == 0) {
    // Note that this case is important. It's valid for a zero-size tensor to
    // have a null data pointer, but in some environments it's invalid to pass a
    // null pointer to memcmp() even when the size is zero.
    return true;
  } else if (bundled_tensor.scalar_type() == ScalarType::Float) {
    return data_is_close<float>(
        bundled_tensor.const_data_ptr<float>(),
        method_output_tensor.const_data_ptr<float>(),
        bundled_tensor.numel(),
        rtol,
        atol);
  } else if (bundled_tensor.scalar_type() == ScalarType::Double) {
    return data_is_close<double>(
        bundled_tensor.const_data_ptr<double>(),
        method_output_tensor.const_data_ptr<double>(),
        bundled_tensor.numel(),
        rtol,
        atol);
  } else if (bundled_tensor.scalar_type() == ScalarType::Half) {
    return data_is_close_half(
        bundled_tensor.const_data_ptr<Half>(),
        method_output_tensor.const_data_ptr<Half>(),
        bundled_tensor.numel(),
        rtol,
        atol);
  } else {
    // Non-floating-point types can be compared bitwise.
    return memcmp(
               bundled_tensor.const_data_ptr(),
               method_output_tensor.const_data_ptr(),
               bundled_tensor.nbytes()) == 0;
  }
}

Result<bundled_program_flatbuffer::BundledMethodTestSuite*>
get_method_test_suite(
    const bundled_program_flatbuffer::BundledProgram* bundled_program,
    Method& method) {
  const char* method_name = method.method_meta().name();
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
ET_NODISCARD Error load_bundled_input(
    Method& method,
    SerializedBundledProgram* bundled_program_ptr,
    size_t testset_idx) {
  ET_CHECK_OR_RETURN_ERROR(
      bundled_program_flatbuffer::BundledProgramBufferHasIdentifier(
          bundled_program_ptr),
      NotSupported,
      "The input buffer should be a bundled program.");

  auto method_test = get_method_test_suite(
      bundled_program_flatbuffer::GetBundledProgram(bundled_program_ptr),
      method);

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
        TensorImpl impl = impl_like(bundled_input_tensor);
        Tensor t = Tensor(&impl);
#endif
        // Use t to create EValue as Method's input.
        e_input = EValue(t);
        // Setting input like this is safe because the `set_input` function only
        // copies the underlying data blob of TensorImpl impl into the method,
        // not the pointer of impl, Tensor t or even the Evalue e_input. So
        // their lifetime will not impact the safety. Also there's a specific
        // memory space with enough lifetime holding the underlying data blob,
        // so the lifetime of the data blob is not an issue.
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
            "Data type %hhu not supported",
            static_cast<uint8_t>(bundled_input->val_type()));
        break;
      }
    }

    ET_CHECK_OR_RETURN_ERROR(
        status == Error::Ok,
        NotSupported,
        "set_input failed during load bundled inputs with status 0%" PRIx32,
        static_cast<uint32_t>(status));
  }

  ::executorch::runtime::internal::event_tracer_set_bundled_input_index(
      method.get_event_tracer(), testset_idx);

  return Error::Ok;
}

ET_NODISCARD Error verify_method_outputs(
    Method& method,
    SerializedBundledProgram* bundled_program_ptr,
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
      method);

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
        TensorImpl impl = impl_like(bundled_expected_output_tensor);
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
            static_cast<uint8_t>(bundled_expected_output->val_type()));
        break;
      }
    }
  }

  return Error::Ok;
}

ET_NODISCARD Error get_program_data(
    void* file_data,
    size_t file_data_len,
    const void** out_program_data,
    size_t* out_program_data_len) {
  if (is_bundled_program(file_data, file_data_len)) {
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

bool is_bundled_program(void* file_data, ET_UNUSED size_t file_data_len) {
  // Even though the flatbuffer API doesn't accept a length, it's important to
  // require one so that we could change the internal representation, or use a
  // future API that does require a length.
  return bundled_program_flatbuffer::BundledProgramBufferHasIdentifier(
      file_data);
}

} // namespace bundled_program
} // namespace executorch
