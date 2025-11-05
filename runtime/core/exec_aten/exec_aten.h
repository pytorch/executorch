/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/tensor_shape_dynamism.h> // @manual
#include <executorch/runtime/platform/compiler.h>
#include <optional>
#ifdef USE_ATEN_LIB
#include <ATen/Tensor.h> // @manual
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h> // @manual
#include <c10/core/Layout.h> // @manual
#include <c10/core/MemoryFormat.h> // @manual
#include <c10/core/Scalar.h> // @manual
#include <c10/util/ArrayRef.h> // @manual
#include <c10/util/BFloat16-math.h> // @manual
#include <c10/util/BFloat16.h> // @manual
#include <c10/util/Half.h> // @manual
#include <c10/util/complex.h> // @manual
#include <c10/util/qint32.h> // @manual
#include <c10/util/qint8.h> // @manual
#include <c10/util/quint2x4.h> // @manual
#include <c10/util/quint4x2.h> // @manual
#include <c10/util/quint8.h> // @manual
#include <c10/util/string_view.h> // @manual
#include <torch/torch.h>
#else // use executor
#include <executorch/runtime/core/array_ref.h> // @manual
#include <executorch/runtime/core/portable_type/bfloat16.h> // @manual
#include <executorch/runtime/core/portable_type/bfloat16_math.h> // @manual
#include <executorch/runtime/core/portable_type/complex.h> // @manual
#include <executorch/runtime/core/portable_type/device.h> // @manual
#include <executorch/runtime/core/portable_type/half.h> // @manual
#include <executorch/runtime/core/portable_type/qint_types.h> // @manual
#include <executorch/runtime/core/portable_type/scalar.h> // @manual
#include <executorch/runtime/core/portable_type/scalar_type.h> // @manual
#include <executorch/runtime/core/portable_type/string_view.h> // @manual
#include <executorch/runtime/core/portable_type/tensor.h> // @manual
#include <executorch/runtime/core/portable_type/tensor_options.h> // @manual

#endif

/**
 * This hack is for separating out ATen mode vs non-ATen mode. In ATen mode,
 * we use the ATen types directly. In non-ATen mode, we use the portable types.
 * To avoid duplicate symbols and/or duplicate operator registration, when a
 * user depends on both the ATen mode and non-ATen mode versions of the
 * ExecuTorch library.
 */
#ifndef ET_RUNTIME_NAMESPACE
#if defined(USE_ATEN_LIB)
#define ET_RUNTIME_NAMESPACE runtime::aten
#else
#define ET_RUNTIME_NAMESPACE runtime
#endif
#endif

namespace executorch {
namespace aten {

using TensorShapeDynamism = executorch::runtime::TensorShapeDynamism;

using std::nullopt;
using std::nullopt_t;
using std::optional;
using std::string_view;

#ifdef USE_ATEN_LIB

using Tensor = at::Tensor;
using TensorList = at::TensorList;
using TensorImpl = at::TensorImpl;
template <typename T>
using ArrayRef = c10::ArrayRef<T>;
using ScalarType = at::ScalarType;
using Scalar = c10::Scalar;
using MemoryFormat = c10::MemoryFormat;
using SizesType = int64_t;
using DimOrderType = uint8_t;
using StridesType = int64_t;
using Device = c10::Device;
using DeviceType = c10::DeviceType;
using Layout = c10::Layout;

// Custom types that map to ScalarType
using Half = c10::Half;
template <typename T>
using complex = c10::complex<T>;
using qint8 = c10::qint8;
using quint8 = c10::quint8;
using qint32 = c10::qint32;
using BFloat16 = c10::BFloat16;
using quint4x2 = c10::quint4x2;
using quint2x4 = c10::quint2x4;
using IntArrayRef = at::IntArrayRef;

template <typename T>
using OptionalArrayRef = c10::OptionalArrayRef<T>;
using OptionalIntArrayRef = OptionalArrayRef<int64_t>;

inline ssize_t compute_numel(const SizesType* sizes, ssize_t dim) {
  return static_cast<ssize_t>(
      c10::multiply_integers(c10::ArrayRef<SizesType>(sizes, dim)));
}

#undef ET_PRI_TENSOR_SIZE
#define ET_PRI_TENSOR_SIZE PRId64

#undef ET_PRI_TENSOR_DIM
#define ET_PRI_TENSOR_DIM PRId64

#undef ET_PRI_TENSOR_NUMEL
#define ET_PRI_TENSOR_NUMEL PRId64

#undef ET_PRI_SIZES_AND_STRIDES
#define ET_PRI_SIZES_AND_STRIDES PRId64

#else // Use executor types

using Tensor = torch::executor::Tensor;
using TensorImpl = torch::executor::TensorImpl;
template <typename T>
using ArrayRef = torch::executor::ArrayRef<T>;
using ScalarType = torch::executor::ScalarType;
using TensorList = ArrayRef<Tensor>;
using Scalar = torch::executor::Scalar;
using MemoryFormat = torch::executor::MemoryFormat;
using SizesType = torch::executor::Tensor::SizesType;
using DimOrderType = torch::executor::Tensor::DimOrderType;
using StridesType = torch::executor::Tensor::StridesType;
using Device = torch::executor::Device;
using DeviceType = torch::executor::DeviceType;
using Layout = torch::executor::Layout;

// Custom types that map to ScalarType
using Half = torch::executor::Half;
template <typename T>
using complex = torch::executor::complex<T>;
using qint8 = torch::executor::qint8;
using quint8 = torch::executor::quint8;
using qint32 = torch::executor::qint32;
using BFloat16 = torch::executor::BFloat16;
using quint4x2 = torch::executor::quint4x2;
using quint2x4 = torch::executor::quint2x4;

using IntArrayRef = torch::executor::IntArrayRef;

template <typename T>
using OptionalArrayRef = std::optional<torch::executor::ArrayRef<T>>;
using OptionalIntArrayRef = OptionalArrayRef<int64_t>;

using torch::executor::compute_numel;

#endif // Use ExecuTorch types

} // namespace aten
} // namespace executorch

// DEPRECATED: The exec_aten:: namespace is deprecated. Use executorch::aten::
// instead.
namespace exec_aten = executorch::aten;

namespace torch {
namespace executor {
using TensorList = ::executorch::aten::TensorList;
} // namespace executor
} // namespace torch
