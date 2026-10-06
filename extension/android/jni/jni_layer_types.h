/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/core/evalue.h>
#include <fbjni/fbjni.h>

namespace executorch::extension {

class JTensor : public facebook::jni::JavaClass<JTensor> {
 public:
  constexpr static const char* kJavaDescriptor =
      "Lorg/pytorch/executorch/Tensor;";

  static facebook::jni::local_ref<JTensor::javaobject> newJTensorFromTensor(
      const executorch::aten::Tensor& tensor);

  static TensorPtr newTensorFromJTensor(
      facebook::jni::alias_ref<JTensor::javaobject> jtensor);
};

class JEValue : public facebook::jni::JavaClass<JEValue> {
 public:
  constexpr static const char* kJavaDescriptor =
      "Lorg/pytorch/executorch/EValue;";

  constexpr static int kTypeCodeTensor = 1;
  constexpr static int kTypeCodeString = 2;
  constexpr static int kTypeCodeDouble = 3;
  constexpr static int kTypeCodeInt = 4;
  constexpr static int kTypeCodeBool = 5;

  static facebook::jni::local_ref<JEValue> newJEValueFromEValue(
      runtime::EValue evalue);

  static TensorPtr JEValueToTensorImpl(
      facebook::jni::alias_ref<JEValue> jevalue);
};

} // namespace executorch::extension
