/*
 * Copyright (c) Intel Corporation
 *
 * Licensed under the BSD License (the "License"); you may not use this file
 * except in compliance with the License. See the license file in the root
 * directory of this source tree for more details.
 */

#ifndef OPENVINO_BACKEND_HPP
#define OPENVINO_BACKEND_HPP

#include <memory>
#include <iostream>
#include <openvino/openvino.hpp>

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

using namespace std;
using executorch::aten::ScalarType;
using executorch::runtime::ArrayRef;
using executorch::runtime::Backend;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::BackendInitContext;
using executorch::runtime::CompileSpec;
using executorch::runtime::DelegateHandle;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::Result;

namespace executorch {
namespace backends {
namespace openvino {

typedef struct {
    std::shared_ptr<ov::CompiledModel> compiled_model;
    std::shared_ptr<ov::InferRequest> infer_request;
} ExecutionHandle;

class OpenvinoBackend final : public ::executorch::runtime::BackendInterface {
 public:
  OpenvinoBackend();
  ~OpenvinoBackend() = default;

  virtual bool is_available() const override;
  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec> compile_specs) const override;
  Error execute(
      BackendExecutionContext& context,
      DelegateHandle* input_handle,
      EValue** args) const override;
  void destroy(DelegateHandle* handle) const override;

 private:
  ov::element::Type convert_to_openvino_type(ScalarType scalar_type) const;
};

} // namespace openvino
} // namespace backends
} // namespace executorch

#endif // OPENVINO_BACKEND_HPP
