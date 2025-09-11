/*  Copyright (c) Intel Corporation
 *
 *  Licensed under the BSD License (the "License"); you may not use this file
 *  except in compliance with the License. See the license file found in the
 *  LICENSE file in the root directory of this source tree.
 */

#ifndef OPENVINO_BACKEND_H
#define OPENVINO_BACKEND_H

#include <openvino/openvino.hpp>
#include <iostream>
#include <memory>

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

namespace exr = executorch::runtime;
namespace exa = executorch::aten;

using namespace std;

namespace executorch {
namespace backends {
namespace openvino {

typedef struct {
  std::shared_ptr<ov::CompiledModel> compiled_model;
  std::shared_ptr<ov::InferRequest> infer_request;
} ExecutionHandle;

class OpenvinoBackend final : public ::exr::BackendInterface {
 public:
  OpenvinoBackend();
  ~OpenvinoBackend() = default;

  virtual bool is_available() const override;
  exr::Result<exr::DelegateHandle*> init(
      exr::BackendInitContext& context,
      exr::FreeableBuffer* processed,
      exr::ArrayRef<exr::CompileSpec> compile_specs) const override;
  exr::Error execute(
      exr::BackendExecutionContext& context,
      exr::DelegateHandle* input_handle,
      exr::Span<exr::EValue*> args) const override;
  void destroy(exr::DelegateHandle* handle) const override;

 private:
  ov::element::Type convert_to_openvino_type(exa::ScalarType scalar_type) const;
};

} // namespace openvino
} // namespace backends
} // namespace executorch

#endif // OPENVINO_BACKEND_H
