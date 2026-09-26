/*  Copyright (c) Intel Corporation
 *
 *  Licensed under the BSD License (the "License"); you may not use this file
 *  except in compliance with the License. See the license file found in the
 *  LICENSE file in the root directory of this source tree.
 */

#ifndef OPENVINO_BACKEND_H
#define OPENVINO_BACKEND_H

#include <memory>
#include <mutex>

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

#include "OpenvinoApi.h"

namespace exr = executorch::runtime;
namespace exa = executorch::aten;

namespace executorch {
namespace backends {
namespace openvino {

struct ExecutionHandle {
  ov_compiled_model_t* compiled_model = nullptr;
  ov_infer_request_t* infer_request = nullptr;
};

class OpenvinoBackend final : public ::exr::BackendInterface {
 public:
  OpenvinoBackend() = default;
  ~OpenvinoBackend() override = default;

  bool is_available() const override;
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
  bool ensure_loaded() const;
  ov_element_type_e convert_to_openvino_type(exa::ScalarType scalar_type) const;
  exr::Result<ov_tensor_t*> create_ov_tensor(const exa::Tensor& tensor) const;

  mutable DlHandle lib_handle_;
  mutable OpenvinoFunctions ov_;
  mutable std::once_flag load_flag_;
  mutable bool loaded_ = false;
};

} // namespace openvino
} // namespace backends
} // namespace executorch

#endif // OPENVINO_BACKEND_H
