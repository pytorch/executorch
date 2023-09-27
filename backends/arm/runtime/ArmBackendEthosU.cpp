/*
 * Copyright 2023 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Arm backend for Ethos-U baremetal driver stack relies on ethos-u-core-driver
 */

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>

#include <ethosu_driver.h>
#include <pmu_ethosu.h>

namespace torch {
namespace executor {

class ArmBackend final : public PyTorchBackendInterface {

public:
	~ArmBackend() = default;

	virtual bool is_available() const override {
		return 1;
	}

	Result<DelegateHandle*> init(
		BackendInitContext& context,
		FreeableBuffer* processed,
		ArrayRef<CompileSpec> compile_specs) const override {
		return Error::Ok;
	}

	Error execute(
		BackendExecutionContext& context,
		DelegateHandle* handle,
		EValue** args) const override {
		return Error::Ok;
	}

	void destroy(DelegateHandle* handle) const override {
		return;
	}

};

namespace {
	auto backend = ArmBackend();
	Backend backend_id{"ArmBackend", &backend};
	static auto registered = register_backend(backend_id);
} // namespace 

} // namespace executor
} // namespace torch
