/*
 * Copyright 2023 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Arm backend for Ethos-U baremetal driver stack relies on ethos-u-core-driver
 */

#include <memory>

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>

#include <ethosu_driver.h>
#include <pmu_ethosu.h>

#include "command_stream.hpp"
using namespace EthosU::CommandStream;

// Required byte alignment of all input pointers
#define ETHOS_U_ALIGN 0xF
char *ethos_align( char *ptr )
{
	return (char*)((uintptr_t)~ETHOS_U_ALIGN & (uintptr_t)(ptr + (ETHOS_U_ALIGN-1)));
}

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

		printf("ArmBackend::init 0x%X\n", processed->data());

		char *data = (char*)processed->data();
		size_t size = processed->size();
		
		//the model should have been placed in sram with
		//__attribute__((section(".sram.data"), aligned(16)))
		void *aligned = ethos_align(data);
		if( data != ethos_align(data)) return Error::InvalidProgram;

		// TODO: Verify address range is accessible to Ethos-U
		// current expectation is the program is in SRAM
		if(0) return Error::InvalidProgram;
		
		// Return the same buffer we were passed - this data will be
		// executed directly
		return processed;
	}

	Error execute(
		BackendExecutionContext& context,
		DelegateHandle* input_handle,
		EValue** args) const override {

		FreeableBuffer* processed = (FreeableBuffer*)input_handle;

		printf("ArmBackend::execute 0x%X\n", processed->data());

		// Command stream - we know at this point it's aligned
		char *handle = (char*)processed->data();
		int command_stream_length = ((int*)handle)[0];
		char *command_stream = ethos_align(handle+sizeof(int));
		
		// Static tensors/weights/model data
		handle = ethos_align( command_stream + command_stream_length );
		int weight_data_length = ((int*)handle)[0];
		char *weight_data = ethos_align(handle+sizeof(int));

		// Activation data, input and output memory
		handle = ethos_align( weight_data + weight_data_length );
		int activation_data_length = ((int*)handle)[0];
		char *activation_data = ethos_align(handle+sizeof(int));


		// Invoke driver using the above pointers
		CommandStream cs(
			DataPointer(command_stream, command_stream_length),
			BasePointers({
					DataPointer(weight_data, weight_data_length),
					DataPointer(activation_data, activation_data_length)
				}),
			PmuEvents({ETHOSU_PMU_CYCLE, ETHOSU_PMU_NPU_IDLE, ETHOSU_PMU_NPU_ACTIVE})
			);

		cs.getPmu().clear();
		int res = cs.run(1);
		if(res == 0)
		{
			uint64_t cycleCount = cs.getPmu().getCycleCount();
			cs.getPmu().print();
			printf("cycleCount=%llu, cycleCountPerJob=%llu\n", cycleCount, cycleCount);
		} else {
			printf("Error, failure executing job\n");
			return Error::InvalidProgram;
		}
	
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
