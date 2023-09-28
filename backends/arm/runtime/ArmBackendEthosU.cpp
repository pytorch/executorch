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

		vela_handles handles = { 0, 0, 0, 0, 0, 0};

		// Command stream - we know at this point it's aligned
		char *data = (char*)processed->data();

		// Read key sections from the vela_bin_stream
		this->vela_read( data, &handles );
		
		printf("Running program data:\n  cmd %p %d\n  weight %p %d\n  scratch %p %d\n",
			   handles.cmd_data, handles.cmd_data_length,
			   handles.weight_data, handles.weight_data_length,
			   handles.scratch_data, handles.scratch_data_length );
		// Invoke driver using the above pointers
		CommandStream cs(
			DataPointer(handles.cmd_data, handles.cmd_data_length),
			BasePointers({
					DataPointer(handles.weight_data, handles.weight_data_length),
					DataPointer(handles.scratch_data, handles.scratch_data_length)
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

private:
	typedef struct {
		const char *cmd_data; int cmd_data_length;
		const char *weight_data; int weight_data_length;
		const char *scratch_data; int scratch_data_length;
	} vela_handles;

	int vela_read(char* data, vela_handles *h ) const {
		if( strncmp( data, "vela_bin_stream", 15 ) ) return 0;
		while( 1 )
		{
			data += 16;
			if( !strncmp( data, "vela_end_stream", 15 ) )
			{
				printf("footer found!\n");
				return 1;
			}
			printf("reading block '%s':\n", data);
			char *block_name = data;
			data += 16;
			int block_length = ((int*)data)[0];
			int block_length_padded = block_length + (15-(block_length-1)%16);
			printf("  length %d\n", block_length );
			printf("  padded length %d\n", block_length_padded );
			char *block_data = data;
			data += block_length_padded;

			if( !strncmp( block_name, "cmd_data", strlen("cmd_data")) )
			{
				printf("Capturing cmd_data\n");
				h->cmd_data = block_data;
				h->cmd_data_length = block_length;
			}
			if( !strncmp( block_name, "weight_data", strlen("weight_data")) )
			{
				printf("Capturing weight_data\n");
				h->weight_data = block_data;
				h->weight_data_length = block_length;
			}
			if( !strncmp( block_name, "scratch_data", strlen("scratch_data")) )
			{
				printf("Capturing scratch_data\n");
				h->scratch_data = block_data;
				h->scratch_data_length = block_length;
			}
		}
	}

};

namespace {
	auto backend = ArmBackend();
	Backend backend_id{"ArmBackend", &backend};
	static auto registered = register_backend(backend_id);
} // namespace 

} // namespace executor
} // namespace torch
