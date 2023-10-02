/*
 * Copyright 2023 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Arm backend for Ethos-U baremetal driver stack, this relies on the
 * ethos-u-core-driver for hardware interaction.
 */

#include <memory>

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>

#include <ethosu_driver.h>
#include <pmu_ethosu.h>

#include "command_stream.hpp"
using namespace EthosU::CommandStream;

namespace torch {
namespace executor {

// TODO we should be in 0x31, not this lower 1MB sRAM
// SRAM (rwx) : ORIGIN = 0x31000000, LENGTH = 0x00200000
#define CS300_SRAM_LOW ((void*)0x11000000)
#define CS300_SRAM_HIGH ((void*)0x110FFFFF)

class ArmBackend final : public PyTorchBackendInterface {

public:
	ArmBackend() {
		ET_LOG(Debug, "Constructing ARM Backend");
	}
	
	~ArmBackend() = default;

	virtual bool is_available() const override {
		return 1;
	}

	Result<DelegateHandle*> init(
		BackendInitContext& context,
		FreeableBuffer* processed,
		ArrayRef<CompileSpec> compile_specs) const override {

        ET_LOG(Info, "ArmBackend::init %p", processed->data() );

		char *data = (char*)processed->data();
		size_t size = processed->size();
		char *foot = data + size - 16;

		// Header and footer both 16 bit aligned suggest valid structure and we
		// wont walk off the end of the chunks and segfault
		if( !((int)data == next_mul_16((int)data)) )
		{
			ET_LOG(Error, "ArmBackend::init header unaligned");
			return Error::InvalidProgram;
		}
		if( !((int)foot == next_mul_16((int)foot)) )
		{
			ET_LOG(Error, "ArmBackend::init header unaligned");
			return Error::InvalidProgram;
		}
		if( !(0 == strncmp( data, "vela_bin_stream", 15 )) )
		{
			ET_LOG(Error, "ArmBackend::init header unaligned");
			return Error::InvalidProgram;
		}
		if( !(0 == strncmp( foot, "vela_end_stream", 15 )) )
		{
			ET_LOG(Error, "ArmBackend::init header unaligned");
			return Error::InvalidProgram;
		}
		// Verify address range is accessible current expectation is the program
		// is wholly stored in SRAM
		if( !(data > CS300_SRAM_LOW || foot < CS300_SRAM_HIGH) );
		
		// Return the same buffer we were passed - this data will be
		// executed directly
		return processed;
	}

	Error execute(
		BackendExecutionContext& context,
		DelegateHandle* input_handle,
		EValue** args) const override {

		FreeableBuffer* processed = (FreeableBuffer*)input_handle;

		ET_LOG(Info, "ArmBackend::execute %p", processed->data() );

		vela_handles handles = { 0, 0, 0, 0, 0, 0 };

		// Command stream - we know at this point it's aligned
		char *data = (char*)processed->data();

		// Read key sections from the vela_bin_stream
		if( !this->vela_read( data, &handles, processed->size() ) )
		{
			ET_LOG(Error, "ArmBackend::vela_read: error, invalid binary layout" );
			return Error::InvalidProgram;
		}

		ET_LOG(Debug, "ArmBackend::execute: Running program data:\n  cmd %p %d\n  weight %p %d\n  scratch %p %d\n",
			   handles.cmd_data, handles.cmd_data_size,
			   handles.weight_data, handles.weight_data_size,
			   handles.scratch_data, handles.scratch_data_size );

		// TMP emit scratch
		printf("Scratch before:\n");
		for( int i=0; i<handles.scratch_data_size; i++ )
		{
			if( i%4 == 0 ) ((char*)handles.scratch_data)[i] = 1;
			printf("%02x ", ((char*)handles.scratch_data)[i]);
			if( !((i+1)%4) ) printf("\n");
		}
		printf("\n");
		
		// Invoke driver using the above pointers
		CommandStream cs(
			DataPointer(handles.cmd_data, handles.cmd_data_size),
			BasePointers({
					DataPointer(handles.weight_data, handles.weight_data_size),
					DataPointer(handles.scratch_data, handles.scratch_data_size)
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

        // TMP emit scratch
        printf("Scratch after:\n");
        for( int i=0; i<handles.scratch_data_size; i++ )
        {
            printf("%02x ", ((char*)handles.scratch_data)[i]);
            if( !((i+1)%4) ) printf("\n");
        }
        printf("\n");
		
		return Error::Ok;
	}

	void destroy(DelegateHandle* handle) const override {
		return;
	}

private:
	typedef struct {
		const char *cmd_data; int cmd_data_size;
		const char *weight_data; int weight_data_size;
		const char *scratch_data; int scratch_data_size;
	} vela_handles;

	typedef struct {
		char name[16];
		int size; char _pad[12];
		char data[];
	} vela_bin_block;

	static int next_mul_16( int n ) {
		return ((n-1)|15)+1;
	}
	
	int vela_read(char* data, vela_handles *h, int size ) const {

		// Read header string
		if( strncmp( data, "vela_bin_stream", 15 ) )
		{
			return 0;
		}
		data += 16;

		// Expect one or more 'vela_bin_block's
		while( 1 )
		{
			vela_bin_block *b = (vela_bin_block*)data;
			data += 16 + 16 + next_mul_16(b->size);

			// Exit with success on finding end of stream
			if( !strncmp( b->name, "vela_end_stream", 15 ) ) return 1;

			if( !strncmp( b->name, "cmd_data", strlen("cmd_data")) )
			{
				// This magic header confirms a valid command stream in binary
				if( strncmp( b->data, "COP1", 4 ) ) return 0;
				h->cmd_data = b->data;
				h->cmd_data_size = b->size;
			}
			if( !strncmp( b->name, "weight_data", strlen("weight_data")) )
			{
				h->weight_data = b->data;;
				h->weight_data_size = b->size;
			}
			if( !strncmp( b->name, "scratch_data", strlen("scratch_data")) )
			{
				h->scratch_data = b->data;
				h->scratch_data_size = b->size;
			}
		}
	}

};

	auto backend = ArmBackend();
	void arm_backend_register() {
		Backend backend_id{"ArmBackend", &backend};
		static auto registered = register_backend(backend_id);
	}

} // namespace executor
} // namespace torch
