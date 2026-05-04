/*
 * Copyright (c) 2026 iote.ai
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * ExecuTorch AXON delegate backend (C++ runtime).
 *
 * Registers as "AxonBackend" with the ExecuTorch runtime. When the .pte
 * contains TOSA-delegated subgraphs tagged for AxonBackend, ExecuTorch
 * calls our init/execute/destroy methods. One delegate handle per
 * subgraph in the .pte; a single .pte may contain many delegated
 * subgraphs.
 *
 * Multi-subgraph wiring
 * ---------------------
 * The Python side (AxonBackend.preprocess) writes one Nordic-compiled C
 * header per delegated subgraph into a generated directory, plus a master
 * table axon_subgraphs_table.h that #includes them all and exposes a
 * const array of {name, &model_<name>} entries. The Python side returns
 * a small marker as the .pte's processed_bytes:
 *
 *   offset  size  field
 *   ------  ----  -----
 *   0       4     magic     "AXNG"
 *   4       4     version   little-endian uint32 = 1
 *   8       4     name_len  little-endian uint32
 *   12      N     name      ASCII subgraph name (no NUL)
 *
 * We parse the marker at init() time, look the matching
 * nrf_axon_nn_compiled_model_s up by name in axon_subgraphs[], and
 * stash a pointer in the per-handle state. execute() then runs
 * nrf_axon_nn_model_infer_sync() with that model.
 */

#if defined(CONFIG_NRF_AXON) && CONFIG_NRF_AXON

#include <zephyr/kernel.h>
#include <zephyr/timing/timing.h>
#include <cstring>

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/log.h>

/* AXON driver */
#include "axon/nrf_axon_platform.h"
#include "drivers/axon/nrf_axon_driver.h"
#include "drivers/axon/nrf_axon_nn_infer.h"

/* Auto-generated subgraph table from the AXON backend export pipeline.
 * Brings in axon_subgraphs[] and AXON_SUBGRAPHS_COUNT. */
#include "generated/axon_subgraphs_table.h"

namespace {

using executorch::runtime::ArrayRef;
using executorch::runtime::Backend;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::BackendInitContext;
using executorch::runtime::BackendInterface;
using executorch::runtime::CompileSpec;
using executorch::runtime::DelegateHandle;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::Result;
using executorch::runtime::Span;
using exec_aten::ScalarType;
using exec_aten::Tensor;

/* Maximum simultaneously-loaded delegate handles. Each handle carries a
 * MAX_PACKED_OUTPUT_BYTES scratch region, so the per-handle cost is
 * dominated by that scratch. */
#define MAX_AXON_DELEGATES 16

/* Maximum int8 packed output bytes for any one delegated subgraph. */
#define MAX_PACKED_OUTPUT_BYTES 1024

struct AxonDelegateHandle {
    const nrf_axon_nn_compiled_model_s *model;
    int8_t packed_output[MAX_PACKED_OUTPUT_BYTES];
    bool initialized;
    /* Profiling: per-handle cumulative cycles spent inside
     * nrf_axon_nn_model_infer_sync() across the entire program run. */
    uint64_t total_infer_cycles;
    uint32_t total_infer_calls;
};

static bool s_platform_initialized = false;
static AxonDelegateHandle s_handles[MAX_AXON_DELEGATES];
static int s_handle_count = 0;

/* Profiling: aggregate AXON delegate cycles across all handles. */
extern "C" {
    uint64_t axon_delegate_total_cycles = 0;
    uint32_t axon_delegate_total_calls = 0;
}

/* ── Marker format (kept in sync with backends/nordic/axon/codegen.py) */
static constexpr uint8_t MARKER_MAGIC[4] = {'A', 'X', 'N', 'G'};
static constexpr uint32_t MARKER_VERSION = 1;
static constexpr size_t MARKER_HEADER_SIZE = 12; /* magic + version + name_len */

static const nrf_axon_nn_compiled_model_s *
parse_marker_and_lookup(const uint8_t *bytes, size_t len, char *out_name, size_t out_name_cap)
{
    if (len < MARKER_HEADER_SIZE) {
        ET_LOG(Error, "AxonBackend: processed_bytes too short (%zu < %zu)",
               len, MARKER_HEADER_SIZE);
        return nullptr;
    }
    if (memcmp(bytes, MARKER_MAGIC, 4) != 0) {
        ET_LOG(Error, "AxonBackend: bad marker magic %02x%02x%02x%02x",
               bytes[0], bytes[1], bytes[2], bytes[3]);
        return nullptr;
    }
    /* Little-endian uint32 reads. */
    uint32_t version =
        (uint32_t)bytes[4] | ((uint32_t)bytes[5] << 8) |
        ((uint32_t)bytes[6] << 16) | ((uint32_t)bytes[7] << 24);
    uint32_t name_len =
        (uint32_t)bytes[8] | ((uint32_t)bytes[9] << 8) |
        ((uint32_t)bytes[10] << 16) | ((uint32_t)bytes[11] << 24);
    if (version != MARKER_VERSION) {
        ET_LOG(Error, "AxonBackend: marker version %u, expected %u",
               (unsigned)version, (unsigned)MARKER_VERSION);
        return nullptr;
    }
    if (MARKER_HEADER_SIZE + name_len > len) {
        ET_LOG(Error, "AxonBackend: marker name overflow (%u + 12 > %zu)",
               (unsigned)name_len, len);
        return nullptr;
    }
    if (name_len + 1 > out_name_cap) {
        ET_LOG(Error, "AxonBackend: marker name too long (%u >= %zu)",
               (unsigned)name_len, out_name_cap);
        return nullptr;
    }
    memcpy(out_name, bytes + MARKER_HEADER_SIZE, name_len);
    out_name[name_len] = '\0';

    /* Linear scan over the generated table — at most ~64 entries. */
    for (size_t i = 0; i < AXON_SUBGRAPHS_COUNT; i++) {
        if (strcmp(axon_subgraphs[i].name, out_name) == 0) {
            return axon_subgraphs[i].model;
        }
    }
    ET_LOG(Error, "AxonBackend: no subgraph named '%s' in generated table "
                  "(%d entries) — did you re-run export but forget to "
                  "rebuild firmware?",
           out_name, (int)AXON_SUBGRAPHS_COUNT);
    return nullptr;
}

class AxonBackendImpl final : public BackendInterface {
public:
    bool is_available() const override {
        return true;
    }

    Result<DelegateHandle*> init(
        BackendInitContext& context,
        FreeableBuffer* processed,
        ArrayRef<CompileSpec> compile_specs
    ) const override {
        ET_LOG(Info, "AxonBackend::init (delegate %d, processed=%zu bytes)",
               s_handle_count, processed->size());

        if (s_handle_count >= MAX_AXON_DELEGATES) {
            ET_LOG(Error, "Too many AXON delegates (max %d)", MAX_AXON_DELEGATES);
            return Error::MemoryAllocationFailed;
        }

        /* Initialize AXON platform once across all delegate handles. */
        if (!s_platform_initialized) {
            nrf_axon_result_e r = nrf_axon_platform_init();
            if (r != NRF_AXON_RESULT_SUCCESS) {
                ET_LOG(Error, "AXON platform init failed: %d", (int)r);
                return Error::InvalidState;
            }
            s_platform_initialized = true;
        }

        /* Parse the marker, look the model up by name. */
        char name_buf[128];
        const nrf_axon_nn_compiled_model_s *model = parse_marker_and_lookup(
            static_cast<const uint8_t*>(processed->data()),
            processed->size(),
            name_buf, sizeof(name_buf));
        if (!model) {
            return Error::InvalidProgram;
        }

        nrf_axon_result_e r = nrf_axon_nn_model_validate(model);
        if (r != NRF_AXON_RESULT_SUCCESS) {
            ET_LOG(Error, "AXON model '%s' validate failed: %d", name_buf, (int)r);
            return Error::InvalidProgram;
        }

        AxonDelegateHandle *handle = &s_handles[s_handle_count++];
        handle->model = model;
        handle->initialized = true;
        handle->total_infer_cycles = 0;
        handle->total_infer_calls = 0;
        memset(handle->packed_output, 0, sizeof(handle->packed_output));

        ET_LOG(Info,
               "  AXON model '%s' bound (out: %ux%ux%u byte_width=%u)",
               name_buf,
               model->output_dimensions.height,
               model->output_dimensions.width,
               model->output_dimensions.channel_cnt,
               (unsigned)model->output_dimensions.byte_width);

        processed->Free();
        return handle;
    }

    Error execute(
        BackendExecutionContext& context,
        DelegateHandle* handle,
        Span<EValue*> args
    ) const override {
        auto *axon_handle = static_cast<AxonDelegateHandle*>(handle);
        const nrf_axon_nn_compiled_model_s *model = axon_handle->model;

        if (args.size() < 2) {
            ET_LOG(Error, "AxonBackend::execute: args=%zu (need >= 2)", args.size());
            return Error::InvalidArgument;
        }

        const auto& input_evalue = args[0];
        if (!input_evalue->isTensor()) {
            ET_LOG(Error, "AxonBackend: input is not a tensor");
            return Error::InvalidArgument;
        }
        const Tensor& input_tensor = input_evalue->toTensor();
        if (input_tensor.scalar_type() != ScalarType::Char) {
            ET_LOG(Error, "AxonBackend: input dtype %d, expected int8",
                   (int)input_tensor.scalar_type());
            return Error::InvalidArgument;
        }
        const int8_t *input_data = input_tensor.const_data_ptr<int8_t>();

        timing_t t_start = timing_counter_get();
        nrf_axon_result_e r = nrf_axon_nn_model_infer_sync(
            model, input_data, axon_handle->packed_output);
        timing_t t_end = timing_counter_get();
        if (r != NRF_AXON_RESULT_SUCCESS) {
            ET_LOG(Error, "AXON inference failed: %d", (int)r);
            return Error::InvalidState;
        }
        uint64_t cyc = timing_cycles_get(&t_start, &t_end);
        axon_handle->total_infer_cycles += cyc;
        axon_handle->total_infer_calls++;
        axon_delegate_total_cycles += cyc;
        axon_delegate_total_calls++;

        /* Copy AXON's packed int8 output into ExecuTorch's output tensor. */
        auto& output_evalue = args[1];
        if (!output_evalue->isTensor()) {
            ET_LOG(Error, "AxonBackend: output is not a tensor");
            return Error::InvalidArgument;
        }
        Tensor& output_tensor = output_evalue->toTensor();
        if (output_tensor.scalar_type() != ScalarType::Char) {
            ET_LOG(Error, "AxonBackend: output dtype %d, expected int8",
                   (int)output_tensor.scalar_type());
            return Error::InvalidArgument;
        }
        int8_t *out_data = output_tensor.mutable_data_ptr<int8_t>();
        size_t copy_bytes = output_tensor.numel();
        if (copy_bytes > sizeof(axon_handle->packed_output)) {
            ET_LOG(Error,
                   "AxonBackend: output tensor (%zu bytes) > packed_output "
                   "scratch (%zu bytes); bump MAX_PACKED_OUTPUT_BYTES",
                   copy_bytes, sizeof(axon_handle->packed_output));
            return Error::MemoryAllocationFailed;
        }
        memcpy(out_data, axon_handle->packed_output, copy_bytes);
        return Error::Ok;
    }

    void destroy(DelegateHandle* handle) const override {
        (void)handle;
    }
};

/* Register the backend with ExecuTorch's runtime. */
static AxonBackendImpl s_axon_backend;
static Backend s_backend_id{"AxonBackend", &s_axon_backend};
static auto s_registered __attribute__((used)) =
    executorch::runtime::register_backend(s_backend_id);

} /* anonymous namespace */

/* Profiling API: zero all cycle counters. */
extern "C" void axon_delegate_reset_profile(void)
{
    axon_delegate_total_cycles = 0;
    axon_delegate_total_calls = 0;
    for (int i = 0; i < s_handle_count; i++) {
        s_handles[i].total_infer_cycles = 0;
        s_handles[i].total_infer_calls = 0;
    }
}

/* Profiling API: dump per-handle AXON cycle counts to the log. */
extern "C" void axon_delegate_dump_profile(void)
{
    ET_LOG(Info, "");
    ET_LOG(Info, "=== AXON delegate profile ===");
    ET_LOG(Info, "handles bound: %d", s_handle_count);
    ET_LOG(Info, "total infer cycles: %llu (%lu calls)",
           (unsigned long long)axon_delegate_total_cycles,
           (unsigned long)axon_delegate_total_calls);
    if (axon_delegate_total_calls > 0) {
        uint64_t avg = axon_delegate_total_cycles / axon_delegate_total_calls;
        ET_LOG(Info, "avg cycles/call: %llu", (unsigned long long)avg);
    }
    for (int i = 0; i < s_handle_count; i++) {
        const AxonDelegateHandle *h = &s_handles[i];
        if (h->total_infer_calls == 0) {
            continue;
        }
        uint64_t avg = h->total_infer_cycles / h->total_infer_calls;
        const auto &dim = h->model->output_dimensions;
        ET_LOG(Info, "  [%2d] %-25s out=%ux%ux%u  calls=%lu  total=%llu  avg=%llu",
               i,
               h->model->model_name ? h->model->model_name : "(unnamed)",
               dim.height, dim.width, dim.channel_cnt,
               (unsigned long)h->total_infer_calls,
               (unsigned long long)h->total_infer_cycles,
               (unsigned long long)avg);
    }
    ET_LOG(Info, "=============================");
}

#else /* not CONFIG_NRF_AXON: stub the profiling symbols so firmware
       * can link without #ifdefs everywhere. */

#include <stdint.h>
extern "C" {
    uint64_t axon_delegate_total_cycles = 0;
    uint32_t axon_delegate_total_calls = 0;
    void axon_delegate_dump_profile(void) {}
    void axon_delegate_reset_profile(void) {}
}

#endif /* CONFIG_NRF_AXON */
