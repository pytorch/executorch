/*
 * Copyright (c) 2026 iote.ai
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Multi-layer inference runner.
 *
 * Loads a .pte with multiple AXON-delegated subgraphs (one per FC layer).
 * Each subgraph has its own compiled command buffer. The AXON delegate
 * binds all subgraphs at init() and dispatches them at execute() time.
 */

#include <cstdint>
#include <cstring>

#include <zephyr/kernel.h>
#include <zephyr/timing/timing.h>

#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/platform/log.h>

#include "model_pte.h"

/* Profiling API from AxonBackend.h */
extern "C" {
    extern uint64_t axon_delegate_total_cycles;
    extern uint32_t axon_delegate_total_calls;
    void axon_delegate_dump_profile(void);
}

namespace et = executorch::runtime;
using et::Error;
using et::EValue;
using et::HierarchicalAllocator;
using et::MemoryAllocator;
using et::MemoryManager;
using et::Method;
using et::Program;
using et::Result;
using et::Span;
using executorch::extension::BufferDataLoader;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using exec_aten::TensorImpl;

static uint8_t method_allocator_pool[32 * 1024];
static uint8_t planned_memory_pool[32 * 1024];
static uint8_t temp_allocator_pool[8 * 1024];

extern "C" int run_inference(void)
{
    if (model_pte_len == 0) {
        ET_LOG(Error, "No model embedded. Run: ./setup_export_env.sh && ./run_export.sh");
        return -1;
    }

    ET_LOG(Info, "Loading model (%u bytes)...", model_pte_len);

    BufferDataLoader loader(model_pte, model_pte_len);
    Result<Program> program = Program::load(&loader);
    if (!program.ok()) {
        ET_LOG(Error, "Program::load failed: 0x%x",
               static_cast<uint32_t>(program.error()));
        return -1;
    }
    ET_LOG(Info, "Program loaded, %zu method(s)", program->num_methods());

    const char *method_name = nullptr;
    {
        auto name_result = program->get_method_name(0);
        if (!name_result.ok()) {
            ET_LOG(Error, "No methods in program");
            return -2;
        }
        method_name = *name_result;
    }
    ET_LOG(Info, "Method: %s", method_name);

    MemoryAllocator method_allocator(
        sizeof(method_allocator_pool), method_allocator_pool);
    MemoryAllocator temp_allocator(
        sizeof(temp_allocator_pool), temp_allocator_pool);

    auto method_meta = program->method_meta(method_name);
    if (!method_meta.ok()) {
        ET_LOG(Error, "Failed to get method meta");
        return -3;
    }

    Span<uint8_t> planned_span(planned_memory_pool,
                               sizeof(planned_memory_pool));
    HierarchicalAllocator planned_allocator({&planned_span, 1});
    MemoryManager memory_manager(
        &method_allocator, &planned_allocator, &temp_allocator);

    Result<Method> method = program->load_method(
        method_name, &memory_manager);
    if (!method.ok()) {
        ET_LOG(Error, "load_method failed: 0x%x",
               static_cast<uint32_t>(method.error()));
        return -4;
    }
    ET_LOG(Info, "Method loaded (AXON delegates bound: %lu)",
           (unsigned long)axon_delegate_total_calls);

    /* Test inputs — 8-dimensional vectors */
    float test_inputs[][8] = {
        { 1.0f,  1.0f,  0.5f, -0.5f,  0.0f,  0.2f, -0.3f,  0.1f},  /* class 3 */
        {-1.0f,  1.0f,  0.3f,  0.7f, -0.1f,  0.0f,  0.4f, -0.2f},  /* class 1 */
        { 1.0f, -1.0f, -0.2f,  0.3f,  0.6f, -0.4f,  0.1f,  0.5f},  /* class 2 */
        {-1.0f, -1.0f,  0.0f, -0.8f,  0.2f,  0.5f, -0.6f,  0.0f},  /* class 0 */
    };
    const int num_tests = sizeof(test_inputs) / sizeof(test_inputs[0]);

    timing_init();
    timing_start();

    for (int t = 0; t < num_tests; t++) {
        Tensor::SizesType sizes[] = {1, 8};
        Tensor::DimOrderType dim_order[] = {0, 1};
        TensorImpl input_impl(
            ScalarType::Float, 2, sizes, test_inputs[t], dim_order);
        Tensor input_tensor(&input_impl);

        Error err = method->set_input(input_tensor, 0);
        if (err != Error::Ok) {
            ET_LOG(Error, "set_input failed: 0x%x", static_cast<uint32_t>(err));
            return -5;
        }

        timing_t t_start = timing_counter_get();
        err = method->execute();
        timing_t t_end = timing_counter_get();

        if (err != Error::Ok) {
            ET_LOG(Error, "execute failed: 0x%x", static_cast<uint32_t>(err));
            return -6;
        }

        uint64_t cycles = timing_cycles_get(&t_start, &t_end);
        uint64_t ns = timing_cycles_to_ns(cycles);

        const EValue &output = method->get_output(0);
        if (output.isTensor()) {
            const auto &out = output.toTensor();
            const float *data = out.const_data_ptr<float>();
            int best = 0;
            for (int i = 1; i < out.numel() && i < 4; i++) {
                if (data[i] > data[best]) best = i;
            }
            ET_LOG(Info, "  input[%d]: class=%d (%.3f, %.3f, %.3f, %.3f) %llu us",
                   t, best,
                   (double)data[0], (double)data[1],
                   (double)data[2], (double)data[3],
                   (unsigned long long)(ns / 1000));
        }
    }

    /* Dump AXON delegate profiling */
    axon_delegate_dump_profile();

    ET_LOG(Info, "Done.");
    return 0;
}
