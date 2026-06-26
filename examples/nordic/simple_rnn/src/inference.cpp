/*
 * Copyright (c) 2026 iote.ai
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Simple RNN inference — demonstrates multi-subgraph delegation.
 *
 * The model has FC layers (AXON) separated by tanh (CPU), producing
 * multiple delegate handles. Each AXON subgraph has its own compiled
 * command buffer.
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
        if (!name_result.ok()) return -2;
        method_name = *name_result;
    }
    ET_LOG(Info, "Method: %s", method_name);

    MemoryAllocator method_allocator(sizeof(method_allocator_pool), method_allocator_pool);
    MemoryAllocator temp_allocator(sizeof(temp_allocator_pool), temp_allocator_pool);
    auto method_meta = program->method_meta(method_name);
    if (!method_meta.ok()) return -3;

    Span<uint8_t> planned_span(planned_memory_pool, sizeof(planned_memory_pool));
    HierarchicalAllocator planned_allocator({&planned_span, 1});
    MemoryManager memory_manager(&method_allocator, &planned_allocator, &temp_allocator);

    Result<Method> method = program->load_method(method_name, &memory_manager);
    if (!method.ok()) {
        ET_LOG(Error, "load_method failed: 0x%x", static_cast<uint32_t>(method.error()));
        return -4;
    }
    ET_LOG(Info, "Method loaded");

    /* The RNN step model takes two inputs: x (1,4) and h (1,8) */
    float input_data[4] = {0.5f, -0.3f, 0.8f, -0.1f};
    float hidden_data[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    Tensor::SizesType input_sizes[] = {1, 4};
    Tensor::SizesType hidden_sizes[] = {1, 8};
    Tensor::DimOrderType dim2[] = {0, 1};

    TensorImpl input_impl(ScalarType::Float, 2, input_sizes, input_data, dim2);
    TensorImpl hidden_impl(ScalarType::Float, 2, hidden_sizes, hidden_data, dim2);
    Tensor input_tensor(&input_impl);
    Tensor hidden_tensor(&hidden_impl);

    timing_init();
    timing_start();

    /* Run 4 RNN steps, feeding hidden state back */
    for (int step = 0; step < 4; step++) {
        Error err = method->set_input(input_tensor, 0);
        if (err != Error::Ok) return -5;
        err = method->set_input(hidden_tensor, 1);
        if (err != Error::Ok) return -5;

        timing_t t_start = timing_counter_get();
        err = method->execute();
        timing_t t_end = timing_counter_get();
        if (err != Error::Ok) {
            ET_LOG(Error, "step %d execute failed: 0x%x", step, (uint32_t)err);
            return -6;
        }

        uint64_t cycles = timing_cycles_get(&t_start, &t_end);
        uint64_t ns = timing_cycles_to_ns(cycles);

        /* Output 0 = out (1,2), Output 1 = h_new (1,8) */
        const EValue &out_val = method->get_output(0);
        if (out_val.isTensor()) {
            const auto &out = out_val.toTensor();
            const float *d = out.const_data_ptr<float>();
            ET_LOG(Info, "  step %d: out=(%.3f, %.3f) %llu us",
                   step, (double)d[0], (double)d[1],
                   (unsigned long long)(ns / 1000));
        }

        /* Feed h_new back as hidden state for next step */
        if (method->outputs_size() > 1) {
            const EValue &h_val = method->get_output(1);
            if (h_val.isTensor()) {
                const auto &h_out = h_val.toTensor();
                const float *h_data = h_out.const_data_ptr<float>();
                for (int i = 0; i < 8 && i < h_out.numel(); i++) {
                    hidden_data[i] = h_data[i];
                }
            }
        }
    }

    axon_delegate_dump_profile();
    ET_LOG(Info, "Done.");
    return 0;
}
