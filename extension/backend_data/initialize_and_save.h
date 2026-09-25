/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <memory>

#include <executorch/extension/backend_data/data_writer.h>
#include <executorch/runtime/core/data_loader.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/event_tracer.h>
#include <executorch/runtime/core/memory_allocator.h>

namespace executorch::extension {

/**
 * Initializes every participating backend once and persists its replacement
 * named data in a replacement PTE.
 *
 * This is a one-shot destructive operation. It consumes every loader and
 * writer, requires exclusive access to the physical source, and destroys the
 * loader before publishing the output.
 *
 * Until segment alignment is serialized, the operation conservatively infers
 * each original alignment as the largest power of two dividing that segment's
 * nonzero absolute file offset. This may add padding, but it cannot weaken a
 * power-of-two alignment honored by the original artifact. A backend may
 * request a stronger alignment for replacement data.
 *
 * @param[in] pte_loader Loader for the original PTE. Must be non-null.
 * @param[in] pte_writer Writer that publishes the replacement PTE. Must be
 *     non-null.
 * @param[in] delegate_temp_allocator Resettable ET-owned allocator. Every
 *     backend-submitted array, key string, and byte span must come from it.
 * @param[in] event_tracer Optional event tracer exposed to each backend.
 * @retval Error::Ok All participating backends were prepared and every
 *     affected source was published, or no backend emitted replacement data.
 * @returns Another error without publishing unfinished output.
 */
runtime::Error initialize_and_save_backend_data(
    std::unique_ptr<runtime::DataLoader> pte_loader,
    std::unique_ptr<DataWriter> pte_writer,
    runtime::MemoryAllocator* delegate_temp_allocator,
    runtime::EventTracer* event_tracer = nullptr);

} // namespace executorch::extension
