/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Arduino's custom <new> header omits <exception>, which breaks
// std::bad_variant_access in <variant>. Include it first.
#include <exception>

#ifndef C10_USING_CUSTOM_GENERATED_MACROS
#define C10_USING_CUSTOM_GENERATED_MACROS
#endif
#ifndef ET_ENABLE_DEPRECATED_CONSTANT_BUFFER
#define ET_ENABLE_DEPRECATED_CONSTANT_BUFFER 0
#endif
#ifndef FLATBUFFERS_MAX_ALIGNMENT
#define FLATBUFFERS_MAX_ALIGNMENT 1024
#endif

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/method_meta.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>
