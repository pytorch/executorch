/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/runtime/backend/backend_options.h>
#include <executorch/runtime/backend/backend_update_context.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/result.h>
#include <cstring>

using executorch::aten::ArrayRef;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::BackendInitContext;
using executorch::runtime::BackendInterface;
using executorch::runtime::BackendOption;
using executorch::runtime::BackendUpdateContext;
using executorch::runtime::CompileSpec;
using executorch::runtime::DelegateHandle;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::Result;

/**
 * A backend class whose methods can be overridden individually.
 */
 class StubBackend final : public BackendInterface {
  public:
 
   // Default name that this backend is registered as.
   static constexpr char kName[] = "StubBackend";
 
   bool is_available() const override {
     return true;
   }
 
   Result<DelegateHandle*> init(
       BackendInitContext& context,
       FreeableBuffer* processed,
       ArrayRef<CompileSpec> compile_specs) const override {
     return nullptr;
   }
 
   Error execute(
       BackendExecutionContext& context,
       DelegateHandle* handle,
       EValue** args) const override {
     return Error::Ok;
    }

    int num_threads() const {
    return num_threads_;
   }

    Error update(
        BackendUpdateContext& context,
        const executorch::runtime::ArrayRef<BackendOption>& backend_options) const override {
        int success_update = 0;
        for (const auto& backend_option : backend_options) {
          if (strcmp(backend_option.key, "NumberOfThreads") == 0) {
              if (std::holds_alternative<int>(backend_option.value)) {
                num_threads_ = std::get<int>(backend_option.value);
                success_update++;
              }
          }
        }
        if (success_update == backend_options.size()) {
          return Error::Ok;
        }
        return Error::InvalidArgument;
    }

   /**
    * Registers the singleton instance if not already registered.
    *
    * Note that this can be used to install the stub as the implementation for
    * any export-time backend by passing in the right name, as long as no other
    * backend with that name has been registered yet.
    */
   static Error register_singleton(const char* name = kName) {
     if (!registered_) {
       registered_ = true;
       return executorch::runtime::register_backend({name, &singleton_});
     }
     return Error::Ok;
   }
 
   /**
    * Returns the instance that was added to the backend registry.
    */
   static StubBackend& singleton() {
     return singleton_;
   }
 
  private:
   static bool registered_;
   static StubBackend singleton_;
   mutable int num_threads_ = 1;
  };

// Static member definitions
bool StubBackend::registered_ = false;
StubBackend StubBackend::singleton_;
