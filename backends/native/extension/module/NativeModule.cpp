// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/extension/module/NativeModule.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <executorch/backends/native/extension/module/MethodMetaBridge.h>
#include <executorch/backends/native/runtime/MethodMeta.h>
#include <executorch/backends/native/runtime/Program.h>
#include <executorch/backends/native/runtime/Validation.h>
#include <executorch/backends/native/runtime/deserialize/DeserializeError.h>
#include <executorch/backends/native/runtime/deserialize/Limits.h>
#include <executorch/backends/native/runtime/deserialize/OwnedBytes.h>
#include <executorch/backends/native/runtime/deserialize/Package.h>
#include <executorch/extension/module/ptn_module.h>
#include <executorch/runtime/platform/log.h>

namespace executorch::extension::native_module {
namespace {

using internal::PtnModule;
using runtime::Error;

struct MethodState {
  std::unique_ptr<MethodMetaBridge> metadata;
};

class NativePtnModule final : public PtnModule {
 public:
  NativePtnModule(
      ptn::Package package,
      ptn::Program program,
      std::unordered_map<std::string, MethodState> methods)
      : package_(std::move(package)),
        program_(std::move(program)),
        methods_(std::move(methods)) {}

  runtime::Result<size_t> num_methods() const override {
    return methods_.size();
  }

  runtime::Result<std::unordered_set<std::string>> method_names()
      const override {
    std::unordered_set<std::string> names;
    names.reserve(methods_.size());
    for (const auto& entry : methods_) {
      names.insert(entry.first);
    }
    return names;
  }

  runtime::Result<ET_RUNTIME_NAMESPACE::MethodMeta> method_meta(
      const std::string& method_name) override {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = methods_.find(method_name);
    if (it == methods_.end()) {
      return Error::InvalidArgument;
    }
    try {
      if (it->second.metadata == nullptr) {
        const ptn::Method& method = program_.get_method(method_name);
        it->second.metadata =
            MethodMetaBridge::create(ptn::MethodMeta::from_method(method));
      }
      return it->second.metadata->view();
    } catch (const ptn::ResourceLimitError& error) {
      ET_LOG(Error, "PTN metadata limit: %s", error.what());
      return Error::OutOfResources;
    } catch (const std::bad_alloc&) {
      return Error::MemoryAllocationFailed;
    } catch (const std::exception& error) {
      ET_LOG(Error, "Invalid PTN metadata: %s", error.what());
      return Error::InvalidProgram;
    } catch (...) {
      ET_LOG(Error, "Unknown exception while reading PTN metadata");
      return Error::Internal;
    }
  }

  runtime::Error load_method(const std::string&) override {
    return Error::NotSupported;
  }

  bool unload_method(const std::string&) override {
    return false;
  }

  bool is_method_loaded(const std::string&) const override {
    return false;
  }

  runtime::Result<std::vector<runtime::EValue>> execute(
      const std::string&,
      const std::vector<runtime::EValue>&) override {
    return Error::NotSupported;
  }

  runtime::Error set_input(const std::string&, const runtime::EValue&, size_t)
      override {
    return Error::NotSupported;
  }

  runtime::Error set_inputs(
      const std::string&,
      const std::vector<runtime::EValue>&) override {
    return Error::NotSupported;
  }

  runtime::Error set_output(const std::string&, runtime::EValue, size_t)
      override {
    return Error::NotSupported;
  }

  runtime::Error set_outputs(
      const std::string&,
      const std::vector<runtime::EValue>&) override {
    return Error::NotSupported;
  }

  runtime::Result<std::vector<runtime::EValue>> get_outputs(
      const std::string&) override {
    return Error::NotSupported;
  }

  runtime::Result<runtime::EValue> get_output(const std::string&, size_t)
      override {
    return Error::NotSupported;
  }

 private:
  ptn::Package package_;
  ptn::Program program_;
  std::unordered_map<std::string, MethodState> methods_;
  std::mutex mutex_;
};

runtime::Result<std::unique_ptr<PtnModule>> load_ptn(
    runtime::DataLoader& loader,
    ET_RUNTIME_NAMESPACE::Program::Verification verification) {
  try {
    auto size = loader.size();
    if (!size.ok()) {
      return size.error();
    }
    if (*size > ptn::detail::kMaxPackageBytes) {
      return Error::OutOfResources;
    }

    std::vector<uint8_t> bytes(*size);
    if (*size != 0) {
      auto loaded = loader.load(
          0,
          *size,
          runtime::DataLoader::SegmentInfo(
              runtime::DataLoader::SegmentInfo::Type::Program));
      if (!loaded.ok()) {
        return loaded.error();
      }
      if (loaded->size() != *size) {
        return Error::InvalidProgram;
      }
      auto data = loaded->data_safe();
      if (!data.ok()) {
        return data.error();
      }
      if (*data == nullptr) {
        return Error::InvalidProgram;
      }
      std::memcpy(bytes.data(), *data, bytes.size());
    }

    ptn::Package package =
        ptn::Package::load(ptn::OwnedBytes::from_vector(std::move(bytes)));
    if (verification ==
        ET_RUNTIME_NAMESPACE::Program::Verification::InternalConsistency) {
      package.verify();
    }
    const ptn::ByteSpan program_bytes = package.program_bytes();
    ptn::Program program =
        ptn::Program::load(program_bytes.data(), program_bytes.size());
    ptn::validate_program_state(program);

    std::unordered_map<std::string, MethodState> methods;
    const std::vector<std::string> method_names = program.method_names();
    methods.reserve(method_names.size());
    for (const std::string& method_name : method_names) {
      methods.try_emplace(method_name);
    }
    std::unique_ptr<PtnModule> module = std::make_unique<NativePtnModule>(
        std::move(package), std::move(program), std::move(methods));
    return module;
  } catch (const ptn::ResourceLimitError& error) {
    ET_LOG(Error, "PTN package limit: %s", error.what());
    return Error::OutOfResources;
  } catch (const ptn::UnsupportedVersionError& error) {
    ET_LOG(Error, "Unsupported PTN version: %s", error.what());
    return Error::NotSupported;
  } catch (const std::bad_alloc&) {
    return Error::MemoryAllocationFailed;
  } catch (const std::exception& error) {
    ET_LOG(Error, "Invalid PTN package: %s", error.what());
    return Error::InvalidProgram;
  } catch (...) {
    ET_LOG(Error, "Unknown exception while loading PTN package");
    return Error::Internal;
  }
}

const internal::PtnHooks kHooks{load_ptn};

struct RegisterHooks {
  RegisterHooks() {
    const Error error = internal::register_ptn_hooks(kHooks);
    if (error != Error::Ok) {
      ET_LOG(Error, "PTN Module hook registration failed: 0x%x", error);
    }
  }
};

const RegisterHooks kRegisterHooks;

} // namespace
} // namespace executorch::extension::native_module

extern "C" void executorch_native_module_ptn_link_anchor() {}
