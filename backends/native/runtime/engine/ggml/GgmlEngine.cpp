// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/engine/ggml/GgmlEngine.h>

#include <algorithm>
#include <cctype>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <executorch/backends/native/runtime/Method.h>
#include <executorch/backends/native/runtime/Program.h>
#include <executorch/backends/native/runtime/deserialize/Package.h>
#include <executorch/backends/native/runtime/engine/ggml/GgmlCache.h>
#include <executorch/backends/native/runtime/engine/ggml/GgmlExecutor.h>
#include <executorch/extension/llm/cache/cache_registry.h>

namespace ptn {

struct GgmlEngineHost::Impl {
  explicit Impl(GgmlEngineOptions value) : options(std::move(value)) {
    if (options.device_index < 0) {
      throw std::invalid_argument("ggml: device index must be non-negative");
    }
    if (options.cpu_threads < 0) {
      throw std::invalid_argument(
          "ggml: CPU thread count must be non-negative");
    }
    if (options.graph_size == 0) {
      throw std::invalid_argument("ggml: graph size must be positive");
    }
    if (options.max_cached_graphs > 16) {
      throw std::invalid_argument(
          "ggml: at most 16 cached graphs per method are supported");
    }
    if (options.device == GgmlDevice::Gpu) {
      ggml_backend_dev_t device = select_ggml_gpu_device(options);
      name = std::string("ggml-") +
          ggml_backend_reg_name(ggml_backend_dev_backend_reg(device));
      std::transform(name.begin(), name.end(), name.begin(), [](char c) {
        return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
      });
      device_name = std::string(ggml_backend_dev_description(device)) + " (" +
          ggml_backend_dev_name(device) + ")";
    } else {
      name = "ggml-cpu";
      device_name = "CPU";
    }
  }

  GgmlEngineOptions options;
  std::string name;
  std::string device_name;
};

GgmlEngineHost::GgmlEngineHost(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

GgmlEngineHost::~GgmlEngineHost() = default;

std::unique_ptr<GgmlEngineHost> GgmlEngineHost::create(
    GgmlEngineOptions options) {
  return std::unique_ptr<GgmlEngineHost>(
      new GgmlEngineHost(std::make_unique<Impl>(std::move(options))));
}

const std::string& GgmlEngineHost::name() const {
  return impl_->name;
}

const std::string& GgmlEngineHost::device_name() const {
  return impl_->device_name;
}

std::unique_ptr<EngineContext> GgmlEngineHost::create_context(
    std::shared_ptr<const Program> program,
    std::shared_ptr<const Package> package) {
  return create_ggml_context(std::move(program), std::move(package));
}

std::unique_ptr<EngineContext> GgmlEngineHost::create_context_with_options(
    std::shared_ptr<const Program> program,
    std::shared_ptr<const Package> package,
    const EngineOptions& options) {
  return create_ggml_context(std::move(program), std::move(package), options);
}

std::unique_ptr<GgmlEngineContext> GgmlEngineHost::create_ggml_context(
    std::shared_ptr<const Program> program,
    std::shared_ptr<const Package> package,
    const EngineOptions& options) {
  namespace cache = executorch::extension::llm::cache;
  std::shared_ptr<GgmlSequenceCache> kv_cache;
  for (const auto& [name, value] : options) {
    if (name != cache::kCacheKeyOption) {
      throw std::invalid_argument(
          impl_->name + ": unsupported context option '" + name + "'");
    }
    std::shared_ptr<cache::Cache> installed =
        cache::CacheRegistry::global().get(value);
    if (!installed) {
      throw std::invalid_argument(
          impl_->name + ": no KV cache is installed under key '" + value + "'");
    }
    GgmlSequenceCache* ggml_cache = installed->as<GgmlSequenceCache>();
    if (ggml_cache == nullptr) {
      throw std::invalid_argument(
          impl_->name + ": KV cache '" + value +
          "' was not built for this engine (CacheFactory backend '" +
          kGgmlCacheBackendId + "', kind 'single')");
    }
    kv_cache = std::shared_ptr<GgmlSequenceCache>(installed, ggml_cache);
  }
  return std::unique_ptr<GgmlEngineContext>(new GgmlEngineContext(
      std::move(program),
      std::move(package),
      impl_->options,
      std::move(kv_cache)));
}

struct GgmlEngineContext::Impl {
  Impl(GgmlEngineOptions value, std::shared_ptr<GgmlSequenceCache> cache)
      : options(std::move(value)), kv_cache(std::move(cache)) {}

  GgmlEngineOptions options;
  std::shared_ptr<GgmlSequenceCache> kv_cache;
  size_t materialized_constant_bytes = 0;
  size_t heap_copied_constant_bytes = 0;
  GgmlMutableKeyOwners mutable_key_owners;
};

GgmlEngineContext::GgmlEngineContext(
    std::shared_ptr<const Program> program,
    std::shared_ptr<const Package> package,
    GgmlEngineOptions options,
    std::shared_ptr<GgmlSequenceCache> cache)
    : EngineContext(std::move(program), std::move(package)),
      impl_(std::make_unique<Impl>(std::move(options), std::move(cache))) {}

GgmlEngineContext::~GgmlEngineContext() = default;

std::unique_ptr<EngineExecutable> GgmlEngineContext::compile_method(
    const Method& method) {
  return make_ggml_executable(
      method,
      package(),
      impl_->options,
      &impl_->materialized_constant_bytes,
      &impl_->heap_copied_constant_bytes,
      impl_->mutable_key_owners,
      impl_->kv_cache.get());
}

size_t GgmlEngineContext::materialized_constant_bytes() const {
  return impl_->materialized_constant_bytes;
}

size_t GgmlEngineContext::heap_copied_constant_bytes() const {
  return impl_->heap_copied_constant_bytes;
}

} // namespace ptn
