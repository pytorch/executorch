// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/cpu/CPUEngine.h>

#include <stdexcept>
#include <utility>
#include <vector>

#include <executorch/backends/cpu/runtime/providers/reference/ReferenceProvider.h>
#include <executorch/backends/cpu/runtime/providers/xnnpack/XNNPACKProvider.h>
#include <executorch/backends/native/runtime/cpu/CPUEngineExecutable.h>
#include <executorch/backends/native/runtime/deserialize/Package.h>
#include <executorch/extension/threadpool/threadpool.h>

namespace ptn {
namespace cpu = executorch::backends::cpu;

struct CPUEngineHost::Impl {
  std::string name = "cpu";
  std::string device_name = "host CPU";
  std::vector<cpu::ProviderFactory> providers;
  cpu::RuntimeConfiguration configuration;
  cpu::ExecutionContext execution;
};

struct CPUEngineContext::Impl {
  cpu::RuntimeConfiguration configuration;
  cpu::ExecutionContext execution;
  CPUConstantStore constants;
};

CPUEngineHost::CPUEngineHost(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

CPUEngineHost::~CPUEngineHost() = default;

std::unique_ptr<CPUEngineHost> CPUEngineHost::create(
    cpu::RuntimeConfiguration configuration) {
  auto impl = std::make_unique<Impl>();
  if (configuration.providers.empty()) {
    impl->providers = {
        cpu::create_xnnpack_provider, cpu::create_reference_provider};
  } else {
    impl->providers.assign(
        configuration.providers.begin(), configuration.providers.end());
  }
  configuration.providers = {impl->providers.data(), impl->providers.size()};
  impl->configuration = std::move(configuration);
  impl->execution.threadpool =
      executorch::extension::threadpool::get_pthreadpool();
  impl->execution.threads = impl->execution.threadpool
      ? pthreadpool_get_threads_count(impl->execution.threadpool)
      : 1;
#if defined(__x86_64__) || defined(__i386__)
  impl->execution.avx2_fma =
      __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#endif
  return std::unique_ptr<CPUEngineHost>(new CPUEngineHost(std::move(impl)));
}

const std::string& CPUEngineHost::name() const {
  return impl_->name;
}
const std::string& CPUEngineHost::device_name() const {
  return impl_->device_name;
}

std::unique_ptr<EngineContext> CPUEngineHost::create_context(
    std::shared_ptr<const Program> program,
    std::shared_ptr<const Package> package) {
  return create_cpu_context(std::move(program), std::move(package));
}

std::unique_ptr<CPUEngineContext> CPUEngineHost::create_cpu_context(
    std::shared_ptr<const Program> program,
    std::shared_ptr<const Package> package) {
  auto impl = std::make_unique<CPUEngineContext::Impl>();
  impl->configuration = impl_->configuration;
  impl->execution = impl_->execution;
  return std::unique_ptr<CPUEngineContext>(new CPUEngineContext(
      std::move(program), std::move(package), std::move(impl)));
}

CPUEngineContext::CPUEngineContext(
    std::shared_ptr<const Program> program,
    std::shared_ptr<const Package> package,
    std::unique_ptr<Impl> impl)
    : EngineContext(std::move(program), std::move(package)),
      impl_(std::move(impl)) {}

CPUEngineContext::~CPUEngineContext() = default;

size_t CPUEngineContext::unique_constant_bytes() const {
  return impl_->constants.unique_bytes();
}

size_t CPUEngineContext::materialized_constant_bytes() const {
  return impl_->constants.materialized_bytes();
}

std::unique_ptr<EngineExecutable> CPUEngineContext::compile_method(
    const Method& method) {
  return create_cpu_engine_executable(
      method,
      package(),
      impl_->constants,
      impl_->configuration,
      impl_->execution);
}

cpu::Buffer CPUConstantStore::acquire(
    const Package& package,
    const std::string& key,
    const cpu::BufferRequirements& requirements) {
  const auto info = package.constant_info(key);
  if (!info) {
    throw std::runtime_error("cpu: missing constant " + key);
  }
  if (requirements.writable_bytes != 0) {
    throw std::runtime_error("cpu: provider requests writable constant " + key);
  }
  auto& versions = constants_[info->owner];
  for (const auto& constant : versions) {
    if (constant->buffer.accepts(requirements)) {
      return constant->buffer;
    }
  }
  auto constant = std::make_unique<Constant>();
  auto buffer = cpu::allocate_buffer(constant->allocator, requirements);
  if (!buffer.ok() || buffer->readable_bytes < info->nbytes) {
    throw std::runtime_error("cpu: cannot allocate constant " + key);
  }
  if (!package.load_constant_into(
          key,
          MutableByteSpan(static_cast<uint8_t*>(buffer->data), info->nbytes))) {
    throw std::runtime_error("cpu: cannot load constant " + key);
  }
  constant->buffer = buffer.get();
  constant->buffer.owner = cpu::StorageOwner::ConstantCopy;
  versions.push_back(std::move(constant));
  if (versions.size() == 1) {
    unique_bytes_ += info->nbytes;
  }
  materialized_bytes_ += info->nbytes;
  return versions.back()->buffer;
}

} // namespace ptn
