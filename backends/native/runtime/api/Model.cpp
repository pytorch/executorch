// cppcheck-suppress-file useStlAlgorithm

// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/api/Model.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <executorch/backends/native/runtime/MethodMeta.h>
#include <executorch/backends/native/runtime/Program.h>
#include <executorch/backends/native/runtime/Validation.h>
#include <executorch/backends/native/runtime/api/Session.h>
#include <executorch/backends/native/runtime/deserialize/CheckedMath.h>
#include <executorch/backends/native/runtime/deserialize/Limits.h>
#include <executorch/backends/native/runtime/deserialize/OwnedBytes.h>
#include <executorch/backends/native/runtime/deserialize/Package.h>
#include <executorch/backends/native/runtime/engine/Engine.h>

namespace ptn {
namespace detail {

class ModelImpl final {
 public:
  std::shared_ptr<const Package> package;
  std::shared_ptr<const Program> program;
  std::vector<std::string> method_names;
  std::unordered_map<std::string, MethodInfo> method_info;
};

// Defined here to keep the private Session implementation out of its API.
// @lint-ignore CLANGTIDY
class SessionImpl final {
 public:
  SessionImpl(
      std::shared_ptr<const ModelImpl> model,
      std::unique_ptr<EngineContext> context)
      : model(std::move(model)), context(std::move(context)) {}

  std::shared_ptr<const ModelImpl> model;
  std::unique_ptr<EngineContext> context;
  std::unordered_map<std::string, std::unique_ptr<EngineExecutable>>
      executables;
};

} // namespace detail
namespace {

std::shared_ptr<const detail::ModelImpl> load_model(OwnedBytes bytes) {
  auto package = std::make_shared<Package>(Package::load(std::move(bytes)));
  const ByteSpan program_bytes = package->program_bytes();
  auto program = std::make_shared<Program>(
      Program::load(program_bytes.data(), program_bytes.size()));
  validate_program_state(*program);

  auto impl = std::make_shared<detail::ModelImpl>();
  impl->package = std::move(package);
  impl->program = std::move(program);
  impl->method_names = impl->program->method_names();
  for (const std::string& name : impl->method_names) {
    const Method& method = impl->program->get_method(name);
    impl->method_info.emplace(name, MethodMeta::from_method(method));
  }
  return impl;
}

bool same_extents(
    std::span<const int64_t> expected,
    std::span<const int64_t> actual) {
  return expected.size() == actual.size() &&
      std::equal(expected.begin(), expected.end(), actual.begin());
}

template <typename View>
void validate_view(const TensorInfo& spec, const View& view) {
  if (spec.dtype() != view.dtype() ||
      !same_extents(spec.sizes(), view.sizes()) ||
      !same_extents(spec.strides(), view.strides())) {
    throw std::invalid_argument("tensor dtype, shape, or strides do not match");
  }
  size_t numel = 1;
  for (const int64_t size : view.sizes()) {
    if (size < 0 ||
        !detail::checked_mul(numel, static_cast<size_t>(size), numel)) {
      throw std::invalid_argument("tensor shape is not representable");
    }
  }
  size_t required = 0;
  if (!detail::checked_mul(numel, element_size(view.dtype()), required)) {
    throw std::invalid_argument("tensor byte size overflows");
  }
  if (view.nbytes() < required || (required != 0 && view.data() == nullptr)) {
    throw std::invalid_argument("tensor storage is too small or null");
  }
}

void validate_executable(
    const MethodInfo& info,
    const EngineExecutable& executable) {
  if (executable.num_inputs() != info.inputs().size() ||
      executable.num_outputs() != info.outputs().size()) {
    throw std::runtime_error(
        "engine executable signature count does not match the method");
  }
  for (size_t i = 0; i < info.inputs().size(); ++i) {
    const std::vector<int64_t> sizes = executable.input_sizes(i);
    if (executable.input_dtype(i) != info.inputs()[i].dtype() ||
        !same_extents(info.inputs()[i].sizes(), sizes)) {
      throw std::runtime_error(
          "engine executable input does not match the method");
    }
  }
  for (size_t i = 0; i < info.outputs().size(); ++i) {
    const std::vector<int64_t> sizes = executable.output_sizes(i);
    if (executable.output_dtype(i) != info.outputs()[i].dtype() ||
        !same_extents(info.outputs()[i].sizes(), sizes)) {
      throw std::runtime_error(
          "engine executable output does not match the method");
    }
  }
}

} // namespace

Model::Model(std::shared_ptr<const detail::ModelImpl> impl)
    : impl_(std::move(impl)) {}

Model::Model(const Model&) = default;
Model& Model::operator=(const Model&) = default;
Model::Model(Model&&) noexcept = default;
Model& Model::operator=(Model&&) noexcept = default;
Model::~Model() = default;

Model Model::load_file(std::string_view path, FileMode mode) {
  if (path.find('\0') != std::string_view::npos) {
    throw std::invalid_argument("model path contains a null byte");
  }
  return Model(load_model(OwnedBytes::from_file(
      std::string(path), mode == FileMode::Mmap, detail::kMaxPackageBytes)));
}

Model Model::load_bytes(std::vector<uint8_t> bytes) {
  return Model(load_model(OwnedBytes::from_vector(std::move(bytes))));
}

std::span<const std::string> Model::method_names() const {
  return impl_ == nullptr ? std::span<const std::string>{}
                          : std::span<const std::string>(impl_->method_names);
}

MethodInfo Model::method_info(std::string_view name) const {
  if (impl_ == nullptr) {
    throw std::logic_error("model has been moved from");
  }
  const auto it = impl_->method_info.find(std::string(name));
  if (it == impl_->method_info.end()) {
    throw std::invalid_argument("unknown method name");
  }
  return it->second;
}

Session Model::create_session(EngineHost& host) const {
  if (impl_ == nullptr) {
    throw std::logic_error("model has been moved from");
  }
  std::unique_ptr<EngineContext> context =
      host.create_context(impl_->program, impl_->package);
  if (context == nullptr) {
    throw std::runtime_error("engine host returned a null context");
  }
  return Session(
      std::make_unique<detail::SessionImpl>(impl_, std::move(context)));
}

Session::Session(std::unique_ptr<detail::SessionImpl> impl)
    : impl_(std::move(impl)) {}

Session::Session(Session&&) noexcept = default;
Session& Session::operator=(Session&&) noexcept = default;
Session::~Session() = default;

void Session::prepare(std::string_view method_name) {
  if (impl_ == nullptr) {
    throw std::logic_error("session has been moved from");
  }
  const std::string name(method_name);
  if (impl_->executables.contains(name)) {
    return;
  }
  const auto metadata = impl_->model->method_info.find(name);
  if (metadata == impl_->model->method_info.end()) {
    throw std::invalid_argument("unknown method name");
  }
  std::unique_ptr<EngineExecutable> executable = impl_->context->compile(name);
  if (executable == nullptr) {
    throw std::runtime_error("engine context returned a null executable");
  }
  validate_executable(metadata->second, *executable);
  impl_->executables.emplace(name, std::move(executable));
}

void Session::release(std::string_view method_name) {
  if (impl_ == nullptr) {
    throw std::logic_error("session has been moved from");
  }
  const std::string name(method_name);
  if (!impl_->model->method_info.contains(name)) {
    throw std::invalid_argument("unknown method name");
  }
  impl_->executables.erase(name);
}

bool Session::is_prepared(std::string_view method_name) const {
  return impl_ != nullptr &&
      impl_->executables.contains(std::string(method_name));
}

void Session::run(
    std::string_view method_name,
    std::span<const ConstTensorView> inputs,
    std::span<MutableTensorView> outputs) {
  if (impl_ == nullptr) {
    throw std::logic_error("session has been moved from");
  }
  const std::string name(method_name);
  const auto metadata = impl_->model->method_info.find(name);
  if (metadata == impl_->model->method_info.end()) {
    throw std::invalid_argument("unknown method name");
  }
  const MethodInfo& info = metadata->second;
  if (inputs.size() != info.inputs().size() ||
      outputs.size() != info.outputs().size()) {
    throw std::invalid_argument("tensor argument count does not match");
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    validate_view(info.inputs()[i], inputs[i]);
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    validate_view(info.outputs()[i], outputs[i]);
  }

  prepare(name);
  EngineExecutable& executable = *impl_->executables.at(name);
  for (size_t i = 0; i < inputs.size(); ++i) {
    const size_t numel =
        info.inputs()[i].nbytes() / element_size(info.inputs()[i].dtype());
    executable.set_input(i, inputs[i].data(), numel, info.inputs()[i].dtype());
  }
  executable.execute();

  std::vector<std::vector<uint8_t>> pending;
  pending.reserve(outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    const size_t numel =
        info.outputs()[i].nbytes() / element_size(info.outputs()[i].dtype());
    pending.emplace_back(info.outputs()[i].nbytes());
    executable.get_output(
        i, pending.back().data(), numel, info.outputs()[i].dtype());
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (!pending.at(i).empty()) {
      std::memcpy(
          outputs[i].data(), pending.at(i).data(), pending.at(i).size());
    }
  }
}

} // namespace ptn
