// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/engine/Engine.h>

#include <stdexcept>
#include <utility>

#include <executorch/backends/native/runtime/Program.h>
#include <executorch/backends/native/runtime/deserialize/Package.h>

namespace ptn {

// Out of line so each vtable is emitted here rather than in every translation
// unit that includes the header.
EngineExecutable::~EngineExecutable() = default;

EngineContext::EngineContext(
    std::shared_ptr<const Program> program,
    std::shared_ptr<const Package> package)
    : program_(std::move(program)), package_(std::move(package)) {
  if (program_ == nullptr || package_ == nullptr) {
    throw std::invalid_argument("engine context requires program and package");
  }
}

EngineContext::~EngineContext() = default;

const Program& EngineContext::program() const {
  return *program_;
}

const Package& EngineContext::package() const {
  return *package_;
}

std::unique_ptr<EngineExecutable> EngineContext::compile(
    const std::string& method_name) {
  std::unique_ptr<EngineExecutable> executable =
      compile_method(program_->get_method(method_name));
  if (executable == nullptr) {
    throw std::runtime_error("engine returned a null executable");
  }
  return executable;
}

EngineHost::~EngineHost() = default;

} // namespace ptn
