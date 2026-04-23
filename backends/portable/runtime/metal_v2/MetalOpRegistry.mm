/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "MetalOpRegistry.h"

// Op implementations registered at construction time.
#include "ops/BinaryOps.h"
#include "ops/UnaryOps.h"
#include "ops/MatMulOp.h"

namespace executorch {
namespace backends {
namespace metal_v2 {

MetalOpRegistry& MetalOpRegistry::shared() {
  static MetalOpRegistry instance;
  return instance;
}

MetalOpRegistry::MetalOpRegistry() {
  // Register binary ops
  registerOp(std::make_unique<AddOp>());
  registerOp(std::make_unique<MulOp>());
  registerOp(std::make_unique<SubOp>());

  // Register unary ops
  registerOp(std::make_unique<ReluOp>());

  // Register matmul ops
  registerOp(std::make_unique<MatMulOp>());
  registerOp(std::make_unique<BatchedMatMulOp>());
  registerOp(std::make_unique<AddMMOp>());
}

void MetalOpRegistry::registerOp(std::unique_ptr<MetalOp> op) {
  ops_[op->name()] = std::move(op);
}

MetalOp* MetalOpRegistry::get(const char* name) const {
  auto it = ops_.find(name);
  return it != ops_.end() ? it->second.get() : nullptr;
}

bool MetalOpRegistry::hasOp(const char* name) const {
  return ops_.find(name) != ops_.end();
}

} // namespace metal_v2
} // namespace backends
} // namespace executorch
