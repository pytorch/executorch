/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "MetalOpRegistry.h"

#include <executorch/runtime/platform/assert.h>

// Op implementations registered at construction time.
#include "../BinaryOps.h"
#include "../UnaryOps.h"
#include "../MatMulOp.h"
#include "../AddMMOp.h"
#include "../BAddBMMOp.h"
#include "../BatchedMatMulOp.h"
#include "../SDPAOp.h"
#include "../AffineQuantizedLinearOp.h"

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
  registerOp(std::make_unique<BAddBMMOp>());

  // Register SDPA op ("aten::scaled_dot_product_attention.default").
  registerOp(std::make_unique<SDPAOp>());

  // Register affine-quantized linear op
  // ("executorch_native::affine_quantized_linear.default"). Custom op;
  // see apple/metal/affine_quantized_linear_op.py for the Python
  // torch.library declaration.
  registerOp(std::make_unique<AffineQuantizedLinearOp>());
}

void MetalOpRegistry::registerOp(std::unique_ptr<MetalOp> op) {
  ET_CHECK_MSG(
      op != nullptr,
      "MetalOpRegistry::registerOp: op is null");
  const char* name = op->name();
  ET_CHECK_MSG(
      ops_.find(name) == ops_.end(),
      "MetalOpRegistry::registerOp: duplicate op name '%s'. "
      "Two ops cannot share the same MetalOp::name() — fix one.",
      name);
  ops_[name] = std::move(op);
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
