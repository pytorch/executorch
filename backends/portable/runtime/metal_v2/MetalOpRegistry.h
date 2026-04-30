/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/portable/runtime/metal_v2/MetalOp.h>

#include <memory>
#include <string>
#include <unordered_map>

namespace executorch {
namespace backends {
namespace metal_v2 {

//===----------------------------------------------------------------------===//
// MetalOpRegistry — global singleton mapping op-name → MetalOp instance
// Populated at process start (constructor registers built-in ops). External
// callers look up ops via MetalOpRegistry::shared().get("aten::add").
//===----------------------------------------------------------------------===//

class MetalOpRegistry {
 public:
  static MetalOpRegistry& shared();

  void registerOp(std::unique_ptr<MetalOp> op);
  MetalOp* get(const char* name) const;
  MetalOp* get(const std::string& name) const {
    return get(name.c_str());
  }
  bool hasOp(const char* name) const;
  bool hasOp(const std::string& name) const {
    return hasOp(name.c_str());
  }

 private:
  MetalOpRegistry();
  std::unordered_map<std::string, std::unique_ptr<MetalOp>> ops_;
};

} // namespace metal_v2
} // namespace backends
} // namespace executorch
