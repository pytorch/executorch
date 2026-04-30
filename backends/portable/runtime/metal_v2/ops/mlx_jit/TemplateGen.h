/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

//===----------------------------------------------------------------------===//
// TemplateGen — port of MLX's `get_template_definition`.
// Builds a single explicit-instantiation Metal kernel directive of the form:
//   template [[host_name("<lib_name>")]] [[kernel]]
//   decltype(<kernel_func><<args...>>) <kernel_func><<args...>>;
// Used by the per-shape JIT path to JIT-emit one instantiation per shape
// instead of pre-baking every (tile × dtype × layout) combination.
// Mirrors MLX 0.31.2's
//   mlx/backend/metal/kernels.h:376-398  get_template_definition<Args...>
// verbatim apart from the namespace.
//===----------------------------------------------------------------------===//

#include <sstream>
#include <string>
#include <string_view>

namespace executorch {
namespace backends {
namespace metal_v2 {
namespace mlx_jit {
namespace TemplateGen {

namespace detail {
// Stream-insertable wrappers so we can pass `bool` as MSL `true`/`false`
// (MLX's get_template_definition relies on iostream's default bool printing
// which is `1`/`0` — but MSL accepts those equally for `bool` non-type
// template args. Keep the same default-streaming behavior as MLX.)
inline void appendArg(std::ostringstream& os, bool first, const char* a) {
  if (!first) os << ", ";
  os << a;
}
inline void appendArg(std::ostringstream& os, bool first, const std::string& a) {
  if (!first) os << ", ";
  os << a;
}
inline void appendArg(std::ostringstream& os, bool first, std::string_view a) {
  if (!first) os << ", ";
  os << a;
}
template <typename T>
inline void appendArg(std::ostringstream& os, bool first, const T& a) {
  if (!first) os << ", ";
  os << a;
}
}  // namespace detail

// Returns the explicit-instantiation directive as a string.
//   makeInstantiation("steel_gemm_fused_nax_nn_bf16_..._bm64_bn128_...",
//                     "gemm",
//                     "bfloat16_t", 64, 128, 256, 2, 4, false, false);
// →   "\ntemplate [[host_name(\"steel_gemm_fused_nax_...\")]] [[kernel]] "
//     "decltype(gemm<bfloat16_t, 64, 128, 256, 2, 4, 0, 0>) "
//     "gemm<bfloat16_t, 64, 128, 256, 2, 4, 0, 0>;\n"
template <typename... Args>
std::string makeInstantiation(
    std::string_view name,
    std::string_view kernel_func,
    Args&&... template_args) {
  std::ostringstream args;
  args << kernel_func << "<";
  bool first = true;
  // Fold expression: emit each arg with a leading ", " from the second on.
  ((detail::appendArg(args, first, template_args), first = false), ...);
  args << ">";
  std::ostringstream out;
  out << "\ntemplate [[host_name(\"" << name << "\")]] [[kernel]] decltype("
      << args.str() << ") " << args.str() << ";\n";
  return out.str();
}

}  // namespace TemplateGen
}  // namespace mlx_jit
}  // namespace metal_v2
}  // namespace backends
}  // namespace executorch
