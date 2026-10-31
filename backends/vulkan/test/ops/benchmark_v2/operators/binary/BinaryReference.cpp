// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ReferenceRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/binary/BinaryReference.h>

#include <algorithm>

namespace vkcompute::shader_profiling::reference {

Tensor binary_reference(
    const Json& spec,
    const TensorMap& tensors,
    const std::string& function,
    size_t threads) {
  std::vector<std::string> bindings;
  for (const Json& binding : spec.at("bindings")) {
    if (binding.contains("tensor")) {
      bindings.push_back(binding.at("tensor"));
    }
  }
  require(
      bindings.size() == (function == "copy" ? 2u : 3u),
      "Copy/add reference requires output followed by tensor input bindings");
  const Tensor& a = tensors.at(bindings[1]);
  const auto sizes = spec.at("tensors")
                         .at(bindings[0])
                         .at("sizes")
                         .get<std::vector<int64_t>>();
  if (function == "copy") {
    require(a.sizes == sizes, "Copy input/output shapes must match");
    return a;
  }
  const Tensor& b = tensors.at(bindings[2]);
  const size_t rank = std::max(a.sizes.size(), b.sizes.size());
  std::vector<int64_t> output_sizes(rank, 1);
  std::vector<size_t> a_strides(rank), b_strides(rank);
  size_t a_stride = 1;
  size_t b_stride = 1;
  for (size_t dim = rank; dim > 0; --dim) {
    const size_t i = dim - 1;
    const int64_t a_size = dim > rank - a.sizes.size()
        ? a.sizes[dim - (rank - a.sizes.size()) - 1]
        : 1;
    const int64_t b_size = dim > rank - b.sizes.size()
        ? b.sizes[dim - (rank - b.sizes.size()) - 1]
        : 1;
    require(
        a_size == b_size || a_size == 1 || b_size == 1,
        "Addition input shapes do not broadcast");
    output_sizes[i] = std::max(a_size, b_size);
    a_strides[i] = a_size == 1 ? 0 : a_stride;
    b_strides[i] = b_size == 1 ? 0 : b_stride;
    a_stride *= a_size;
    b_stride *= b_size;
  }
  require(
      output_sizes == sizes,
      "Addition output shape does not match broadcast shape");
  double alpha = 1;
  if (spec.contains("push_constants") && !spec.at("push_constants").empty()) {
    alpha = spec.at("push_constants").at(0).at("data").at(0);
  }
  Tensor output{sizes, std::vector<double>(tensor_numel(sizes))};
  constexpr size_t kChunk = 4096;
  parallel_for(
      (output.values.size() + kChunk - 1) / kChunk, threads, [&](size_t chunk) {
        const size_t end = std::min(output.values.size(), (chunk + 1) * kChunk);
        for (size_t index = chunk * kChunk; index < end; ++index) {
          size_t remainder = index;
          size_t ai = 0;
          size_t bi = 0;
          for (size_t dim = rank; dim > 0; --dim) {
            const size_t coordinate = remainder % output_sizes[dim - 1];
            remainder /= output_sizes[dim - 1];
            ai += coordinate * a_strides[dim - 1];
            bi += coordinate * b_strides[dim - 1];
          }
          output.values[index] = a.values[ai] + alpha * b.values[bi];
        }
      });
  return output;
}

namespace {

Tensor
copy_reference(const Json& spec, const TensorMap& tensors, size_t threads) {
  return binary_reference(spec, tensors, "copy", threads);
}

Tensor
add_reference(const Json& spec, const TensorMap& tensors, size_t threads) {
  return binary_reference(spec, tensors, "add", threads);
}

} // namespace

REGISTER_REFERENCE_FUNCTIONS {
  REGISTER_REFERENCE_FN(copy, copy_reference);
  REGISTER_REFERENCE_FN(add, add_reference);
  REGISTER_SHADER_REFERENCE(buffer_to_buffer_float, copy);
  REGISTER_SHADER_REFERENCE(buffer_to_buffer_half, copy);
  REGISTER_SHADER_REFERENCE(binary_add_texture3d_float, add);
  REGISTER_SHADER_REFERENCE(binary_add_texture3d_half, add);
}

} // namespace vkcompute::shader_profiling::reference
