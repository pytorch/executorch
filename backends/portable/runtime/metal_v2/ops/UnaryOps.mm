/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "UnaryOps.h"
#include <executorch/backends/portable/runtime/metal_v2/MetalStream.h>
#include <executorch/runtime/platform/log.h>

namespace executorch {
namespace backends {
namespace metal_v2 {

using runtime::Error;

//===----------------------------------------------------------------------===//
// ReluOp
//===----------------------------------------------------------------------===//

void ReluOp::dispatch(
    MetalStream* stream,
    EValuePtrSpan inputs,
    EValuePtrSpan outputs) {
  
  auto& input = inputs[0]->toTensor();
  auto& output = outputs[0]->toTensor();
  
  auto err = resizeOutput(inputs, outputs[0]);
  if (err != Error::Ok) {
    ET_LOG(Error, "ReluOp: failed to resize output");
    return;
  }
  
  ScalarType dtype = output.scalar_type();
  std::string kname = std::string("relu_") + dtypeSuffix(dtype);
  
  auto* kernel = getKernel(stream, kname.c_str());
  uint32_t numel = static_cast<uint32_t>(input.numel());
  
  stream->dispatch(kernel, {
    {input.mutable_data_ptr(), input.nbytes()},
    {output.mutable_data_ptr(), output.nbytes()},
    numel
  }, computeGrid(output), uvec3(256, 1, 1));
}

const char* ReluOp::kernelSource() const {
  return R"(
#include <metal_stdlib>
using namespace metal;

template<typename T>
kernel void relu_kernel(
    device const T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant uint& numel [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  if (i < numel) {
    output[i] = max(input[i], T(0));
  }
}

template [[host_name("relu_f32")]] kernel void relu_kernel<float>(device const float*, device float*, constant uint&, uint);
template [[host_name("relu_f16")]] kernel void relu_kernel<half>(device const half*, device half*, constant uint&, uint);
)";
}

} // namespace metal_v2
} // namespace backends
} // namespace executorch
