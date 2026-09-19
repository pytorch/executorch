/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/apple/metal/runtime/ops/common.h>

#include <cstring>

namespace executorch {
namespace backends {
namespace metal {

extern "C" {

// out = beta * self + alpha * (mat1 @ mat2)
//
// AOTInductor's MPS backend re-fuses ``mm + bias`` into ``aten.addmm`` during
// codegen (see torch/_inductor/fx_passes/post_grad.py), so the Metal backend
// must provide this c-shim in addition to ``aoti_torch_mps_mm_out``. The
// signature mirrors torch's generated ``c_shim_mps.h``:
//   aoti_torch_mps_addmm_out(out, self, mat1, mat2, beta, alpha)
// where ``self`` is the bias/addend, ``mat1`` is [M, K] and ``mat2`` is [K, N].
AOTITorchError aoti_torch_mps_addmm_out(
    AOTITensorHandle out,
    AOTITensorHandle self,
    AOTITensorHandle mat1,
    AOTITensorHandle mat2,
    double beta,
    double alpha) {
  ET_LOG(
      Debug,
      "aoti_torch_mps_addmm_out: out=%p, self=%p, mat1=%p, mat2=%p, beta=%f, alpha=%f",
      out,
      self,
      mat1,
      mat2,
      beta,
      alpha);

  if (!out || !self || !mat1 || !mat2) {
    ET_LOG(Error, "aoti_torch_mps_addmm_out: null tensor handles");
    return Error::InvalidArgument;
  }

  @autoreleasepool {
    try {
      auto out_tensor = reinterpret_cast<Tensor*>(out);
      auto bias_tensor = reinterpret_cast<Tensor*>(self);
      auto mat1_tensor = reinterpret_cast<Tensor*>(mat1);
      auto mat2_tensor = reinterpret_cast<Tensor*>(mat2);

      // Validate matmul operand dimensions.
      if (mat1_tensor->dim() != 2 || mat2_tensor->dim() != 2) {
        std::string error_msg =
            "aoti_torch_mps_addmm_out: mat1/mat2 must be 2-D, got " +
            std::to_string(mat1_tensor->dim()) + " and " +
            std::to_string(mat2_tensor->dim());
        ET_LOG(Error, "%s", error_msg.c_str());
        throw std::runtime_error(error_msg);
      }

      int64_t M = mat1_tensor->sizes()[0]; // rows of mat1
      int64_t K = mat1_tensor->sizes()[1]; // cols of mat1 / rows of mat2
      int64_t N = mat2_tensor->sizes()[1]; // cols of mat2

      if (mat1_tensor->sizes()[1] != mat2_tensor->sizes()[0]) {
        std::string error_msg =
            "aoti_torch_mps_addmm_out: incompatible matrix sizes (" +
            std::to_string(M) + "x" + std::to_string(K) + " and " +
            std::to_string(mat2_tensor->sizes()[0]) + "x" +
            std::to_string(N) + ")";
        ET_LOG(Error, "%s", error_msg.c_str());
        throw std::runtime_error(error_msg);
      }

      // All operands must share mat1's dtype. The AOTI addmm fusion guarantees
      // this, but guard against silently reinterpreting mismatched buffers.
      if (mat2_tensor->scalar_type() != mat1_tensor->scalar_type() ||
          bias_tensor->scalar_type() != mat1_tensor->scalar_type() ||
          out_tensor->scalar_type() != mat1_tensor->scalar_type()) {
        ET_LOG(
            Error,
            "aoti_torch_mps_addmm_out: dtype mismatch across operands "
            "(mat1=%d, mat2=%d, self=%d, out=%d)",
            static_cast<int>(mat1_tensor->scalar_type()),
            static_cast<int>(mat2_tensor->scalar_type()),
            static_cast<int>(bias_tensor->scalar_type()),
            static_cast<int>(out_tensor->scalar_type()));
        return Error::InvalidArgument;
      }

      // Detect transposed mat2 (column-major), same as aoti_torch_mps_mm_out.
      bool mat2_is_transposed = false;
      int64_t mat2_stride_0 = mat2_tensor->strides()[0];
      int64_t mat2_stride_1 = mat2_tensor->strides()[1];
      if (mat2_stride_0 == 1 && mat2_stride_1 != 1) {
        mat2_is_transposed = true;
      }

      ETMetalStream* stream = getCurrentMetalStream();
      if (!stream) {
        ET_LOG(Error, "aoti_torch_mps_addmm_out: no current Metal stream");
        return Error::Internal;
      }

      id<MTLDevice> device = get_metal_device();
      if (!device) {
        throw std::runtime_error("Failed to get Metal device");
      }

      id<MTLBuffer> bias_buffer =
          get_mtl_buffer(bias_tensor, "aoti_torch_mps_addmm_out", "self");
      id<MTLBuffer> mat1_buffer =
          get_mtl_buffer(mat1_tensor, "aoti_torch_mps_addmm_out", "mat1");
      id<MTLBuffer> mat2_buffer =
          get_mtl_buffer(mat2_tensor, "aoti_torch_mps_addmm_out", "mat2");
      id<MTLBuffer> out_buffer =
          get_mtl_buffer(out_tensor, "aoti_torch_mps_addmm_out", "out");

      stream->endKernelCoalescing();

      int32_t dtype = static_cast<int32_t>(mat1_tensor->scalar_type());
      MPSDataType mps_dtype;
      if (dtype == static_cast<int32_t>(SupportedDTypes::FLOAT32)) {
        mps_dtype = MPSDataTypeFloat32;
      } else if (dtype == static_cast<int32_t>(SupportedDTypes::BFLOAT16)) {
        mps_dtype = MPSDataTypeBFloat16;
      } else {
        ET_LOG(Error, "aoti_torch_mps_addmm_out: unsupported dtype %d", dtype);
        throw std::runtime_error("Unsupported data type for addmm");
      }

      NSArray<NSNumber*>* mat1Shape = @[ @(M), @(K) ];
      NSArray<NSNumber*>* mat2PhysicalShape =
          mat2_is_transposed ? @[ @(N), @(K) ] : @[ @(K), @(N) ];

      // Bias may be 1-D [N] or 2-D [M, N]; feed its physical shape and rely on
      // MPSGraph broadcasting in the addition.
      NSMutableArray<NSNumber*>* biasShape = [NSMutableArray array];
      for (size_t i = 0; i < static_cast<size_t>(bias_tensor->dim()); ++i) {
        [biasShape addObject:@(bias_tensor->sizes()[i])];
      }
      if ([biasShape count] == 0) {
        // 0-D scalar bias.
        [biasShape addObject:@(1)];
      }

      // beta/alpha are baked into the cached MPSGraph as constants, so the
      // cache key must capture their exact values (bit-reinterpreted to int64),
      // not just whether they equal 1.
      int64_t beta_bits = 0, alpha_bits = 0;
      std::memcpy(&beta_bits, &beta, sizeof(double));
      std::memcpy(&alpha_bits, &alpha, sizeof(double));

      GraphCacheKey cache_key;
      cache_key.op_name = "addmm";
      cache_key.shape_params = {M, K, N, beta_bits, alpha_bits};
      // Include the full bias shape (rank + each dim), not just the rank, so
      // graphs built for differently-shaped but equal-rank biases (e.g. [N]
      // vs [1], or [M, N] vs [1, N]) don't collide and reuse a biasPlaceholder
      // with the wrong shape.
      cache_key.shape_params.push_back(bias_tensor->dim());
      for (size_t i = 0; i < static_cast<size_t>(bias_tensor->dim()); ++i) {
        cache_key.shape_params.push_back(bias_tensor->sizes()[i]);
      }
      cache_key.dtype = dtype;
      cache_key.transpose_flag = mat2_is_transposed;

      MPSGraph* mpsGraph = nullptr;
      MPSGraphTensor* addmmOutput = nil;
      MPSGraphTensor* biasPlaceholder = nil;
      MPSGraphTensor* mat1Placeholder = nil;
      MPSGraphTensor* mat2Placeholder = nil;

      auto cache_it = graph_cache.find(cache_key);
      if (cache_it != graph_cache.end()) {
        CachedGraph& cached = cache_it->second;
        mpsGraph = cached.graph;
        mat1Placeholder = cached.input1;
        mat2Placeholder = cached.input2;
        biasPlaceholder = cached.input3;
        addmmOutput = cached.output;
        cache_stats.hits++;
      } else {
        mpsGraph = [MPSGraph new];
        cache_stats.misses++;

        mat1Placeholder = [mpsGraph placeholderWithShape:mat1Shape
                                                dataType:mps_dtype
                                                    name:@"mat1"];
        mat2Placeholder = [mpsGraph placeholderWithShape:mat2PhysicalShape
                                                dataType:mps_dtype
                                                    name:@"mat2_physical"];
        biasPlaceholder = [mpsGraph placeholderWithShape:biasShape
                                                dataType:mps_dtype
                                                    name:@"bias"];

        MPSGraphTensor* mat2Logical = mat2Placeholder;
        if (mat2_is_transposed) {
          mat2Logical = [mpsGraph transposeTensor:mat2Placeholder
                                        dimension:-2
                                    withDimension:-1
                                             name:@"mat2_transposed"];
        }

        MPSGraphTensor* mmOutput =
            [mpsGraph matrixMultiplicationWithPrimaryTensor:mat1Placeholder
                                            secondaryTensor:mat2Logical
                                                       name:@"matmul"];

        // alpha * (mat1 @ mat2)
        MPSGraphTensor* scaledMM = mmOutput;
        if (alpha != 1.0) {
          MPSGraphTensor* alphaConst =
              [mpsGraph constantWithScalar:alpha dataType:mps_dtype];
          scaledMM = [mpsGraph multiplicationWithPrimaryTensor:mmOutput
                                               secondaryTensor:alphaConst
                                                          name:@"alpha_scale"];
        }

        // beta * self(bias)
        MPSGraphTensor* scaledBias = biasPlaceholder;
        if (beta != 1.0) {
          MPSGraphTensor* betaConst =
              [mpsGraph constantWithScalar:beta dataType:mps_dtype];
          scaledBias = [mpsGraph multiplicationWithPrimaryTensor:biasPlaceholder
                                                 secondaryTensor:betaConst
                                                            name:@"beta_scale"];
        }

        addmmOutput = [mpsGraph additionWithPrimaryTensor:scaledMM
                                          secondaryTensor:scaledBias
                                                     name:@"addmm"];

        CachedGraph cached_graph;
        cached_graph.graph = mpsGraph;
        cached_graph.input1 = mat1Placeholder;
        cached_graph.input2 = mat2Placeholder;
        cached_graph.input3 = biasPlaceholder;
        cached_graph.output = addmmOutput;
        graph_cache[cache_key] = cached_graph;
      }

      NSArray<NSNumber*>* outShape = @[ @(M), @(N) ];

      MPSGraphTensorData* mat1Data =
          [[MPSGraphTensorData alloc] initWithMTLBuffer:mat1_buffer
                                                  shape:mat1Shape
                                               dataType:mps_dtype];
      MPSGraphTensorData* mat2Data =
          [[MPSGraphTensorData alloc] initWithMTLBuffer:mat2_buffer
                                                  shape:mat2PhysicalShape
                                               dataType:mps_dtype];
      MPSGraphTensorData* biasData =
          [[MPSGraphTensorData alloc] initWithMTLBuffer:bias_buffer
                                                  shape:biasShape
                                               dataType:mps_dtype];

      NSMutableDictionary* feeds = [NSMutableDictionary dictionary];
      feeds[mat1Placeholder] = mat1Data;
      feeds[mat2Placeholder] = mat2Data;
      feeds[biasPlaceholder] = biasData;

      MPSGraphTensorData* outputData =
          [[MPSGraphTensorData alloc] initWithMTLBuffer:out_buffer
                                                  shape:outShape
                                               dataType:mps_dtype];
      NSDictionary* results = @{addmmOutput : outputData};

      @try {
        stream->executeMPSGraph(mpsGraph, feeds, results, SyncType::COMMIT);
      } @catch (NSException* exception) {
        ET_LOG(
            Error,
            "aoti_torch_mps_addmm_out: NSException during executeMPSGraph: %s - %s",
            [[exception name] UTF8String],
            [[exception reason] UTF8String]);
        throw std::runtime_error("MPSGraph execution failed with NSException");
      } @finally {
        // Runs on success and on both ObjC and C++ exception unwind, so the
        // MPSGraphTensorData objects are never leaked on the failure path.
        [mat1Data release];
        [mat2Data release];
        [biasData release];
        [outputData release];
      }

      ET_LOG(Debug, "aoti_torch_mps_addmm_out: executed successfully");
      return Error::Ok;

    } catch (const std::exception& e) {
      ET_LOG(Error, "aoti_torch_mps_addmm_out exception: %s", e.what());
      return Error::Internal;
    } catch (...) {
      ET_LOG(Error, "aoti_torch_mps_addmm_out: unknown exception");
      return Error::Internal;
    }
  }
}

} // extern "C"

} // namespace metal
} // namespace backends
} // namespace executorch
