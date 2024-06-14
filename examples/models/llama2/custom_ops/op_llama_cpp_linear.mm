//
//  ATenQuantizedLinear.m
//  CustomLinear
//
//  Created by Mengwei Liu on 5/10/24.
//
// #include <iostream>
// #include <fstream>
#include <executorch/backends/apple/mps/runtime/operations/OperationUtils.h>
#include <executorch/backends/apple/mps/runtime/MPSStream.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include "op_llama_cpp_linear.h"
#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>

// #define _CAPTURE_KERNEL 1
namespace torch {
namespace executor {

namespace native {


using namespace mps;
using RuntimeContext = torch::executor::RuntimeContext;
using MPSStream = mps::delegate::MPSStream;

Tensor& _llama_cpp_mm_int8_out(
  RuntimeContext& ctx,
  const Tensor& A, 
  const Tensor& B, 
  const Tensor& scales, 
  Tensor& C) {
    (void)ctx;
    auto M = A.size(0);
    auto N = B.size(0);
    auto K = A.size(1);

    // ET_KERNEL_CHECK(A.dtype() == exec_aten::ScalarType:: || A.dtype() == kHalf || A.dtype() == kFloat,
    //             __func__,
    //             " : expect A to be either 32-bit or 16-bit float tensor.");
    // ET_KERNEL_CHECK(A.dim() == 2, __func__, " : expect A to be 2D tensor.");

    // TORCH_CHECK(B.dtype() == kChar, __func__, " : expect B to be int8 tensor.");
    // TORCH_CHECK(B.is_contiguous(), __func__, " : expect B to be contiguous.");
    // TORCH_CHECK(B.size(1) == K, __func__, " : expect B.size(1) == ", K);

  // auto C = at::empty({M, N}, A.options());
  MPSStream* mpsStream = mps::delegate::getCurrentMPSStream();
  std::array<uint32_t, 3> sizes = {static_cast<uint32_t>(M), static_cast<uint32_t>(K), static_cast<uint32_t>(N)};
  dispatch_sync(mpsStream->queue(), ^() {
    @autoreleasepool {
      // mpsStream->endKernelCoalescing();
      NSError *error = nil;
      id<MTLDevice> device = mpsStream->device();
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      // Load the custom linear shader
      id<MTLLibrary> customKernelLibrary = [device newLibraryWithSource:[NSString stringWithUTF8String:QUANTIZED_KERNEL]
                                                                  options:nil
                                                                    error:&error];
      ET_CHECK_MSG(customKernelLibrary, "Error creating custom kernel library: ", error.localizedDescription.UTF8String);
      std::string kernel;
      if (M < 12) {
        kernel = "int8pack_mm_" + mps::delegate::scalarToMetalTypeString(A.scalar_type());
      } else {
        kernel = "large_m_int8pack_mm_" + mps::delegate::scalarToMetalTypeString(A.scalar_type());
      }
      // Create a function
      id<MTLFunction> customQuantizedLinearFunction = [customKernelLibrary newFunctionWithName:[NSString stringWithUTF8String:kernel.c_str()]];
      ET_CHECK_MSG(customQuantizedLinearFunction, "Error creating custom kernel function: %s", kernel.c_str());

      id<MTLComputePipelineState> quantizedPSO = [device newComputePipelineStateWithFunction:customQuantizedLinearFunction error:&error];
      // ET_CHECK_MSG(quantizedPSO != nil, "%s", error.localizedDescription.UTF8String.c_str());

      [computeEncoder setComputePipelineState:quantizedPSO];
      [computeEncoder setBuffer:mps::delegate::getMTLBufferStorage(A) offset:0 atIndex:0];
      [computeEncoder setBuffer:mps::delegate::getMTLBufferStorage(B) offset:0 atIndex:1];
      [computeEncoder setBuffer:mps::delegate::getMTLBufferStorage(scales) offset:0 atIndex:2];
      [computeEncoder setBuffer:mps::delegate::getMTLBufferStorage(C) offset:0 atIndex:3];
      [computeEncoder setBytes:sizes.data() length:16 atIndex:4];
      if (M < 12) {
        [computeEncoder setThreadgroupMemoryLength:32 atIndex:0];
        [computeEncoder dispatchThreadgroups:MTLSizeMake((N + 7) / 8, M, 1)
                       threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      } else {
        [computeEncoder setThreadgroupMemoryLength:12288 atIndex:0];
        [computeEncoder dispatchThreadgroups:MTLSizeMake((M + 31) / 32, (N + 63) / 64, 1)
                       threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      }
        ET_CHECK(mpsStream->synchronize(mps::delegate::SyncType::COMMIT_AND_WAIT) == Error::Ok);
    }
  });
  // std::ofstream logfile("logfile_no_backend.txt", std::ios_base::app);
  // if (logfile.is_open()) {
  //     logfile << "A address: " << A.const_data_ptr() << ", values [" << A.const_data_ptr<float>()[0] << ", " << A.const_data_ptr<float>()[1] << ", " << A.const_data_ptr<float>()[2] << ", " << A.const_data_ptr<float>()[3] << ", ...]" << std::endl;
  //     logfile << "C address: " << C.const_data_ptr() << ", values [" << C.const_data_ptr<float>()[0] << ", " << C.const_data_ptr<float>()[1] << ", " << C.const_data_ptr<float>()[2] << ", " << C.const_data_ptr<float>()[3] << ", ...]" << std::endl;

  //     logfile.close();
  // } else {
  //     std::cerr << "Unable to open log file" << std::endl;
  // }
  return C;
}

} // namespace native
} // namespace executor
} // namespace torch

EXECUTORCH_LIBRARY(
    llama_cpp,
    "_weight_int8pack_mm.out",
    torch::executor::native::_llama_cpp_mm_int8_out);