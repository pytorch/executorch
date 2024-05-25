//
//  ATenQuantizedLinear.m
//  CustomLinear
//
//  Created by Mengwei Liu on 5/10/24.
//

#include <executorch/backends/apple/mps/runtime/operations/OperationUtils.h>
#include <op_llama_cpp_linear.h>

// #define _CAPTURE_KERNEL 1
namespace torch {
namespace executor {

namespace native {


using namespace mps;

Tensor _llama_cpp_mm_int8(const Tensor& A, const Tensor& B, const Tensor& scales) {
    auto M = A.size(0);
    auto N = B.size(0);
    auto K = A.size(1);

    TORCH_CHECK(A.dtype() == kBFloat16 || A.dtype() == kHalf || A.dtype() == kFloat,
                __func__,
                " : expect A to be either 32-bit or 16-bit float tensor.");
    TORCH_CHECK(A.is_contiguous(), __func__, " : expect A to be contiguous.");
    TORCH_CHECK(A.dim() == 2, __func__, " : expect A to be 2D tensor.");

    TORCH_CHECK(B.dtype() == kChar, __func__, " : expect B to be int8 tensor.");
    TORCH_CHECK(B.is_contiguous(), __func__, " : expect B to be contiguous.");
    TORCH_CHECK(B.size(1) == K, __func__, " : expect B.size(1) == ", K);

  auto C = at::empty({M, N}, A.options());
  MPSStream* mpsStream = getCurrentMPSStream();
  std::array<uint32_t, 3> sizes = {static_cast<uint32_t>(M), static_cast<uint32_t>(K), static_cast<uint32_t>(N)};
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      NSError *error = nil;
#if _CAPTURE_KERNEL
      if (getMPSProfiler().isCaptureEnabled()) {
        getMPSProfiler().startCapture(__func__, mpsStream);
      }
#endif
      id<MTLDevice> device = mpsStream->device();
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      // Load the custom linear shader
      id<MTLLibrary> customKernelLibrary = [device newLibraryWithSource:[NSString stringWithUTF8String:QUANTIZED_KERNEL]
                                                                  options:nil
                                                                    error:&error];
      TORCH_CHECK(customKernelLibrary, "Error creating custom kernel library: ", error.localizedDescription.UTF8String);
      const std::string kernel = "kernel_mul_mm_" + scalarToMetalTypeString(A.scalar_type()) + "_char";

      // Create a function
      id<MTLFunction> customQuantizedLinearFunction = [customKernelLibrary newFunctionWithName:[NSString stringWithUTF8String:kernel.c_str()]];
      TORCH_CHECK(customQuantizedLinearFunction, "Error creating custom kernel function: ", kernel);

      id<MTLComputePipelineState> quantizedPSO = [device newComputePipelineStateWithFunction:customQuantizedLinearFunction error:&error];
      TORCH_CHECK(quantizedPSO, error.localizedDescription.UTF8String);

      [computeEncoder setComputePipelineState:quantizedPSO];
      mtl_setBuffer(computeEncoder, A, 0);
      mtl_setBuffer(computeEncoder, B, 1);
      mtl_setBuffer(computeEncoder, scales, 2);
      mtl_setBuffer(computeEncoder, C, 3);
      [computeEncoder setBytes:sizes.data() length:16 atIndex:4];
        [computeEncoder setThreadgroupMemoryLength:12288 atIndex:0];
        [computeEncoder dispatchThreadgroups:MTLSizeMake( (M + 31)/32, (N + 63)/64, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        mpsStream->synchronize(SyncType::COMMIT_AND_WAIT);
        
#if _CAPTURE_KERNEL
      if (getMPSProfiler().isCapturing()) {
        getMPSProfiler().stopCapture(mpsStream);
      }
#endif
    }
  });

  return C;
}

} // namespace native
} // namespace executor
} // namespace torch