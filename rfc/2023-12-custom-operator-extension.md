# Custom operator extension

**Status:** { **RFC** | ~~Final | POR | Inactive~~ }

**Author:** Denis Vieriu

**Last Update:** 2023-12-05

# Summary
[summary]: #summary

Today, custom kernels have no knowledge of the device they are running, and it makes very hard for ExecuTorch delegates to detect when a custom operator is being used. This document proposes two solutions on how to extend the custom kernels, so that a delegate can intercept when a custom op is running, and if it can improve it.

# Motivation
[motivation]: #motivation

Custom kernels implemented by the users currently have no knowledge about the device they are running on and when they are being dispatch. This makes it very hard to share resources between a delegate and a custom operation. For example, consider 3 lowered modules, running in following order:
- **`lowered_module_1`**: *MPS* delegate
- **`lowered_module_2`**: custom operation implemented on the GPU (using [Metal](https://developer.apple.com/metal/) kernel)
- **`lowered_module_3`**: *CPU* interpreter

Since **`lowered_module_2`** is implemented as a custom [Metal](https://developer.apple.com/metal/) kernel, the exact same set of resources that the *MPS* delegate is using could be shared with the Metal kernel.
Taking it one step further, if the MPS delegate would know that the next module that is going to run is a *Metal* kernel, it could enable additional optimizations, such as adaptive committing.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

Current custom operators are assumed to be always running on the **CPU**. If the delegate invocation would have of next knowledge of the device that is going to run (e.g *MPS delegate*, *CPU interpreter*, *XNNPack*), then based on this flag, it could enable adaptive committing.

What is **adaptive committing**? **Adaptive committing** means that the kernel invocations could share the same set of resources as the delegate itself. For example, consider we have the following list of invocations:
```
1. MPS_DELEGATE  # Start committing
2. METAL_KERNEL  # adaptive commit
3. METAL_KERNEL
4. MPS_DELEGATE  # Next op is a non-Metal kernel and neither a MPS delegate call -> break adaptive committing
4. CPU_KERNEL
5. CPU_KERNEL
6. METAL_KERNEL  # Start commiting, there is no other delegate call / custom operator -> break adaptive committing
```
In the above example, the 2nd dispatch (`METAL_KERNEL 2`) can reuse the same resources as the MPS delegate itself([Command Buffer](https://developer.apple.com/documentation/metal/mtlcommandbuffer?language=objc), [Command Queue](https://developer.apple.com/documentation/metal/mtlcommandqueue?language=objc), [Command Encoder](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder?language=objc) and adaptively commit the work to the GPU based on the workload). Similarly, the `METAL_KERNEL 2`, `METAL_KERNEL 3` and `MPS_DELEGATE 4` would be able to share the resources across them, since they are running on the same device.

Currently, each operation in the above list is executed one by one, and there is a hard wait after each call. After `MPS_DELEGATE 1` call runs, it will wait until it finishes running, then it will run the `METAL_KERNEL 2` for which it waits again, and so on for the remaining operations. The proposed solution would remove any synchronization between the `MPS_DELEGATE` and the `METAL_KERNEL` calls, if the next operation is known to be a Metal kernel or another MPS delegate invocation.

Below is an example of a Metal kernel for [Softshrink](https://pytorch.org/docs/stable/generated/torch.nn.Softshrink.html) operator, implemented as a custom operator.

**Metal** kernel implementation:
```c++
// SoftShrinkage(x) = x - lambda, if x > lambda
//                    x + lambda, if x < -lambda
//                    0,          otherwise
template<typename T>
kernel void softshrink_kernel(constant T*     input  [[buffer(0)]],
                              device   T*     output [[buffer(1)]],
                              constant float& lambda [[buffer(2)]],
                              uint index [[thread_position_in_grid]]) {
    output[index] = input[index] >  lambda ? input[index] - lambda :
                    input[index] < -lambda ? input[index] + lambda : 0;
}
template
[[host_name("softshrink_kernel_half")]]
kernel void softshrink_kernel<half>(constant half*  input  [[buffer(0)]],
                                    device   half*  output [[buffer(1)]],
                                    constant float& lambda [[buffer(2)]],
                                    uint index [[thread_position_in_grid]]);
template
[[host_name("softshrink_kernel_float")]]
kernel void softshrink_kernel<float>(constant float*  input  [[buffer(0)]],
                                     device   float*  output [[buffer(1)]],
                                     constant float& lambda  [[buffer(2)]],
                                     uint index [[thread_position_in_grid]]);
```

Consider the following module using the above custom `Softshrink` custom operator together with `MV2` model:
```python
class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.mv2_lowered_module_mps = lowered_module

  def forward(self, input, lambd=0.5):
    # Custom MV2 model
    out = self.mv2_lowered_module_mps(input) # Lowered to MPS delegate
    for i in range(2):
        out = torch.ops.my_ops.mps_softshrink.default(out, lambd) # Custom Metal kernel
    return out

# Once lowered, this model will have the following structure:
# 1. MPS_DELEGATE
# 2. METAL_KERNEL
# 3. METAL_KERNEL
```

Custom operator implementation using the Softshrink **Metal** kernel:
```obj-c++
Tensor& mps_softshrink_out_impl(RuntimeContext& ctx, const Tensor& input, double lambd, Tensor& output) {
  (void)ctx;

  @autoreleasepool {
    id<MTLDevice> device = mps::MPSDevice::getInstance()->device();
    NSError* error = nil;

    // Set the number of threads equal to the number of elements within the input tensor.
    int numThreads = input.numel();

    // Load the custom softshrink kernel.
    id<MTLLibrary> customKernelLibrary = [device newLibraryWithSource:[NSString stringWithUTF8String:CUSTOM_KERNEL]
                                                              options:nil
                                                                error:&error];
    ET_CHECK_MSG(customKernelLibrary, "Failed to to create custom kernel library, error: %s", error.localizedDescription.UTF8String);

    std::string kernel_name = std::string("softshrink_kernel_") + (input.scalar_type() == ScalarType::Float ? "float" : "half");
    id<MTLFunction> customSoftShrinkFunction = [customKernelLibrary newFunctionWithName:[NSString stringWithUTF8String:kernel_name.c_str()]];
    ET_CHECK_MSG(customSoftShrinkFunction, "Failed to create function state object for %s", kernel_name.c_str());
    auto mpsStream = getDefaultMPSStream();

    // Create a compute pipeline state object for the soft shrink kernel.
    id<MTLComputePipelineState> softShrinkPSO = [device newComputePipelineStateWithFunction:customSoftShrinkFunction error:&error];
    ET_CHECK_MSG(softShrinkPSO != nil, "Failed to create softshrink PSO %s", error.localizedDescription.UTF8String);

    id<MTLComputeCommandEncoder> computeEncoder;
    if (mpsStream->commitAndContinueEnabled()) {
        computeEncoder = mpsStream->commandEncoder();
    } else {
        // Get a reference to the command buffer for the MPSStream.
        id<MTLCommandBuffer> commandBuffer = getDefaultMPSStream()->commandBuffer();
        ET_CHECK_MSG(commandBuffer, "Failed to retrieve command buffer reference");
        computeEncoder = [commandBuffer computeCommandEncoder];
    }

    ET_CHECK_MSG(computeEncoder, "Failed to create compute command encoder");

    float lambda = (float)lambd;
    // Encode the pipeline state object and its parameters.
    [computeEncoder setComputePipelineState:softShrinkPSO];
    [computeEncoder setBuffer:mps::getMTLBufferStorage(input) offset:0 atIndex:0];
    [computeEncoder setBuffer:mps::getMTLBufferStorage(output) offset:0 atIndex:1];
    [computeEncoder setBytes:&lambda length:sizeof(float) atIndex:2];

    MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);

    // Calculate a thread group size.
    NSUInteger threadGroupSize = softShrinkPSO.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > numThreads) {
          threadGroupSize = numThreads;
    }
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

    // Encode the compute command.
    [computeEncoder dispatchThreads:gridSize
            threadsPerThreadgroup:threadgroupSize];

    // If commitAndContinue is enabled, coalesce all metal kernels into a single encoder
    // Otherwise, commit the current work and create a new command buffer and a new command encoder
    if (!mpsStream->commitAndContinueEnabled()) {
        [computeEncoder endEncoding];
        getDefaultMPSStream()->synchronize(SyncType::COMMIT_AND_WAIT);
    }
  }
  return output;
}
```

Executing the above model with adaptive committing enabled between the MPS delegate and the Metal kernels, the performance increases by **2-3 times** on a M2 Max machine (compared to the current approach, where after a delegate/custom operator execution, there is a hard sync). This difference could get even higher when a model uses lots of custom Metal kernels interleaved with MPS delegate calls.

# Proposed APIs (Demo Purpose Only)
[proposed-apis]: #proposed-apis

This documents proposes two solutions:

## 1. Pass metadata regarding next operator directly in the delegate / kernel call
- Backend `init` call receives metadata regarding next operator.
Current signature for `init` is the following:
```c++
  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec> compile_specs);
```

- `BackendInitContext` doesn't currently hold any information regarding next operator. One proposed solution would be that `BackendInitContext` include metadata information about next operator, such as the device it will be running (e.g XNNPack, MPS Delegate, CPU). If it's a custom kernel, in order to know if it's implemented as a Metal kernel, the device could be passed directly from the `*.yaml` file registration, e.g:
```yaml
- func: my_ops::mps_softshrink.out(Tensor input, float lambd, *, Tensor(a!) output) -> Tensor(a!)
  kernels:
    - arg_meta: null
      kernel_name: custom::mps_softshrink_out_impl
      device: mps # new device field indicating the device it will run on
```

The **device** field from the `*.yaml` file registration can be passed into the `BackendInitContext` in order to know that next operator is a Metal kernel.

Similarly, custom operators need to know what next call is (**custom kernel call** / **delegate call**). This is needed in order to know if adaptive committing needs to be kept enabled or it should be disabled and submit all the encoded work to the GPU. This information can be passed through the `RuntimeContext` variable:
```obj-c++
Tensor& mps_softshrink_out_impl(RuntimeContext& ctx, const Tensor& input, double lambd, Tensor& output) {
    // `ctx` holds information regarding what next operator is
```
Similar to `BackendInitContext` for the delegates, the metadata regarding next operator/delegate should be passed through the `RuntimeContext& ctx` variable when the kernel is invoked.

## 2. Record the metadata regarding delegate / operator executing in the delegate itself
Second approach is similar to first one, but consists in creating the metadata regarding delegate / custom kernel invocations and their order directly at AOT time. Considering the previous example:
```python
class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.mv2_lowered_module_mps = lowered_module

  def forward(self, input, lambd=0.5):
    # Custom MV2 model
    out = self.mv2_lowered_module_mps(input) # Lowered to MPS Delegate
    for i in range(2):
        out = torch.ops.my_ops.mps_softshrink.default(out, lambd) # Custom Metal kernel
    out = torch.add(out, 1) # CPU operation
    return out

# Once lowered, this model will have the following structure:
# 1. MPS_DELEGATE
# 2. METAL_KERNEL
# 3. METAL_KERNEL
# 4. CPU_INTERPRETER
```

The list and order of operations is created at AOT time, and the delegate/kernel looks directly in this list in order to know when adaptive committing should be enabled / disabled. This list could be passed similarly through the `BackendInitContext` / `RuntimeContext` variables For example:
- `1. MPS_DELEGATE` executes and based on the list created at AOT it sees that next operation is a Metal kernel, keeps adaptive committing enabled
- `2. METAL_KERNEL` executes, based on the list keeps adaptive committing enabled
- `3. METAL_KERNEL` executed, and based on the list it sees that next operation is a CPU interpreter invocation, it disabled adaptive committing and encodes all the work to the GPU. This is the only place where synchronization is introduced.
