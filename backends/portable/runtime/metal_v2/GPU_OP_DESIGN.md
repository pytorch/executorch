# GpuOp Design Proposal for metal_v2

## Current State Analysis

### v1 (metal/) Strengths
1. **Rich dtype dispatch**: `kernelNameForDtype()` generates f32/f16/bf16 variants
2. **Op variants**: BinaryOp has ss/sv/vs/vv/g for different input shapes
3. **TensorMeta**: Full tensor info (sizes, strides, dtype, numel)
4. **Shape inference**: `inferOutputShapes()` for output allocation
5. **MatMul optimization**: 
   - Device-specific config (`MatMulConfig::forDevice`)
   - Multiple kernels: naive/tiled/simd/batched/gemv/transposed
   - Automatic kernel selection based on M,N,K
   - Simdgroup MMA with double buffering

### v2 (metal_v2/) Current State
1. **Simple interface**: `dispatch(stream, inputs, outputs)`
2. **EValue-based**: Works directly with ExecuTorch tensors
3. **Single kernel**: One kernel per op (no variants)
4. **f32 only**: No dtype dispatch
5. **No shape inference**: Expects pre-allocated outputs

### Key Design Differences

| Aspect | v1 | v2 |
|--------|----|----|
| Buffer model | `mem_obj_id → MTLBuffer` | `cpu_ptr → MTLBuffer` |
| Encoding | `encode(encoder, device, ...)` | `dispatch(stream, ...)` |
| Kernel cache | Per-op cache | Stream's compiler cache |
| ICB support | No | Yes |
| Tensor info | TensorMeta struct | EValue access |

---

## Proposed Long-Term Design

### 1. TensorMeta for v2

Add lightweight tensor metadata to avoid EValue coupling:

```cpp
struct TensorInfo {
  void* data_ptr;
  size_t nbytes;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  DType dtype;
  size_t numel;
  
  // Helpers
  bool is_contiguous() const;
  bool is_scalar() const { return numel == 1; }
  
  // Create from EValue
  static TensorInfo from(const runtime::EValue& ev);
};
```

### 2. GpuOp Base Class

```cpp
class GpuOp {
public:
  virtual ~GpuOp() = default;
  
  /// Op name for registry lookup
  virtual const char* name() const = 0;
  
  /// Supported dtypes
  virtual bool supports(DType dtype) const { 
    return dtype == DType::Float32; 
  }
  
  /// Infer output shapes from inputs (for allocation)
  virtual std::vector<std::vector<int64_t>> inferOutputShapes(
      Span<const TensorInfo> inputs,
      Span<const ScalarParam> params) const = 0;
  
  /// Main dispatch method
  virtual void dispatch(
      GpuStream* stream,
      Span<const TensorInfo> inputs,
      Span<TensorInfo> outputs,
      Span<const ScalarParam> params) = 0;

protected:
  /// Get kernel for dtype (compiles on first use)
  GpuKernel* getKernel(GpuStream* stream, const char* kernelName);
  
  /// Kernel source (subclass provides)
  virtual const char* kernelSource() const = 0;
};
```

### 3. BinaryOp Base for Elementwise

```cpp
class BinaryOp : public GpuOp {
public:
  /// Short name for kernel (e.g., "add", "mul")
  virtual const char* opName() const = 0;
  
  /// Does this op have alpha parameter?
  virtual bool hasAlpha() const { return false; }
  
  // Detect input pattern
  enum class Variant { SS, SV, VS, VV, General };
  Variant detectVariant(const TensorInfo& a, const TensorInfo& b) const;
  
  // Generate kernel name: {variant}_{op}_{dtype}
  std::string kernelName(Variant v, DType dtype) const;
  
  // Shape inference: broadcast rules
  std::vector<std::vector<int64_t>> inferOutputShapes(
      Span<const TensorInfo> inputs,
      Span<const ScalarParam> params) const override;
  
  // Dispatch with automatic variant selection
  void dispatch(
      GpuStream* stream,
      Span<const TensorInfo> inputs,
      Span<TensorInfo> outputs,
      Span<const ScalarParam> params) override;
  
protected:
  // Compute broadcast strides for General kernel
  static std::vector<int64_t> broadcastStrides(
      const std::vector<int64_t>& shape,
      const std::vector<int64_t>& outShape);
};

// Concrete ops just define opName and hasAlpha
class AddOp : public BinaryOp {
public:
  const char* name() const override { return "aten::add"; }
  const char* opName() const override { return "add"; }
  bool hasAlpha() const override { return true; }
  const char* kernelSource() const override;
};
```

### 4. MatMulOp with Kernel Selection

```cpp
class MatMulOp : public GpuOp {
public:
  const char* name() const override { return "aten::mm"; }
  
  // Shape inference: [M,K] x [K,N] → [M,N]
  std::vector<std::vector<int64_t>> inferOutputShapes(
      Span<const TensorInfo> inputs,
      Span<const ScalarParam> params) const override;
  
  void dispatch(
      GpuStream* stream,
      Span<const TensorInfo> inputs,
      Span<TensorInfo> outputs,
      Span<const ScalarParam> params) override;

protected:
  // Select kernel based on M,N,K and device
  enum class KernelType { Naive, Tiled, Simd, Batched, GEMV };
  KernelType selectKernel(int M, int N, int K, int batch) const;
  
  // Device-specific thresholds
  struct Config {
    int simdThreshold;
    int tiledThreshold;
    int gemvThreshold;
  };
  static Config configForDevice(id<MTLDevice> device);
  
  const char* kernelSource() const override;
};
```

### 5. Stream Integration

The GpuStream provides:
- Kernel compilation (`compiler()->compile()`)
- Buffer registration (`registerExternalBuffer()`)
- Dispatch encoding (`dispatch(kernel, args, grid, block)`)
- ICB support (automatic with same API)

```cpp
// Op doesn't know about ICB - stream handles it
void AddOp::dispatch(
    GpuStream* stream,
    Span<const TensorInfo> inputs,
    Span<TensorInfo> outputs,
    Span<const ScalarParam> params) {
  
  auto variant = detectVariant(inputs[0], inputs[1]);
  auto dtype = outputs[0].dtype;
  auto name = kernelName(variant, dtype);
  
  auto* kernel = getKernel(stream, name.c_str());
  
  // Stream handles ICB vs direct encoding transparently
  stream->dispatch(kernel, {
    {inputs[0].data_ptr, inputs[0].nbytes},
    {inputs[1].data_ptr, inputs[1].nbytes},
    {outputs[0].data_ptr, outputs[0].nbytes},
    static_cast<int64_t>(outputs[0].numel)
  }, computeGrid(outputs[0]), computeBlock(variant));
}
```

### 6. Kernel Source Organization

MLX-style kernels with template instantiation:

```metal
// binary_kernels.metal
template<typename T, typename Op>
kernel void binary_vv(
    device const T* a [[buffer(0)]],
    device const T* b [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant int64_t& numel [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  uint idx = i * 4;
  if (idx + 4 <= numel) {
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      out[idx+j] = Op()(a[idx+j], b[idx+j]);
    }
  } else {
    for (uint j = idx; j < numel; j++) {
      out[j] = Op()(a[j], b[j]);
    }
  }
}

// Instantiate all variants
#define INSTANTIATE_BINARY(name, op) \
  template [[host_name("vv_" name "_f32")]] kernel void binary_vv<float, op>(...); \
  template [[host_name("vv_" name "_f16")]] kernel void binary_vv<half, op>(...);

INSTANTIATE_BINARY("add", AddOp)
INSTANTIATE_BINARY("mul", MulOp)
```

---

## Migration Path

### Phase 1: Add TensorInfo
- Create `TensorInfo` struct
- Add `from(EValue&)` factory
- Update dispatch signature

### Phase 2: BinaryOp Base
- Port v1's variant detection logic
- Add dtype dispatch
- Keep simple VV kernel for now

### Phase 3: Multi-dtype Support
- Add f16/bf16 kernel instantiations
- Update compiler to handle dtype suffix
- Test with half precision models

### Phase 4: Broadcast Support
- Add General kernel with strides
- Port `computeBroadcastStrides`
- Test with broadcasting models

### Phase 5: MatMul Optimization
- Port v1's kernel selection
- Add simd/tiled/gemv kernels
- Device-specific tuning

---

## Summary

The key insight is that v2's **stream-centric design** (ICB, batching) is good, but the **op abstraction** should be richer like v1:

| Keep from v2 | Add from v1 |
|--------------|-------------|
| Stream-based dispatch | TensorMeta/TensorInfo |
| ICB transparency | Dtype dispatch |
| CPU ptr → buffer mapping | Shape inference |
| Compiler caching | Op variants (ss/sv/vs/vv/g) |
| | Kernel selection (matmul) |
| | Device-specific tuning |

The result is a design that:
1. Works with ICB replay (v2)
2. Has full dtype support (v1)
3. Handles broadcasts efficiently (v1)
4. Selects optimal kernels (v1)
5. Maintains simple op implementation
