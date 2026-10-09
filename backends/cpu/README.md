# CPU Backend

> [!NOTE]
> Do not use.

The initial workload is pretrained ConvNeXt-Tiny in FP32 on a Linux server.
The registered `CPURecipeType.FP32` (`cpu_fp32`) Recipe retains a semantic Native
graph inside one `CpuBackend` delegate in a regular ExecuTorch `.pte`.
This first delegate version supports pure operations on statically shaped,
contiguous FP32 tensors.

The `cpu_delegate_version` compile spec stores `CPU_DELEGATE_VERSION` as a
four-byte little-endian unsigned integer. The loader accepts explicitly
supported versions. Bump the version when changing serialized semantics or
storage guarantees incompatibly; provider additions, kernel optimizations, and
runtime planning changes do not require a bump. Native graph schema versioning
is separate.

AOT export preserves semantic operations and aligned constants without encoding
provider coverage or implementation choices. Runtime provider selection happens
when the delegate is loaded.
