# buck2 shims

The `shim/` subtree helps the Meta-internal buck2 build system also work in the
open-source repo.

Shims are how open-source buck2 supports a [line
like](https://github.com/pytorch/executorch/blob/50aa517549d10324147534d91d04a923b76421d6/kernels/optimized/targets.bzl#L1):

```
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
```

In the open-source repo, `fbsource//xplat` (a Meta-internal root) doesn't exist.
The `shim = shim` line in `../.buckconfig` tells buck2 to look in
[`shim/xplat/executorch/build/runtime_wrapper.bzl`](https://github.com/pytorch/executorch/blob/main/shim/xplat/executorch/build/runtime_wrapper.bzl)
instead.
