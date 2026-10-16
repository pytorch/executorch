<!--
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-->

# ExecuTorch for ESP-IDF (`executorch` component)

**This tree is generated. Do not edit it by hand.** It is produced by
`examples/espressif/build_idf_component.sh` in the
[ExecuTorch](https://github.com/pytorch/executorch) repository:

```sh
executorch/examples/espressif/build_idf_component.sh \
  --profiles @PROFILES@ --out <this directory>
```

`VERSION` in this directory records the ExecuTorch commit, the op
profiles baked in, and each profile's op-set hash. A `.pte` is only
valid against the profile whose hash its export was checked against;
mismatches surface as `OperatorMissing` at load time.

## Using it

Add the component to your project's component requirements (`REQUIRES
executorch`), pick an op profile in `menuconfig` (ExecuTorch -> Op
profile, or `CONFIG_ET_OP_PROFILE_<NAME>=y` in your sdkconfig), then:

- `#include <et_model.h>` for the small C inference API (flash XIP
  model bytes, heap_caps method, planned and optional scratch arenas
  with internal/PSRAM placement). This is what most applications should
  use. The `.pte` bytes must be 16-byte aligned; `et_model_create`
  rejects misaligned data. Application pipelines built on it (audio
  front ends, detectors, wake word logic) live with the application,
  not in this runtime component.
- `#include <executorch/...>` for the full C++ runtime, including
  `extension/module` (`Module`). Note two ESP-IDF differences from
  upstream: `MmapDataLoader` exists but always fails with
  `NotSupported` (ESP-IDF has no mmap; use `BufferDataLoader` or
  `FileDataLoader`), and runtime logging goes through ESP_LOG under
  the `executorch` tag.

`CONFIG_ET_ENABLE` (default `y`) gates the whole component. A project
that vendors it before using it can set `CONFIG_ET_ENABLE=n` to keep
its firmware unchanged; the component then registers empty and
`et_model.h` is not on the include path.

## Op profiles

@PROFILE_TABLE@

Only the selected profile's kernels are compiled in. Switching
profiles is a Kconfig change and a rebuild; no regeneration.

## Logging

`CONFIG_ET_ENABLE_LOGGING=n` compiles ExecuTorch logging out entirely.
Otherwise runtime messages go to ESP_LOG (tag `executorch`), and
`CONFIG_ET_LOG_LEVEL_*` is a compile-time floor (`ET_MIN_LOG_LEVEL`)
on top of the usual IDF runtime levels. `et_model` logs its own
errors under the `et_model` tag.

## Updating

Re-run the generator against a newer ExecuTorch checkout and replace
this whole directory. Review the `VERSION` diff (commit, op-set
hashes, file counts, size); do not line-review vendored sources.
