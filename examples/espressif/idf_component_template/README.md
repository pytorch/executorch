<!--
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-->

# ExecuTorch for ESP-IDF (`executorch` component)

**This tree is generated. Do not edit it by hand.** It is produced by
`examples/espressif/build_idf_component.sh` in the
[ExecuTorch](https://github.com/pytorch/executorch) repository.

`VERSION` in this directory records the ExecuTorch commit, the generator's
options, and the kernel libraries the component ships.
`examples/hello_executorch` is a complete project that runs a small model
with its kernel bundle; its README says how they were made.

## Using it

Depend on one exact version and commit `dependencies.lock`:

```yaml
dependencies:
  pytorch/executorch: "==@ET_COMPONENT_VERSION@"
```

A development copy, versioned `X.Y.Z-dev.YYYYMMDD`, is not on the
registry; use it from your project's `components/` or through
`override_path`.

A kernel bundle is made for one component version, a `~N` respin
included, and the build refuses a bundle made for another version, or
for the same version from another commit. With a range, the build
breaks once the lock is re-solved to a newer version (for example by
`idf.py update-dependencies`), until the bundle is regenerated.

The component carries the runtime, every kernel of its kernel
libraries (portable, quantized, the Cadence backend's generic kernels
and the `esp_nn::` op library) and the ESP-NN delegate
(`EspNNBackend`), but registers only the kernels your models use and
links the delegate only when one of them uses it. It depends on
`espressif/esp-nn`, which the component manager fetches; a project
that builds without the component manager provides it as
`components/esp-nn`. The kernels come
from a kernel bundle that `tools/gen_kernels.py` makes from your
`.pte` files. Run it with the Python environment you exported the
models with, after every export, ExecuTorch upgrade or update of this
component, and keep the bundle in your project:

```sh
python <this directory>/tools/gen_kernels.py --pte main/model.pte --out main/et_kernels
```

Then declare the models and the bundle, from a component's
`CMakeLists.txt` after its `idf_component_register()` or from the
project's after `project()`, and add `executorch` to the requirements
of the components that use it:

```cmake
idf_component_register(SRCS "main.c" PRIV_REQUIRES executorch)
executorch_add_model(PTE "${CMAKE_CURRENT_SOURCE_DIR}/model.pte")
executorch_add_kernels(DIR "${CMAKE_CURRENT_SOURCE_DIR}/et_kernels")
```

If the Python ESP-IDF builds with can import `executorch` (or you
point `EXECUTORCH_PYTHON` at one that can, with `-D` or in the
environment, for example
`idf.py -DEXECUTORCH_PYTHON=$HOME/venvs/et/bin/python build`), you can
leave out `executorch_add_kernels`: the build then makes the bundle in
its build directory from the declared models, with the same result as
running `gen_kernels.py` yourself.

The build fails at configure time if the bundle doesn't match the
declared models (a model was re-exported, added or removed) or was
made for another version of this component, and prints the
`gen_kernels.py` command that fixes it. For models that are not
available at build time (for example, delivered over the air), pass
`--ops <namespace::op.overload>` or `--oplist <selected_operators.yaml>`
to `gen_kernels.py` instead of, or as well as, `--pte`, and call
`executorch_add_backend(EspNNBackend)` if those models use the ESP-NN
delegate. A project that declares only `executorch_add_backend` links
just that delegate and no CPU kernels or prim ops, so its models must run
entirely in the delegate; a model that also needs a prim op such as a
view needs a bundle made with `--ops`. A project that declares nothing
links nothing from this component, unless it calls `et_model`: that
fails to link with an undefined `et_requires_executorch_add_kernels`.

With a bundle that registers kernels, the build compiles every vendored
kernel source, and the link keeps only the ones the bundle registers.
The extra kernels cost clean-build time, not flash.

- `#include <et_model.h>` for the small C inference API (flash XIP
  model bytes; method, planned and optional scratch arenas either
  passed in or allocated with heap_caps internal/PSRAM placement). This
  is what most applications should use. The `.pte` bytes must be
  16-byte aligned; `et_model_create` rejects misaligned data. Application pipelines built on it (audio
  front ends, detectors, wake word logic) live with the application,
  not in this runtime component.
- `#include <executorch/...>` for the core C++ runtime
  (`Program`/`Method`) and `BufferDataLoader`. `extension::Module`, the
  file and mmap data loaders, and external `.ptd` data maps are not
  included. Runtime logging goes through ESP_LOG under the
  `executorch` tag.

## Size and debug options

`CONFIG_ET_OPTIMIZE_SIZE` (default `y`) builds the runtime and kernels
with `-Os` regardless of the project's optimization level.
`CONFIG_ET_PROGRAM_VERIFICATION` compiles in full `.pte` verification,
which `et_model_config_t.verify` requires. `CONFIG_ET_DEBUG_ASSERTS`
keeps ExecuTorch's debug assertions (otherwise the runtime builds with
`NDEBUG`). `CONFIG_ET_MAX_KERNEL_NUM` sizes the kernel registry for an
application that registers kernels of its own; the default, 0, sizes it
from the bundle, and a size below the bundle's is a configure error. The
ESP-NN delegate's options are under ExecuTorch -> ExecuTorch ESP-NN
backend.

## Logging

`CONFIG_ET_ENABLE_LOGGING=n` compiles ExecuTorch logging out entirely.
Otherwise runtime messages go to ESP_LOG (tag `executorch`), and
`CONFIG_ET_LOG_LEVEL_*` is a compile-time floor (`ET_MIN_LOG_LEVEL`)
on top of the usual IDF runtime levels. `et_model` logs its own
errors under the `et_model` tag.

## Updating

Bump the `==` pin, or, for a vendored copy, re-run the generator
against a newer ExecuTorch checkout and replace this whole directory.
Then regenerate your kernel bundle with the matching `executorch`
Python package. Review the `VERSION` diff (commit, kernel library
hashes); do not line-review vendored sources.

Projects moving from the op profiles of earlier versions remove
`CONFIG_ET_ENABLE` and `CONFIG_ET_OP_PROFILE_*` from their sdkconfig,
which no longer do anything, and declare their models and a kernel
bundle as above.
