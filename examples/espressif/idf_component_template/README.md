<!--
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-->

# ExecuTorch for ESP-IDF

[ExecuTorch](https://github.com/pytorch/executorch) is PyTorch's open
source stack for running AI locally on phones, wearables, laptops,
browsers, embedded systems and microcontrollers. This component brings
its runtime to ESP-IDF 6.0 and later, for any ESP target, as source that
builds with your project.

## How ExecuTorch works

A model is prepared ahead of time, on your computer, with the
`executorch` Python package. You export it with `torch.export`,
optionally quantize it, and lower it to a `.pte` program. Lowering can
hand the parts of the model that a backend supports to that backend's
delegate, and it plans where the model's intermediate tensors live in
memory, so the device keeps them in an arena whose size is known before
it starts.

On the device, the runtime reads the `.pte` in place from flash and runs
it: each operator with a kernel from a kernel library, and each
delegated part with its backend.

[How ExecuTorch works](https://docs.pytorch.org/executorch/stable/intro-how-it-works.html)
and [Model export and lowering](https://docs.pytorch.org/executorch/stable/using-executorch-export.html)
cover this in more depth.

## What the component provides

- The ExecuTorch runtime, and `et_model.h`, a small C API that runs a
  `.pte` from flash.
- Kernel libraries: portable (reference kernels for the core ATen
  operators), quantized, and the Cadence backend's generic kernels.
- The ESP-NN delegate, `EspNNBackend`, which runs the int8 parts of a
  model lowered for it with Espressif's
  [ESP-NN](https://github.com/espressif/esp-nn) kernels. ESP-NN has SIMD
  kernels for the ESP32-S3 and portable C kernels for the other chips.
  The delegate is a prototype under active development.
- `tools/gen_kernels.py`, which makes the kernel bundle that registers
  the kernels your models use.
- `examples/hello_executorch`, a complete project, in every published
  version.

## Try it

`hello_executorch` runs a small model and checks its outputs. Its `.pte`
and kernel bundle are already made, so it needs only ESP-IDF:

```sh
idf.py create-project-from-example "pytorch/executorch==@ET_COMPONENT_VERSION@:hello_executorch"
cd hello_executorch
idf.py set-target esp32s3
idf.py build flash monitor
```

It prints `ET_COMPONENT_SMOKE: PASS` when every output matches. A
release candidate is on the staging registry only, so for one, first
`export IDF_COMPONENT_REGISTRY_URL=https://components-staging.espressif.com`
in the shell, since the build resolves the version again. In a copy of
this component you generated yourself, build `examples/hello_executorch`
in place instead. The generator leaves it out only of an
`--allow-dev-stamp` copy made without the executorch package built from
that commit.

## Using it in your project

### Add the dependency

Depend on one exact version and commit `dependencies.lock`:

```yaml
dependencies:
  pytorch/executorch: "==@ET_COMPONENT_VERSION@"
```

A release candidate's dependency also needs
`registry_url: https://components-staging.espressif.com`.

A kernel bundle is made for one component version, a `~N` respin
included, and the build refuses a bundle made for another version, or
for the same version from another commit. With a range, the build
breaks once the lock is re-solved to a newer version (for example by
`idf.py update-dependencies`), until the bundle is regenerated.

The component depends on `espressif/esp-nn`, which the component manager
fetches from the production registry, components.espressif.com, even
for a project that names another registry. A project that builds
without the component manager provides it as `components/esp-nn`.

### Set up Python

Exporting models and making kernel bundles need the executorch this
component was built from, and the torch that `VERSION` names as
`torch_requirement`, from its `torch_index`, since ExecuTorch's wheels
declare no torch:

```sh
@ET_INSTALL@
```

### Export a model

To run a model with ESP-NN, quantize it with `EspNNQuantizer` and lower
it with `to_espnn_edge`, which delegates the int8 regions ESP-NN
supports and leaves the rest to the CPU kernels:

```python
import torch
from executorch.backends.espressif import EspNNQuantizer, to_espnn_edge
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

model = MyModel().eval()  # your torch.nn.Module
example_inputs = (torch.randn(1, 3, 96, 96),)

prepared = prepare_pt2e(
    torch.export.export(model, example_inputs).module(), EspNNQuantizer()
)
# Tuples of representative inputs, from which the int8 scales are set.
for inputs in calibration_inputs:
    prepared(*inputs)
quantized = torch.export.export(convert_pt2e(prepared), example_inputs)
edge, report = to_espnn_edge(quantized)
print(report["cpu_ops"])  # the operators left to the CPU kernels
with open("main/model.pte", "wb") as f:
    f.write(edge.to_executorch().buffer)
```

### Make the kernel bundle

`tools/gen_kernels.py` makes the kernel bundle from your `.pte` files.
Run it with the Python environment you exported the models with, after
every export, ExecuTorch upgrade or update of this component, and keep
the bundle in your project:

```sh
python <this directory>/tools/gen_kernels.py --pte main/model.pte --out main/et_kernels
```

### Declare the models and the bundle

From a component's `CMakeLists.txt` after its `idf_component_register()`,
or from the project's after `project()`, declare the models and the
bundle, and add `executorch` to the requirements of the components that
use it:

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

### Run the model

`et_model.h` loads a `.pte` and runs it:

```c
#include <esp_log.h>
#include <et_model.h>

et_model_config_t cfg = ET_MODEL_CONFIG_DEFAULT(model_pte, model_pte_size);
cfg.planned_location = ET_MEM_INTERNAL;  // planned tensors in internal SRAM
et_model_t *m = et_model_create(&cfg);
if (!m) {
  return;  // et_model_create logs why
}
size_t n;
const void *out = NULL;
if (!et_model_load_method(m, NULL) || !et_model_set_input(m, 0, frame, sizeof(frame)) ||
    !et_model_execute(m) || !(out = et_model_output_data(m, 0, &n))) {
  ESP_LOGE("app", "%s", et_model_last_error(m));
  et_model_destroy(m);
  return;
}
```

`model_pte` and `model_pte_size` are the `.pte` bytes, read in place,
typically from flash. They must be 16-byte aligned, and
`et_model_create` rejects misaligned data. ESP-IDF's `EMBED_FILES` does
not align what it embeds, so `examples/hello_executorch/main/CMakeLists.txt`
embeds its `.pte` with an assembler `.balign 16` and `.incbin` instead.
The method, planned and optional scratch arenas are either passed in or
allocated with heap_caps in internal RAM or PSRAM. Application pipelines
built on it (audio front ends, detectors, wake word logic) live with the
application, not in this component.

For the core C++ runtime (`Program`/`Method`) and `BufferDataLoader`,
`#include <executorch/...>`. `extension::Module`, the file and mmap data
loaders, and external `.ptd` data maps are not included.

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

## About this component

**This tree is generated. Do not edit it by hand.** It is produced by
`examples/espressif/build_idf_component.sh` in the ExecuTorch
repository. Releases are versioned as ExecuTorch's, `X.Y.Z`, with
`X.Y.Z-rc.N` for release candidates and `X.Y.Z~N` for respins of this
component. The registry's dev versions, `X.Y.Z-dev.YYYYMMDD`, come from
the `nightly` branch. A copy you generate yourself is not published; use
it from your project's `components/` or through `override_path`.

`VERSION` records the ExecuTorch commit, the generator's options, the
kernel libraries the component ships and the torch to install with
executorch. The README of `examples/hello_executorch` says how its model
and kernel bundle were made. Vendored third-party code is listed in
`THIRD_PARTY_LICENSES` and the `sbom*.yml` files.
