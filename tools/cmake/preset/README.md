# ExecuTorch Build Presets

Build presets are a collection of cmake arguments that configure flavors of ExecuTorch libraries. Build presets are intended to help answer questions like:

- "How do I build the ET library used in the pip wheel?" → use [pybind.cmake](./pybind.cmake)
- "How do I build the iOS library?" → use [ios.cmake](./ios.cmake)
- "How do I build what's required to run an LLM?" → use [llm.cmake](./llm.cmake)

## Why?

See: https://github.com/pytorch/executorch/discussions/10661. tl;dr instead of turning on multiple opaque/unknown flags, you can reduce it to:

```bash
$ cmake --preset macos
$ cmake --build cmake-out -j100 --target executor_runner
```

## Working with Presets

> [!TIP]
> Configurable presets options and their default values are stored in [default.cmake](./default.cmake).

### Within ExecuTorch

If you're developing ExecuTorch directly, you can use the `cmake --preset` command:

```bash
# List all the presets buildable on your machine
$ cmake --list-presets

# Build a preset
$ cmake --preset llm

# Build a preset with one-off configuration change
$ cmake -DEXECUTORCH_BUILD_MPS=OFF --preset llm
```

The cmake presets roughly map to the ExecuTorch presets and are explicitly listed in [CMakePresets.json](../../../CMakePresets.json). Note that you are encouraged to rely on presets when build locally and adding build/tests in CI — CI should do what a developer would do and nothing more!

### Including ExecuTorch as Third-party Library

#### Choose a built-in preset

You can include ExecuTorch like any other cmake project in your project:
```cmake
add_subdirectory(executorch)
```
However, note that since a preset isn't specified, it will use the [default](./default.cmake) options. This likely not what you want — you likely want to use a preset. Check for presets available in [tools/cmake/preset](.) and explicitly choose which preset to use:

```cmake
set(EXECUTORCH_BUILD_PRESET_FILE executorch/tools/cmake/preset/llm.cmake)
add_subdirectory(executorch)
```

Even when using a preset, you can explicitly override specific preset configurations:

```cmake
set(EXECUTORCH_BUILD_PRESET_FILE executorch/tools/cmake/preset/llm.cmake)

# Although llm.cmake might have turned on `EXECUTORCH_BUILD_MPS`, you can turn if off
set(EXECUTORCH_BUILD_MPS OFF)

add_subdirectory(executorch)
```

#### Creating your own preset

Preset files are just cmake files that turn on/off ExecuTorch configs. For example if we want to build the runner with a CoreML backend, create your cmake preset file (i.e. `my_projects_custom_preset.cmake`):

```cmake
# File: my_projects_custom_preset.cmake

set_overridable_option(EXECUTORCH_BUILD_COREML ON)
set_overridable_option(EXECUTORCH_BUILD_EXECUTOR_RUNNER ON)
```

We use `set_overridable_option` so that it allows the option to be overriden via the command line using `-D`. Feel free to use the standard `set(...)` function to prevent overriding options.

Once the preset is created, set `EXECUTORCH_BUILD_PRESET_FILE` before adding ExecuTorch:

```cmake
set(EXECUTORCH_BUILD_PRESET_FILE my_projects_custom_preset.cmake)
add_subdirectory(executorch)
```
