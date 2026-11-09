# Partitioner API

`CoreAIPartitioner` decides which parts of a model are delegated to Core AI, and how they're compiled. With no arguments, it delegates every part of the model that Core AI supports, using default settings. That's the right choice for most models.

## Usage

```python
import torch
from executorch.backends.apple.coreai import (
    CoreAIPartitioner,
    get_default_compile_config,
    get_default_passes,
)
from executorch.exir import to_edge_transform_and_lower

et_program = to_edge_transform_and_lower(
    torch.export.export(model, example_inputs),
    transform_passes=get_default_passes(),
    partitioner=[CoreAIPartitioner()],
    compile_config=get_default_compile_config(),
).to_executorch()
```

::::{important}
`CoreAIPartitioner` must be used with `to_edge_transform_and_lower()`. The legacy `to_edge()` + `to_backend()` workflow is **not supported**, because it decomposes ops that Core AI lowers directly, such as fused SDPA.
::::

The partitioner groups the supported ops into *partitions*, connected subgraphs that each become one Core AI delegate. An op Core AI can't run splits the graph, so a method can have several partitions.

## Options

All options are keyword arguments.

| Option | Default | Description |
|---|---|---|
| `aot_compile_config` | `None` | Compile the model for specific device architectures at export time. See [Ahead-of-time compilation](#ahead-of-time-compilation). |
| `min_deployment_version` | `None` | The minimum OS version the exported model targets, as `"major[.minor[.patch]]"`, such as `"27.0"`. |
| `input_enumerations` | `None` | Compile dynamic input dimensions to a fixed set of sizes. See [Enumerated shapes](#enumerated-shapes). |
| `multifunction` | `False` | Put the partitions of every method into one shared Core AI model. See [Multi-method models](#multi-method-models). |
| `take_over_constant_data` | `True` | Store constants, such as weights, inside the Core AI model. If `False`, they're passed to the delegate as inputs instead. |
| `take_over_mutable_buffer` | `True` | Move the model's mutable buffers, and the in-place updates to them, into the Core AI model. The runtime doesn't run stateful Core AI models yet, so set this to `False` for models that update buffers in place. See [Stateful models](#stateful-models). |

## Stateful Models

Models that update a buffer in place, such as a KV cache, export fine with the default `take_over_mutable_buffer=True`, but the runtime doesn't run the resulting stateful Core AI model yet. Loading fails with `NotSupported`, and the log shows `Stateful functions are not supported`.

To run these models today, pass `take_over_mutable_buffer=False`:

```python
partitioner = CoreAIPartitioner(take_over_mutable_buffer=False)
```

The buffers then stay in ExecuTorch. Core AI computes each new buffer value, and ExecuTorch writes it back to the buffer after the delegate runs, so the state carries over from one call to the next. This adds a copy of each updated buffer per call.

## Excluding Nodes from Delegation

To keep a node out of the Core AI delegate, mark it with `do_not_delegate`; it then runs on ExecuTorch's CPU kernels. Mark nodes in a transform pass, which runs on the edge graph just before partitioning. Marks set on the output of `torch.export` are lost when it's converted to the edge dialect.

```python
from executorch.backends.apple.coreai.partition.partitioner import do_not_delegate
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassResult


def keep_relu_on_cpu(graph_module):
    for node in graph_module.graph.nodes:
        if node.op == "call_function" and node.target == exir_ops.edge.aten.relu.default:
            do_not_delegate(node)
    return PassResult(graph_module, False)


et_program = to_edge_transform_and_lower(
    torch.export.export(model, example_inputs),
    transform_passes=[*get_default_passes(), keep_relu_on_cpu],
    partitioner=[CoreAIPartitioner()],
    compile_config=get_default_compile_config(),
).to_executorch()
```

## Ahead-of-Time Compilation

By default, the `.pte` contains a portable Core AI model (`.aimodel`) that runs on any supported device. With `aot_compile_config`, export instead compiles the model with `xcrun coreai-build` for specific Core AI device architectures, and the `.pte` contains one compiled model (`.aimodelc`) per architecture:

```python
from executorch.backends.apple.coreai import AOTCompileConfig, CoreAIPartitioner

partitioner = CoreAIPartitioner(
    aot_compile_config=AOTCompileConfig(platform="iOS", architectures=["h17p"]),
)
```

`AOTCompileConfig` has these fields:

| Field | Default | Description |
|---|---|---|
| `platform` | `macOS` | The target platform: `iOS`, `macOS`, `watchOS`, `visionOS` or `tvOS`. |
| `architectures` | All | The Core AI device architectures to compile for, such as `h17p`. These are Core AI's names, not CPU architectures like `arm64`. If omitted, the compiler builds every architecture it supports for the platform. |
| `preferred_compute` | `none` | A compute unit to optimize for: `gpu`, `neural-engine` or `none`. |
| `expect_frequent_reshapes` | `False` | Optimize for input shapes that change between calls. Set the [runtime option](coreai-runtime.md#frequent-reshapes) of the same name too. |

This requires `xcrun coreai-build`, which comes with the Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`).

At load time, the runtime asks Core AI for the device's architecture and uses the compiled model with exactly that name. There's no fallback: if the `.pte` doesn't include the device's architecture, loading fails with an error naming the device's architecture and the ones the `.pte` contains. Compiled models can still need specialization on the device, which the runtime [caches](coreai-runtime.md#cache) as usual.

## Dynamic and Enumerated Shapes

Dimensions exported as dynamic, with `torch.export.Dim`, stay dynamic in the Core AI model. If a model's input shapes change between calls, also set the [`coreai_expect_frequent_reshapes`](coreai-runtime.md#frequent-reshapes) runtime option.

### Enumerated shapes

If a dynamic dimension only ever takes a few sizes, `input_enumerations` compiles the model for exactly those sizes instead. Pass one entry per model input: `None` for an input without enumerated dimensions, or a dict that maps each dimension index to its sizes. Each enumerated dimension must be exported as dynamic:

```python
batch = torch.export.Dim("batch", min=1, max=64)
ep = torch.export.export(model, (x,), dynamic_shapes={"x": {0: batch}})

et_program = to_edge_transform_and_lower(
    ep,
    transform_passes=get_default_passes(),
    partitioner=[CoreAIPartitioner(input_enumerations=[{0: [4, 16, 32]}])],
    compile_config=get_default_compile_config(),
).to_executorch()
```

This model accepts a batch of 4, 16 or 32. How it works:

- **One static function per shape.** Core AI replaces the dynamic model with one static function for each combination of enumerated sizes. If you enumerate several dimensions, the number of functions is the product of their size counts.
- **Calls are routed by shape.** At each call, the delegate runs the function whose input shapes exactly match the inputs. Any other shape fails with `InvalidArgument` before anything runs.
- **Weights are shared.** All the functions belong to one Core AI model, with one set of weights, one specialization and one cache entry.
- **Load time grows with the number of functions.** The delegate prepares every function when the model loads. On an M4 Max, loading a small, already-cached model took 7 ms with 1 function, 41 ms with 16 and 248 ms with 64.

Export fails if an enumerated size would make another dimension computed from it zero or negative, such as `batch - 1` with a batch of 1.

## Multi-Method Models

By default, each partition becomes a separate Core AI model, so methods that share weights each store their own copy. With `multifunction=True`, the partitions of every method go into one Core AI model instead, as separate functions named `<method>_<partition index>`:

```python
et_program = to_edge_transform_and_lower(
    {"forward": forward_ep, "prefill": prefill_ep},
    transform_passes=get_default_passes(),
    partitioner=[CoreAIPartitioner(multifunction=True)],
    compile_config=get_default_compile_config(),
).to_executorch()
```

Here `forward` runs the function `forward_0` and `prefill` runs `prefill_0`. If a method has several partitions, they become `forward_0`, `forward_1` and so on. Weights the methods have in common are stored once, so two methods that use the same weights produce a `.pte` about the size of one.

On the device, all the methods share one cache entry: the first load specializes the model, and the others restore it from the cache. Each method still loads its own copy of the model, with its own weights in memory, unless you also set the [`coreai_share_multifunction_model`](coreai-runtime.md#sharing-a-multi-method-model) runtime option.

Rules for multi-method exports:

- **Every method needs `multifunction=True`.** Export fails otherwise, naming the methods without it.
- **Other options must match.** Every method's partitioner must use the same `aot_compile_config` and `min_deployment_version`. Only `input_enumerations` can differ.
- **Function names must be unique.** If two partitions would get the same function name, export fails. Rename one of the methods.

### Enumerated shapes in a multi-method model

To enumerate shapes for only some methods, or use different sizes per method, give each method its own partitioner:

```python
et_program = to_edge_transform_and_lower(
    {"decode": decode_ep, "prefill": prefill_ep},
    transform_passes=get_default_passes(),
    partitioner={
        "decode": [CoreAIPartitioner(multifunction=True)],
        "prefill": [
            CoreAIPartitioner(multifunction=True, input_enumerations=[{0: [64, 128]}])
        ],
    },
    compile_config=get_default_compile_config(),
).to_executorch()
```

Here `decode` stays a single function, `decode_0`. Each partition of `prefill` becomes one function per enumerated shape, named after the partition with a suffix for the shape.
