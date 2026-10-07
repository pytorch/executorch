# Memory Planning

Audience: Backend integrators and embedded developers who are interested in customizing the regions of memory ExecuTorch programs operate in.

## Overview

MemoryPlanning is the very last action taken before taking an `ExportedProgram` and undergoing emission to an ExecuTorch program. During this process, ExecuTorch takes the size and lifespan of each mutable tensor, and plans out their location in fixed size memory arenas.

Concretely, there are three passes related to memory planning:
* `SpecPropPass` computes a TensorSpec for each tensor in the graph (inputs, intermediates or outputs). The most important field of the tensor spec is a symbolic expression of the shapes of the tensor, where the initial set of symbols comes from the dimensions of input tensors, intermediate tensor shapes’ symbolic expression is propagated via tensor operations. The dimensions can be marked as either dynamic or static by users and when the dims are dynamic, users are required to annotate the dim with a ValueRange.

* `SymShapeEvalPass` evaluates the symbolic expressions to concrete integers with their upper bounds. There are two ways to doing the upper bound specialization:
HintBasedSymShapeEval (to be deprecated) is the old way of evaluating the upper bound. It doesn’t look at the ValueRange of the symbols but uses the shapes of example inputs to replace all the symbols. We call it “hint based“ because the example inputs’ shapes are just hints of what the input shapes might be at run time and are used for tracing only. ValueRangeBasedSymShapeEval is the recommended way of doing UpperBoundMemory planning. It will actually look at the ValueRange of the symbols and do an inference over the ranges to get a real upper bound.

* `MemoryPlanningPass` does the actual memory planning given all tensors get a TensorSpec with concrete integer shapes.

## Algorithms

ExecuTorch provides two options for memory planning algorithms out of the box, but users can define their own if the provided options are inappropriate or insufficient for their use case.

* The naive algorithm simply concatenates all the tensors together in a linear memory block without considering memory re-use. It serves as an upper bound for total memory consumption and serves as a baseline.

* The Greedy algorithm tries to re-use the already allocated memory based on the best-fit criteria. Specifically:
When there isn’t an allocated memory whose lifetime doesn’t overlap with the current tensor that we try to do memory planning for, we allocate a new memory buffer with the same size and lifetime as the current tensor. When there is one or more allocated memory buffer, whose lifetime overlaps with the current tensor, we pick the buffer that has the closest size with current tensor so as to reduce memory fragmentation. Finally, we allocate these memory buffers linearly in memory.


## Method Inputs and Outputs

The `MemoryPlanningPass` exposes the option to not memory plan program inputs and outputs. If the IO is not planned then users will be expected to provide data buffers to back these values at runtime. Example:

```python
program = edge_program.to_executorch(
            exir.ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(
                    alloc_graph_input=False, # Inputs will not be memory planned, the data_ptr for input tensors after model load will be nullptr
                    alloc_graph_output=True, # Outputs will be memory planned, the data_ptr for output tensors after model load will be in the `planned_memory`.
                )
            )
        )
```

One common set-up would be for models where the outputs of the model are provided as inputs to subsequent inferences. In that situation, it would generally be better to not memory plan the IO, and instead provide the same buffer to both the input and output at runtime to avoid a copy.

## Custom Memory Plans

Users can write custom memory plans to take advantage of multiple memory locations (like SRAM and DRAM), place the outputs of specific nodes in specific locations, or even change the planning algorithm itself. The following example shows how you could reuse the provided planning algorithms, but with multiple hierarchies and placing the outputs of specific ops in specific memory arenas.

```python
class CustomPoolMemoryPlanningPass(MemoryPlanningPass):
    def run(self, graph_module: GraphModule, graph_signature: Optional[ExportGraphSignature]) -> PassResult:
        for subgm in graph_module.modules():
            if not isinstance(subgm, GraphModule):
                continue
            for node in subgm.graph.nodes:
                # mem_id = 1 placeholder and outputs of mul
                # mem_id = 2 for outputs of add
                # parent class will copy spec will to alloc nodes
                if node.op == "placeholder":
                    node.meta["spec"].mem_id = 1
                    continue

                if node.op != "call_function":
                    continue

                if node.target == torch.ops.aten.add.out:
                    node.meta["spec"].mem_id = 2
                elif node.target == torch.ops.aten.mul.out:
                    node.meta["spec"].mem_id = 1

        return super().run(graph_module, graph_signature)
```

Then later when lowering to ExecuTorch you can use your custom plan in the following way:

```python
program = edge_program.to_executorch(
            exir.ExecutorchBackendConfig(
                memory_planning_pass=CustomPoolMemoryPlanningPass(
                    memory_planning_algo=greedy,
                )
            )
        )
```

> **Note:** Custom pool passes that pre-assign `mem_id` are not yet compatible
> with `enable_non_cpu_memory_planning=True`.  When per-device planning is
> enabled, device buffers are appended after the CPU buffers in the global
> `bufsizes` array.  If a custom pass has already set `mem_id` values (e.g.
> `mem_id=2` or `mem_id=3`), those slots may collide with the device-buffer
> slots, leading to incorrect memory layout.  If both features are enabled
> simultaneously, `apply_algo` will raise a `NotImplementedError`.

Users attempting to write a custom memory planning algorithm should start by looking at [the greedy algorithm's implementation](https://github.com/pytorch/executorch/blob/main/exir/memory_planning.py#L801).

## Device-Aware Memory Planning

`ExecutorchBackendConfig.enable_non_cpu_memory_planning` is `True` by default, so
the memory planning pass partitions tensor specs by their device type and runs
the planning algorithm independently for each device.  This produces separate
memory buffers for each device (e.g. CPU vs. CUDA), ensuring that device memory
and host memory are never mixed.  Set it to `False` for the legacy behavior,
where every tensor is planned into one host pool whatever its device:

```python
program = edge_program.to_executorch(
            exir.ExecutorchBackendConfig(
                enable_non_cpu_memory_planning=False,
            )
        )
```

The resulting `bufsizes` array layout depends on which devices are present:

| Scenario | bufsizes | Description |
|---|---|---|
| CPU only | `[0, cpu_size]` | Same as legacy behavior |
| CUDA only | `[0, cuda_size]` | Buffer 1 is CUDA, no wasted CPU slot |
| CPU + CUDA | `[0, cpu_size, cuda_size]` | Buffer 1 is CPU, buffer 2 is CUDA |

**Current limitations:**
- Not compatible with custom pool passes that pre-assign `spec.mem_id` (see note above).
- Submodule buffer sizes (from control-flow submodules like `cond`/`while`/`map`)
  are applied only to the CPU partition, and that partition exists only when the
  top-level graph has a CPU tensor of its own.  A method whose whole top level is
  on an accelerator therefore loses the reservation: the branch tensors keep
  arena indices the top-level plan has given to live device tensors, and the two
  overlap with no error and no change in arena size.  Keep at least one
  top-level tensor on the host, or set `enable_non_cpu_memory_planning=False`.

## Sharing a Buffer Across Methods

A multi-method program holds one copy of each registered buffer, but memory
planning runs once per method, so by default each method places that buffer
wherever its own plan puts it.  `shared_buffer_fqns` names the buffers that have
to sit at one address: the named buffers are withheld from the planning
algorithm and given a memory arena of their own on each device that owns one, at
an arena index and offset that mean the same thing in every method.

```python
program = edge_program.to_executorch(
            exir.ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(
                    share_mutable_buffers=True,
                    shared_buffer_fqns=frozenset({"cache"}),
                ),
            )
        )
```

The argument is checked rather than interpreted loosely:

- It requires `share_mutable_buffers=True`, and an empty set is rejected — pass
  `None` for the legacy behavior, which shares *every* mutable buffer through a
  single CPU arena at `mem_id=2` and requires every other tensor on arena 1, so
  it refuses a custom pool or a device arena.
- It *replaces* that legacy behavior rather than adding to it.  Naming any
  buffer hands every other mutable buffer back to the planning algorithm, which
  plans each method on its own and has no reason to give one buffer the same
  address twice; a write through one method is then not what another method
  reads.  A method that only *reads* such a buffer is served differently again:
  the buffer is const in that method and is emitted with its `state_dict` data,
  so every call returns the registered value and none ever returns what another
  method wrote — stable and plausible rather than visibly stale.  Export warns,
  naming the buffers more than one method reaches: one method has no second
  placement to disagree with.  List every mutable buffer that has to be shared,
  or drop `shared_buffer_fqns` for the legacy path above.
- One pass instance plans the whole program, because the arena numbering is
  agreed across methods.  A per-method `dict` of passes is rejected — and so is
  a dict of plain `share_mutable_buffers` passes, which agrees a `mem_id=2`
  placement the same way: each instance sees one method, so the export succeeds
  having shared nothing.
- It takes effect only where memory planning is given a graph signature, which
  is what says which placeholders are buffers.  `EdgeProgramManager.to_executorch`
  — the path shown above — passes one.  `ExirExportedProgram.to_executorch`,
  `LoweredBackendModule.program()` and `LoweredBackendModule.buffer()` do not, so
  a pass carrying these names plans there exactly as it would with
  `shared_buffer_fqns=None`: nothing is shared and none of the checks below run.
  Export warns rather than raising.
- No named buffer may have zero elements.  There is no state in such a buffer
  for two methods to share, and where it is the only buffer declared on its
  device it leaves an arena of zero bytes.
- Every named buffer must be a buffer of at least one method, and must be
  mutated by at least one of them.  A placement on a buffer placeholder is what
  tells the emitter that the buffer is mutable, and a mutable buffer is by
  default emitted without its `state_dict` data, so a buffer nothing writes to
  would come back uninitialized.  Mutation by *some* method is all that is
  checked, and that leaves a hazard for the rest: a reader run before the writer
  sees uninitialized planned memory rather than the registered value.  Refusing
  this shape is not an option — a prefill method that fills a cache and a decode
  method that reads it is what the feature is for — so export warns instead,
  naming each declared buffer that some method only reads, and telling the
  caller to run a writer first.
- No named buffer may carry `et_init_buffer`, which is what
  `InitializedMutableBufferPass` sets.  The runtime copies the serialized
  initial value into the buffer's planned allocation every time it loads a
  method that has it, and the shared arena makes that one allocation for the
  whole program, so any second load puts the initial value back over the live
  one.  Dropping `shared_buffer_fqns` does not make an initialized buffer safe:
  the legacy `mem_id=2` path puts every mutable buffer at one address in every
  method too — it is just not checked there.
- A named buffer that a custom pool pass has already pinned is rejected: the
  dedicated arena and the pool are two different placements for one tensor.
- Every method that has a named buffer must describe it the same way — the same
  `dtype`, shape, stride, layout and byte count, on the same device.  A single
  slot cannot be two tensors.
- The planning algorithm must not declare `plans_against_a_memory_budget`.  A
  dedicated arena is appended after the algorithm has returned, so it is not one
  of the arenas that budget describes and nothing is charged for its bytes.
  The attribute is read off the algorithm the pass was handed and off the
  entries of its `algo_list`, and nothing is looked through: an algorithm
  wrapped in a `functools.partial` or handed over as a bound method declares
  nothing here.  `BankedGreedy` sets the attribute; `banked_memory_planning_pass`
  refuses `share_mutable_buffers` outright, so this check is what catches a pass
  built around `BankedGreedy` by hand.

Three graph shapes are refused rather than laid out:

- A tensor placed inside another tensor's storage — a `TensorSpec` carrying a
  `storage_base`, which memory planning sets from the `_share_alloc_with_arg_idx`
  node meta that `reinplace_pass` writes (`NotImplementedError`).  The declared
  buffers are placed by hand after the algorithm has returned, so an alias the
  algorithm is asked to fit inside one has no base allocation to be an offset
  into.  Which aliases reach a declared buffer is a question about the whole
  chain, and no walk of one is made here, so an alias of a tensor no declared
  name mentions — which the algorithm would have placed, inside its base at the
  offset it asked for — is refused with the rest.  Dropping `shared_buffer_fqns`
  *and* `share_mutable_buffers` returns every tensor to the algorithm; dropping
  `shared_buffer_fqns` alone leaves the legacy sharing path, which withholds
  every mutable buffer from the algorithm just the same.
- A method that has a control-flow submodule of any kind
  (`NotImplementedError`): the submodule's tensors are planned by a recursive
  `apply_algo` into arena indices nothing renumbers afterwards, so the common
  numbering would have to hand every one of them back unchanged, and it has no
  way to promise that.  That shape is already broken without this argument — a
  `cond` method planned with `alloc_graph_input` and `alloc_graph_output` off
  emits a zero-byte arena its submodules name and fails to load.
- A host tensor at a *non-zero* device index under
  `enable_non_cpu_memory_planning=True` (`NotImplementedError`): a non-zero host
  device index is out of this argument's scope, and no arena layout here is
  designed or tested for one.

Apart from a named buffer the pool has already pinned, which is rejected above, a
custom memory pool is compatible with this: arena indices are handed out in
order, so a pool is one more arena of the CPU block and the dedicated arena is
appended after it.  With per-device planning on, `apply_algo` refuses a
pre-assigned `mem_id` only where the program also has a tensor on another
device, and that is unchanged by this argument.

### What the runtime has to do

A shared arena is a *compile-time* agreement: the emitted program gives the
buffer the same `memory_id` and offset in every method.  Nothing in the `.pte`
tells a runtime to back that arena with one allocation, so a shared arena is
honored only by a runtime that implements the sharing — on a device, that means
a runtime written for it.  On the host, two in-tree paths already share arenas,
and they do not agree about which.

`extension/module`'s `Module`, in the three bullets below, means a `Module`
allocating the planned memory itself.  All of this lives behind
`if (!planned_memory)` in `Module::load_method`, so a caller that supplies its
own `HierarchicalAllocator` reaches none of it — neither the sharing nor the
refusal.

- `Module(..., share_memory_arenas=true)` shares `mem_id=1` and `mem_id=2` only.
  A CPU-only program with no custom pool puts the shared arena at `mem_id=2`,
  which that `Module` does share.
- A custom pool widens the CPU block, so the shared arena follows it at
  `mem_id=3` or above.  `Module` treats any `mem_id>2` as a custom memory plan
  and gives it a fresh per-method buffer, so the methods do *not* share it.
- A shared arena on a device is refused: `Module` returns
  `Error::NotSupported` when any of a method's planned buffers is off CPU and
  `share_memory_arenas` is set.

The Python `extension/pybindings` entry points:

- `_load_for_executorch_*` build a `Module` with `share_memory_arenas` defaulted
  to false, so nothing is shared there at all — which is equally true of the
  legacy `mem_id=2` path.
- `_load_program*` do not build a `Module`.  They size each host arena to the
  largest any host-only method needs and hand every one of those methods the
  same allocation, so *every* host arena is shared whatever its index: a CPU
  shared arena at `mem_id=3` or above is honored here even though `Module`
  would not honor it.  A method with a planned buffer off CPU is given arenas
  of its own instead and shares none of them.

Leaving the default runtime alone is deliberate: the exporter places the arena
where the layout requires rather than where one runtime happens to look, so the
pass does not refuse the configurations `Module` declines.

## Debugging Tool

Please refer to [Memory Planning Inspection](memory-planning-inspection.md) for a tool to inspect the result of memory planning.
