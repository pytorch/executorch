# torch.export.export()

Captures a PyTorch `nn.Module` into an `ExportedProgram` — a fully traced
ATen-dialect graph with no Python runtime dependency.

```python
from torch.export import export
exported = export(model.eval(), (example_input,))
```

## How export works

### The tracing mechanism

`torch.export` traces `forward()` by replacing real tensors with **fake
tensors** — objects that carry shapes, dtypes, and device metadata but no data.
These are wrapped in proxy objects that record every ATen operation into a graph.
The result is a fixed DAG of operations that can be executed without Python.

Fake tensors are implemented as meta-device tensor subclasses that lie about
their device. Each operator has a **fake implementation** (meta kernel) that
computes output shapes and metadata from input shapes — without touching data.
For example, `torch.cat`'s fake kernel sums input sizes along the cat dimension.
When you see `UnsupportedOperatorException`, it means an operator is missing
its fake implementation.

Two tracing modes:

- **Non-strict** (`strict=False`, recommended): the Python interpreter runs
  normally with fake tensors. Most Python patterns work. Conditions on tensor
  shapes are captured as guards.
- **Strict** (`strict=True`, current default): TorchDynamo analyzes Python
  bytecode symbolically without executing it. More restrictive — some Python
  patterns are unsupported. Will stop being the default.

### Static vs dynamic values

Every value encountered during tracing is classified as either **static** (fixed
at export time, burned into the graph as a constant) or **dynamic** (variable
across runs, tracked symbolically in the graph).

| Value | Classification |
|-------|---------------|
| Tensor data | Always dynamic |
| Tensor shapes | Static by default; dynamic via `dynamic_shapes` |
| Tensor metadata (dtype, device, layout) | Always static |
| Python primitives (int, float, bool, str, None) | Always static |
| Container structure (list length, dict keys) | Always static |
| Module parameters and buffers | Dynamic data, always-static shapes |
| Module attributes of primitive type | Always static |

**Static values are constant-folded.** If `y = 3` is a static input and the
model computes `x + y + 7`, the graph contains `x + 10`. The graph is
*specialized* to that value — passing a different one at runtime raises an
error. Loops over static values are unrolled; branches on static values are
resolved to a single path.

**Dynamic values flow through the graph as symbolic variables.** The tracer
must prove that all operations on them are valid for all possible runtime
values.

This classification is the single concept from which all export behavior
follows. Control flow, shape constraints, and failure modes are all
consequences of how the tracer handles static vs dynamic values.

### Symbolic shapes

When a dimension is marked dynamic, the tracer assigns it a **symbol** (e.g.,
`s0`). As operations transform tensors, output shapes become **symbolic
expressions** computed by fake implementations: `torch.cat([x, x], dim=0)`
produces shape `[2*s0, ...]`; a stride-2 conv on length `L` produces
`1 + ((L - 1) // 2)`.

Symbols are either **backed** or **unbacked**:

- **Backed symbols** come from input dimensions marked as dynamic. They have
  **hint values** — the concrete sizes from the example inputs. The tracer uses
  hints to evaluate conditions, pick branches, and emit guards.

- **Unbacked symbols** come from operations whose output shape depends on
  tensor *data*, not tensor shapes (e.g., `torch.nonzero()`, `.item()`). They
  have no hint value, so the tracer can't evaluate conditions involving them
  without explicit help (see `torch._check` below).

### Guards

When the tracer encounters a condition on a backed dynamic shape, it uses
the hint to evaluate it, picks a branch, and records a **guard** — a symbolic
boolean expression asserting that the branch choice is correct. For example,
`if x.size(0) > 4` with hint `s0 = 8` produces the guard `s0 > 4`.

After tracing, all guards must be **provable** from the `Dim` specification
(the `min`/`max` ranges). If the solver can't prove a guard, export fails with
`ConstraintViolationError` and suggests how to fix the `Dim` spec.

The solver uses algebraic reasoning over the `Dim` ranges. It handles **linear
arithmetic well** (addition, multiplication, comparison) but is **weak at
integer division (`//`) and modular arithmetic (`%`)**. Any shape expression
involving these operators is likely to produce guards the solver can't prove.
This is the root cause of the most difficult export failures.

## API

```python
torch.export.export(
    model,          # nn.Module in eval mode
    args=(),        # tuple of example positional inputs
    kwargs=None,    # dict of example keyword inputs
    dynamic_shapes=None,  # shape dynamism specification
    strict=True,    # True: strict tracing. False: non-strict (recommended)
)
```

### Dynamic shapes

By default, every dimension is static. `dynamic_shapes` maps inputs to symbolic
dimension specs. Dynamic rank (changing the number of dimensions) is not
supported.

```python
from torch.export import Dim, export

batch = Dim("batch", min=1, max=32)
seq_len = Dim("seq_len", min=1, max=2048)

exported = export(
    model.eval(),
    (example_input,),
    dynamic_shapes={"x": {0: batch, 1: seq_len}},
)
```

Rules:
- Keys match kwarg names or positional index
- Values map dimension indices → `Dim` objects
- Unspecified dimensions remain static
- `{}` for an input means all dims are static
- Same `Dim` object across inputs asserts equal size
- Linear relationships between dims: `{0: dx}`, `{0: 2 * dx}`

#### Dim.DYNAMIC and Dim.AUTO

Instead of explicit `Dim("name", min=, max=)`:

- **`Dim.DYNAMIC`**: infer range automatically. **Errors** if the dimension
  turns out to be static (i.e., requires it to truly be dynamic).
- **`Dim.AUTO`**: infer range automatically. **Does not error** if the dimension
  is static. "Best effort" — marks dynamic if possible, keeps static otherwise.

Both often produce unbounded ranges (`int_oo` upper bound). Downstream
consumers that need concrete upper bounds for memory planning will fail. Always
prefer explicit `Dim(min=, max=)`.

#### ShapesCollection

For complex nested inputs (dicts, dataclasses), `ShapesCollection` lets you
assign dynamic shapes directly to tensor objects instead of mirroring the input
structure:

```python
sc = torch.export.ShapesCollection()
sc[tensor_x] = (dim, dim + 1, 8)     # tuple: per-dimension specs
sc[tensor_y] = {0: dim * 2}          # dict: only specified dims are dynamic
ep = export(M(), (args,), dynamic_shapes=sc)
```

#### 0/1 specialization

By default, the system specializes on dimensions of size 0 or 1. This avoids
complex guards for broadcasting and contiguity edge cases. Dynamic symbols
start with an implicit range of `[2, int_oo]` unless the `Dim` spec explicitly
includes 0 or 1.

This means: if your `Dim(min=1)` and the example input has size 1 on that
dimension, export may specialize it as static. Use `min=2` or larger, or
provide an example input with size > 1.

## The ExportedProgram

```python
exported = export(model.eval(), (x,))

print(exported.graph)            # torch.fx.Graph — the traced DAG
gm = exported.graph_module       # torch.fx.GraphModule wrapping the graph
out = exported.module()(x)       # run eagerly for correctness testing
```

### Key attributes

- `exported.graph_signature` — describes which graph inputs are user inputs,
  parameters, or buffers, and which outputs are buffer mutations
- `exported.range_constraints` — maps symbols (`s0`, `s1`) to their proven
  value ranges, reflecting guards from tracing
- `exported.state_dict` — parameters and buffers

### Unflattening

Export produces a flattened graph (all submodules inlined). To reconstruct the
original module hierarchy:

```python
from torch.export import unflatten
unflat = unflatten(exported)  # returns UnflattenedModule with original hierarchy
```

Module swaps on the unflattened module won't work unless you set
`preserve_module_call_signature` during export:

```python
ep = export(model, args, preserve_module_call_signature=("encoder",))
```

### Node metadata

Every `call_function` node has:
- `node.meta["val"]` — a `FakeTensor` or `SymInt` describing the output shape
  and dtype. This is the source of truth for shape information in the graph.
- `node.meta["stack_trace"]` — Python source location
- `node.meta["nn_module_stack"]` — which `nn.Module` the op came from
- `node.meta["source_fn_stack"]` — the original torch function before
  decomposition

### IR levels

Export produces three IR levels via `run_decompositions`:

| IR | How | Properties | Op count |
|----|-----|-----------|----------|
| Training IR | `export()` (default) | May contain mutations | ~3000 |
| Inference IR | `ep.run_decompositions(decomp_table={})` | Purely functional | ~2000 |
| Core ATen IR | `ep.run_decompositions(decomp_table=None)` | Highly decomposed | ~180 |

Training IR preserves in-place ops (`aten.add_`) for autograd compatibility.
Inference IR functionalizes them (`aten.add`). Core ATen IR decomposes further
(e.g., `conv2d` → `convolution`). Most deployment backends use Core ATen IR.

Use `torch.no_grad()` when producing inference or Core ATen IR:
```python
with torch.no_grad():
    inference_ep = exported.run_decompositions(decomp_table={})
```

Custom decompositions: start from the default table and override specific ops:
```python
decomp_table = torch.export.default_decompositions()

def _linear_decomp(input, weight, bias=None):
    out = input @ weight.T
    return out + bias if bias is not None else out

decomp_table[torch.ops.aten.linear.default] = _linear_decomp

with torch.no_grad():
    exported = exported.run_decompositions(decomp_table)
```

### Serialization

```python
torch.export.save(exported, "model.pt2")
loaded = torch.export.load("model.pt2")
```

The `.pt2` file is a zipfile containing the graph, state dict, and metadata.

## Control flow

How control flow is handled follows directly from the static/dynamic
classification of the condition:

### Static control flow

Branches on static values (Python primitives, static shapes) are resolved at
trace time. The taken branch is traced; the other is discarded. Loops over
static ranges are unrolled. This is transparent — no special handling needed.

### Shape-dependent control flow

When a condition involves **backed** dynamic shapes, the tracer uses the hint
value to evaluate it, traces the taken branch, and emits a **guard**. The guard
must be provable from the `Dim` spec.

```python
def forward(self, x):
    if x.size(0) > 4:      # guard: s0 > 4
        return x.sin()
    return x.cos()
```

If `Dim("batch", min=2, max=32)`, the solver proves `s0 > 4` is not always
true → `ConstraintViolationError`. Fix: use `min=5`, or restructure to
avoid the branch, or use `torch.cond`.

### Data-dependent control flow

When a condition involves **unbacked** dynamic shapes (from `.item()`,
`.nonzero()`, etc.), the tracer has no hint to evaluate it →
`GuardOnDataDependentSymNode`.

Two solutions:

**`torch._check`** — assert a fact about the unbacked symbol so the tracer can
continue on one branch. Creates a runtime assertion in the graph.

```python
nz = x.nonzero()
torch._check(nz.shape[0] > 0)    # refines u0 range to [1, inf]
if nz.shape[0] > 0:               # tracer now picks True
    return x.sin()
```

**`torch.cond`** — trace both branches. The graph contains both subgraphs and
selects at runtime. Use when you genuinely need both paths.

```python
return torch.cond(
    pred=x.sum() > 0,
    true_fn=lambda x: x.sin(),
    false_fn=lambda x: x.cos(),
    operands=(x,),
)
```

`torch.cond` lowers to `torch.ops.higher_order.cond` — a special node with two
`GraphModule` attributes for the branches. No closures; all captured values
become explicit operands. No mutations allowed in branches.

**`torch.map`** — trace a loop body over a dynamic number of iterations. Use for
data-dependent loops where unrolling isn't possible:

```python
return torch.map(
    f=lambda x_i: x_i.sin(),
    xs=(x,),  # map over first dim of each tensor
)
```

**`torch._check_is_size`** — like `torch._check`, but additionally marks the
unbacked symbol as "size-like." Size-like symbols are assumed to never be 0 or
1 under `guard_size_oblivious` checks, which eliminates many framework-internal
guards. Use when passing an unbacked symbol to a function that expects a tensor
dimension.

## Module state

Rules for module state during tracing:

- **Parameters**: read freely, **cannot be updated** during forward. Export with
  `torch.no_grad()` to avoid autograd-related updates.
- **Buffers**: read freely, **can be updated** (in-place or reassignment).
  Updated buffers are lifted as additional graph outputs.
- **Attributes of primitive type**: static — read freely, updates are burned in
  (no graph instructions emitted for gets/sets).
- **Attributes of Tensor type**: can be read but **cannot be updated** during
  forward. Register as a buffer if updates are needed.
- **All module state shapes are static** — parameter and buffer shapes cannot be
  dynamic.

## When export fails

Every failure traces back to the tracer encountering something the static graph
contract can't represent. Diagnose by asking: **is this a control flow problem,
a shape constraint problem, a side effect, or a missing kernel?**

### Shape constraint failures

The most common and subtle failures. When a dimension is symbolic, every
downstream shape operation generates guards. The solver must prove each guard
from the `Dim` min/max bounds. Most difficult failures involve `//` or `%` in
symbolic expressions (see Guards above).

When export raises `ConstraintViolationError`, the error message includes
suggested fixes. Apply them automatically:

```python
from torch.export.dynamic_shapes import refine_dynamic_shapes_from_suggested_fixes

try:
    ep = export(model, args, dynamic_shapes=ds)
except Exception as e:
    ds = refine_dynamic_shapes_from_suggested_fixes(str(e), ds)
    ep = export(model, args, dynamic_shapes=ds)
```

**Modular branches from padding/alignment logic.**
```
Not all values of seq_len satisfy ((1 + (seq_len // S)) % N) != 0
```
Models that pad or align to multiples of N create guards like
`if frames % N == 0`. The solver can't prove modular conditions for arbitrary
`Dim` values.
Fix: disable the alignment logic before export (e.g., set the pad-to-multiple
config to 0, or override the padding method to be a no-op). Alternatively,
constrain the `Dim` so the condition is always true.

**reshape with `-1` inference on symbolic dimensions.**
```
(D + D*(((-1) + L) // S)) == (1 + (((-1) + L) // S))
```
`reshape(b, t, -1)` needs the solver to prove `numel / (b*t)` is a whole
number. When `t` involves `//`, the solver can't.
Fix: `flatten(start_dim)` on *static* dimensions instead of `reshape`.
`x.permute(0, 2, 1, 3).flatten(2)` flattens dims 2+3 (both static) without
generating guards on the dynamic dimension.

**Linear/matmul on conv-derived symbolic expressions.**
```
(1 % (1 + (((-1) + L) // S))) != 0
```
After strided convolutions, the output dimension becomes
`1 + ((L-1)//S)`. Applying `nn.Linear` triggers a contiguity/divisibility
guard the solver can't prove. Known limitation — `torch._check`,
`.contiguous()`, `.clone()` don't help. The `//` in the symbolic expression is
the root cause. A plain `Dim` works fine with `Linear`; the issue is
specifically expressions produced by strided convolutions.

**view/reshape mixing two correlated dynamic dimensions.**
```
(((-1) + ((2*T*T) // T)) % T) != 0
```
`view(b, h, -1, T)` on a `(b, h, T, 2T)` tensor creates numel `2*T*T`.
The `-1` resolves to `(2*T*T)//T` which the solver can't simplify to `2*T`.
This arises in relative position attention where pad+view is used to shift
position scores across two sequence-length-dependent dimensions.
Fix: replace the pad+reshape+slice pattern with `torch.gather` using
explicitly computed indices:
```python
arange = torch.arange(seq_len, device=x.device)
idx = arange.unsqueeze(0) - arange.unsqueeze(1) + (seq_len - 1)
idx = idx.unsqueeze(0).unsqueeze(0).expand(b, h, -1, -1)
result = torch.gather(x, 3, idx)  # (b, h, T, T) — no view needed
```
The gather operates element-wise without view guards. Using explicit
dimensions in `reshape` (no `-1`) doesn't help because the stride
computation in the view algorithm also generates unprovable guards.

### Side effect failures

The graph captures a pure function. Conditional buffer mutations, distributed
calls, or in-place state updates during `forward()` break this contract.

```python
def forward(self, x):
    if x.size(1) > self.max_len:
        self.pos_enc = make_pos_enc(x.size(1))   # conditional mutation
```

Fix: pre-compute state at maximum size before export, then no-op the updater
so `forward()` is pure. Note that *unconditional* buffer updates are supported
— the graph lifts them as additional outputs.

### Missing fake kernels

Custom operators need a fake (meta) implementation that computes output shapes
without data:

```python
@torch.library.register_fake("mylib::custom_op")
def _(x):
    return torch.empty_like(x)

# For data-dependent output shapes:
@torch.library.register_fake("mylib::custom_nonzero")
def _(x):
    nnz = torch.library.get_ctx().new_dynamic_size()  # unbacked symbol
    return x.new_empty([nnz, x.dim()], dtype=torch.int64)
```

### Making untraceable code traceable

When the problematic code is in a module you can't modify:

**Wrapper modules** — present a traceable `forward()` around a subgraph:

```python
class EncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder
        self.proj = model.projection

    def forward(self, x, length):
        enc_out, _ = self.encoder(x=x, length=length)
        return self.proj(enc_out)
```

**Re-implementing untraceable layers** — when a layer's `forward()` has
data-dependent masking, generator-based loops, or other untraceable patterns,
re-implement the forward logic in the wrapper. Extract sub-modules as
`nn.ModuleList` so weights remain in the graph:

```python
class ConvWrapper(nn.Module):
    def __init__(self, conv_block):
        super().__init__()
        self.conv_layers = nn.ModuleList(list(conv_block.layers))
        self.out = conv_block.out

    def forward(self, x):
        x = x.unsqueeze(1)
        for layer in self.conv_layers:
            x = layer(x)
        x = x.permute(0, 2, 1, 3).flatten(2)
        return self.out(x)
```

The key question: *which operations in the original forward() actually matter
for inference?* Training-only logic (masking for batched inputs, dynamic
padding for variable batches, gradient checkpointing) can be dropped for
single-sample inference.

**Monkey-patching individual methods** — when the problem is a single method
buried deep in a module hierarchy, wrapping the outer module doesn't help.
Use `types.MethodType` to surgically replace just that method on each
affected sub-module:

```python
import types

def _fixed_method(self, x):
    # export-safe reimplementation
    ...

for layer in model.encoder.layers:
    layer.attn.problematic_method = types.MethodType(
        _fixed_method, layer.attn
    )
```

This preserves the module hierarchy and all weights — only the forward logic
of the targeted method changes. Useful when the same problematic pattern
repeats across many layers (e.g., attention layers in a transformer stack).

**Overriding device-dependent branches** — some models
have `if torch.cuda.is_available():` branches that route to CUDA-specific
code paths. During tracing on CPU, these branches can still cause issues if
the tracer sees both paths. Override before export:

```python
old = torch.cuda.is_available
torch.cuda.is_available = lambda: False
ep = export(wrapper, args, strict=False)
torch.cuda.is_available = old
```

## Debugging

### draft_export

When export fails repeatedly and you want to see all issues at once instead of
fixing them one by one:

```python
ep = torch.export.draft_export(model, args)
print(ep._report)    # detailed report with per-issue debug info
```

`draft_export` always produces a graph, even with soundness issues. It uses
**real tensor tracing** alongside fake tensors — when the fake tracer can't
evaluate a condition, it falls back to the real tensor value and emits a runtime
assert. It also catches missing fake kernels (using real kernels as fallback)
and incorrect fake kernels (by comparing real vs fake outputs). Not for
production — for diagnosis.

### TORCH_LOGS and tlparse

Enable verbose logging and parse into an HTML report:
```bash
TORCH_LOGS="+dynamo,+export" python export_script.py 2>&1 | tlparse
```

The report shows guard failures, graph breaks, symbol creation, and the traced
graph. Key log patterns:
- `create_symbol s0 = 5 for L['x'].size()[0]` — symbol allocated with hint
- `eval Eq(s0, 5) [guard added]` — guard generated from a condition
- `GuardOnDataDependentSymNode` — unbacked symbol hit a branch

For deeper debugging:
- `TORCHDYNAMO_EXTENDED_DEBUG_CREATE_SYMBOL="s0"` — full trace for one symbol
- `TORCHDYNAMO_EXTENDED_DEBUG_GUARD_ADDED="Eq(s0, 5)"` — trace a specific guard
- `TORCHDYNAMO_EXTENDED_DEBUG_CPP=1` — C++ backtraces for guards from kernels

### Incremental testing

Isolate failures by controlling one variable at a time:

1. **Static shapes first** (no `dynamic_shapes`). If this fails, the problem is
   control flow or side effects — not shape constraints.
2. **Add one dynamic dimension at a time**. The dimension that causes failure
   tells you which downstream shape expression is unprovable.
3. **Test each wrapper independently** before combining them.

```python
# Step 1: catches control flow + side effects
ep = export(wrapper, args, strict=False)

# Step 2: catches shape constraint violations
ep = export(wrapper, args, dynamic_shapes=ds, strict=False)
```

### Static shapes as deliberate fallback

When a dimension is fundamentally blocked from being dynamic — typically due
to `//` or `%` in the symbolic expression (e.g., conv-derived `1+((L-1)//S)`
hitting `nn.Linear`) — keeping that dimension static is a valid strategy, not
a failure.

Design the export around it: use a fixed input size for the affected method
and have the caller handle variable-length inputs via padding/truncation or
chunking. This is practical when the model already operates on fixed-size
chunks (e.g., streaming models with fixed chunk sizes per config).

```python
# Dynamic shapes blocked → use static shapes, caller pads to fixed size
ep = export(
    wrapper, (fixed_size_input, length),
    strict=False,   # no dynamic_shapes
)
```
